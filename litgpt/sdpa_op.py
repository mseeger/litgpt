from typing import Optional, List, Union

import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from litgpt.attention_utils import (
    filter_sdpa_kernels,
    attention_compute_scores,
    minus_infinity,
    build_mask_cache,
    build_mask_slice,
    mask_cache_bool,
    mask_slice_bool,
)


class SDPAFunction(Function):
    """
    Provides `scaled_dot_product_attention` as an `autograd` operator,
    ensuring that only its inputs are stored in the `autograd` graph, not the
    intermediates.

    """
    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_positions: Optional[torch.Tensor],
        input_pos: int,
        scale_factor: float,
        sliding_window_size: Optional[int] = None,
        sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]] = None,
    ) -> torch.Tensor:
        # Check dimensions
        assert query.ndim == 4 and key.ndim == 4 and value.ndim == 4
        assert key.shape == value.shape
        batch_size, n_query_groups, kv_len, head_size = key.shape
        assert query.shape[0] == batch_size and query.shape[3] == head_size
        _, n_head, q_len, _ = query.shape
        assert q_len <= kv_len
        assert n_query_groups <= n_head and n_head % n_query_groups == 0
        is_causal = input_pos == 0
        if is_causal:
            assert q_len == kv_len
            assert token_positions is None
        else:
            assert token_positions is not None
            assert token_positions.shape == key.shape[:-1]
        # Computations are done in `float32`
        query = query.to(dtype=torch.float32)
        key = key.to(dtype=torch.float32)
        value = value.to(dtype=torch.float32)
        mask_kwargs = dict(dtype=torch.float32, device=query.device)
        if is_causal:
            if sliding_window_size is not None:
                attn_mask = build_mask_cache(
                    max_seq_length=kv_len,
                    sliding_window_size=sliding_window_size,
                    **mask_kwargs,
                ).view(1, 1, kv_len, kv_len)
            else:
                attn_mask = None
        else:
            attn_mask = build_mask_slice(
                input_pos=input_pos,
                num=q_len,
                token_positions=token_positions,
                n_head=n_head,
                **mask_kwargs,
                sliding_window_size=sliding_window_size,
            )
        # Run the right version of `F.scaled_dot_product_attention`
        kwargs = dict(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=scale_factor,
            is_causal=attn_mask is None,
            enable_gqa=n_query_groups < n_head,
        )
        if sdpa_kernels is not None:
            if not isinstance(sdpa_kernels, list):
                sdpa_kernels = [sdpa_kernels]
            # Filter out kernels which are not supported
            sdpa_kernels = filter_sdpa_kernels(sdpa_kernels, **kwargs)
            with sdpa_kernel(sdpa_kernels):
                y = F.scaled_dot_product_attention(**kwargs)
        else:
            y = F.scaled_dot_product_attention(**kwargs)

        return y.to(dtype=key.dtype)

    @staticmethod
    def setup_context(ctx, inputs, output):
        query, key, value, token_positions, input_pos, scale_factor, sliding_window_size, sdpa_kernels = inputs
        ctx.save_for_backward(query, key, value, token_positions)
        ctx.extra_args = dict(
            input_pos=input_pos,
            scale_factor=scale_factor,
            sliding_window_size=sliding_window_size,
        )

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        # Inputs from context
        query, key, value, token_positions = ctx.saved_tensors
        input_pos = ctx.extra_args["input_pos"]
        scale_factor = ctx.extra_args["scale_factor"]
        sliding_window_size = ctx.extra_args["sliding_window_size"]
        is_causal = input_pos == 0
        grad_query = grad_key = grad_value = None
        # Prepare inputs
        device = query.device
        dtype = query.dtype
        query = query.to(dtype=torch.float32)
        key = key.to(dtype=torch.float32)
        value = value.to(dtype=torch.float32)
        need_query = ctx.needs_input_grad[0]
        need_key = ctx.needs_input_grad[1]
        need_value = ctx.needs_input_grad[2]
        if need_query or need_key or need_value:
            # We will use two buffers `tmp_array1`, `tmp_array2` of shape
            # `(batch_size, n_head, q_len, kv_len)`, which is in general the
            # largest size. We make sure that no extra copies of this size
            # are created.
            batch_size, n_head, q_len, head_size = query.shape
            _, n_query_groups, kv_len, _ = key.shape
            q_per_kv = n_head // n_query_groups
            if grad_y.shape != query.shape:
                raise ValueError(f"grad_y.shape = {grad_y.shape}, query.shape = {query.shape}, must be the same")
            grad_y = grad_y.to(dtype=torch.float32)
            # Compute attention weights f(S)
            tmp_array1 = attention_compute_scores(
                query=query, key=key,
            )
            tmp_array1 *= scale_factor  # S without masking
            # Attention masking
            if is_causal:
                bool_mask = mask_cache_bool(
                    max_seq_length=kv_len,
                    sliding_window_size=sliding_window_size,
                    device=device,
                    dtype=dtype,
                ).bool()[None, None, :, :].expand_as(tmp_array1)
            else:
                bool_mask = mask_slice_bool(
                    input_pos=input_pos,
                    num=q_len,
                    token_positions=token_positions,
                    n_head=n_head,
                    device=device,
                    sliding_window_size=sliding_window_size,
                )
                assert bool_mask.shape == tmp_array1.shape
            tmp_array1[bool_mask] = minus_infinity(dtype=tmp_array1.dtype)  # S
            # Softmax
            tmp_array2 = F.softmax(tmp_array1, dim=-1)  # f(S)
            if need_value:
                # Avoid transpose of `tmp_array2`, which may create copy
                grad_value = torch.matmul(grad_y.mT, tmp_array2)
                if q_per_kv > 1:
                    grad_value = grad_value.view(
                        batch_size, n_query_groups, q_per_kv, head_size, kv_len,
                    ).sum(dim=2)
                grad_value = grad_value.mT.to(dtype=dtype)
            if need_query or need_key:
                if q_per_kv == 1:
                    torch.matmul(grad_y, value.mT, out=tmp_array1)
                else:
                    q_shape = (batch_size, n_query_groups, q_per_kv, q_len, head_size)
                    _arg1 = grad_y.view(*q_shape)
                    _arg2 = value.unsqueeze(2).mT
                    o_shape = q_shape[:3] + (q_len, kv_len)
                    # _arg1: (bs, nh_k, q_per_kv, q_len, hs)
                    # _arg2: (bs, nh_k, 1, hs, kv_len)
                    torch.matmul(
                        _arg1,
                        _arg2,
                        out=tmp_array1.view(*o_shape)
                    )
                tmp_array1 *= tmp_array2
                tmp_array2 *= tmp_array1.sum(dim=-1, keepdim=True)  # (diag e) f(S)
                tmp_array1 -= tmp_array2  # E
                if need_query:
                    # Compute matmul(E, K)
                    if q_per_kv == 1:
                        grad_query = torch.matmul(tmp_array1, key)
                    else:
                        e_shape = (batch_size, n_query_groups, q_per_kv, q_len, kv_len)
                        _arg1 = tmp_array1.view(*e_shape)
                        _arg2 = key.unsqueeze(2)
                        # _arg1: (bs, nh_k, q_per_kv, q_len, kv_len)
                        # _arg2: (bs, nh_k, 1, kv_len, hs)
                        grad_query = torch.matmul(_arg1, _arg2).view(*query.shape)
                    grad_query *= scale_factor
                    grad_query = grad_query.to(dtype=dtype)
                if need_key:
                    # Compute matmul(E.mT, Q)
                    # Avoid transpose of `tmp_array1`, which may create copy
                    grad_key = torch.matmul(query.mT, tmp_array1)
                    if q_per_kv > 1:
                        grad_key = grad_key.view(
                            batch_size, n_query_groups, q_per_kv, head_size, kv_len,
                        ).sum(dim=2)
                    grad_key *= scale_factor
                    grad_key = grad_key.mT.to(dtype=dtype)

        return grad_query, grad_key, grad_value, None, None, None, None, None

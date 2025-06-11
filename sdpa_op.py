from typing import Optional, List, Union

import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from litgpt.attention import (
    reorder_keys_values,
    needs_reordering_keys_values,
    filter_sdpa_kernels,
    _attention_compute_scores,
    _minus_infinity,
)


class SDPAFunction(Function):
    """
    Provides `scaled_dot_product_attention` in the basic `is_causal=True`
    variant as an `autograd` operator, ensuring that only its inputs are
    stored in the `autograd` graph, not the intermediates.

    """
    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_positions: torch.Tensor,
        input_pos: int,
        scale_factor: float,
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
        assert token_positions.shape == key.shape[:-1]
        # Prepare inputs (reordering, casting)
        query, key, value, _ = SDPAFunction._prepare_inputs(
            query, key, value, token_positions, input_pos,
        )
        # Run the right version of `F.scaled_dot_product_attention`
        kwargs = dict(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
            dropout_p=0.0,
            scale=scale_factor,
            is_causal=True,
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
    def _prepare_inputs(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_positions: torch.Tensor,
        input_pos: int,
    ):
        _, _, q_len, head_size = query.shape
        # Reordering (if necessary)
        if needs_reordering_keys_values(
            input_pos, q_len, token_positions,
        ):
            reorder_index = reorder_keys_values(
                input_pos, q_len, token_positions,
            ).unsqueeze(-1).expand(-1, -1, -1, head_size)
            key = key.gather(dim=2, index=reorder_index)
            value = value.gather(dim=2, index=reorder_index)
        else:
            reorder_index = None
        # Computations are done in `float32`
        query = query.to(dtype=torch.float32)
        key = key.to(dtype=torch.float32)
        value = value.to(dtype=torch.float32)
        return query, key, value, reorder_index

    @staticmethod
    def setup_context(ctx, inputs, output):
        query, key, value, token_positions, input_pos, scale_factor, sdpa_kernels = inputs
        ctx.save_for_backward(query, key, value, token_positions)
        ctx.mark_non_differentiable(token_positions)
        ctx.extra_args = dict(
            input_pos=input_pos,
            scale_factor=scale_factor,
        )

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        # Inputs from context
        query, key, value, token_positions = ctx.saved_tensors
        input_pos = ctx.extra_args["input_pos"]
        scale_factor = ctx.extra_args["scale_factor"]
        grad_query = grad_key = grad_value = None
        # Prepare inputs
        device = query.device
        dtype = query.dtype
        query, key, value, reorder_index = SDPAFunction._prepare_inputs(
            query, key, value, token_positions, input_pos,
        )
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
            tmp_array1 = _attention_compute_scores(
                query=query, key=key,
            )
            tmp_array1 *= scale_factor  # S without masking
            # Causal masking
            # If the mask is (i, j), i over Q, j over K:
            # - If same unit: q_i depends on k_j for i >= j.
            #   So mask out i < j
            # - Q, K are aligned at end: If offset = kv_len - q_len, then mask
            #   out i + offset < j
            offset = kv_len - q_len
            kwargs = dict(device=device, dtype=torch.int)
            mask_index = torch.arange(offset, kv_len, **kwargs).unsqueeze(-1) < torch.arange(kv_len, **kwargs).unsqueeze(0)
            tmp_array1[
                mask_index.view(1, 1, -1, -1).expand_as(tmp_array1)
            ] = _minus_infinity(dtype=tmp_array1.dtype)  # S
            # Softmax
            tmp_array2 = F.softmax(tmp_array1, dim=-1)  # f(S)
            if need_value:
                # Avoid transpose of `tmp_array2`, which may create copy
                grad_value = torch.matmul(grad_y.mT, tmp_array2).mT.to(dtype=dtype)
                if reorder_index is not None:
                    grad_value = torch.scatter(
                        grad_value, dim=2, index=reorder_index, src=grad_value,
                    )
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
                if need_key:
                    # Compute matmul(E.mT, Q)
                    # Avoid transpose of `tmp_array1`, which may create copy
                    grad_key = torch.matmul(query.mT, tmp_array1)
                    if q_per_kv > 1:
                        grad_key = grad_key.view(
                            batch_size, n_query_groups, q_per_kv, head_size, kv_len,
                        ).sum(dim=2)
                    grad_key = grad_key.mT.to(dtype=dtype)
                    if reorder_index is not None:
                        grad_key = torch.scatter(
                            grad_key, dim=2, index=reorder_index, src=grad_key,
                        )

        return grad_query, grad_key, grad_value, None, None, None, None

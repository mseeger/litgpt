import math
from typing import Optional, Tuple, List, Union

import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from litgpt.attention import (
    reorder_keys_values,
    needs_reordering_keys_values,
    filter_sdpa_kernels,
)


class SDPAFunction(Function):
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
        batch_size, n_query_groups, kv_len, head_size = key.shape
        assert key.shape == value.shape
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
        query, key, value, _ = SDPAFunction._prepare_inputs(
            query, key, value, token_positions, input_pos,
        )
        need_query = ctx.needs_input_grad[0]
        need_key = ctx.needs_input_grad[1]
        need_value = ctx.needs_input_grad[2]
        if need_query or need_key or need_value:
            # We will use two buffers of shape
            # `(batch_size, n_head, q_len, kv_len)`, which is in general the
            # largest size
            batch_size, n_head, q_len, _ = query.shape
            _, n_query_groups, kv_len, _ = key.shape
            # Compute attention weights f(S)
            # HIER!

        return grad_query, grad_key, grad_value, None, None, None, None

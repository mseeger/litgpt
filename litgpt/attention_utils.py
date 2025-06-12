from typing import List, Optional

import torch
from torch.backends.cuda import (
    can_use_flash_attention,
    can_use_efficient_attention,
    can_use_cudnn_attention,
)
from torch.nn.attention import SDPBackend, SDPAParams


def needs_reordering_keys_values(
    input_pos: int,
    num: int,
    token_positions: torch.Tensor,
) -> bool:
    """
    See :func:`reorder_keys_values`. Here, we check some common conditions
    under which the ordering already supports `is_causal=True`. If this is
    the case, we return `False` (no reordering is needed).

    Args:
        input_pos: Position of first new token
        num: Number of new tokens 'q_len`
        token_positions: Token positions in KV cache

    Returns:
        Do keys and values buffers have to be reordered?

    """
    if num == 1:
        return False
    should_be = torch.arange(
        input_pos,
        input_pos + num,
        device=token_positions.device,
        dtype=token_positions.dtype,
    ).view(1, 1, -1).expand(*token_positions.shape[:-1], -1)
    return (token_positions[:, :, (-num):] != should_be).any().sum().item() > 0


def reorder_keys_values(
    input_pos: int,
    num: int,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    """
    In order to call `F.scaled_dot_product_attention` with `is_causal=True`,
    the key and value buffers must be ordered such that the final `num`
    entries correspond to the most recent tokens
    `input_pos:(input_pos + num)`. Here, we create an index of the same
    shape as `token_positions`, which performs such a reordering.

    Args:
        input_pos: Position of first new token
        num: Number of new tokens 'q_len`
        token_positions: Token positions in KV cache

    Returns:
        Index to reorder key and value buffers

    """
    assert token_positions.ndim == 3
    batch_size, n_query_groups, cache_length = token_positions.shape
    assert cache_length >= num
    reorder_index = torch.empty_like(token_positions)
    dtype = token_positions.dtype
    # Distinguish between token positions >= input_pos (1), and the rest
    # (2). For (1), each token_position[b, h, :] is a permutation of
    # input_pos:(input_pos + num). We use their argsort shifted to
    # (cache_length - num):cache_length. The content of (2) does not
    # matter, we map each token_position[b, h, :] to 0:(cache_length - num).
    is_new = token_positions >= input_pos
    thresh = cache_length - num
    temp_pos = token_positions[is_new]
    should_be = batch_size * n_query_groups * num
    assert temp_pos.numel() == should_be, f"{temp_pos.numel()} token_positions entries >= {input_pos}, should be {should_be}"
    sort_ind = temp_pos.view(
        batch_size, n_query_groups, -1,
    ).argsort(dim=-1) + thresh
    reorder_index[is_new] = sort_ind.flatten().to(dtype=dtype)
    reorder_index[~is_new] = torch.arange(
        thresh, dtype=dtype, device=token_positions.device,
    ).unsqueeze(0).expand(batch_size * n_query_groups, -1).flatten()
    return reorder_index


def filter_sdpa_kernels(
    sdpa_kernels: List[SDPBackend],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    enable_gqa: bool,
    **kwargs,
) -> List[SDPBackend]:
    params = SDPAParams(
        query, key, value, attn_mask, dropout_p, is_causal, enable_gqa
    )
    new_kernels = []
    for kernel in sdpa_kernels:
        if kernel == SDPBackend.FLASH_ATTENTION and not can_use_flash_attention(params):
            continue
        elif kernel == SDPBackend.EFFICIENT_ATTENTION and not can_use_efficient_attention(params):
            continue
        elif kernel == SDPBackend.CUDNN_ATTENTION and not can_use_cudnn_attention(params):
            continue
        new_kernels.append(kernel)
    return new_kernels


def attention_compute_scores(
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    assert query.ndim == key.ndim == 4
    assert query.shape[0] == key.shape[0] and query.shape[3] == key.shape[3]
    nh_q = query.shape[1]
    nh_k = key.shape[1]
    assert nh_q % nh_k == 0
    # - query: (bs, nh_q, T_q, hs)
    # - key: (bs, nh_k, T_k, hs)
    q_per_kv = nh_q // nh_k
    key_transposed = key.mT  # (bs, nh_k, hs, T_k)
    if q_per_kv == 1:
        return query @ key_transposed
    else:
        assert q_per_kv > 1
        q_shape = query.shape[:1] + (nh_k, q_per_kv) + query.shape[2:]
        _query = query.view(*q_shape)
        key_transposed = key_transposed.unsqueeze(2)
        # At this point:
        # - _query: (bs, nh_k, q_per_kv, T_q, hs)
        # - key_transposed: (bs, nh_k, 1, hs, T_k)
        # - scores: (bs, nh_k, q_per_kv, T_q, T_k)
        scores = torch.matmul(_query, key_transposed)
        s_shape = query.shape[:-1] + (key.shape[2],)
        return scores.view(*s_shape)


def attention_compute_weighted_values(
    scores: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    assert scores.ndim == value.ndim == 4
    assert scores.shape[0] == scores.shape[0] and scores.shape[3] == value.shape[2]
    nh_q = scores.shape[1]
    nh_k = value.shape[1]
    assert nh_q % nh_k == 0
    # - scores: (bs, nh_q, T_q, T_k)
    # - value: (bs, nh_k, T_k, hs)
    q_per_kv = nh_q // nh_k
    if q_per_kv == 1:
        return scores @ value
    else:
        s_shape = scores.shape[:1] + (nh_k, q_per_kv) + scores.shape[2:]
        _scores = scores.view(*s_shape)
        _value = value.unsqueeze(2)
        # At this point:
        # - _scores: (bs, nh_k, q_per_kv, T_q, T_k)
        # - _value: (bs, nh_k, 1, T_k, hs)
        # - result: (bs, nh_k, q_per_kv, T_q, hs)
        result = torch.matmul(_scores, _value)
        r_shape = scores.shape[:-1] + (value.shape[-1],)
        return result.view(*r_shape)


def minus_infinity(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


def build_mask_cache(
    max_seq_length: int,
    sliding_window_size: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
          Global Window              Sliding window             Sliding window
          attention mask      +            bias          =      attention mask
    ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
    │ True False False False │  │ True  True  True True │  │ True  False False False │
    │ True True  False False │  │ True  True  True True │  │ True  True  False False │
    │ True True  True  False │  │ False True  True True │  │ False True  True  False │
    │ True True  True  True  │  │ False False True True │  │ False False True  True  │
    └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
    """
    # Usual causal mask:
    mask = torch.ones(
        max_seq_length, max_seq_length, device=device, dtype=dtype,
    ).triu(diagonal=1)
    if sliding_window_size is not None:
        mask += torch.ones_like(mask).tril(diagonal=-sliding_window_size)
    mask.masked_fill_(mask.bool(), minus_infinity(dtype))
    return mask


def build_mask_slice(
    input_pos: int,
    num: int,
    token_positions: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns mask for case `input_pos > 0` in :class:`MultiHeadSelfAttention`.

    Args:
        input_pos: Position in input sequence, must be positive
        num: Length of query argument `q_len`
        token_positions: Token positions in KV cache, shape
            `(eff_batch_size, n_query_groups, cache_length)`
        dtype: Data type of the output mask
        device: Device of the output mask
        sliding_window_size: Size of sliding window (if any)

    Returns:
        Mask tensor, shape `(eff_batch_size, n_query_groups, num, cache_length)`

    """
    # Build boolean mask, then map False -> 0, True -> -infty
    # If (i, j) indexes the complete (seq_len, seq_len) mask matrix,
    # causality is given by I(i < j). If `sliding_window_size` is given,
    # this translates to I(i >= j + sws) if sws = sliding_window_size.
    assert token_positions.ndim == 3
    tp_dtype = token_positions.dtype
    token_positions = token_positions.unsqueeze(2).to(device=device)
    kwargs = dict(device=device, dtype=tp_dtype)
    bool_mask = torch.arange(
        input_pos, input_pos + num, **kwargs,
    ).view(1, 1, -1, 1) < token_positions
    if sliding_window_size is not None:
        extra_mask = torch.arange(
            input_pos - sliding_window_size,
            input_pos + num - sliding_window_size,
            **kwargs,
        ).view(1, 1, -1, 1) >= token_positions
        bool_mask += extra_mask
    mask = torch.zeros(bool_mask.shape, dtype=dtype, device=device)
    mask.masked_fill_(bool_mask, minus_infinity(dtype))
    return mask

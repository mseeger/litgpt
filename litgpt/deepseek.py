import math
from typing import Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from litgpt.config import Config as BaseConfig
from litgpt.kvcache import KVCache, DefaultKeysAndValues
from litgpt.model import apply_rope


@dataclass
class Config(BaseConfig):
    """
    Additional parameters of multi-head latent attention (MLA)

    Note that `head_size` is `d_h'`.
    """
    kv_low_rank: int = 512  # d_c
    q_low_rank: int = 1024  # d_c'
    rope_head_size: int = 64  # d_r


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__()
        # Maps input X to low-rank C_KV, K_R, C_Q
        self.qkv_encode = nn.Linear(
            config.n_embd,
            config.kv_low_rank + config.rope_head_size + config.q_low_rank,
            bias=config.bias or config.attn_bias,
        )
        # Maps C_Q to Q-equivalent input to dot products
        self.q_decode = nn.Linear(
            config.q_low_rank,
            config.n_head * (config.kv_low_rank + config.rope_head_size),
            bias=config.bias,
        )
        # Maps output of dot product attention to U
        self.v_proj = nn.Linear(
            config.kv_low_rank,
            config.n_head * config.head_size,
            bias=config.bias,
        )
        # Output projection
        self.output_proj = nn.Linear(
            config.head_size * config.n_head,
            config.n_embd,
            bias=config.bias or config.attn_bias,
        )
        # KV cache (needed for inference)
        self.kv_cache = kv_cache
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None and
            block_idx % config.sliding_window_layer_stride == 0
        )
        if config.norm_qk:
            self.norm_q = config.norm_class(config.q_low_rank, eps=config.norm_eps)
            self.norm_k = config.norm_class(config.kv_low_rank, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None
        self.config = config
        self.block_idx = block_idx

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        input_pos: Optional[int] = None,
        for_prefill: bool = False,
        mask_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            cos: RoPE parameters
            sin: RoPE parameters
            input_pos: See :meth:`GPT.forward`
            for_prefill: See :meth:`GPT.forward`
            mask_cache: Used for building mask in special case

        Returns:
            Output tensor
        """
        head_size = self.config.head_size
        n_head = self.config.n_head
        q_low_rank = self.config.q_low_rank
        kv_low_rank = self.config.kv_low_rank
        rope_head_size = self.config.rope_head_size
        qk_head_size = kv_low_rank + rope_head_size

        B, T, _ = x.size()  # batch size, sequence length, n_embd
        if input_pos is not None:
            if self.kv_cache is None or self.kv_cache.next_token_pos is None:
                raise ValueError("If input_pos is given, KV cache must exist and be initialized (call with for_prefill=True)")
            if self.kv_cache.next_token_pos != input_pos:
                raise ValueError(f"input_pos = {input_pos} != {self.kv_cache.next_token_pos} = kv_cache.next_token_pos")
            if self.kv_cache.max_tokens_forward < T:
                raise ValueError(
                    f"T = {T}, must be <= max_tokens_forward = {self.kv_cache.max_tokens_forward}")
            for_prefill = False
        elif for_prefill:
            # Sanity check
            if self.kv_cache is None:
                raise ValueError("If for_prefill is True, KV cache must exist")

        # Initial map to [C_KV, K_R, C_Q]
        latents = self.qkv_encode(x)
        c_kv, k_r, c_q = latents.split(
            (kv_low_rank, rope_head_size, q_low_rank), dim=-1
        )
        if self.config.norm_qk:
            c_kv = self.norm_k(c_kv)
            c_q = self.norm_q(c_q)

        # Map to QA equivalent
        q_equiv = self.q_decode(c_q)
        q_nope, q_r = q_equiv.split((kv_low_rank, rope_head_size), dim=-1)

        # RoPE
        k_r = apply_rope(k_r, cos, sin)
        q_r = apply_rope(q_r, cos, sin)

        # Reshape and transpose
        k_equiv = torch.cat((c_kv, k_r), dim=-1).view(B, 1, T, qk_head_size)
        q_equiv = torch.cat((q_nope, q_r), dim=-1).view(
            B, T, n_head, qk_head_size).transpose(1, 2)
        v_equiv = c_kv.view(B, 1, T, head_size)

        if input_pos is not None:
            # Extend KV cache and retrieve key, value tensors to be used.
            # Note: A KV cache for MLA only stores the K equivalent tensor, the
            # second argument is ignored
            k_and_v = self.kv_cache(k_equiv, v_equiv)
            # k: (B, n_head, cache_length, qk_head_size)
        else:
            if for_prefill:
                # Prefill KV cache
                # Only the key argument is really used, the other is ignored
                self.kv_cache.prefill(key=k_equiv, value=v_equiv)
            # Only k is really used
            k_and_v = DefaultKeysAndValues(k_equiv, v_equiv)

        # HIER!!
        # We need the attention mask if there is sliding window attention,
        # or if `input_pos` is given and T > 1.
        is_causal = input_pos is None
        use_mask = self.apply_sliding_window_attention or (not is_causal and T > 1)
        mask = None
        if use_mask:
            # Special case requires building a mask. `mask_cache` is only needed
            # then.
            assert mask_cache is not None, "mask_cache must be given if sliding window attention is used, or if input_pos given and T > 1"
            if is_causal:
                mask = mask_cache[:T, :T].view(1, 1, T, T)
                is_causal = False
            else:
                # We need a mask if T > 1, since inference needs to be causal
                # for the new tokens
                mask = batched_index_select(
                    mask_cache[:, input_pos:(input_pos + T)],
                    dim=0,
                    idx=self.kv_cache.token_positions(),
                )

        # Efficient attention using Flash Attention CUDA kernels.
        # NOTE: efficient implementation is disabled if `mask` is not None or softcapping is enabled.
        # ↓ (B, nh, T, hs) @ (B, nh, T, hs).mT --> (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        return_scores = input_pos is not None and self.kv_cache.update_requires_attn_weights()
        y, scores = self.scaled_dot_product_attention(
            q, k_and_v, mask, is_causal, return_scores
        )

        if return_scores:
            # Pass attention weights to KV cache
            self.kv_cache.update(attn_weights=scores)

        # Re-assemble all head outputs side by side.
        y = y.reshape(B, T, head_size * n_head)

        # Output projection.
        return self.proj(y)  # (B, T, C)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k_and_v: KeysAndValues,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert mask is None or not is_causal, "Cannot have mask and is_causal=True"
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

        # with softcapping we cannot use SDPA
        if return_scores or self.config.attention_logit_softcapping is not None or not k_and_v.both_in_parallel():
            y, scores = scaled_dot_product_attention(
                query=q,
                k_and_v=k_and_v,
                scale=scale,
                mask=mask,
                attention_logit_softcapping=self.config.attention_logit_softcapping,
                is_causal=is_causal,
            )
            if not return_scores:
                scores = None
        else:
            # We need `key` and `value` at the same time here. For the training
            # use case, this will be the case, since `k_and_v` is the default
            # in this case.
            key = k_and_v.keys()
            value = k_and_v.values()
            for retry in range(2):
                try:
                    y = F.scaled_dot_product_attention(
                        query=q,
                        key=key,
                        value=value,
                        attn_mask=mask,
                        dropout_p=0.0,
                        scale=scale,
                        is_causal=is_causal,
                    )
                    break
                except RuntimeError as ex:
                    if retry == 1 or self.config.n_query_groups == self.config.n_head:
                        raise ex  # Re-throw
                    # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention
                    # `scaled_dot_product_attention` is supposed to support
                    # `query.shape = (bs, nh_q, ...), key.shape = (bs, nh_k, ...)`
                    # and `nh_k < nh_q` if `nh_q` is a multiple of `nh_k`. But
                    # this seems not yet supported, so have to lift K, V here.
                    # This is annoying, as it wastes memory.
                    q_per_kv = self.config.n_head // self.config.n_query_groups
                    key = k_and_v.keys().repeat_interleave(q_per_kv, dim=1)
                    value = k_and_v.values().repeat_interleave(q_per_kv, dim=1)
            scores = None
        return y.transpose(1, 2), scores

    def create_default_kv_cache(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: Optional[int] = None,
    ):
        self.kv_cache = DenseKVCache(
            config=self.config,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with legacy checkpoints."""

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.{attr}"
            current_key = f"{prefix}qkv.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(state_dict.pop(legacy_key), self.config)

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

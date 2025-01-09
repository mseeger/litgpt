from typing import Optional, Tuple

import torch

from litgpt import Config
from litgpt.kvcache.base import KVCacheParams
from litgpt.kvcache.attn_weights import AttnWeightsKVCache
from litgpt.kvcache.utils import bitsize_of, bits_for_torch_dtype


class H2OKVCache(AttnWeightsKVCache):
    """
    Implements some variants of the heavy hitter oracle (H2O) KV cache, see

        Zhang et al
        H2O: Heavy-hitter oracle for efficient generative inference of large language models
        Advances in Neural Information Processing Systems 37, 2024
        https://openreview.net/forum?id=RkRrPp7GKO

    Our implementation contains some improvements over their code:

    * They average scores over the batch dimension and occupy slots
      independent of the batch dimension. We make eviction decisions
      for each batch entry independently, which is not more expensive
    * They sum scores over all rounds, which may favor earlier tokens.
      We allow this as well, but also support normalization of
      cumulative scores by the number of rounds a token is in the
      cache (if `normalize_scores=True`).

    The original H2O method as published is provided in
    :class:`H2OOriginalKVCache`.
    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        cache_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        normalize_scores: bool = False,
    ):
        super().__init__(config, batch_size, cache_length, device, dtype)
        self.normalize_scores = normalize_scores
        shape = (batch_size, self.n_query_groups, cache_length)
        self.register_buffer("scores", torch.zeros(shape, device=device, dtype=torch.float), persistent=False)
        # Length of key, value tensors in last recent :meth:`prefill` call. This
        # is needed for normalization
        self.prefill_length = None

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.next_position is not None:
            # Reset score values for slots where the new token will be written
            index = self.next_position.unsqueeze(-1)
            self.scores[:self.eff_batch_size, ...].scatter_(-1, index, 0.0)
        return super().forward(key, value)

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        # We do not compute H2O scores and cumulate scores for tokens inserted
        # by the prefill. Instead, we set the scores to 0. We also store the
        # length of key, value in `prefill_length`, so that the normalization
        # can be done properly, in that a token with position `t` entered the
        # cache in round `max(t, prefill_length - 1)`.
        super().prefill(key, value)
        self.prefill_length = self.current_length
        self.scores = 0.0

    def _update(self, attn_weights: torch.Tensor):
        # Scores are computed in `torch.float`. Also, deal with
        # `n_query_groups < n_head` by averaging
        attn_weights = self._average_attn_weights(attn_weights.to(torch.float))
        self.scores[:self.eff_batch_size, :, :self.current_length] += attn_weights
        if self.current_length == self.cache_length:
            # Set `next_position` to score minimizers
            scores = self.scores[:self.eff_batch_size, ...]
            if self.normalize_scores:
                # Normalize cumulative scores
                token_pos = self.token_pos[:self.eff_batch_size, ...]
                other = torch.full(
                    (1, 1, 1),
                    self.prefill_length - 1,
                    dtype=self.token_pos.dtype,
                    device=self.device
                ).expand(*token_pos.shape)
                token_pos = token_pos.maximum(other)
                denom = (self.next_token_pos - token_pos).to(torch.float)
                scores = scores / denom
            self.next_position = scores.argmin(dim=-1)

    def size_estimate(self) -> int:
        return super().size_estimate() + bitsize_of(self.scores)

    @staticmethod
    def size_estimate_apriori(params: KVCacheParams, **kwargs) -> int:
        cache_length = kwargs.get("cache_length")
        if cache_length is None:
            raise IndexError("Argument 'cache_length' is missing")
        else:
            cache_length = int(cache_length)
        dtype = params.dtype
        if dtype is None:
            raise ValueError("params.dtype must be provided")
        numel = params.batch_size * params.n_query_groups * cache_length
        add_here = numel * bits_for_torch_dtype(dtype)
        return super().size_estimate_apriori(params, **kwargs) + add_here


class H2OOriginalKVCache(AttnWeightsKVCache):
    """
    Implements the heavy hitter oracle (H2O) KV cache, see

        Zhang et al
        H2O: Heavy-hitter oracle for efficient generative inference of large language models
        Advances in Neural Information Processing Systems 37, 2024
        https://openreview.net/forum?id=RkRrPp7GKO

    This is the original version, equivalent to their published code. This
    class is mostly for comparisons, we recommend to use :class:`H2OKVCache`
    instead, which has some simple improvements.

    The original version sums scores over the batch dimension and makes
    decisions independent of this dimension. This is why `self.next_position`
    and `self.token_pos` are broadcast here over the batch dimension. Their
    shapes remain the same, for compatibility with the parent class.
    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        cache_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(config, batch_size, cache_length, device, dtype)
        shape = (self.n_query_groups, cache_length)
        self.register_buffer("scores", torch.zeros(shape, device=device, dtype=torch.float), persistent=False)

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.next_position is not None:
            # Reset score values for slots where the new token will be written
            # Note: `next_position` is broadcast over the batch dimension
            index = self.next_position[0].unsqueeze(-1)
            self.scores.scatter_(-1, index, 0.0)
        return super().forward(key, value)

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        # We do not compute H2O scores and cumulate scores for tokens inserted
        # by the prefill. Instead, we set the scores to 0.
        super().prefill(key, value)
        self.scores = 0.0

    def _update(self, attn_weights: torch.Tensor):
        # Scores are computed in `torch.float`. Also, deal with
        # `n_query_groups < n_head` by averaging
        attn_weights = self._average_attn_weights(attn_weights.to(torch.float))
        # Sum over the batch dimension
        aggregated_weights = attn_weights.sum(0)
        self.scores[:, :self.current_length] += aggregated_weights
        if self.current_length == self.cache_length:
            # Set `next_position` to score minimizers
            # Note: `next_position` is broadcast over the batch dimension
            self.next_position = self.scores.argmin(
                dim=-1
            ).unsqueeze(0).expand(self.eff_batch_size, -1)

    def size_estimate(self) -> int:
        return super().size_estimate() + bitsize_of(self.scores)

    @staticmethod
    def size_estimate_apriori(params: KVCacheParams, **kwargs) -> int:
        cache_length = kwargs.get("cache_length")
        if cache_length is None:
            raise IndexError("Argument 'cache_length' is missing")
        else:
            cache_length = int(cache_length)
        dtype = params.dtype
        if dtype is None:
            raise ValueError("params.dtype must be provided")
        numel = params.n_query_groups * cache_length
        add_here = numel * bits_for_torch_dtype(dtype)
        return super().size_estimate_apriori(params, **kwargs) + add_here

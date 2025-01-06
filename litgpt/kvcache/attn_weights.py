from typing import Optional, Tuple

import torch

from litgpt import Config
from litgpt.kvcache.base import KVCache


class AttnWeightsKVCache(KVCache):
    """
    Base class for key-value caches which need attention weights to be passed
    (via :meth:`update`) in every round. In general, these weights are used to
    compute scores, based on which eviction decisions are taken. All of this
    happens in :meth:`_update`, which subclasses need to implement.
    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        cache_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            config: Model config
            batch_size: Inference batch size
            cache_length: Number of slots (i.e., tokens) in cache
            device: Device for buffers
            dtype: Data type for buffers
        """
        super().__init__(config, batch_size, device, dtype)
        if cache_length <= 0:
            raise ValueError("cache_length must be positive integer")
        self.cache_length = cache_length
        shape = (batch_size, self.n_query_groups, cache_length, self.head_size)
        self.register_buffer("k", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("token_pos", torch.zeros(shape[:-1], device=device, dtype=torch.int), persistent=False)
        # Slot positions where :meth:`forward` writes new key, value tensors.
        # Integer array of shape `(batch_size, n_query_groups)`.
        # Initialized by :meth:`prefill`.
        self.next_position = None
        # Next token position :meth:`forward` is called for
        self._next_token_pos = None
        # Number of slots which are occupied. Grows until `cache_length`, then
        # stays there. Initialized by :meth:`prefill`.
        self.current_length = None
        self.just_updated = False

    @property
    def next_token_pos(self) -> Optional[int]:
        return self._next_token_pos

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.next_position is None:
            if self.current_length == self.cache_length:
                raise IndexError("Need to call 'update' before calling 'forward'")
            else:
                raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        shape = (self.batch_size, self.n_query_groups, self.head_size)
        if key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        if self.current_length == self.cache_length and not self.just_updated:
            raise IndexError("Need to call 'update' before calling 'forward'")
        key = key.to(device=self.device, dtype=self.dtype)
        value = value.to(device=self.device, dtype=self.dtype)
        # Write new content into slot
        # `next_position` is batched index, shape `(batch_size, n_query_groups)`
        index = self.next_position.unsqueeze(-1)
        # `index[i, j, 0] = next_position[i, j]`
        self.token_pos.scatter_(-1, index, self._next_token_pos)
        index = index.unsqueeze(-1).expand(-1, -1, 1, self.head_size)
        # `index[i, j, 0, k] = next_position[i, j]`
        self.k.scatter_(-2, index, key.unsqueeze(2))
        self.v.scatter_(-2, index, value.unsqueeze(2))
        self.current_length = min(self.cache_length, self.current_length + 1)
        self.just_updated = False
        self.next_position = None  # Set by next :meth:`update` call
        self._next_token_pos += 1
        return self.k[:, :, :self.current_length, :], self.v[:, :, :self.current_length, :]

    def update(self, *args, **kwargs):
        """
        Needs argument `attn_weights` to be passed. This method needs to set
        `self.next_position` to the slot position where :meth:`forward` is to
        write the new key, value information.

        Args:
            attn_weights: Attention weights for the multi-head attention
                computation done just after the last recent :meth:`forward` call.
                Shape must be `(batch_size, n_head, current_length)`
        """
        if len(args) >= 1:
            attn_weights = args[0]
        else:
            attn_weights = kwargs.get("attn_weights")
            if attn_weights is None:
                raise ValueError("Need to pass attn_weights argument")
        if not isinstance(attn_weights, torch.Tensor):
            raise TypeError("attn_weights argument needs to be torch.Tensor")
        if attn_weights.device != self.device:
            raise ValueError(f"attn_weights.device = {attn_weights.device}, self.device = {self.device}. Must be the same")
        shape = (self.batch_size, self.n_head, self.current_length)
        if attn_weights.shape != shape:
            raise ValueError(f"Shape of attn_weights must be {shape}, but attn_weights.shape = {attn_weights.shape}")
        self._update(attn_weights)
        # Check post-conditions
        if self.current_length < self.cache_length:
            # Set `next_position`
            self._set_next_position_constant(self.current_length)
        elif self.next_position is None:
            raise IndexError("Error in '_update': self.next_position must be set")
        self.just_updated = True

    def _update(self, attn_weights: torch.Tensor):
        """
        Implementation of :meth:`update`, given the `attn_weights` array.
        This method needs to set `next_position`, a batched index of shape
        `(batch_size, n_query_groups)`. If `current_length < cache_length`,
        this must be constant equal to `current_length` (see
        :meth:`_set_next_position_constant`). This assignment is done at the
        end of :meth:`update`, so this method does not have to do it.

        Args:
            attn_weights: Attention weights for the multi-head attention
                computation done just after the last recent :meth:`forward` call.
                Shape must be `(batch_size, n_head, current_length)`
        """
        raise NotImplementedError()

    @property
    def max_prefill_length(self) -> int:
        return self.cache_length - 1

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        """
        Starts a generation loop by passing key and value tensors coming from
        a prefill with embeddings coming from the prompts. The length `T` must
        be smaller than `cache_length`.

        Note: The largest `T` this method can be called for is `cache_length - 1`.
        This is because we cannot fill the cache and call :meth:`forward`
        before calling :meth:`update`, since otherwise `next_position` is not
        properly set. The underlying issue is that the prefill computation uses
        the efficient "training" computation of multi-head attention, for which
        we do not obtain the attention weights. But these are needed (for the
        final token filling the cache) in :meth:`update`.

        Args:
            key: Prefill keys, `(batch_size, n_query_groups, T, head_size)`
            value: Prefill values, `(batch_size, n_query_groups, T, head_size)`
        """
        if key.dim() != 4:
            raise ValueError("key must have 4 dimensions")
        init_length = key.shape[2]
        if init_length > self.max_prefill_length:
            raise ValueError(f"key.shape[2] = {init_length}, must be at most {self.max_prefill_length}")
        shape = (self.batch_size, self.n_query_groups, init_length, self.head_size)
        if key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        # Initialize cache buffers
        self.k[:, :, :init_length, :] = key.to(device=self.device, dtype=self.dtype)
        self.v[:, :, :init_length, :] = value.to(device=self.device, dtype=self.dtype)
        self.current_length = init_length
        self._next_token_pos = init_length
        # The cache is not completely full, so that :meth:`forward` can be
        # called without having to call :meth:`update` before
        self._set_next_position_constant(init_length)

    def _set_next_position_constant(self, val: int):
        self.next_position = torch.full(
            (1, 1), val, dtype=torch.int, device=self.device
        ).expand(self.batch_size, self.n_query_groups)

    def _average_attn_weights(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        `attn_weights` has shape `(batch_size, n_head, current_length)`, while
        `v` has shape `(batch_size, n_query_groups, cache_length, head_size)`.
        If `n_query_groups < n_head`, we average over the heads per query group.
        Otherwise, we just return `attn_weights`.

        Args:
            attn_weights: Argument to :meth:`_update`

        Returns:
            See above, shape `(batch_size, n_query_groups, current_length)`
        """
        if self.n_query_groups == self.n_head:
            return attn_weights
        else:
            return attn_weights.view(
                self.batch_size, self.n_query_groups, -1, self.current_length
            ).mean(dim=2)

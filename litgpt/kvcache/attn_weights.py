from ast import Index
from typing import Optional, Tuple

import torch

from litgpt import Config
from litgpt.kvcache import KVCache


class AttnWeightsKVCache(KVCache):
    """
    Base class for key-value caches which need attention weights to be passed
    (via :meth:`update`) in every round. In general, these weights are used to
    compute scores, based on which eviction decisions are taken.
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
            cache_length: Number of slots (i.e., tokens) of cache
            device: Device for buffers
            dtype: Data type for buffers
        """
        super().__init__(config, batch_size, device, dtype)
        if cache_length <= 0:
            raise ValueError("cache_length must be positive integer")
        self.cache_length = cache_length
        # Slot position where :meth:`forward` writes new key, value tensors.
        # Initialized by :meth:`prefill`.
        self.next_position = None
        # Number of slots which are occupied. Grows until `cache_length`, then
        # stays there. Initialized by :meth:`prefill`.
        self.current_length = None
        self.just_updated = False

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.next_position is None:
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        shape = (self.batch_size, self.n_query_groups, self.head_size)
        if key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError(f"key, value must have dtype {self.dtype}, but key.dtype = {key.dtype}, value.dtype = {key.dtype}")
        if self.current_length == self.cache_length and not self.just_updated:
            raise IndexError("Need to call 'update' before calling 'forward'")
        # Write new content into slot
        self.k[:, :, self.next_position, :] = key
        self.v[:, :, self.next_position, :] = value
        self.current_length = max(self.cache_length, self.current_length + 1)
        self.just_updated = False
        return self.k[:, :, :self.current_length, :], self.v[:, :, :self.current_length, :]

    def update(self, *args, **kwargs):
        """
        Needs argument `attn_weights` to be passed. This method needs to set
        `self.next_position` to the slot position where :meth:`forward` is to
        write the new key, value information.

        Args:
            attn_weights: Attention weights for the multi-head attention
                computation done just after the last recent :meth:`forward` call.
                Shape must be `(
        """
        if len(args) >= 1:
            attn_weights = args[0]
        else:
            attn_weights = kwargs.get("attn_weights")
            if attn_weights is None:
                raise ValueError("Need to pass attn_weights argument")
        if not isinstance(attn_weights, torch.Tensor):
            raise TypeError("attn_weights argument needs to be torch.Tensor")
        shape = (self.batch_size, self.n_head, self.current_length)
        if attn_weights.shape != shape:
            raise ValueError(f"Shape of attn_weights must be {shape}, but attn_weights.shape = {attn_weights.shape}")
        self._update(attn_weights)
        # Check post-conditions
        if self.next_position is None or not (0 <= self.next_position < self.cache_length):
            raise IndexError("Error in '_update': self.next_position needs to be set")
        if self.current_length < self.cache_length and self.next_position != self.current_length:
            raise IndexError(f"Error in '_update': next_position = {self.next_position}, current_length = {self.current_length}, must be equal as long as cache not full")

    def _update(self, attn_weights: torch.Tensor):
        raise NotImplementedError()

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        """
        Starts a generation loop by passing key and value tensors coming from
        a prefill with embeddings coming from the prompts. The length `T` must
        be smaller or equal to `cache_length`.

        Args:
            key: Prefill keys, `(batch_size, n_query_groups, T, head_size)`
            value: Prefill values, `(batch_size, n_query_groups, T, head_size)`
        """
        if key.dim() != 4:
            raise ValueError("key must have 4 dimensions")
        init_length = key.shape[2]
        if init_length > self.cache_length:
            raise ValueError(f"key.shape[2] = {init_length}, must be at most {self.cache_length}")
        shape = (self.batch_size, self.n_query_groups, init_length, self.head_size)
        if key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError(f"key, value must have dtype {self.dtype}, but key.dtype = {key.dtype}, value.dtype = {key.dtype}")
        # Initialize cache buffers
        self.k[:, :, :init_length, :] = key
        self.v[:, :, :init_length, :] = value
        self.current_length = init_length
        if init_length < self.cache_length:
            self.next_position = init_length

from typing import Tuple, Optional

import torch

from litgpt.config import Config


class KVCache(torch.nn.Module):
    """
    Base class for key-value caches.

    Buffers have shapes
    `(batch_size, config.n_query_groups, cache_length, head_size)`, where
    `head_size = config.n_embed // config.n_head`.

    Note: In general, key tensors need to be position-encoded.
    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            config: Model config
            batch_size: Inference batch size
            device: Device for buffers
            dtype: Data type for buffers
        """
        super().__init__()
        self.batch_size = batch_size
        self.n_query_groups = config.n_query_groups
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.device = device
        self.dtype = dtype

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accepts key and value tensors for new token position. These are written
        into the cache. If the cache is full, they overwrite the slot for a
        token which is evicted. In general, the eviction decision is taken in
        the last recent call of :meth:`update`.

        Args:
            key: New keys, `(batch_size, n_query_groups, head_size)`
            value: New values, `(batch_size, n_query_groups, head_size)`

        Returns:
            key_cached, value_cached, `(batch_size, n_query_groups, T, head_size)`,
                where `T <= cache_length` is the current cache length

        """
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        """
        To be called after each `forward`, passing extra information depending
        on the subclass. In general, this method updates internal scores and
        takes a decision which slot is evicted upon the next `forward` call
        (if the cache is full).

        Args:
            *args: Depends on subclass
            **kwargs: Depends on subclass
        """
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
        raise NotImplementedError()


class DenseKVCache(KVCache):
    """
    Key-value cache for dense attention. Key and value tensors for all
    past tokens are maintained. The cache length is the maximum sequence
    length. This cache requires a lot of memory, it can only be used for
    moderate cache lengths.

    Note: If the cache is full, :meth:`forward` raises an exception. The cache
    buffers are allocated up front and are not enlarged later on.
    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: Optional[int] = None,
    ):
        """
        Args:
            config: Model config
            batch_size: Inference batch size
            device: Device for buffers
            dtype: Data type for buffers
            max_sequence_length: Cache length. If not given, we use
            `config.block_size`
        """
        super().__init__(config, batch_size, device, dtype)
        if max_sequence_length is None:
            max_sequence_length = config.block_size
        self.cache_length = max_sequence_length
        shape = (batch_size, self.n_query_groups, max_sequence_length, self.head_size)
        self.register_buffer("k", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.next_position = None

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.next_position is None:
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        if self.next_position >= self.cache_length:
            raise IndexError("Cache is full, cannot add further content")
        shape = (self.batch_size, self.n_query_groups, self.head_size)
        if key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        # Move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(key.dtype)
        self.v = self.v.to(value.dtype)
        # Append new content to cache
        self.k[:, :, self.next_position, :] = key
        self.v[:, :, self.next_position, :] = value
        self.next_position += 1
        return self.k[:, :, :self.next_position, :], self.v[:, :, :self.next_position, :]

    def update(self, *args, **kwargs):
        pass

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        if key.dim() != 4:
            raise ValueError("key must have 4 dimensions")
        init_length = key.shape[2]
        if init_length > self.cache_length:
            raise ValueError(f"key.shape[2] = {init_length}, must be at most {self.cache_length}")
        shape = (self.batch_size, self.n_query_groups, init_length, self.head_size)
        if key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        # Initialize cache content
        self.k = self.k.to(key.dtype)
        self.v = self.v.to(value.dtype)
        self.k[:, :, :init_length, :] = key
        self.v[:, :, :init_length, :] = value
        self.next_position = init_length

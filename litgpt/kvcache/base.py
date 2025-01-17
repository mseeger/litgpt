from typing import Tuple, Optional, Dict
from dataclasses import dataclass

import torch

from litgpt.config import Config
from litgpt.kvcache.utils import bitsize_of, bits_for_torch_dtype


@dataclass(frozen=True)
class KVCacheParams:
    batch_size: int
    n_query_groups: int
    head_size: int
    n_head: int
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]
    max_prefill_length: int

    @staticmethod
    def from_config(
        config: Config,
        batch_size: Optional[int] = None,
        max_prefill_length: Optional[int] = None,
    ) -> "KVCacheParams":
        return KVCacheParams(
            batch_size=batch_size,
            n_query_groups = config.n_query_groups,
            head_size = config.n_embd // config.n_head,
            n_head = config.n_head,
            device=None,
            dtype=None,
            max_prefill_length=max_prefill_length,
        )


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
        Note that `batch_size` is the maximum batch size the cache can be used
        with. The effective batch size is determined when calling
        :meth:`prefill` and can change with any such call.

        Args:
            config: Model config
            batch_size: Inference batch size (maximum)
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

    @property
    def next_token_pos(self) -> Optional[int]:
        """
        Returns:
            Input position for next token to be generated, or `None` if cache
            has not been initialized yet (call of :meth:`prefill`).
        """
        raise NotImplementedError()

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accepts key and value tensors for new token position. These are written
        into the cache. If the cache is full, they overwrite the slot for a
        token which is evicted. In general, the eviction decision is taken in
        the last recent call of :meth:`update`.

        Args:
            key: New keys, `(eff_batch_size, n_query_groups, head_size)`
            value: New values, `(eff_batch_size, n_query_groups, head_size)`

        Returns:
            key_cached, value_cached, `(eff_batch_size, n_query_groups, T,
                head_size)`, where `T <= cache_length` is the current cache
                length

        """
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        """
        Some caches require this method to be called after each `forward`,
        passing extra information depending on the subclass. In general,
        this method updates internal scores and takes a decision which slot
        is evicted upon the next `forward` call (if the cache is full).

        Args:
            *args: Depends on subclass
            **kwargs: Depends on subclass
        """
        raise NotImplementedError()

    def update_requires_attn_weights(self) -> bool:
        """
        Returns:
            If `True`, :meth:`update` requires aergument `attn_weights`, which
            passes current attention weights as
            `(eff_batch_size, n_query_groups,T)` tensor, where `T <= cache_length`
            is the current cache length

        """
        return False

    @property
    def max_prefill_length(self) -> int:
        """
        Returns:
            Maximum sequence length for `key`, `value` tensors passed to
            :meth:`prefill`.

        """
        raise NotImplementedError()

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        """
        Starts a generation loop by passing key and value tensors coming from
        a prefill with embeddings coming from the prompts. The length must be
        `T <= max_prefill_length`. The effective batch size must be
        `eff_batch_size <= batch_size`. This batch size is then fixed for
        subsequent calls of :meth:`forward` and :meth:`update`.

        Args:
            key: Prefill keys, `(eff_batch_size, n_query_groups, T, head_size)`
            value: Prefill values, `(eff_batch_size, n_query_groups, T, head_size)`
        """
        raise NotImplementedError()

    def get_params(self) -> KVCacheParams:
        return KVCacheParams(
            batch_size=self.batch_size,
            n_query_groups=self.n_query_groups,
            head_size=self.n_head,
            n_head=self.n_head,
            device=self.device,
            dtype=self.dtype,
            max_prefill_length=self.max_prefill_length,
        )

    def token_positions(self) -> torch.Tensor:
        """
        Returns:
            Token positions in slots of the cache, shape
            `(eff_batch_size, n_query_groups, T)`.where `T <= cache_length`
            is the current cache length.
        """
        raise NotImplementedError()

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        """
        This is a theoretical estimate of the main buffers (which should all
        be allocated up front), it does not cover temporary storage used in
        the methods (make sure these are small compared to the main buffers).
        Also, real memory usage may be larger due to alignment issues.

        Returns:
            num_bits_total, bits_by_part (unit is bit)
        """
        raise NotImplementedError()

    @staticmethod
    def size_estimate_apriori(params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        """
        Same semantics as :meth:`size_estimate`, but can be called without a
        cache being created. Results may not be exactly the same, but should
        be very close.

        Args:
            params: KV cache parameters
            **kwargs: Extra arguments (optional)

        Returns:
            num_bits_total, bits_by_part (unit is bit)
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
        self.eff_batch_size = None

    @property
    def next_token_pos(self) -> Optional[int]:
        return self.next_position

    @property
    def max_prefill_length(self) -> int:
        return self.cache_length

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.next_position is None:
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        if self.next_position >= self.cache_length:
            raise IndexError("Cache is full, cannot add further content")
        shape = (self.eff_batch_size, self.n_query_groups, self.head_size)
        if key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        if key.dtype != value.dtype:
            raise ValueError(f"key.dtype = {key.dtype} != {value.dtype} = value.dtype")
        # Move the buffer to the activation dtype for when AMP is used
        if key.dtype != self.dtype:
            self.dtype = key.dtype
            self.k = self.k.to(self.dtype)
            self.v = self.v.to(self.dtype)
        # Append new content to cache
        self.k[:self.eff_batch_size, :, self.next_position, :] = key
        self.v[:self.eff_batch_size, :, self.next_position, :] = value
        self.next_position += 1
        return (
            self.k[:self.eff_batch_size, :, :self.next_position, :],
            self.v[:self.eff_batch_size, :, :self.next_position, :],
        )

    def update(self, *args, **kwargs):
        pass

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        if key.dim() != 4:
            raise ValueError("key must have 4 dimensions")
        init_length = key.shape[2]
        if init_length > self.cache_length:
            raise ValueError(f"key.shape[2] = {init_length}, must be at most {self.cache_length}")
        eff_batch_size = key.shape[0]
        if eff_batch_size > self.batch_size:
            raise ValueError(f"key.shape[0] = {eff_batch_size} must be at most batch_size = {self.batch_size}")
        shape = (eff_batch_size, self.n_query_groups, init_length, self.head_size)
        if key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        # Initialize cache content
        self.k = self.k.to(key.dtype)
        self.v = self.v.to(value.dtype)
        self.k[:eff_batch_size, :, :init_length, :] = key
        self.v[:eff_batch_size, :, :init_length, :] = value
        self.next_position = init_length
        self.eff_batch_size = eff_batch_size

    def token_positions(self) -> torch.Tensor:
        return torch.arange(self.next_position, device=self.device).reshape(
            1, 1, -1
        ).expand(self.eff_batch_size, self.n_query_groups, -1)

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        sz_buffs = bitsize_of(self.k) + bitsize_of(self.v)
        return sz_buffs, dict(buffers=sz_buffs)

    @staticmethod
    def size_estimate_apriori(params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        max_sequence_length = kwargs.get("max_sequence_length")
        if max_sequence_length is None:
            raise IndexError("Argument 'max_sequence_length' is missing")
        else:
            max_sequence_length = int(max_sequence_length)
        dtype = params.dtype
        if dtype is None:
            raise ValueError("params.dtype must be provided")
        numel = params.batch_size * params.n_query_groups * max_sequence_length * params.head_size
        sz_buffs = 2 * numel * bits_for_torch_dtype(dtype)
        return sz_buffs, dict(buffers=sz_buffs)


class MostRecentKVCache(KVCache):
    """
    Baseline key-value cache which stores the most recent `cache_length` key,
    value tensors.
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
            cache_length: Number of slots of cache
            device: Device for buffers
            dtype: Data type for buffers
        """
        super().__init__(config, batch_size, device, dtype)
        if cache_length <= 0:
            raise ValueError("cache_length must be positive")
        self.cache_length = cache_length
        shape = (batch_size, self.n_query_groups, cache_length, self.head_size)
        self.register_buffer("k", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("token_pos", torch.zeros(cache_length, device=device, dtype=torch.int), persistent=False)
        self.next_position = None
        self.eff_batch_size = None
        self.current_length = None
        self._next_token_pos = None

    @property
    def next_token_pos(self) -> Optional[int]:
        return self._next_token_pos

    @property
    def max_prefill_length(self) -> int:
        return self.cache_length

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.next_position is None:
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        shape = (self.eff_batch_size, self.n_query_groups, self.head_size)
        if key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        # Move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(key.dtype)
        self.v = self.v.to(value.dtype)
        # Append new content to cache
        self.k[:self.eff_batch_size, :, self.next_position, :] = key
        self.v[:self.eff_batch_size, :, self.next_position, :] = value
        self.token_pos[self.next_position] = self._next_token_pos
        self.next_position = (self.next_position + 1) % self.cache_length
        self.current_length = min(self.current_length + 1, self.cache_length)
        self._next_token_pos += 1
        return (
            self.k[:self.eff_batch_size, :, :self.current_length, :],
            self.v[:self.eff_batch_size, :, :self.current_length, :],
        )

    def update(self, *args, **kwargs):
        pass

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        if key.dim() != 4:
            raise ValueError("key must have 4 dimensions")
        init_length = key.shape[2]
        if init_length > self.max_prefill_length:
            raise ValueError(f"key.shape[2] = {init_length}, must be at most {self.max_prefill_length}")
        eff_batch_size = key.shape[0]
        if eff_batch_size > self.batch_size:
            raise ValueError(f"key.shape[0] = {eff_batch_size} must be at most batch_size = {self.batch_size}")
        shape = (eff_batch_size, self.n_query_groups, init_length, self.head_size)
        if key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        # Initialize cache content
        self.k = self.k.to(key.dtype)
        self.v = self.v.to(value.dtype)
        self.k[:eff_batch_size, :, :init_length, :] = key
        self.v[:eff_batch_size, :, :init_length, :] = value
        self.token_pos[:init_length] = torch.arange(
            init_length, dtype=self.token_pos.dtype, device=self.token_pos.device
        )
        self.current_length = init_length
        self._next_token_pos = init_length
        self.next_position = init_length % self.cache_length
        self.eff_batch_size = eff_batch_size

    def token_positions(self) -> torch.Tensor:
        return self.token_pos[:self.current_length].reshape(1, 1, -1).expand(
            self.eff_batch_size, self.n_query_groups, -1
        )

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        sz_buffs = bitsize_of(self.k) + bitsize_of(self.v)
        sz_pos = bitsize_of(self.token_pos)
        return sz_buffs + sz_pos, dict(buffers=sz_buffs, token_pos=sz_pos)

    @staticmethod
    def size_estimate_apriori(params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        cache_length = kwargs.get("cache_length")
        if cache_length is None:
            raise IndexError("Argument 'cache_length' is missing")
        else:
            cache_length = int(cache_length)
        dtype = params.dtype
        if dtype is None:
            raise ValueError("params.dtype must be provided")
        numel = params.batch_size * params.n_query_groups * cache_length * params.head_size
        k_and_v = 2 * numel * bits_for_torch_dtype(dtype)
        tk_p = cache_length * bits_for_torch_dtype(torch.int)
        return k_and_v + tk_p, dict(buffers=k_and_v, token_pos=tk_p)

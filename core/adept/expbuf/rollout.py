from __future__ import annotations

import typing

import torch
from torch import Tensor
from torch.multiprocessing import Queue

from adept.alias import Experience
from adept.config import configurable

if typing.TYPE_CHECKING:
    from adept.module import ExpBuf


class Rollout(ExpBuf):
    def __init__(self, rollout_len: int = 20):
        self._rollout_len = rollout_len
        self._buffer: dict[str, Tensor] = {}
        self._cur_ix = 0

    def step(self, actor_entry: Experience, env_entry: Experience) -> bool:
        # Reset the buffer if full
        if self.ready():
            self.reset()
        # Lazily initialize the buffer
        for key, tensor in {**actor_entry, **env_entry}.keys():
            if key not in self._buffer:
                self._buffer[key] = torch.empty(
                    self._rollout_len,
                    *tensor.shape,
                    device=tensor.device,
                    dtype=tensor.dtype
                )
            self._buffer[key][self._cur_ix] = actor_entry[key]
        self._cur_ix += 1
        return self.ready()

    def reset(self) -> bool:
        for tensor in self._buffer.values():
            tensor.detach_()
        self._cur_ix = 0
        return self.ready()

    def to_dict(self) -> Experience:
        assert self.ready()
        return self._buffer

    def ready(self) -> bool:
        return self._cur_ix == len(self)

    def __len__(self):
        return self._rollout_len


class OffPolicyRollout(Rollout):
    @configurable
    def __init__(self, rollout_len: int = 20):
        super().__init__(rollout_len)
        self.is_initialized = False

    def step(self, actor_entry: Experience, env_entry: Experience) -> bool:
        result = super().step(actor_entry, env_entry)
        if not self.is_initialized:
            for tensor in self._buffer.values():
                tensor.share_memory_()
            self.is_initialized = True
        return result

    def get(self, q: Queue) -> OffPolicyRollout:
        """Copy data into experience buffer from a queue."""
        self._buffer = q.get()
        return self

    def put(self, q: Queue) -> OffPolicyRollout:
        """Copy data into a queue from the experience buffer."""
        q.put(self._buffer)
        return self

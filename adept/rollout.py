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
        # Add an extra entry for bootstrapping
        self._rollout_len = rollout_len + 1
        self._buffer: dict[
            str, Tensor | typing.Iterable[Tensor] | dict[str, Tensor]
        ] = {}
        self._cur_ix = 0

    def step(self, actor_entry: Experience, env_entry: Experience) -> bool:
        # Reset the buffer if full
        if self.ready():
            self.reset()
        # Lazily initialize the buffer
        for rollout_buf, item in self._zip_with_buffer({**actor_entry, **env_entry}):
            rollout_buf[self._cur_ix] = item
        self._cur_ix += 1
        return self.ready()

    def reset(self) -> bool:
        for tensor in self._iter_buffer():
            # Shift the bootstrap to the front
            tensor[0] = tensor[-1]
            tensor[1:] = tensor[1:].detach()
        self._cur_ix = 1
        return self.ready()

    def to_dict(self) -> Experience:
        assert self.ready()
        return self._buffer

    def ready(self) -> bool:
        return self._cur_ix == len(self)

    def __len__(self):
        return self._rollout_len

    def _zip_with_buffer(self, experience: Experience) -> typing.Iterable[tuple[Tensor, Tensor]]:
        """Yields a tuple of (rollout buffer, item)

        The rollout buffer is lazily initialized to the correct shape and dtype.
        """
        def make_empty(t: Tensor) -> Tensor:
            return torch.empty(
                self._rollout_len, *t.shape, device=t.device, dtype=t.dtype
            )

        for key, item in experience.items():
            if isinstance(item, Tensor):
                if key not in self._buffer:
                    self._buffer[key] = make_empty(item)
                yield self._buffer[key], item
            elif isinstance(item, dict):
                if key not in self._buffer:
                    self._buffer[key] = {}
                for k, v in item.items():
                    if k not in self._buffer[key]:
                        self._buffer[key][k] = make_empty(v)
                    yield self._buffer[key][k], v
            elif isinstance(item, typing.List):
                if key not in self._buffer:
                    self._buffer[key] = []
                for i in range(len(item)):
                    if i >= len(self._buffer[key]):
                        self._buffer[key].append(make_empty(item[i]))
                    yield self._buffer[key][i], item[i]

    def _iter_buffer(self) -> typing.Iterator[Tensor]:
        for item in self._buffer.values():
            if isinstance(item, Tensor):
                yield item
            elif isinstance(item, dict):
                for v in item.values():
                    yield v
            elif isinstance(item, typing.Iterable):
                for v in item:
                    yield v


class OffPolicyRollout(Rollout):
    @configurable
    def __init__(self, rollout_len: int = 20):
        super().__init__(rollout_len)
        self.is_initialized = False

    def step(self, actor_entry: Experience, env_entry: Experience) -> bool:
        result = super().step(actor_entry, env_entry)
        if not self.is_initialized:
            for t in self._iter_buffer():
                t.share_memory_()
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

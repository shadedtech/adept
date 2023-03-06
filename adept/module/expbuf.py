from __future__ import annotations

import abc

from torch.multiprocessing import Queue

from adept.alias import Experience, Observation, HiddenStates


class ExpBuf(abc.ABC):
    @abc.abstractmethod
    def step(
        self,
        actor_entry: Experience,
        env_entry: Experience,
        next_obs: Observation,
        next_hiddens: HiddenStates,
    ) -> bool:
        ...

    @abc.abstractmethod
    def reset(self) -> bool:
        ...

    @abc.abstractmethod
    def to_dict(self) -> Experience:
        ...


class OffPolicyExpBuf(ExpBuf, abc.ABC):
    @abc.abstractmethod
    def get(self, q: Queue) -> OffPolicyExpBuf:
        """Copy data into experience buffer from a queue."""
        ...

    @abc.abstractmethod
    def put(self, q: Queue) -> OffPolicyExpBuf:
        """Copy data into a queue from the experience buffer."""
        ...

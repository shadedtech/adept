import abc

import torch

from adept.alias import Observation, Spec


class Preprocessor:
    @abc.abstractmethod
    def __call__(self, obs: Observation) -> Observation:
        ...

    @property
    @abc.abstractmethod
    def observation_spec(self) -> Spec:
        ...

    @abc.abstractmethod
    def to(self, device: torch.device):
        ...

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        ...

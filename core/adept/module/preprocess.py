import abc

import torch

from adept.alias import Observation, Spec


class Preprocessor:
    @abc.abstractmethod
    def __call__(self, obs: Observation) -> Observation:
        ...

    @classmethod
    @abc.abstractmethod
    def spec(cls, in_spec: Spec) -> Spec:
        ...

    @abc.abstractmethod
    def to(self, device: torch.device):
        ...

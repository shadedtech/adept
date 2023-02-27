import abc

import torch
from torch import nn

from adept.alias import Observation, Spec


class Preprocessor(nn.Module):
    @abc.abstractmethod
    def __call__(self, obs: Observation) -> Observation:
        ...

    @property
    @abc.abstractmethod
    def observation_spec(self) -> Spec:
        ...

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        ...

from __future__ import annotations

import abc

import torch
from torch import nn, Tensor
from torch.distributions import Distribution as TorchDistribution


class Distribution(nn.Module, abc.ABC):
    def __init__(self, n_output: int):
        super().__init__()
        self.n_output = n_output

    @abc.abstractmethod
    def forward(self, logits: Tensor) -> TorchDistribution:
        ...


class Normal(Distribution):
    """A module that builds a Diagonal Gaussian distribution from means.

    Standard deviations are learned parameters in this module.
    """
    def __init__(self, n_output: int):
        super().__init__(n_output)
        self.log_stdev = nn.Parameter(torch.zeros(n_output))

    def forward(self, logits: Tensor) -> TorchDistribution:
        dist = torch.distributions.Normal(loc=logits, scale=self.log_stdev.exp())
        # Log prob comes out with an extra dimension, so we sum over it
        old_log_prob = dist.log_prob
        dist.log_prob = lambda x: old_log_prob(x).sum(-1)
        return dist


class Categorical(Distribution):
    """A module that builds a Categorical distribution from logits."""
    def forward(self, logits: Tensor) -> TorchDistribution:
        dist = torch.distributions.Categorical(logits=logits)
        return dist

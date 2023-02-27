from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.distributions import Distribution


class Normal(nn.Module):
    """A module that builds a Diagonal Gaussian distribution from means.

    Standard deviations are learned parameters in this module.
    """
    def __init__(self, n_output: int):
        super().__init__()
        self.log_stdev = nn.Parameter(torch.zeros(n_output))

    def forward(self, logits: Tensor) -> Distribution:
        dist = torch.distributions.Normal(loc=logits, scale=self.log_stdev.exp())
        return dist

from typing import Tuple

import torch
from adept.alias import HiddenStates, HiddenState
from adept.alias import Shape
from adept.net.base import NetMod1D
from torch import nn


class OutputLayer1D(NetMod1D):
    def __init__(self, name: str, input_shape: Shape, output_shape: Shape):
        super().__init__(name, input_shape)
        self._out_shp = output_shape
        self.linear_out = nn.Linear(input_shape[0], output_shape[0])

    def _forward(
        self, x: torch.Tensor, hiddens: HiddenStates = None
    ) -> tuple[torch.Tensor, HiddenState]:
        return self.linear_out(x), torch.tensor([])

    def _output_shape(self) -> Shape:
        return self._out_shp

from typing import Optional

import torch
from torch import nn

from adept.alias import HiddenState
from adept.alias import Shape
from adept.net.base import NetMod4D


class OutputLayer4D(NetMod4D):
    def __init__(self, name: str, input_shape: Shape, output_shape: Shape):
        super().__init__(name, input_shape)
        self._out_shp = output_shape
        f_in, f_out = input_shape[0], output_shape[0]
        self.conv_out = nn.Conv3d(f_in, f_out, kernel_size=1)

    def _forward(
        self, x: torch.Tensor, hiddens: HiddenState = None
    ) -> tuple[torch.Tensor, Optional[HiddenState]]:
        return self.conv_out(x), None

    def _output_shape(self) -> Shape:
        return self._out_shp

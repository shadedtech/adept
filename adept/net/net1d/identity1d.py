import torch

from adept.alias import HiddenState
from adept.alias import Shape
from adept.net.base import NetMod1D

from typing import Optional


class Identity1D(NetMod1D):
    def _output_shape(self) -> Shape:
        return self.input_shape

    def _forward(
        self, x: torch.Tensor, hiddens: HiddenState = None
    ) -> tuple[torch.Tensor, Optional[HiddenState]]:
        return x, None

import torch

from typing import Optional

from adept.alias import HiddenState
from adept.alias import Shape
from adept.net.base import NetMod2D


class Identity2D(NetMod2D):
    def _forward(
        self, x: torch.Tensor, hiddens: HiddenState
    ) -> tuple[torch.Tensor, Optional[HiddenState]]:
        return x, None

    def _output_shape(self) -> Shape:
        return self.input_shape

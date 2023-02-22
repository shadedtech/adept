from typing import Optional

import torch

from adept.alias import HiddenState
from adept.alias import Shape
from adept.net.base import NetMod3D


class Identity3D(NetMod3D):
    def _output_shape(self) -> Shape:
        return self.input_shape

    def _forward(
        self, x: torch.Tensor, hiddens: HiddenState
    ) -> tuple[torch.Tensor, Optional[HiddenState]]:
        return x, None

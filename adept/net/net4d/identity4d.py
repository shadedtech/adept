import torch

from adept.alias import HiddenState
from adept.alias import Shape
from adept.net.base import NetMod4D


class Identity4D(NetMod4D):
    def _forward(
        self, x: torch.Tensor, hiddens: HiddenState
    ) -> tuple[torch.Tensor, HiddenState]:
        return x, torch.tensor([])

    def _output_shape(self) -> Shape:
        return self.input_shape

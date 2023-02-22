import torch
from torch import nn
from torch.nn import functional as F

from adept.alias import HiddenState
from adept.alias import Shape
from adept.config import configurable
from adept.net.base import NetMod1D

from typing import Optional


class LinearNet(NetMod1D):
    @configurable
    def __init__(
        self,
        name: str,
        input_shape: Shape,
        n_layer: int = 3,
        n_hidden: int = 512,
    ):
        super().__init__(name, input_shape)
        self._n_layer = n_layer
        self._n_hidden = n_hidden
        (f,) = input_shape
        for i in range(self._n_layer):
            in_feat = f if i == 0 else self._n_hidden
            self.add_module(
                f"linear{i}", nn.Linear(in_feat, self._n_hidden, bias=False)
            )
            self.add_module(f"norm{i}", nn.BatchNorm1d(self._n_hidden))

    def _forward(
        self, x: torch.Tensor, hiddens: HiddenState = None
    ) -> tuple[torch.Tensor, Optional[HiddenState]]:
        for i in range(self._n_layer):
            linear = self._modules[f"linear{i}"]
            norm = self._modules[f"norm{i}"]
            x = F.relu(norm(linear(x)))
        return x, None

    def _output_shape(self) -> Shape:
        return (self._n_hidden,)


if __name__ == "__main__":
    linearnet = LinearNet("source1d", (5,))
    print(linearnet._n_layer)
    print(linearnet._n_hidden)
    test = torch.ones(4, 5)
    result = linearnet.forward(test)
    print(result)

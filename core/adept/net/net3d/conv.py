from typing import Optional
from typing import Tuple

import torch
from adept import util
from adept.alias import HiddenState
from adept.alias import Shape
from adept.config import configurable
from adept.net.base import NetMod3D
from torch import nn
from torch.nn import functional as F

from adept.util import torch_util


@configurable
class ConvNet(NetMod3D):
    def __init__(
        self,
        name: str,
        input_shape: Shape,
        n_layer: int = 4,
        n_channel: int = 32,
        first_kernel_sz: int = 7,
    ):
        super().__init__(name, input_shape)
        cfg = self.get_local_cfg()
        self._n_layer = n_layer or cfg.n_layer
        self._n_channel = n_channel or cfg.n_channel
        self._first_kernel_sz = first_kernel_sz or cfg.first_kernel_sz
        f, h, w = input_shape
        relu_gain = nn.init.calculate_gain("relu")
        for i in range(self._n_layer):
            in_feat = f if i == 0 else self._n_channel
            kernel_sz = self._first_kernel_sz if i == 0 else 3
            self.add_module(
                f"conv{i}",
                nn.Conv2d(
                    in_feat, self._n_channel, kernel_sz, stride=2, padding=1
                ),
            )
            self.add_module(f"norm{i}", nn.BatchNorm2d(self._n_channel))
            self._modules[f"conv{i}"].weight.data.mul_(relu_gain)
        self._output_shape_cache = None

    def _forward(
        self, x: torch.Tensor, hiddens: HiddenState
    ) -> tuple[torch.Tensor, HiddenState]:
        for i in range(self._n_layer):
            conv = self._modules[f"conv{i}"]
            norm = self._modules[f"norm{i}"]
            x = F.relu(norm(conv(x)))
        return x, torch.tensor([])

    def _output_shape(self) -> Shape:
        if self._output_shape_cache is None:
            f, h, w = self.input_shape
            dim_sz_h = h
            dim_sz_w = w
            for i in range(self._n_layer):
                kernel_sz = self._first_kernel_sz if i == 0 else 3
                dim_sz_h = torch_util.calc_conv_dim(dim_sz_h, kernel_sz, 2, 1, 1)
                dim_sz_w = torch_util.calc_conv_dim(dim_sz_w, kernel_sz, 2, 1, 1)
            self._output_shape_cache = (self._n_channel, dim_sz_h, dim_sz_w)
        return self._output_shape_cache


if __name__ == "__main__":
    convnet = ConvNet("source3d", (32, 84, 84))
    print(convnet._n_layer)
    print(convnet._n_channel)
    print(convnet._first_kernel_sz)
    print(convnet.output_shape())
    test = torch.ones(4, 32, 84, 84)
    result = convnet.forward(test, torch.tensor([]))
    print(result)

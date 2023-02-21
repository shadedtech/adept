import abc
from typing import Optional

import torch
from torch import nn

from adept.alias import HiddenState
from adept.alias import Shape


class NetMod(nn.Module, metaclass=abc.ABCMeta):
    is_configurable: bool = False

    """Network module.

    Similar to torch.nn.Module but provides support for dimension casting,
    and hidden state management.
    """

    @classmethod
    @abc.abstractmethod
    def dim(cls) -> int:
        """Dimensionality of the network module at the input.

        1, 2, 3, or 4.
        """
        ...

    @abc.abstractmethod
    def _forward(
        self, x: torch.Tensor, hiddens: HiddenState
    ) -> tuple[torch.Tensor, HiddenState]:
        """Perform a forward pass.

        If the module doesn't use hidden states, return an empty tensor.
        """
        ...

    @abc.abstractmethod
    def _output_shape(self) -> Shape:
        """Return the output shape of the module."""
        ...

    @abc.abstractmethod
    def _shape_1d(self) -> Shape:
        ...

    @abc.abstractmethod
    def _shape_2d(self) -> Shape:
        ...

    @abc.abstractmethod
    def _shape_3d(self) -> Shape:
        ...

    @abc.abstractmethod
    def _shape_4d(self) -> Shape:
        ...

    def __init__(self, name: str, input_shape: Shape):
        super().__init__()
        self.name = name
        self.input_shape = input_shape

    def new_hidden_states(
        self, device: torch.device, batch_sz: int = 1
    ) -> HiddenState:
        """Initialize hidden states.

        Override this method if your module uses hidden states.
        """
        return torch.tensor([])

    def forward(
        self, x: torch.Tensor, hiddens: HiddenState = None, dim: Optional[int] = None
    ):
        if dim is None:
            dim = len(self._output_shape())
        if hiddens is None:
            hiddens = torch.tensor([])
        out, nxt_hiddens = self._forward(x, hiddens)
        if dim == 1:
            out = self._to_1d(out)
        elif dim == 2:
            out = self._to_2d(out)
        elif dim == 3:
            out = self._to_3d(out)
        elif dim == 4:
            out = self._to_4d(out)
        else:
            raise Exception(f"Invalid dim: {dim}")
        return out, nxt_hiddens

    def output_shape(self, dim: Optional[int] = None) -> Shape:
        if dim is None:
            dim = len(self._output_shape())
        if dim == 1:
            shp = self._shape_1d()
        elif dim == 2:
            shp = self._shape_2d()
        elif dim == 3:
            shp = self._shape_3d()
        elif dim == 4:
            shp = self._shape_4d()
        else:
            raise Exception(f"Invalid dim: {dim}")
        return shp

    def _to_1d(self, feature_map: torch.Tensor) -> torch.Tensor:
        b = feature_map.size()[0]
        return feature_map.view(b, *self._shape_1d())

    def _to_2d(self, feature_map: torch.Tensor) -> torch.Tensor:
        b = feature_map.size()[0]
        return feature_map.view(b, *self._shape_2d())

    def _to_3d(self, feature_map: torch.Tensor) -> torch.Tensor:
        b = feature_map.size()[0]
        return feature_map.view(b, *self._shape_3d())

    def _to_4d(self, feature_map: torch.Tensor) -> torch.Tensor:
        b = feature_map.size()[0]
        return feature_map.view(b, *self._shape_4d())

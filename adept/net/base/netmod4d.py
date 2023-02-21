import abc

from adept.alias import Shape
from adept.module import NetMod


class NetMod4D(NetMod, metaclass=abc.ABCMeta):
    @classmethod
    def dim(cls) -> int:
        return 4

    def _shape_1d(self) -> Shape:
        f, d, h, w = self._output_shape()
        return (f * d * h * w,)

    def _shape_2d(self) -> Shape:
        f, d, h, w = self._output_shape()
        return f, d * h * w

    def _shape_3d(self) -> Shape:
        f, d, h, w = self._output_shape()
        return f * d, h, w

    def _shape_4d(self) -> Shape:
        return self._output_shape()

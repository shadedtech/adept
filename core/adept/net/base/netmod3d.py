import abc

from adept.alias import Shape
from adept.module import NetMod


class NetMod3D(NetMod, metaclass=abc.ABCMeta):
    @classmethod
    def dim(cls) -> int:
        return 3

    def _shape_1d(self) -> Shape:
        f, h, w = self._output_shape()
        return (f * h * w,)

    def _shape_2d(self) -> Shape:
        f, h, w = self._output_shape()
        return f, h * w

    def _shape_3d(self) -> Shape:
        return self._output_shape()

    def _shape_4d(self) -> Shape:
        f, h, w = self._output_shape()
        return f, 1, h * w

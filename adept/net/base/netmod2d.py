import abc

from adept.alias import Shape
from adept.module import NetMod


class NetMod2D(NetMod, metaclass=abc.ABCMeta):
    @classmethod
    def dim(cls) -> int:
        return 2

    def _shape_1d(self) -> Shape:
        f, s = self._output_shape()
        return (f * s,)

    def _shape_2d(self) -> Shape:
        return self._output_shape()

    def _shape_3d(self) -> Shape:
        f, s = self._output_shape()
        return f * s, 1, 1

    def _shape_4d(self) -> Shape:
        f, s = self._output_shape()
        return f, s, 1, 1

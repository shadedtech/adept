from __future__ import annotations

import abc

import torch
from torch import Tensor


class Space(abc.ABC):
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    @abc.abstractmethod
    def sample(self) -> Tensor:
        ...


class Box(Space):
    def __init__(
        self,
        lows: Tensor,
        highs: Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        super().__init__(shape, dtype)
        self.low = lows
        self.high = highs

    def sample(self) -> Tensor:
        return (
            torch.rand(self.shape, dtype=self.dtype, device=self.low.device)
            * (self.high - self.low)
            + self.low
        )


class Discrete(Space):
    def __init__(
        self,
        n: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        super().__init__(shape, dtype)
        self.n = n

    def sample(self) -> Tensor:
        return torch.randint(self.n, self.shape, dtype=self.dtype)


if __name__ == "__main__":
    print(type(torch.int))

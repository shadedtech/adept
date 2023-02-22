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
        shape: tuple[int, ...],
        lows: Tensor,
        highs: Tensor,
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

    def zeros(self) -> Tensor:
        return torch.zeros(self.shape, dtype=self.dtype)


class Discrete(Space):
    def __init__(
        self,
        n_category: int,
        shape: tuple[int, ...] = (1, ),
        dtype: torch.dtype = torch.long,
    ):
        super().__init__(shape, dtype)
        self.n = n_category

    def sample(self) -> Tensor:
        return torch.randint(self.n, self.shape, dtype=self.dtype)

    def zeros(self) -> Tensor:
        return torch.zeros(self.shape, dtype=self.dtype)


class MultiDiscrete(Space):
    def __init__(
        self,
        n_category: tuple[int, ...],
        shape: tuple[int, ...] = (1, ),
        dtype: torch.dtype = torch.long,
    ):
        super().__init__((*shape, len(n_category)), dtype)
        self.n_category = n_category

    def sample(self) -> Tensor:
        return torch.stack(
            [
                torch.randint(n, self.shape[:-1], dtype=self.dtype)
                for n in self.n_category
            ],
            dim=-1,
        )

    def zeros(self) -> Tensor:
        return torch.zeros(self.shape, dtype=self.dtype)


if __name__ == "__main__":
    print(type(torch.int))

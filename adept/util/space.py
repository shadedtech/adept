from __future__ import annotations

import abc
import math

from numbers import Number

import torch
from torch import Tensor


class Space(abc.ABC):
    def __init__(
        self,
        batch_size: int | tuple[int, ...],
        non_batch_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        if type(batch_size) is int:
            batch_size = (batch_size,)

        self.batch_size: tuple[int, ...] = batch_size
        self.flat_batch_size: int = math.prod(batch_size)
        self.dtype = dtype

        self._non_batch_shape: tuple[int, ...] = non_batch_shape

    @abc.abstractmethod
    def sample(self, with_batch: bool = True) -> Tensor:
        ...

    @abc.abstractmethod
    def logit_shape(self, with_batch: bool) -> tuple[int, ...]:
        ...

    def shape(self, with_batch: bool = True) -> tuple[int, ...]:
        return (
            (*self.batch_size, *self._non_batch_shape)
            if with_batch
            else self._non_batch_shape
        )

    def zeros(self, with_batch: bool = True) -> Tensor:
        return torch.zeros(self.shape(with_batch), dtype=self.dtype)

    def set_batch_size(self, batch_size: int | tuple[int, ...]) -> Space:
        if type(batch_size) is int:
            batch_size = (batch_size,)

        self.batch_size = batch_size
        self.flat_batch_size: int = math.prod(batch_size)
        return self


class Box(Space):
    def __init__(
        self,
        shape: int | tuple[int, ...],
        lows: Number | Tensor,
        highs: Number | Tensor,
        dtype: torch.dtype = torch.float32,
        n_batch_dim: int = 0,
    ):
        if type(shape) is int:
            shape = (shape,)
        if isinstance(lows, Number):
            lows = torch.empty(shape).fill_(lows)
        if isinstance(highs, Number):
            highs = torch.empty(shape).fill_(highs)
        super().__init__(shape[:n_batch_dim], shape[n_batch_dim:], dtype)
        self.low = lows
        self.high = highs

    def sample(self, with_batch: bool = True) -> Tensor:
        return (
            torch.rand(self.shape(with_batch), dtype=self.dtype, device=self.low.device)
            * (self.high - self.low)
            + self.low
        )

    def logit_shape(self, with_batch: bool = True) -> tuple[int, ...]:
        # If non batch shape is empty, unsqueeze to make it 1D
        non_batch_shape = self._non_batch_shape or (1,)
        return (
            (*self.batch_size, *non_batch_shape)
            if with_batch
            else self._non_batch_shape
        )

    def __repr__(self):
        return f"Box({self.shape()}, batch_size={self.batch_size})"


class Discrete(Space):
    def __init__(
        self,
        n_category: int,
        batch_size: int | tuple[int, ...] = tuple(),
        dtype: torch.dtype = torch.long,
    ):
        super().__init__(batch_size, tuple(), dtype)
        self.n_category = n_category

    def sample(self, with_batch: bool = True) -> Tensor:
        return torch.randint(self.n_category, self.shape(with_batch), dtype=self.dtype)

    def logit_shape(self, with_batch: bool = True) -> tuple[int, ...]:
        return (*self.batch_size, self.n_category) if with_batch else (self.n_category,)

    def __repr__(self):
        return f"Discrete({self.n_category}, batch_size={self.batch_size})"


class MultiDiscrete(Space):
    def __init__(
        self,
        n_categories: tuple[int, ...],
        batch_size: int | tuple[int, ...] = tuple(),
        dtype: torch.dtype = torch.long,
    ):
        super().__init__(batch_size, (len(n_categories),), dtype)
        self.n_categories = n_categories

    def sample(self, with_batch: bool = True) -> Tensor:
        return torch.stack(
            [
                torch.randint(n_category, self.shape(with_batch), dtype=self.dtype)
                for n_category in self.n_categories
            ],
            dim=-1,
        )

    def logit_shape(self, with_batch: bool = True) -> tuple[int, ...]:
        logit_size = sum(self.n_categories)
        return (*self.batch_size, logit_size) if with_batch else (logit_size,)


if __name__ == "__main__":
    print(type(torch.int))

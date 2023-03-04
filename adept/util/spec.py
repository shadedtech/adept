from __future__ import annotations

from enum import Enum
from typing import Iterable, Dict, TypeVar, Iterator

from adept.alias import Spec, Shape
from adept.util import space
from adept.util.space import Space

T = TypeVar("T")


class SpecImpl:
    def __init__(self, spec: Spec, name: str = None):
        if isinstance(spec, space.Space) and name is None:
            raise Exception("Must provide name if spec is not a dict or iterable")
        self.spec = spec
        self.name = name

    def to_dict(self) -> dict[str, Space]:
        return to_dict(self.spec, self.name)

    def items(self) -> Iterable[tuple[str, Space]]:
        return items(self.spec, self.name)

    def values(self) -> Iterable[Space]:
        return (v for _, v in self.items())

    @property
    def batch_size(self) -> tuple[int, ...]:
        return next(iter(self.values())).batch_size


def to_dict(x: T | Iterable[T] | dict[str, T], name: str = None) -> Dict[str, T]:
    return dict(items(x, name))


def items(
    x: T | Iterable[T] | dict[str, T], name: str = None
) -> Iterable[tuple[str, T]]:
    if isinstance(x, dict):
        for k, v in x.items():
            yield k, v
    elif isinstance(x, Iterable):
        for i, v in enumerate(x):
            yield str(i), v
    if not name:
        raise Exception("Must provide name if x is not a dict or iterable")
    yield name, x


if __name__ == "__main__":
    import torch

    my_spec = {
        "discrete": space.Discrete(2, batch_size=(2, 3)),
        "box": space.Box(
            (2, 3, 5),
            torch.zeros(2, 3, 5),
            torch.ones(2, 3, 5),
            n_batch_dim=2,
        ),
    }
    spec = SpecImpl(my_spec)
    print(spec.batch_size())


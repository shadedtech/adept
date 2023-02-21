from __future__ import annotations

from typing import Iterable, Dict, TypeVar

from adept.alias import Spec, Shape

T = TypeVar("T")


def to_dict(
    x: T | Iterable[T] | dict[int | str, T], name: str = None
) -> Dict[int | str, T]:
    if isinstance(x, dict):
        return x
    elif isinstance(x, Iterable):
        return {i: v for i, v in enumerate(x)}
    if not name:
        raise Exception("Must provide name if x is not a dict or iterable")
    return {name: x}


def input_shapes(input_spec: Spec, name: str = "obs") -> dict[int | str, Shape]:
    if isinstance(input_spec, dict):
        return {k: v.shape for k, v in input_spec.items()}
    elif isinstance(input_spec, Iterable):
        return {i: v.shape for i, v in enumerate(input_spec)}
    return {name: input_spec.shape}

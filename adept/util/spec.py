from __future__ import annotations

from typing import Iterable, Dict, TypeVar

from adept.alias import Spec, Shape
from adept.util import space
from adept.util.space import Space

T = TypeVar("T")


def to_dict(
    x: T | Iterable[T] | dict[str, T], name: str = None
) -> Dict[str, T]:
    if isinstance(x, dict):
        return x
    elif isinstance(x, Iterable):
        return {str(i): v for i, v in enumerate(x)}
    if not name:
        raise Exception("Must provide name if x is not a dict or iterable")
    return {name: x}


def obs_shapes(input_spec: Spec, name: str = "obs") -> dict[str, Shape]:
    if isinstance(input_spec, dict):
        return {k: v.shape() for k, v in input_spec.items()}
    elif isinstance(input_spec, Iterable):
        return {str(i): v.shape() for i, v in enumerate(input_spec)}
    return {name: input_spec.shape()}


def logit_shapes(action_spec: Spec, name: str = "action") -> dict[str, Shape]:
    if isinstance(action_spec, dict):
        return {k: v.logit_shape(with_batch=False) for k, v in action_spec.items()}
    elif isinstance(action_spec, Iterable):
        return {str(i): v.logit_shape(with_batch=False) for i, v in enumerate(action_spec)}
    return {name: action_spec.logit_shape(with_batch=False)}

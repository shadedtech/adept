from __future__ import annotations

from typing import Any

from torch import Tensor, IntTensor

from adept.util.space import Space

Action = Tensor | tuple[Tensor, ...] | dict[str, Tensor]
Observation = Tensor | tuple[Tensor, ...] | dict[str, Tensor]
Reward = Tensor
Done = IntTensor
Spec = Space | dict[str, Space] | tuple[Space, ...]
Shape = tuple[int, ...]
HiddenState = Tensor
HiddenStates = dict[str, HiddenState]
Experience = dict[str, Tensor]
Info = dict[str, Any]
Losses = dict[str, Tensor]
Metrics = dict[str, int | float]
# TODO Info should be a class

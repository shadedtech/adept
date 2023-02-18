from __future__ import annotations

import abc
import typing
from typing import Tuple

from adept.alias import Spec, Observation, Action, HiddenStates, Reward, Done, Experience

if typing.TYPE_CHECKING:
    from adept.net import StaticNet


class Actor(abc.ABC):
    """Forward pass logic."""

    def __init__(self, action_spec: Spec):
        super().__init__()
        self._action_spec = action_spec

    @abc.abstractmethod
    def step(
        self,
        obs: Observation,
        hiddens: HiddenStates,
        net: StaticNet,
        preprocess_gpu: typing.Callable[[Observation], Observation] = lambda x: x,
    ) -> tuple[Action, Experience, HiddenStates]:
        """Decide actions, data for backprop, and the next hidden states.
        """
        ...

    @abc.abstractmethod
    def observe(
        self,
        next_obs: Observation,
        rewards: Reward,
        dones: Done,
    ) -> Experience:
        """Observe the next state transition and return any necessary info to be
        saved in the experience buffer.
        """
        ...

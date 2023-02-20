from __future__ import annotations

import abc
import typing

from adept.alias import Losses, Experience
from adept.alias import Metrics

if typing.TYPE_CHECKING:
    from adept.net import AutoNetwork
    from adept.run._base import Updater


class Learner(abc.ABC):
    """Handles backwards pass logic."""

    @abc.abstractmethod
    def step(
        self, net: AutoNetwork, updater: Updater, expbuf: Experience, step_count: int
    ) -> tuple[Losses, Metrics]:
        ...
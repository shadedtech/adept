import abc

import typing

from adept.alias import Spec, Action, Observation, Reward, Done, Info

if typing.TYPE_CHECKING:
    from adept.module import Preprocessor


class Environment(abc.ABC):
    def __init__(self, seed: int):
        self._seed = seed

    @property
    @abc.abstractmethod
    def observation_spec(self) -> Spec:
        """Observation specification of the environment."""
        ...

    @property
    @abc.abstractmethod
    def action_spec(self) -> Spec:
        """Action specification of the environment."""
        ...

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def n_reward_component(self) -> int:
        """Number of reward components.

        Specify reward components for eval / tensorboard logging.
        """
        ...

    @abc.abstractmethod
    def step(self, action: Action) -> tuple[Observation, Reward, Done, Info]:
        ...

    @abc.abstractmethod
    def reset(self) -> Observation:
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Closes any simulation instances and connections."""
        ...

    @abc.abstractmethod
    def get_preprocessor(self) -> Preprocessor:
        """Get the GPU preprocessor."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

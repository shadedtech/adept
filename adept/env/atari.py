import warnings

import numpy as np
import torch

from adept.alias import Observation, Action, Reward, Done, Info, Spec
from adept.module import Environment, Preprocessor
from adept.util import space

try:
    import gymnasium as gym
except ImportError:
    warnings.warn("gymnasium not installed, AtariEnv will not work")
    gym = None


class AtariPreprocessor(Preprocessor):
    def __init__(self, observation_spec: Spec):
        super().__init__()
        self._observation_spec = observation_spec

    def __call__(self, obs: Observation) -> Observation:
        pass

    @property
    def observation_spec(self) -> Spec:
        return self._observation_spec

    @property
    def batch_size(self) -> int:
        return 1


class AtariEnv(Environment):
    def __init__(
        self,
        seed: int,
        env_id: str = "PongNoFrameskip-v4",
        episode_len: float = 10_000,
        skip_rate: int = 4,
        frame_stack: bool = False,
        noop_max: int = 30,
    ):
        super().__init__(seed)

        self._env = gym.make(env_id)
        self._observation_space = self._to_adept_space(self._env.observation_space)
        self._action_space = self._to_adept_space(self._env.action_space)

    @property
    def observation_spec(self) -> Spec:
        return self._observation_space

    @property
    def action_spec(self) -> Spec:
        return self._action_space

    @property
    def n_reward_component(self) -> int:
        return 1

    def step(self, action: Action) -> tuple[Observation, Reward, Done, Info]:
        gym_obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        obs = torch.from_numpy(gym_obs).view(1, *gym_obs.shape)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1)
        return obs, reward, done, info

    def reset(self) -> Observation:
        pass

    def close(self) -> None:
        pass

    def get_preprocessor(self) -> Preprocessor:
        return AtariPreprocessor(self._observation_space)

    def _to_adept_space(self, gym_space) -> Spec:
        if isinstance(gym_space, gym.spaces.Box):
            torch_lows = torch.from_numpy(gym_space.low)
            torch_highs = torch.from_numpy(gym_space.high)
            return space.Box(
                (1,) + gym_space.shape,
                torch_lows,
                torch_highs,
                torch.float32,
                n_batch_dim=1
            )
        elif isinstance(gym_space, gym.spaces.Discrete):
            return space.Discrete(gym_space.n, batch_size=1)


if __name__ == '__main__':
    env = AtariEnv(0)
    print(env.observation_spec.sample())
    print(env.action_spec.sample())
    env.close()

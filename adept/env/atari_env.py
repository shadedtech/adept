from __future__ import annotations

import logging

import gymnasium as gym
import torch
from torch import Tensor

from adept.alias import Observation, Action, Reward, Done, Info, Spec
from adept.env.base.gym_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
    NumpyObsWrapper,
)
from adept.module import Environment, Preprocessor
from adept.util import space

logger = logging.getLogger(__name__)


class AtariPreprocessor(Preprocessor):
    def __init__(self, observation_spec: Spec, batch_size: int = 1):
        super().__init__()
        self._observation_spec = observation_spec
        self._batch_size = batch_size
        self._device = torch.device("cpu")
        # self.transforms = torch.nn.Sequential(
        #     transforms.Grayscale(),
        #     transforms.Resize((84, 84), interpolation=2),
        # )
        # self.scripted_transforms = torch.jit.script(self.transforms)

    @torch.no_grad()
    def __call__(self, obs: Tensor) -> Tensor:
        # Move feature dimension to front for pytorch convention
        # obs = obs.permute(0, 3, 1, 2)
        # obs = self.scripted_transforms(obs)
        obs = obs.to(self._device)
        obs = obs.float() / 255.0
        return obs

    @property
    def observation_spec(self) -> Spec:
        return self._observation_spec

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def to(self, device: torch.device) -> AtariPreprocessor:
        self._device = device
        super().to(device)
        return self


class AtariEnv(Environment):
    def __init__(
        self,
        seed: int,
        env_id: str = "PongNoFrameskip-v4",
        episode_len: float = 10_000,
        skip_rate: int = 4,
        noop_max: int = 30,
        framestack: int = 4,
    ):
        super().__init__(seed)

        env = gym.make(env_id)
        # Same as Sample Factory
        env._max_episode_steps = episode_len
        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=skip_rate)
        env = EpisodicLifeEnv(env)
        # noinspection PyUnresolvedReferences
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, framestack)
        env = NumpyObsWrapper(env)
        env.seed(self._seed)
        self._env = env
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
        # New gym API is weird
        done = terminated or truncated
        # Add batch dimension
        obs = torch.from_numpy(gym_obs)
        reward = torch.tensor(reward).view(1,)
        done_t = torch.tensor(done).int()
        if done:
            obs = self.reset()
        return obs, reward, done_t, info

    def reset(self) -> Observation:
        gym_obs, _ = self._env.reset()
        obs = torch.from_numpy(gym_obs)
        return obs

    def close(self) -> None:
        self._env.close()

    def get_preprocessor(self) -> Preprocessor:
        return AtariPreprocessor(self._observation_space)

    def _to_adept_space(self, gym_space) -> Spec:
        if isinstance(gym_space, gym.spaces.Box):
            torch_lows = torch.from_numpy(gym_space.low)
            torch_highs = torch.from_numpy(gym_space.high)
            return space.Box(
                gym_space.shape,
                torch_lows,
                torch_highs,
            )
        elif isinstance(gym_space, gym.spaces.Discrete):
            return space.Discrete(gym_space.n)


if __name__ == "__main__":
    from adept.util.log_util import setup_logging
    setup_logging(logger)
    env = AtariEnv(0)
    preprocess = env.get_preprocessor()
    print(env.observation_spec.sample())
    print(env.action_spec.sample())
    print(env.reset())
    obs, reward, done, info = env.step(env.action_spec.sample())
    print(obs.shape)
    print(preprocess(obs))
    env.close()

from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Tuple

import numpy as np
import torch
from torch import multiprocessing as mp, Tensor

from adept.alias import Spec, Observation, Reward, Done, Info
from adept.config import configurable
from adept.env.atari import AtariEnv, AtariPreprocessor
from adept.module import Environment, Preprocessor



class AtariPool(Environment):
    n_reward_component: int = 1

    @configurable
    def __init__(
        self,
        env_id: str = "PongNoFrameskip-v4",
        n_sim: int = 64,
        start_seed: int = 0,
    ):
        super().__init__(start_seed)
        mp.set_start_method("spawn", force=True)
        dummy_env = AtariEnv(0)
        dummy_env.close()
        self._observation_spec = dummy_env.observation_spec.set_batch_size(n_sim)
        self._action_spec = dummy_env.action_spec.set_batch_size(n_sim)

        self.batch_size = n_sim
        self.n_sim = n_sim

        self._start_seed = start_seed
        self._obs_sm = self._observation_spec.zeros().share_memory_()
        self._reward_sm = torch.zeros(
            n_sim, self.n_reward_component
        ).share_memory_()
        self._done_sm = torch.zeros(n_sim, dtype=torch.int)
        self._action_sm = self._action_spec.zeros().share_memory_()

        self._procs = []
        self._cxns = []
        for sim_ix in range(self.n_sim):
            seed = start_seed + sim_ix
            parent_cxn, child_cxn = mp.Pipe()
            proc = mp.Process(
                target=worker,
                args=(
                    env_id,
                    sim_ix,
                    child_cxn,
                    seed,
                    self._obs_sm,
                    self._reward_sm,
                    self._done_sm,
                    self._action_sm,
                ),
                daemon=True,
            )
            proc.start()
            self._cxns.append(parent_cxn)
            self._procs.append(proc)

    def step(self, actions: Tensor) -> Tuple[Observation, Reward, Done, Info]:
        self._action_sm.copy_(actions)
        for env_ix in range(self.n_sim):
            self._cxns[env_ix].send("step")
        for cxn in self._cxns:
            cxn.recv()
        return self._obs_sm, self._reward_sm, self._done_sm, {}

    def reset(self) -> Observation:
        for cxn in self._cxns:
            cxn.send("reset")
        for cxn in self._cxns:
            cxn.recv()
        return self._obs_sm

    def close(self) -> None:
        for c in self._cxns:
            c.send("close")
        for p in self._procs:
            p.join()

    def get_preprocessor(self) -> Preprocessor:
        return AtariPreprocessor(self.observation_spec)

    @property
    def observation_spec(self) -> Spec:
        return self._observation_spec

    @property
    def action_spec(self) -> Spec:
        return self._action_spec


def worker(
    env_id: str,
    sim_ix: int,
    cxn: Connection,
    seed: int,
    obs_sm: Tensor,
    reward_sm: Tensor,
    done_sm: Tensor,
    action_sm: Tensor,
):
    env = AtariEnv(seed, env_id)
    torch.manual_seed(seed)
    np.random.seed(seed)
    running = True
    while running:
        try:
            msg = cxn.recv()
            if msg == "step":
                obs, reward, done, info = env.step(action_sm[sim_ix])
                obs_sm[sim_ix].copy_(obs)
                reward_sm[sim_ix].copy_(reward)
                done_sm[sim_ix] = done
                cxn.send(True)
            elif msg == "reset":
                _ = env.reset()
                cxn.send(True)
            elif msg == "close":
                env.close()
                cxn.close()
                running = False
            else:
                raise Exception(f"Unexpected message: {msg}")
        except KeyboardInterrupt:
            env.close()
            cxn.close()
            running = False


if __name__ == '__main__':
    pool = AtariPool()
    obs = pool.reset()
    for _ in range(100):
        obs, reward, done, _ = pool.step(pool.action_spec.sample())
        print(reward)
    pool.close()

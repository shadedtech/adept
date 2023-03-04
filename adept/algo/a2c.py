from __future__ import annotations

import typing

import torch
from torch import nn, Tensor

from adept import util
from adept.algo import _base
from adept.alias import (
    Spec,
    Observation,
    Reward,
    Done,
    Experience,
    HiddenStates,
    Action,
    Losses,
    Metrics,
)
from adept.config import configurable
from adept.module import Actor, Learner
from adept.util import spec, space

if typing.TYPE_CHECKING:
    from adept.net import AutoNetwork
    from adept.run import Updater


class ActorExperience(typing.TypedDict):
    log_probs: torch.Tensor
    entropies: torch.Tensor
    critic: torch.Tensor


class EnvExperience(typing.TypedDict):
    reward: Reward
    done: Done


class A2CActor(Actor):
    @configurable
    def __init__(self, action_spec: Spec, is_training: bool = True):
        super().__init__(action_spec)
        self.action_distributions = nn.ModuleDict(_build_distributions(action_spec))

        self.batch_size = spec.SpecImpl(action_spec, "action").batch_size
        self._output_spec = {
            **spec.to_dict(action_spec, "action"),
            "critic": space.Box((*self.batch_size, 1), -float("inf"), float("inf")),
        }
        if is_training:
            self.train()
        else:
            self.eval()

    def step(
        self,
        obs: Observation,
        hiddens: HiddenStates,
        net: AutoNetwork,
    ) -> tuple[Action, ActorExperience, HiddenStates]:
        out, nxt_hiddens = net.forward(obs, hiddens)
        actions, exp = self._get_experience(out)
        return actions, exp, nxt_hiddens

    def _get_experience(self, out: dict[str, Tensor]) -> tuple[Action, ActorExperience]:
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        actions: dict[str, Tensor] = {}
        for action_key, dist_mod in self.action_distributions.items():
            dist = dist_mod.forward(out[action_key])
            action = dist.sample()
            log_probs.append(normalize_shape_bf(dist.log_prob(action), self.batch_size))
            entropies.append(normalize_shape_bf(dist.entropy(), self.batch_size))
            actions[action_key] = action.cpu()
        log_probs_b = torch.cat(log_probs, dim=-1).sum(-1)
        entropies_b = torch.cat(entropies, dim=-1).sum(-1)
        return spec.from_dict(actions), {
            "log_probs": log_probs_b,
            "entropies": entropies_b,
            "critic": out["critic"].squeeze(1),
        }

    def observe(
        self, next_obs: Observation, rewards: Reward, dones: Done
    ) -> EnvExperience:
        return {"reward": rewards, "done": dones}

    @property
    def output_spec(self) -> Spec:
        return self._output_spec


class A2CLearner(Learner):
    @configurable
    def __init__(self, value_weight: float = 0.5, entropy_weight: float = 3e-4, discount: float = 0.99):
        self._value_weight = value_weight
        self._entropy_weight = entropy_weight
        self._discount = discount

    def step(
        self, net: AutoNetwork, updater: Updater, xp: Experience, step_count: int
    ) -> tuple[Losses, Metrics]:
        bootstrap_values = xp["critic"][-1].detach()
        est_values_rb = xp["critic"][:-1]
        rewards_rb = xp["reward"][:-1].sum(-1)
        dones_rb = xp["done"][:-1]
        log_probs_rb = xp["log_probs"][:-1]
        entropies_rb = xp["entropies"][:-1]
        with torch.no_grad():
            true_values_rb = _base.calc_returns(
                bootstrap_values, rewards_rb, dones_rb, self._discount
            ).detach()
        value_loss = (self._value_weight * (true_values_rb - est_values_rb) ** 2).mean()
        advantages_rb = true_values_rb - est_values_rb.detach()
        policy_loss = (-log_probs_rb * advantages_rb.detach()).mean()
        entropy_loss = (-entropies_rb * self._entropy_weight).mean()
        updater.step(value_loss + policy_loss + entropy_loss)
        losses = {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }
        metrics = {}
        return losses, metrics


def _build_distributions(action_spec: Spec) -> dict[str, util.Distribution]:
    ds = {}
    for k, v in spec.items(action_spec, "action"):
        if type(v) is space.Discrete:
            ds[k] = util.distribution.Categorical(v.n_category)
        elif type(v) is space.Box:
            ds[k] = util.distribution.Normal(v.shape()[-1])
    return ds


def normalize_shape_bf(t: Tensor, batch_size: tuple[int, ...]) -> Tensor:
    """Normalize shape to (B, F)

    If F dimension is missing, it is added.
    """
    if t.dim() < len(batch_size):
        raise ValueError(
            f"Expected tensor to have at least {len(batch_size)} dimensions, "
            f"but got {t.dim()}"
        )
    if t.dim() > len(batch_size) + 1:
        raise ValueError(
            f"Expected tensor to have at most {len(batch_size) + 1} dimensions, "
            f"but got {t.dim()}"
        )
    if t.dim() == len(batch_size):
        t = t.unsqueeze(-1)
    return t


if __name__ == "__main__":
    action_spec = {
        "discrete": util.space.Discrete(2, batch_size=5),
        "continuous": util.space.Box((5, 2), -1, 1, n_batch_dim=1),
    }
    out = {k: v.sample() for k, v in action_spec.items()}
    out["discrete"] = out["discrete"].unsqueeze(-1)
    out["critic"] = torch.rand(5, 1)
    actor = A2CActor(action_spec)
    print(actor._get_experience(out))

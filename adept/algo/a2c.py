import typing

from torch import nn

from adept import util
from adept.alias import (
    Spec,
    Observation,
    Reward,
    Done,
    Experience,
    HiddenStates,
    Action,
    Losses,
    Metrics, )
from adept.config import configurable
from adept.module import Actor, Learner
from adept.util import spec, space

if typing.TYPE_CHECKING:
    from adept.net import AutoNetwork
    from adept.run import Updater


class A2CActor(Actor):
    @configurable
    def __init__(self, action_spec: Spec):
        super().__init__(action_spec)
        self.distributions = nn.ModuleDict(_build_distributions(action_spec))

        batch_size = spec.SpecImpl(action_spec, "action").batch_size
        self._output_spec = {
            **spec.to_dict(action_spec, "action"),
            "critic": space.Box(batch_size, -float("inf"), float("inf")),
        }

    def step(
        self,
        obs: Observation,
        hiddens: HiddenStates,
        net: AutoNetwork,
    ) -> tuple[Action, Experience, HiddenStates]:
        pass

    def observe(
        self, next_obs: Observation, rewards: Reward, dones: Done
    ) -> Experience:
        pass

    @property
    def output_spec(self) -> Spec:
        return self._output_spec


class A2CLearner(Learner):
    @configurable
    def __init__(self, entropy_weight: float = 0.01, discount: float = 0.99):
        self._entropy_weight = entropy_weight
        self._discount = discount

    def step(
        self, net: AutoNetwork, updater: Updater, xp: Experience, step_count: int
    ) -> tuple[Losses, Metrics]:
        pass
        # returns = _base.calc_returns(  # Shape: (R, B)
        #     bootstrap_values, rewards, dones, self._discount
        # ).detach()
        # advantages = returns - values.detach()  # Shape: (R, B)
        # value_loss = (0.5 * (returns - values) ** 2).mean()
        # policy_loss = (-log_probs * advantages.unsqueeze(-1).detach()).mean()
        # entropy_loss = (-entropies * self._entropy_weight).mean()
        # updater.step(value_loss + policy_loss + entropy_loss)
        # losses = {
        #     "value_loss": value_loss.item(),
        #     "policy_loss": policy_loss.item(),
        #     "entropy_loss": entropy_loss.item(),
        # }
        # metrics = {}
        # return losses, metrics


def _build_distributions(action_spec: Spec) -> dict[str, util.Distribution]:
    ds = {}
    for k, v in spec.items(action_spec, "action"):
        if type(v) is space.Discrete:
            ds[k] = util.distribution.Categorical(v.n_category)
        elif type(v) is space.Box:
            ds[k] = util.distribution.Normal(v.shape()[-1])
    return ds

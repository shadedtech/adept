import logging
import os

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from adept import util
from adept.config import CONFIG_MANAGER
from adept.config import configurable
from adept.module import Environment, Actor, Learner, ExpBuf, Preprocessor
from adept.net import AutoNetwork
from adept.run import _base

logger = logging.getLogger(__name__)


@configurable
def main(
    gpu_id: int = 0,
    env: str = "adept.env.AtariEnv",
    actor: str = "adept.algo.A2CActor",
    learner: str = "adept.algo.A2CLearner",
    expbuf: str = "adept.expbuf.Rollout",
    optimizer: str = "adept.optim.Adam",
    seed: int = 0,
    logdir: str = "/tmp/adept_logs",
    experiment_tag: str = None,
    resume_path: str = None,
    n_step: int = 10_000_000,
    checkpoint_interval: int = 1_000_000,
):
    _base.check_omp_num_threads()
    with util.import_object(env)() as env:
        actor = util.import_object(actor)(env.action_spec)
        learner = util.import_object(learner)()
        expbuf = util.import_object(expbuf)()
        print(CONFIG_MANAGER.to_yaml())
        device = _base.get_device(gpu_id)
        preprocessor = env.get_preprocessor().to(device)
        net = AutoNetwork(
            spec.input_shapes(preprocessor.observation_spec), actor.output_shapes()
        ).to(device)
        net = _base.init_network(net, is_train=True, rundir=resume_path)
        optimizer = util.import_object(optimizer)(net.parameters())
        run(
            device,
            env,
            actor,
            learner,
            expbuf,
            preprocessor,
            net,
            optimizer,
            seed,
            logdir,
            experiment_tag,
            resume_path,
        )


def run(
    device: torch.device,
    env: Environment,
    actor: Actor,
    learner: Learner,
    expbuf: ExpBuf,
    preprocessor: Preprocessor,
    network: AutoNetwork,
    optimizer: Optimizer,
    seed: int = 0,
    logdir: str = "/tmp/adept_logs",
    experiment_tag: str = None,
    resume_path: str = None,
    n_step: int = 10_000_000,
    checkpoint_interval: int = 1_000_000,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rundir_path = _base.get_rundir(__file__, logdir, experiment_tag, resume_path)
    os.makedirs(rundir_path, exist_ok=True)
    logger.info(f"Logging to {rundir_path}")
    updater = _base.BasicUpdater(network, optimizer)
    writer = CheckpointWriter(rundir_path)
    tb_writer = SummaryWriter(rundir_path)
    step_count = _base.get_step_count(rundir_path, resume_path)
    next_save = _base.get_next_save(checkpoint_interval, step_count)
    ep_rewards = torch.zeros(env.batch_size())
    hidden_states = network.new_hidden_states(device, env.batch_size())
    obs = preprocessor(env.reset())
    while step_count < n_step:
        actions, actor_xp, hidden_states = actor.step(
            obs,
            hidden_states,
            network,
        )
        nxt_obs, rewards, dones, infos = env.step(actions)
        nxt_obs = preprocessor(nxt_obs)
        env_xp = actor.observe(nxt_obs, rewards.to(device), dones.to(device))
        ready = expbuf.step(actor_xp, env_xp)
        step_count += env.batch_size()
        ep_rewards += rewards.sum(dim=-1)
        obs = nxt_obs
        for batch_ix, done in enumerate(dones):
            if done:
                tb_writer.add_scalar("reward", ep_rewards[batch_ix], step_count)
                ep_rewards[batch_ix].zero_()
                # TODO reset hidden states
        if ready:
            # TODO backwards pass and housekeeping
            losses, metrics = learner.step(network, updater, expbuf.to_dict(), step_count)

if __name__ == "__main__":
    from adept.util import log_util, spec, CheckpointWriter

    log_util.setup_logging()
    main()

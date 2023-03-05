import logging
import os
import time
from collections import deque

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from adept import util
from adept.config import CONFIG_MANAGER
from adept.config import configurable
from adept.module import Environment, Actor, Learner, ExpBuf, Preprocessor
from adept.net import AdeptNetwork
from adept.run import _base

logger = logging.getLogger(__name__)


@configurable
def main(
    gpu_id: int = 0,
    env: str = "adept.env.AtariPool",
    actor: str = "adept.algo.A2CActor",
    learner: str = "adept.algo.A2CLearner",
    expbuf: str = "adept.rollout.Rollout",
    optimizer: str = "adept.optim.Adam",
    seed: int = 0,
    logdir: str = "/tmp/adept_logs",
    experiment_tag: str = None,
    resume_path: str = None,
    n_step: int = 10_000_000,
    checkpoint_interval: int = 1_000_000,
):
    _base.check_omp_num_threads()
    device = _base.get_device(gpu_id)
    env = util.import_object(env)()
    actor = util.import_object(actor)(env.action_spec).to(device)
    learner = util.import_object(learner)()
    expbuf = util.import_object(expbuf)()
    print(CONFIG_MANAGER.to_yaml())
    preprocessor = env.get_preprocessor().to(device)
    net = AdeptNetwork(
        preprocessor.observation_spec,
        actor.output_spec,
    ).to(device)
    net = _base.init_network(net, is_train=True, rundir=resume_path)
    optimizer = util.import_object(optimizer)(net.parameters())
    # TODO init optimizer, actor, preprocessor from resume
    try:
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
            n_step,
            checkpoint_interval,
        )
    finally:
        env.close()


def run(
    device: torch.device,
    env: Environment,
    actor: Actor,
    learner: Learner,
    expbuf: ExpBuf,
    preprocessor: Preprocessor,
    network: AdeptNetwork,
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
    writer = util.CheckpointWriter(rundir_path)
    tb_writer = SummaryWriter(rundir_path)
    step_count = _base.get_step_count(rundir_path, resume_path)
    next_save = _base.get_next_save(checkpoint_interval, step_count)
    ep_rewards = torch.zeros(preprocessor.batch_size)
    hidden_states = network.new_hidden_states(device, preprocessor.batch_size)
    obs = preprocessor(env.reset())
    delta_times = deque(maxlen=100)
    while step_count < n_step:
        start_time = time.perf_counter()
        actions, actor_xp, hidden_states = actor.step(
            obs,
            hidden_states,
            network,
        )
        nxt_obs, rewards, dones, infos = env.step(actions)
        nxt_obs = preprocessor(nxt_obs)
        env_xp = actor.observe(nxt_obs, rewards.to(device), dones.to(device))
        ready = expbuf.step(actor_xp, env_xp)
        step_count += preprocessor.batch_size
        ep_rewards += rewards.sum(dim=-1)
        obs = nxt_obs
        for batch_ix, done in enumerate(dones):
            if done:
                tb_writer.add_scalar("reward", ep_rewards[batch_ix], step_count)
                ep_rewards[batch_ix].zero_()
                hidden_states = _base.reset_hidden_states(
                    batch_ix, hidden_states, network.new_hidden_states(device)
                )
                logger.info(
                    f"STEP: {step_count} "
                    f"REWARD: {ep_rewards[batch_ix]} "
                    f"SPS: {_base.get_steps_per_second(delta_times, preprocessor.batch_size)}"
                )
        if ready:
            losses, metrics = learner.step(
                network, updater, expbuf.to_dict(), step_count
            )
            expbuf.reset()
            for t in hidden_states.values():
                t.detach_()
            log_util.write_summaries(
                tb_writer, step_count, losses, metrics, network.named_parameters()
            )
        if step_count >= next_save:
            writer.save_network(network, step_count)
            writer.save_optimizer(optimizer, step_count)
            writer.save_actor(actor, step_count)
            writer.save_preprocessor(preprocessor, step_count)
            next_save += checkpoint_interval
        delta_times.append(time.perf_counter() - start_time)


if __name__ == "__main__":
    from adept.util import log_util

    log_util.setup_logging()
    main()

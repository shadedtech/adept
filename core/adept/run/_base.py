from __future__ import annotations

import abc
import datetime
import logging
import os
import statistics
import warnings
from collections import deque

import torch
from torch import nn, optim
from typing import Optional

from adept.config import configurable
from adept.util import log_util
from adept.util.log_util import RunDir

logger = logging.getLogger(__name__)


def is_omp_num_threads_set() -> bool:
    if "OMP_NUM_THREADS" in os.environ:
        num_threads = os.environ["OMP_NUM_THREADS"]
        logger.debug(f"OMP_NUM_THREADS set to {num_threads}")
        return num_threads == "1"
    else:
        return False


def check_omp_num_threads():
    if not is_omp_num_threads_set():
        warnings.warn(
            "OMP_NUM_THREADS is not set. `export OMP_NUM_THREADS=1` for optimal performance.",
            stacklevel=2,
        )


def get_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device("cuda:{}".format(gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    return device


def init_network(
    net: nn.Module,
    is_train: bool,
    rundir: Optional[str] = None,
    netpath: Optional[str] = None,
) -> nn.Module:
    if rundir:
        rundir = log_util.RunDir(rundir)
        path = rundir.latest_network_path()
    elif netpath:
        path = netpath
    if rundir or netpath:
        net.load_state_dict(
            torch.load(
                path,
                map_location=lambda storage, loc: storage,
            )
        )
    return net.train() if is_train else net.eval()


def get_run_id(
    script_path: str, tag: Optional[str], timestamp: Optional[str] = None
) -> str:
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_path, _ = os.path.splitext(script_path)
    script_name = os.path.split(script_path)[-1]
    parts = [tag, script_name, timestamp] if tag else [script_name, timestamp]
    return "_".join(parts)


def get_rundir(
    script_path: str,
    logdir: str,
    tag: Optional[str],
    resume: Optional[str],
    timestamp: [Optional[str]] = None,
) -> str:
    if resume:
        rundir = os.path.abspath(resume)
    else:
        run_id = get_run_id(script_path, tag, timestamp)
        rundir = os.path.join(os.path.abspath(logdir), run_id)
    return rundir


def get_next_save(epoch_sz: float, step_count: int) -> int:
    if step_count == 0:
        return step_count
    epoch_sz = int(epoch_sz)
    epoch_ix = step_count // epoch_sz
    return (epoch_ix + 1) * epoch_sz


def get_step_count(rundir_path: str, resume: Optional[str]) -> int:
    return RunDir(rundir_path).latest_epoch() if resume else 0


def get_steps_per_second(dts: deque, batch_sz: int) -> str:
    return (
        str(batch_sz / statistics.mean(dts))
        if len(dts) == dts.maxlen
        else "Calculating..."
    )


class Updater(abc.ABC):
    @abc.abstractmethod
    def step(self, loss: torch.Tensor) -> Updater:
        ...


class BasicUpdater(Updater):
    @configurable
    def __init__(
        self,
        network: nn.Module,
        optimizer: optim.Optimizer,
        grad_norm_clip: float = 0.5,
    ):
        self._network = network
        self._optimizer = optimizer
        self._grad_norm_clip = grad_norm_clip

    def step(self, loss: torch.Tensor) -> BasicUpdater:
        self._optimizer.zero_grad()
        loss.backward()
        if self._grad_norm_clip:
            nn.utils.clip_grad_norm_(
                self._network.parameters(), self._grad_norm_clip
            )
        self._optimizer.step()
        return self

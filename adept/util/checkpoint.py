from __future__ import annotations

import logging
import os
import typing

import torch
from torch import nn
from torch.optim import Optimizer

if typing.TYPE_CHECKING:
    from adept.net import AutoNetwork
    from adept.module import Actor, Preprocessor

logger = logging.getLogger(__name__)


class CheckpointWriter:
    def __init__(self, rundir: str):
        self._run_dir = rundir

    def save_network(self, network: AutoNetwork, step_count: int) -> None:
        save_dir = os.path.join(self._run_dir, str(step_count))
        save_module(network, save_dir, f"net_{step_count}.pth")
        logger.info("Network saved on step", step_count)

    def save_actor(self, network: Actor, step_count: int) -> None:
        save_dir = os.path.join(self._run_dir, str(step_count))
        save_module(network, save_dir, f"actor_{step_count}.pth")

    def save_preprocessor(self, network: Preprocessor, step_count: int) -> None:
        save_dir = os.path.join(self._run_dir, str(step_count))
        save_module(network, save_dir, f"preprocessor_{step_count}.pth")

    def save_optimizer(self, optimizer: Optimizer, step_count: int) -> None:
        save_dir = os.path.join(self._run_dir, str(step_count))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            optimizer.state_dict(),
            os.path.join(save_dir, f"optimizer_{step_count}.pth"),
        )


def save_module(network: nn.Module, path: str, filename: str):
    os.makedirs(path, exist_ok=True)
    torch.save(
        network.state_dict(),
        os.path.join(path, filename),
    )

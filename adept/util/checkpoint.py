from __future__ import annotations

import os
import typing

import torch
from torch.optim import Optimizer
import logging

if typing.TYPE_CHECKING:
    from adept.net import AutoNetwork

logger = logging.getLogger(__name__)


class CheckpointWriter:
    def __init__(self, rundir: str):
        self._run_dir = rundir

    def save_network(self, network: AutoNetwork, step_count: int) -> None:
        save_dir = os.path.join(self._run_dir, str(step_count))
        save_network(network, save_dir, f"model_{step_count}.pth")
        logger.info("Network saved on step", step_count)

    def save_optimizer(self, optimizer: Optimizer, step_count: int) -> None:
        save_dir = os.path.join(self._run_dir, str(step_count))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            optimizer.state_dict(),
            os.path.join(save_dir, f"optimizer_{step_count}.pth"),
        )


def save_network(network: AutoNetwork, path: str, filename: str):
    os.makedirs(path, exist_ok=True)
    torch.save(
        network.state_dict(),
        os.path.join(path, filename),
    )

import datetime
from os import listdir
from os import path
from typing import List
from typing import Optional

from omegaconf import DictConfig
from omegaconf import OmegaConf


def get_rundir(
    script_path: str,
    logdir: str,
    tag: Optional[str],
    resume: Optional[str],
    timestamp: [Optional[str]] = None,
) -> str:
    if resume:
        rundir = path.abspath(resume)
    else:
        run_id = get_run_id(script_path, tag, timestamp)
        rundir = path.join(path.abspath(logdir), run_id)
    return rundir


def get_run_id(
    script_path: str, tag: Optional[str], timestamp: Optional[str] = None
) -> str:
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_path, _ = path.splitext(script_path)
    script_name = path.split(script_path)[-1]
    parts = [tag, script_name, timestamp] if tag else [script_name, timestamp]
    return "_".join(parts)


class RunDir:
    def __init__(self, rundir_path: str):
        self._rundir_path = path.abspath(rundir_path)

    def epochs(self) -> List[int]:
        epochs = []
        for item in listdir(self._rundir_path):
            item_path = path.join(self._rundir_path, item)
            if path.isdir(item_path):
                if item.isnumeric():
                    item_int = int(item)
                    if item_int >= 0:
                        epochs.append(item_int)
        return list(sorted(epochs))

    def latest_epoch(self) -> int:
        epochs = self.epochs()
        return max(epochs) if epochs else 0

    def latest_epoch_path(self) -> str:
        return path.join(self._rundir_path, str(self.latest_epoch()))

    def latest_network_path(self) -> str:
        epoch_path = self.latest_epoch_path()
        net_file = [f for f in listdir(epoch_path) if ("model" in f)][0]
        return path.join(epoch_path, net_file)

    def latest_optimizer_path(self) -> str:
        epoch_path = self.latest_epoch_path()
        optim_file = [f for f in listdir(epoch_path) if ("optimizer" in f)][0]
        return path.join(epoch_path, optim_file)

    def epoch_path(self, epoch: int) -> str:
        return path.join(self._rundir_path, str(epoch))

    def network_path(self, epoch: int) -> str:
        epoch_path = self.epoch_path(epoch)
        net_file = [f for f in listdir(epoch_path) if ("model" in f)][0]
        return path.join(epoch_path, net_file)

    def optimizer_path(self, epoch: int) -> str:
        epoch_path = self.epoch_path(epoch)
        optim_file = [
            f for f in listdir(self.latest_epoch_path()) if ("optimizer" in f)
        ][0]
        return path.join(epoch_path, optim_file)

    def cfg_path(self) -> str:
        return path.join(self._rundir_path, "cfg.yaml")

    def load_cfg(self) -> DictConfig:
        return OmegaConf.load(self.cfg_path())

    def eval_path(self) -> str:
        return path.join(self._rundir_path, "eval.csv")

    def best_eval_path(self) -> str:
        return path.join(self._rundir_path, "best.csv")

    def debug_path(self):
        return path.join(self._rundir_path, "debug")

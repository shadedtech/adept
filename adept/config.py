import inspect
from functools import wraps
from os.path import abspath
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import TypeVar

from omegaconf import DictConfig
from omegaconf import OmegaConf


_T = TypeVar("_T")


class _ConfigManager:
    def __init__(self):
        self.cli_conf = OmegaConf.from_cli()
        self.user_conf = {}
        if "config" in self.cli_conf:
            self.user_conf = _load_user_conf(self.cli_conf["config"])
            del self.cli_conf["config"]
        self._cfg = self.provided()

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(self._cfg, sort_keys=True)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self._cfg.items()}

    def contains(self, tag: str) -> bool:
        return tag in self._cfg

    def cfg(self) -> DictConfig:
        return self._cfg

    def provided(self) -> DictConfig:
        return OmegaConf.create({**self.user_conf, **self.cli_conf})

    def _add_config(self, conf: Dict[str, Any]):
        self._cfg = OmegaConf.create({**self._cfg, **conf})


def _load_user_conf(user_conf_path: Optional[str]) -> DictConfig:
    return OmegaConf.load(abspath(user_conf_path))


CONFIG_MANAGER = _ConfigManager()


def configurable(f: Type[_T]) -> Type[_T]:
    f.tag = f.__name__
    if inspect.isclass(f):
        raise NotImplementedError("Configurable class not implemented, decorate the __init__ method instead")

    @wraps(f)
    def wrapper(*args, **kwargs):
        if f.__name__ == "__init__":
            f.tag = args[0].__class__.__name__
        params = inspect.signature(f).parameters
        # params = {k: v for k, v in params.items() if k != "self"}
        if "tag" in kwargs:
            f.tag = kwargs["tag"]
            del kwargs["tag"]
        conf = {k: v for k, v in CONFIG_MANAGER.provided().get(f.tag, {}).items()}
        for i, (name, param) in enumerate(params.items()):
            if param.default is not param.empty:
                if i < len(args):
                    conf[name] = args[i]
                elif name in kwargs:
                    conf[name] = kwargs[name]
                elif name in conf:
                    kwargs[name] = conf[name]
                else:
                    conf[name] = param.default
        CONFIG_MANAGER._add_config({f.tag: conf})
        return f(*args, **kwargs)

    wrapper.is_configurable = True
    return wrapper


if __name__ == "__main__":

    class Network:
        @configurable
        def __init__(self, n_layer: int = 3, n_hidden: int = 5):
            print("n_layer", n_layer)
            print("n_hidden", n_hidden)

    network = Network(12)

    print(CONFIG_MANAGER.to_yaml())

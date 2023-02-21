from typing import Iterable

import torch
from torch.optim import Adam as TorchAdam

from adept.config import configurable


class Adam(TorchAdam):
    @configurable
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 2.5e-4,
        beta0: float = 0.9,
        beta1: float = 0.999,
        eps: float = 1e-5,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ):
        super().__init__(
            params,
            lr,
            (beta0, beta1),
            eps,
            weight_decay,
            amsgrad,
        )

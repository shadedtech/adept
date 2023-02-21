from typing import List, Optional

import torch


def calc_returns(
    bootstrap_values_b: torch.Tensor,
    rewards_rb: torch.Tensor,
    dones_rb: torch.IntTensor | torch.BoolTensor,
    discount: float,
) -> torch.Tensor:
    dones_rb = dones_rb.bool()
    target_returns_b = bootstrap_values_b  # Shape: (B)
    rollout_len = len(dones_rb)
    nstep_returns_b: List[Optional[torch.Tensor]] = [
        None for _ in range(rollout_len)
    ]
    for r_ix in reversed(range(rollout_len)):
        reward = rewards_rb[r_ix]  # Shape: (B)
        dones_mask = 1.0 - dones_rb[r_ix].float()  # Shape: (B)
        target_returns_b = reward + discount * target_returns_b * dones_mask
        nstep_returns_b[r_ix] = target_returns_b
    return torch.stack(nstep_returns_b)

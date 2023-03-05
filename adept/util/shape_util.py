from __future__ import annotations

from torch import Tensor


def norm_tensor_bf(t: Tensor, batch_size: tuple[int, ...]) -> Tensor:
    """Normalize shape to (B, F)

    If F dimension is missing, it is added.
    """
    if t.dim() < len(batch_size):
        raise ValueError(
            f"Expected tensor to have at least {len(batch_size)} dimensions, "
            f"but got {t.dim()}"
        )
    if t.dim() > len(batch_size) + 1:
        raise ValueError(
            f"Expected tensor to have at most {len(batch_size) + 1} dimensions, "
            f"but got {t.dim()}"
        )
    if t.dim() == len(batch_size):
        t = t.unsqueeze(-1)
    return t


def norm_shape_bf(shape: tuple[int, ...], batch_size: tuple[int, ...]) -> tuple[int, ...]:
    """Normalize shape to (B, F)

    If F dimension is missing, it is added.
    """
    if len(shape) < len(batch_size):
        raise ValueError(
            f"Expected shape to have at least {len(batch_size)} dimensions, "
            f"but got {len(shape)}"
        )
    if len(shape) > len(batch_size) + 1:
        raise ValueError(
            f"Expected shape to have at most {len(batch_size) + 1} dimensions, "
            f"but got {len(shape)}"
        )
    if len(shape) == len(batch_size):
        shape = (*shape, 1)
    return shape

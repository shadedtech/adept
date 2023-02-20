from torch import nn


def get_num_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def calc_conv_dim(
    dim_size: int, kernel_size: int, stride: int, padding: int, dilation: int
) -> int:
    numerator = dim_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return numerator // stride + 1

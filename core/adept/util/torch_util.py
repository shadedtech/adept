from torch import nn


def get_num_params(model: nn.Module, non_embedding: bool = True):
    """
    Return the number of parameters in the model.
    For non-embedding count (default), the position embeddings get subtracted.
    The token embeddings would too, except due to the parameter sharing these
    params are actually used as weights in the final layer, so we include them.
    """
    n_params = sum(p.numel() for p in model.parameters())
    if non_embedding:
        n_params -= model.position_encoder.weight.numel()
    return n_params


def calc_conv_dim(
    dim_size: int, kernel_size: int, stride: int, padding: int, dilation: int
) -> int:
    numerator = dim_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return numerator // stride + 1

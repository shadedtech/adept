"""NanoGPT with light modification for RL.

Mainly,
* Ability to turn off causal attention
* Input is not tokenized (raw feature vector)
* Type annotations and dimensions are more explicit
* Add padding if input sequence is too short

Original code by Andrej Karpathy (https://github.com/karpathy/nanoGPT)
"""
import math
from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from adept.alias import HiddenState, Shape
from adept.config import configurable
from adept.net.base import NetMod2D


# @torch.jit.script good, but don't use with torch.compile
def new_gelu(x: Tensor) -> Tensor:
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class GPT2(NetMod2D):
    @configurable
    def __init__(
        self,
        name: str,
        input_shape: Tuple[int, ...],
        n_head: int = 8,
        n_layer: int = 12,
        seq_len: int = 32,
        n_feature: int = 128,
        bias: bool = True,
        p_dropout: float = 0.2,
        is_causal: bool = False,
        pad_value: float = 0.0
    ):
        super().__init__(name, input_shape)
        self.in_proj = nn.Linear(seq_len, n_feature, bias=False)
        self.position_encoder = nn.Embedding(seq_len, n_feature)
        self.blocks = nn.ModuleList(
            [
                Block(n_head, seq_len, n_feature, bias, p_dropout, is_causal)
                for _ in range(n_layer)
            ]
        )

        self._n_feature = n_feature
        self._seq_len = seq_len
        self._pad_value = pad_value

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (LayerNorm, nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _forward(self, x_bfs: Tensor, hiddens: HiddenState) -> tuple[Tensor, Optional[HiddenState]]:
        b, f, s = x_bfs.shape
        x_bsf = x_bfs.permute(0, 2, 1)
        # zero pad if sequence is shorter than seq_len
        if s < self._seq_len:
            tmp = torch.empty(b, self._seq_len, f, device=x_bsf.device).fill_(self._pad_value)
            tmp[:, :s, :] = x_bsf
            x_bsf = tmp
        # x_bsf = self.in_proj(x_bsf)
        # x_bsf = x_bsf + self.position_encoder(
        #     torch.arange(x_bsf._non_batch_shape[1], device=x_bsf.device)
        # )
        for block in self.blocks:
            x_bsf = block(x_bsf)
        return x_bsf.permute(0, 2, 1), None

    def _output_shape(self) -> Shape:
        return self._n_feature, self._seq_len


class Block(nn.Module):
    def __init__(
        self,
        n_head: int,
        seq_len: int = 128,
        n_feature: int = 512,
        bias: bool = True,
        p_dropout: float = 0.2,
        is_causal: bool = False,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(n_feature, bias=bias)
        self.attn = SelfAttention(
            n_head, seq_len, n_feature, bias, p_dropout, is_causal
        )
        self.ln_2 = LayerNorm(n_feature, bias=bias)
        self.mlp = MLP(n_feature, n_feature, bias=bias, p_dropout=p_dropout)

    def forward(self, x_bsf: Tensor) -> Tensor:
        x_bsf = x_bsf + self.attn(self.ln_1(x_bsf))
        x_bsf = x_bsf + self.mlp(self.ln_2(x_bsf))
        return x_bsf


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, bias: bool, p_dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(n_in, n_out * 4, bias=bias)
        self.c_proj = nn.Linear(n_out * 4, n_out, bias=bias)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.c_proj(new_gelu(self.c_fc(x))))


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(
        self,
        n_head: int = 12,
        seq_len: int = 128,
        n_feature: int = 512,
        bias: bool = True,
        dropout: float = 0.2,
        is_causal: bool = False,
    ):
        super().__init__()
        assert n_feature % n_head == 0
        self.n_feature = n_feature
        self.n_head = n_head
        self.dropout = dropout
        self.is_causal = is_causal
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_feature, 3 * n_feature, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_feature, n_feature, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(F, "scaled_dot_product_attention") and dropout == 0.0
        if not self.flash and is_causal:
            print(
                "WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len),
            )

    def forward(self, x_bsf: Tensor) -> Tensor:
        b, s, f = x_bsf.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x_bsf).split(self.n_feature, dim=2)
        k_bhsf = k.view(b, s, self.n_head, f // self.n_head).transpose(1, 2)
        q_bhsf = q.view(b, s, self.n_head, f // self.n_head).transpose(1, 2)
        v_bhsf = v.view(b, s, self.n_head, f // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, H, S, F) x (B, H, F, S) -> (B, H, S, S)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y_bhsf = F.scaled_dot_product_attention(
                q_bhsf,
                k_bhsf,
                v_bhsf,
                attn_mask=None,
                dropout_p=self.dropout,
                is_causal=self.is_causal,
            )
        else:
            # manual implementation of attention
            att_bhss = (q_bhsf @ k_bhsf.transpose(-2, -1)) * (
                1.0 / k_bhsf.size(-1) ** 0.5
            )
            if self.is_causal:
                att_bhss = att_bhss.masked_fill(
                    self.bias[:, :, :s, :s] == 0, float("-inf")
                )
            att_bhss = F.softmax(att_bhss, dim=-1)
            att_bhss = self.attn_dropout(att_bhss)
            y_bhsf = att_bhss @ v_bhsf
        y_bhsf = (
            y_bhsf.transpose(1, 2).contiguous().view(b, s, f)
        )  # re-assemble all head outputs side by side

        # output projection
        y_bhsf = self.resid_dropout(self.c_proj(y_bhsf))
        return y_bhsf

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from adept.alias import Shape, HiddenState
from adept.config import configurable
from adept.net.base import NetMod1D


class LSTM(NetMod1D):
    @configurable
    def __init__(
        self, name: str, input_shape: Shape, n_hidden: int = 512
    ):
        super().__init__(name, input_shape)
        self._n_hidden = n_hidden
        (f,) = input_shape
        self.lstm_cell = nn.LSTMCell(f, self._n_hidden)
        self.lstm_cell.bias_ih.data.fill_(0)
        self.lstm_cell.bias_hh.data.fill_(0)

    def _forward(
        self, x: torch.Tensor, hiddens: HiddenState
    ) -> tuple[torch.Tensor, Optional[HiddenState]]:
        h_state, cell_state = self.lstm_cell(x, torch.unbind(hiddens, dim=1))
        return h_state, torch.stack([h_state, cell_state], dim=1)

    def _output_shape(self) -> Shape:
        return (self._n_hidden,)

    def new_hidden_states(
        self, device: torch.device, batch_sz: int = 1
    ) -> Optional[HiddenState]:
        return torch.zeros(batch_sz, 2, self._n_hidden, device=device)


if __name__ == "__main__":
    lstm = LSTM("body", (256,))
    print(lstm._n_hidden)
    test = torch.ones(4, 256)
    test_hiddens = torch.zeros(4, 2, 512)
    result = lstm.forward(test, test_hiddens)
    print(result)

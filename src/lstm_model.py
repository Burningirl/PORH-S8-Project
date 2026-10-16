# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:43:35 2025

@author: Burningirl
"""

import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=2, output_size=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, _) = self.lstm(packed)
        out = self.fc(h_n[-1])  # Use last hidden state
        return out


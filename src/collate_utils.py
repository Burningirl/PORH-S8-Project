# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 17:22:43 2025

@author: Burningirl
"""

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    X = [item[0] for item in batch]  # list of (T, 2)
    y = [item[1] for item in batch]  # list of (num_targets,)

    lengths = torch.tensor([len(seq) for seq in X])
    X_padded = pad_sequence(X, batch_first=True)  # shape: (B, T, 2)
    y_batch = torch.stack(y)

    return X_padded, y_batch, lengths


# ===== TEST RAPIDE =====
'''
from torch.utils.data import DataLoader
from dataset_creation import PORHDataset

ds = PORHDataset(
    "C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/processed",
    target_keys=[
        "PF - Peak flow",
        "TM - Time to max*",
        "AO - Occlusion area (Unit*sec.)",
        "AH - Hyperaemia area* (Unit*sec.)"
    ],
    split_file="C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/split/train_split.json"
)

loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)

for X, y, lengths in loader:
    print(X.shape, y.shape, lengths)
    break
'''


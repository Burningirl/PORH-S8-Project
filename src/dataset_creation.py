# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:06:54 2025

@author: Burningirl

"""

import os
import json
import torch
import numpy as np
import pandas as pd  # requis pour conversion de temps
from torch.utils.data import Dataset

class PORHDataset(Dataset):
    def __init__(self, data_dir, target_keys, split_file=None):
        self.data_dir = data_dir
        self.target_keys = target_keys
        self.split_file = split_file

        # Explicitly define the loading of splits
        with open(split_file, 'r') as f:
            self.file_list = [os.path.join(data_dir, fname + "_signal.npy") for fname in json.load(f)]

        # Load normalization stats
        norm_path = os.path.join(data_dir, "normalization.json")
        with open(norm_path, "r") as f:
            self.norm = json.load(f)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        signal_path = self.file_list[idx]
        base_name = os.path.basename(signal_path).replace("_signal.npy", "")

        # Load signal
        signal = np.load(signal_path)
        X_tensor = torch.tensor(signal, dtype=torch.float32)
        length = X_tensor.shape[0]

        # Load labels
        label_path = os.path.join(self.data_dir, f"{base_name}_params.json")
        with open(label_path, "r") as f:
            params = json.load(f)

        # Convert labels properly and normalize
        def to_seconds(v):
            try:
                if isinstance(v, str) and ":" in v:
                    t = pd.to_datetime(v).time()
                    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
                return float(v)
            except:
                return np.nan

        y = []
        for key in self.target_keys:
            val = to_seconds(params.get(key, np.nan))
            if not np.isnan(val) and not np.isinf(val):
                mean, std = self.norm[key]["mean"], self.norm[key]["std"]
                norm_val = 0 if std == 0 else (val - mean) / std
            else:
                norm_val = np.nan
            y.append(norm_val)
        #print("Normalized labels (y):", y)

        return X_tensor, torch.tensor(y, dtype=torch.float32), length


# ===== TEST RAPIDE =====
'''
ds = PORHDataset(
    data_dir="C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/processed",
    target_keys=[
        "PF - Peak flow",
        "TM - Time to max*",
        "AO - Occlusion area (Unit*sec.)",
        "AH - Hyperaemia area* (Unit*sec.)"
    ],
    split_file="C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/split/train_split.json"
)

print("Nombre d'échantillons dans ce split :", len(ds))

'''
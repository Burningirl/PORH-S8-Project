# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:02:09 2025

@author: Burningirl
"""

import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
import pandas as pd

class AlignedPORHDataset(Dataset):
    def __init__(self, aligned_dir, params_dir, normalization_file, target_keys):
        self.aligned_dir = aligned_dir
        self.params_dir = params_dir
        self.target_keys = target_keys

        # Get list of base filenames
        self.base_names = [f.replace("_aligned_signal.npy", "")
                           for f in os.listdir(aligned_dir)
                           if f.endswith("_aligned_signal.npy")]

        # Load normalization stats
        with open(normalization_file, "r") as f:
            self.norm = json.load(f)

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        base = self.base_names[idx]

        # Load aligned signal
        signal_path = os.path.join(self.aligned_dir, f"{base}_aligned_signal.npy")
        signal = np.load(signal_path)  # shape: (T, 2) — filtered + derivative
        signal = signal[::2] 
        X = torch.tensor(signal, dtype=torch.float32)

        # Load target parameters
        params_path = os.path.join(self.params_dir, f"{base}_params.json")
        with open(params_path, "r") as f:
            params = json.load(f)

        # Convert time-like values to seconds
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
                mean = self.norm[key]["mean"]
                std = self.norm[key]["std"]
                norm_val = 0 if std == 0 else (val - mean) / std
            else:
                norm_val = np.nan  # will need to be masked or skipped
            y.append(norm_val)

        y_tensor = torch.tensor(y, dtype=torch.float32)

        return X, y_tensor

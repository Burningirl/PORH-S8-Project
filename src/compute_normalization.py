# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:39:05 2025

@author: Burningirl
"""

import os
import json
import numpy as np
import pandas as pd

data_dir = "C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/new_processed"
target_keys = [
    "RF - Rest flow",
    "BZ - Biological zero",
    "PF - Peak flow",
    "TM - Time to max*",
    "AO - Occlusion area (Unit*sec.)",
    "AH - Hyperaemia area* (Unit*sec.)"
]

values_by_key = {k: [] for k in target_keys}

def to_seconds(v):
    try:
        if isinstance(v, str) and ":" in v:
            t = pd.to_datetime(v).time()
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
        return float(v)
    except:
        return np.nan

for filename in os.listdir(data_dir):
    if filename.endswith("_params.json"):
        with open(os.path.join(data_dir, filename), "r") as f:
            params = json.load(f)
        for k in target_keys:
            val = params.get(k, None)
            val = to_seconds(val)
            if isinstance(val, (int, float)) and not (np.isnan(val) or np.isinf(val)):
                values_by_key[k].append(val)

# Check values explicitly
for k, vals in values_by_key.items():
    if not vals:
        print(f"⚠️ No valid values found for {k}.")
    else:
        print(f"{k}: {len(vals)} valid values found.")

# Compute stats
norm_stats = {
    k: {
        "mean": float(np.mean(values)) if values else 0,
        "std": float(np.std(values)) if values else 1
    }
    for k, values in values_by_key.items()
}

# Save to file
with open(os.path.join(data_dir, "normalization.json"), "w") as f:
    json.dump(norm_stats, f, indent=4)

print("✅ Normalization stats saved correctly.")

# -*- coding: utf-8 -*-
"""
Created on Thu May 15 22:54:16 2025

@author: Burningirl
"""
# data_separation.py (synchronisé avec load_and_preprocess à 3 retours)

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from preprocessing_script2 import load_and_preprocess

input_dir = Path("C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/new_raw/")
output_dir = Path("C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/new_processed/")
output_dir.mkdir(parents=True, exist_ok=True)

def json_compatible(obj):
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)): return str(obj)
    if hasattr(obj, '__str__'): return str(obj)
    return obj

for file in input_dir.glob("*.xlsx"):
    print(f"\n🌀 Processing {file.name}")

    time, signal_2d, params = load_and_preprocess(file)
    if time is None:
        print("⛔️ Skipped (error or empty)")
        continue

    base_name = file.stem

    # Sauvegarde CSV pour inspection du signal filtré uniquement
    df_out = pd.DataFrame({'time': time, 'filtered_signal': signal_2d[:, 0]})
    df_out.to_csv(output_dir / f"{base_name}_filtered.csv", index=False)

    # Sauvegarde signal+feature au format .npy
    np.save(output_dir / f"{base_name}_time.npy", time)
    np.save(output_dir / f"{base_name}_signal.npy", signal_2d)

    # Sauvegarde paramètres clés
    params_clean = {k: json_compatible(v) for k, v in params.items()}
    with open(output_dir / f"{base_name}_params.json", 'w') as f:
        json.dump(params_clean, f, indent=2)

    print(f"✅ Saved: {base_name}_signal.npy (shape {signal_2d.shape}), {base_name}_params.json")

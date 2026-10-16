# -*- coding: utf-8 -*-
"""
Created on Wed May 28 03:10:02 2025

@author: Burningirl
"""

import os
import numpy as np
import json
from hmmlearn import hmm
from scipy.signal import savgol_filter

# === Configuration ===
DATA_DIR = "C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/new_processed"  # Folder where *_signal.npy and *_time.npy are stored
OUTPUT_DIR = "C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/new_events"   # Folder to save *_events.json
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Utility: Segment and extract key times ===
def detect_occlusion_times(time, signal):
    filtered_signal = signal[:, 0]
    first_derivative = np.gradient(filtered_signal, time)
    second_derivative = savgol_filter(filtered_signal, window_length=11, polyorder=2, deriv=2)

    features = np.column_stack([
        filtered_signal,
        first_derivative,
        np.abs(first_derivative),
        second_derivative
    ])
    features -= features.mean(axis=0)
    features /= features.std(axis=0)

    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(features)
    hidden_states = model.predict(features)

    # Remap states based on mean signal
    state_means = [filtered_signal[hidden_states == i].mean() for i in range(3)]
    sorted_states = np.argsort(state_means)  # 0 = occlusion
    state_map = {old: new for new, old in enumerate(sorted_states)}
    remapped_states = np.array([state_map[s] for s in hidden_states])
    hidden_states = remapped_states

    occlusion_state = 0
    baseline_state = 1

    # Detect start
    occlusion_start_idx = next(
        (i for i in range(1, len(hidden_states))
         if hidden_states[i-1] != occlusion_state and hidden_states[i] == occlusion_state), None)
    occlusion_start_time = time[occlusion_start_idx] if occlusion_start_idx is not None else None

    # Detect end
    peak_idx = np.argmax(filtered_signal)
    occlusion_end_idx = None
    for i in range(1, peak_idx):
        if hidden_states[i-1] == occlusion_state and hidden_states[i] != occlusion_state:
            occlusion_end_idx = i
    occlusion_end_time = time[occlusion_end_idx] if occlusion_end_idx is not None else None

    # Detect peak
    peak_time = time[peak_idx]

    return {
        "occlusion_start_time": float(occlusion_start_time) if occlusion_start_time else None,
        "occlusion_end_time": float(occlusion_end_time) if occlusion_end_time else None,
        "peak_time": float(peak_time)
    }

# === Batch Processing ===
all_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_signal.npy")]

for f in all_files:
    base = f.replace("_signal.npy", "")
    try:
        signal = np.load(os.path.join(DATA_DIR, f))
        time = np.load(os.path.join(DATA_DIR, f"{base}_time.npy"))

        events = detect_occlusion_times(time, signal)
        out_path = os.path.join(OUTPUT_DIR, f"{base}_events.json")

        with open(out_path, 'w') as outfile:
            json.dump(events, outfile, indent=4)

        print(f"✅ Processed {base}")
    except Exception as e:
        print(f"❌ Failed {base}: {e}")

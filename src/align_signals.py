# -*- coding: utf-8 -*-
"""
Created on Wed May 28 03:33:09 2025

@author: Burningirl
"""

import os
import numpy as np
import json

# === Configuration ===
DATA_DIR = "C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/new_processed"     # Contains *_signal.npy and *_time.npy
EVENTS_DIR = "C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/new_events"      # Contains *_events.json
OUTPUT_DIR = "C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/new_aligned"     # Output directory for aligned signals
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Parameters ===
start_offset = -300  # seconds before occlusion start
end_offset = 600     # seconds after occlusion start
target_duration = end_offset - start_offset

# === Utility: align and crop a signal ===
def align_signal(time, signal, t0, start_offset, end_offset):
    t_start = t0 + start_offset
    t_end = t0 + end_offset
    mask = (time >= t_start) & (time <= t_end)
    if np.sum(mask) < 2:
        return None, None  # Not enough data

    cropped_time = time[mask] - t0  # Recenter time to t = 0
    cropped_signal = signal[mask]
    return cropped_time, cropped_signal

# === Batch Process ===
for file in os.listdir(DATA_DIR):
    if not file.endswith("_signal.npy"):
        continue

    base = file.replace("_signal.npy", "")
    try:
        # Load signal and time
        signal = np.load(os.path.join(DATA_DIR, f"{base}_signal.npy"))
        time = np.load(os.path.join(DATA_DIR, f"{base}_time.npy"))

        # Load events
        with open(os.path.join(EVENTS_DIR, f"{base}_events.json"), "r") as f:
            events = json.load(f)
        t0 = events.get("occlusion_start_time")
        if t0 is None:
            print(f"⛔ No occlusion start in {base}, skipping.")
            continue

        # Align and crop
        cropped_time, cropped_signal = align_signal(time, signal, t0, start_offset, end_offset)
        if cropped_time is None:
            print(f"⚠️  Not enough data for {base}, skipping.")
            continue

        # Save aligned output
        np.save(os.path.join(OUTPUT_DIR, f"{base}_aligned_time.npy"), cropped_time)
        np.save(os.path.join(OUTPUT_DIR, f"{base}_aligned_signal.npy"), cropped_signal)
        with open(os.path.join(OUTPUT_DIR, f"{base}_events.json"), "w") as f:
            json.dump(events, f, indent=4)
        print(f"✅ Aligned {base}")

    except Exception as e:
        print(f"❌ Error processing {base}: {e}")

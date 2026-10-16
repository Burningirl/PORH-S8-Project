# -*- coding: utf-8 -*-
"""
Created on Sat May 17 01:50:54 2025

@author: Burningirl
"""
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from scipy.signal import savgol_filter
import os

# === Configuration ===
DATA_DIR = "C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/processed/"  # Adjust as needed
FILENAME = "PORH050" # Adjust to your specific file
SIGNAL_FILE = os.path.join(DATA_DIR, f"{FILENAME}_signal.npy")
TIME_FILE = os.path.join(DATA_DIR, f"{FILENAME}_time.npy")

# === Load Data ===
time = np.load(TIME_FILE)
signal = np.load(SIGNAL_FILE)
filtered_signal = signal[:, 0]

# === Feature Engineering ===
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

# === HMM Training ===
model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
model.fit(features)
hidden_states = model.predict(features)

# === Remap states based on signal mean ===
state_means = [filtered_signal[hidden_states == i].mean() for i in range(3)]
sorted_states = np.argsort(state_means)  # 0=occlusion, 1=baseline, 2=hyperemia
state_map = {old: new for new, old in enumerate(sorted_states)}
remapped_states = np.array([state_map[s] for s in hidden_states])
hidden_states = remapped_states

# === Occlusion State Detection ===
occlusion_state = 0
baseline_state = 1
hyperemia_state = 2

# Detect occlusion start
occlusion_start_idx = next(
    (i for i in range(1, len(hidden_states))
     if hidden_states[i-1] != occlusion_state and hidden_states[i] == occlusion_state),
    None
)
occlusion_start_time = time[occlusion_start_idx] if occlusion_start_idx is not None else None

# Detect occlusion end (last exit before peak)
peak_idx = np.argmax(filtered_signal)
peak_time = time[peak_idx]

occlusion_end_idx = None
for i in range(1, peak_idx):
    if hidden_states[i-1] == occlusion_state and hidden_states[i] != occlusion_state:
        occlusion_end_idx = i
occlusion_end_time = time[occlusion_end_idx] if occlusion_end_idx is not None else None

# === Apply domain-informed correction ===
# Everything before occlusion start → baseline
# Occlusion period → occlusion
# After occlusion end → hyperemia

corrected_states = hidden_states.copy()
if occlusion_start_idx:
    corrected_states[:occlusion_start_idx] = baseline_state
if occlusion_end_idx:
    corrected_states[occlusion_start_idx:occlusion_end_idx] = occlusion_state
    corrected_states[occlusion_end_idx:] = hyperemia_state


# === Print results ===
print("==== Detected Events (with corrected states) ====")
print(f"Occlusion Start Time: {occlusion_start_time:.2f} s" if occlusion_start_time else "❌ Start not detected")
print(f"Occlusion End Time:   {occlusion_end_time:.2f} s" if occlusion_end_idx else "❌ End not detected")
print(f"Peak Flow Time:       {peak_time:.2f} s")

# === Plot ===
plt.figure(figsize=(14, 6))
colors = ['orange', 'blue', 'green']
labels = ['Occlusion', 'Baseline', 'Hyperaemia']
for state in range(3):
    idx = corrected_states == state
    plt.plot(time[idx], filtered_signal[idx], '.', color=colors[state], alpha=0.4, label=f"State {state} ({labels[state]})")

if occlusion_start_time:
    plt.axvline(occlusion_start_time, color='red', linestyle='--', label="Occlusion Start")
if occlusion_end_time:
    plt.axvline(occlusion_end_time, color='green', linestyle='--', label="Occlusion End")
plt.axvline(peak_time, color='black', linestyle='--', label="Peak Flow")

plt.title(f"HMM + Hybrid Correction - {FILENAME}")
plt.xlabel("Time (s)")
plt.ylabel("Filtered Signal (PU)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

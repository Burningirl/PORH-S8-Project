# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 11:34:05 2025

@author: Burningirl
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lstm_model import LSTM_Model
import numpy as _np

# === Config
model_path = "C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/models_short_weighted/checkpoint_epoch_24.pt"
aligned_dir = "new_aligned"
params_dir = "new_processed"
target_keys = ["PF - Peak flow", "TM - Time to max*", "RF - Rest flow", "BZ - Biological zero"]

# === Load model
model = LSTM_Model(input_size=2, hidden_size=32, num_layers=2, output_size=4)
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# === Load normalization values
with open("processed/normalization.json", "r") as f:
    norm = json.load(f)
mean = np.array([norm[k]["mean"] for k in target_keys])
std = np.array([norm[k]["std"] for k in target_keys])

# === Evaluate each file
for fname in os.listdir(aligned_dir):
    if not fname.endswith("_aligned_signal.npy"):
        continue

    base = fname.replace("_aligned_signal.npy", "")
    print(f"\n🔍 Evaluating {base}...")

    X = np.load(os.path.join(aligned_dir, fname))
    X_tensor = torch.tensor(X).unsqueeze(0).float()
    lengths = torch.tensor([X.shape[0]])

    # Load true targets
    with open(os.path.join(params_dir, f"{base}_params.json")) as f:
        target_dict = json.load(f)

    def to_seconds(value):
        try:
            if value in ["nan", "NaN", None] or str(value).strip().lower() == "nan":
                return np.nan
            if isinstance(value, str) and ":" in value:
                t = pd.to_datetime(value).time()
                return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
            return float(value)
        except Exception as e:
            print(f"⚠️ Conversion échouée pour la valeur : {value} → {e}")
            return np.nan

    y_true = np.array([to_seconds(target_dict.get(k, np.nan)) for k in target_keys])

    # Predict
    with torch.no_grad():
        pred = model(X_tensor, lengths)

    # Denormalize predictions
    y_pred = pred.squeeze(0).numpy() * std + mean

    # === MAE
    mae = np.abs(y_true - y_pred)
    print("MAE par cible :")
    for short, val in zip(["PF", "TM", "RF", "BZ"], mae):
        print(f"  {short} : {val:.2f}")
    print(f"➡️ MAE Moyenne : {mae[~np.isnan(mae)].mean():.3f}")

    # === Per-file bar plot with annotations
    short_keys = ["PF", "TM", "RF", "BZ"]
    x = np.arange(len(short_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars_true = ax.bar(x - width / 2, y_true, width, label="True", color="tab:blue", alpha=0.7, edgecolor="k", linewidth=0.3)
    bars_pred = ax.bar(x + width / 2, y_pred, width, label="Predicted", color="tab:orange", alpha=0.7, edgecolor="k", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(short_keys)
    ax.set_ylabel("Value")
    ax.set_title(f"{base}: True vs Predicted")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bars (handle NaNs)
    all_vals = _np.array(list(y_true) + list(y_pred))
    max_val = _np.nanmax(all_vals) if all_vals.size > 0 else 1.0
    if _np.isnan(max_val) or max_val == 0:
        max_val = 1.0
    offset = max_val * 0.02 + 1e-6

    for i, (tv, pv) in enumerate(zip(y_true, y_pred)):
        xt = x[i] - width / 2
        xp = x[i] + width / 2
        if not _np.isnan(tv):
            ax.text(xt, tv + offset, f"{tv:.2f}", ha="center", va="bottom", fontsize=8, color="k", zorder=5)
        else:
            ax.text(xt, offset, "nan", ha="center", va="bottom", fontsize=8, color="k", zorder=5)
        if not _np.isnan(pv):
            ax.text(xp, pv + offset, f"{pv:.2f}", ha="center", va="bottom", fontsize=8, color="darkorange", zorder=6)
        else:
            ax.text(xp, offset, "nan", ha="center", va="bottom", fontsize=8, color="darkorange", zorder=6)

    plt.tight_layout()

    # optional: save plot to ./plots/
    os.makedirs("plots", exist_ok=True)
    fig_path = os.path.join("plots", f"{base}_true_vs_pred.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

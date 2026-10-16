# -*- coding: utf-8 -*-
"""
Created on Sat May 31 13:38:57 2025

@author: Burningirl
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lstm_model import LSTM_Model
from aligned_dataset import AlignedPORHDataset
from collate_utils import collate_fn

# === Config
target_keys = [
    "RF - Rest flow", "BZ - Biological zero", "PF - Peak flow",
    "TM - Time to max*", "AO - Occlusion area (Unit*sec.)", "AH - Hyperaemia area* (Unit*sec.)"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20
model_dir = "models_weighted"
os.makedirs(model_dir, exist_ok=True)
log_path = os.path.join(model_dir, "loss_log_weighted.txt")

# === Load weights from normalization.json
with open("processed/normalization.json", "r") as f:
    norm = json.load(f)
weights = torch.tensor([1 / norm[k]["std"]**2 for k in target_keys], dtype=torch.float32).to(device)

# === Dataset
dataset = AlignedPORHDataset(
    aligned_dir="aligned",
    params_dir="processed",
    normalization_file="processed/normalization.json",
    target_keys=target_keys
)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# === Model
model = LSTM_Model(input_size=2, hidden_size=32, num_layers=2, output_size=len(target_keys)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Training loop
for epoch in range(epochs):
    model.train()
    total_loss, count = 0, 0

    for X, y, lengths in loader:
        try:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            print("Batch loaded:", X.shape, y.shape)

            mask = ~torch.isnan(y).any(dim=1)
            print("Valid samples:", mask.sum().item())
            if mask.sum() == 0:
                continue

            pred = model(X, lengths)
            loss = torch.abs(pred[mask] - y[mask]) * weights
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        except Exception as e:
            print(f"❌ Error at epoch {epoch+1}, batch skipped: {e}")
            continue

    avg = total_loss / count if count > 0 else float("nan")
    print(f"✅ Epoch {epoch+1}: Weighted MAE = {avg:.4f}")

    # Save model
    torch.save({
        "epoch": epoch+1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg
    }, os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt"))

    # Save log
    with open(log_path, "a") as f:
        f.write(f"Epoch {epoch+1}: Weighted MAE = {avg:.4f}\n")

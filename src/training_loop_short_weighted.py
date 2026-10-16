# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:05:54 2025

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
target_keys = ["PF - Peak flow", "TM - Time to max*", "RF - Rest flow", "BZ - Biological zero"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 25
model_dir = "models_short_weighted"
os.makedirs(model_dir, exist_ok=True)

# === Chargement des poids depuis normalization.json
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

# === Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    mae_per_target = torch.zeros(len(target_keys)).to(device)
    count = 0

    for X, y, lengths in loader:
        X, y, lengths = X.to(device), y.to(device), lengths.to(device)
        
        # === Debug masquage
        print("Batch loaded:", X.shape, y.shape)
        mask = ~torch.isnan(y).any(dim=1)
        print("Valid samples:", mask.sum().item())
        
        if mask.sum() == 0:
            continue

        pred = model(X, lengths)
        y_valid = y[mask]
        pred_valid = pred[mask]

        loss_per_sample = torch.abs(pred_valid - y_valid)
        weighted_loss = loss_per_sample * weights
        loss = weighted_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        mae_per_target += loss_per_sample.sum(dim=0)
        count += mask.sum().item()

    avg_loss = total_loss / count if count > 0 else float("nan")
    avg_mae_per_target = mae_per_target / count

    # === Log to terminal
    print(f"\n🧪 Epoch {epoch+1}: Weighted MAE = {avg_loss:.4f}")
    for k, v in zip(target_keys, avg_mae_per_target):
        print(f"  {k} : {v.item():.4f}")

    # === Save checkpoint
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt"))

    # === Save loss log
    with open(os.path.join(model_dir, "loss_log.txt"), "a") as f:
        f.write(f"Epoch {epoch+1}: Weighted MAE = {avg_loss:.4f}\n")
        for k, v in zip(target_keys, avg_mae_per_target):
            f.write(f"  {k} : {v.item():.4f}\n")

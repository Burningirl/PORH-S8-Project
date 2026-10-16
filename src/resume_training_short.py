# -*- coding: utf-8 -*-
"""
Created on Sat May 31 13:59:31 2025

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
model_dir = "models_short"
checkpoint_path = os.path.join(model_dir, "checkpoint_epoch_25.pt")
resume_from = 25
epochs = 50  # jusqu’à epoch 50

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
loss_fn = nn.L1Loss(reduction="none")


# === Resume checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

print(f"✅ Checkpoint loaded from epoch {checkpoint['epoch']} (loss={checkpoint['loss']:.4f})")

# === Training resume
for epoch in range(resume_from, epochs):
    model.train()
    total_loss, count = 0, 0

    for X, y, lengths in loader:
        X, y, lengths = X.to(device), y.to(device), lengths.to(device)
        
        # === Debug masquage
        print("Batch loaded:", X.shape, y.shape)
        mask = ~torch.isnan(y).any(dim=1)
        print("Valid samples:", mask.sum().item())

        if mask.sum() == 0:
            continue

        preds = model(X, lengths)
        y_valid = y[mask]
        preds_valid = preds[mask]

        loss = loss_fn(preds_valid, y_valid).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    avg = total_loss / count if count > 0 else float("nan")
    print(f"Epoch {epoch+1}: MAE = {avg:.4f}")

    # Save checkpoint
    torch.save({
        "epoch": epoch+1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg
    }, os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt"))

    with open(os.path.join(model_dir, "loss_log.txt"), "a") as f:
        f.write(f"Epoch {epoch+1}: MAE = {avg:.4f}\n")

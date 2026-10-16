# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:24:44 2025

@author: Burningirl
"""

import os
import json
import random
from pathlib import Path

# === Dossiers de travail ===
input_dir = Path("C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/processed/")
output_dir = Path("C:/Users/ELTEBOOK/Documents/MScT SCE Bio/M1/S8/LSTM-PORH/split/")
output_dir.mkdir(parents=True, exist_ok=True)
base_names = []

# === Liste des Fichiers ===
for file in input_dir.glob("*_signal.npy"):
    
    base_name = file.stem.replace("_signal", "")
    base_names.append(base_name)

print(base_names)

# ===  shuffle + split ===
random.seed(42)
random.shuffle(base_names)

n = len(base_names)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_split = base_names[:train_end]
val_split = base_names[train_end:val_end]
test_split = base_names[val_end:]

print(len(train_split))
print(len(val_split))
print(len(test_split))

# ===  Sauvegarder des splits dans des fichiers ===

# Dictionnaire contenant les splits et leurs noms de fichier correspondants
splits = {
    "train_split": ("train_split.json", train_split),
    "val_split": ("val_split.json", val_split),
    "test_split": ("test_split.json", test_split),
}

# Écriture des fichiers JSON dans output_dir
for split_name, (file_name, data) in splits.items():
    file_path = output_dir / file_name
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Fichier '{file_path}' écrit avec succès.")

# ===== TEST RAPIDE =====
'''
# Nom du fichier à vérifier
file_path = output_dir / "train_split.json"

# Charger et vérifier le contenu
try:
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            print(f"Le fichier '{file_name}' contient bien une liste de noms : {data}")
        else:
            print(f"Le fichier '{file_name}' ne contient pas une liste valide.")
except FileNotFoundError:
    print(f"Le fichier '{file_name}' est introuvable.")
except json.JSONDecodeError:
    print(f"Erreur lors du décodage du fichier '{file_name}'.")
    
'''
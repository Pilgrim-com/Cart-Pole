"""Trim existing training results to first 1000 episodes."""
import numpy as np
import csv
import os

BASE = "model/Stabilize"
ALGOS = ["Linear_Q", "DQN", "MC_REINFORCE"]
MAX_EP = 1000

for algo in ALGOS:
    d = os.path.join(BASE, algo)
    if not os.path.isdir(d):
        print(f"[SKIP] {d} not found")
        continue

    # Trim durations.npy
    npy = os.path.join(d, f"{algo}_durations.npy")
    if os.path.exists(npy):
        data = np.load(npy)
        orig = len(data)
        trimmed = data[:MAX_EP]
        np.save(npy, trimmed)
        print(f"[OK] {npy}: {orig} -> {len(trimmed)}")
    else:
        print(f"[SKIP] {npy} not found")

    # Trim training_log.csv
    csv_path = os.path.join(d, "training_log.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = list(csv.DictReader(f))
        orig = len(reader)
        trimmed = reader[:MAX_EP]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=reader[0].keys())
            writer.writeheader()
            writer.writerows(trimmed)
        print(f"[OK] {csv_path}: {orig} -> {len(trimmed)}")
    else:
        print(f"[SKIP] {csv_path} not found")

print("\nDone! Model weights (.pth) are unchanged.")

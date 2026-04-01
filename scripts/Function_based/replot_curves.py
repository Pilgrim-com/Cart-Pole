"""Regenerate individual training_curve.png from trimmed durations.npy."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os

BASE = "model/Stabilize"

# All possible run labels
RUN_LABELS = [
    "Linear_Q", "DQN", "MC_REINFORCE", "AC", "A2C",
    "DQN_buf100", "DQN_buf1000",
    "A2C_envs4", "A2C_envs16",
]

for label in RUN_LABELS:
    d = os.path.join(BASE, label)
    npy = os.path.join(d, f"{label}_durations.npy")
    if not os.path.exists(npy):
        print(f"[SKIP] {npy} not found")
        continue

    data = np.load(npy)
    durations_t = torch.tensor(data, dtype=torch.float)

    plt.figure(figsize=(10, 5))
    plt.title(f"{label} — Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy(), alpha=0.6, label="Raw")
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label="100-ep avg", linewidth=2)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(d, f"{label}_training_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out} ({len(data)} episodes)")

print("\nDone!")

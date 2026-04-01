#!/bin/bash

# Cart-Pole HW3 — Train REMAINING experiments only
echo "========================================="
echo " Training Remaining 6 Experiments"
echo "========================================="

TASK="Stabilize-Isaac-Cartpole-v0"

declare -a EXPERIMENTS=(
    "--algo AC"
    "--algo A2C"
    "--algo DQN --buffer_size 100"
    "--algo DQN --buffer_size 1000"
    "--algo A2C --num_envs 4 --num_envs_a2c 4"
    "--algo A2C --num_envs 16 --num_envs_a2c 16"
)

echo -e "\n[Phase 1] Training Agents sequentially..."
for EXP in "${EXPERIMENTS[@]}"; do
    echo ">>> Training: python scripts/Function_based/train.py --task ${TASK} ${EXP}"
    python scripts/Function_based/train.py --task ${TASK} ${EXP}
    echo ">>> Finished."
done

# Plot results
echo -e "\n[Phase 2] Generating Unified Learning Efficiency Plot..."
python scripts/Function_based/plot_all_algorithms.py

echo -e "\n[Phase 3] Done! All remaining experiments completed."

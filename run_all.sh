#!/bin/bash

# Cart-Pole HW3 Auto-Experiment Script
echo "========================================="
echo " Starting Cart-Pole Full Automation Test"
echo " (Includes Default 5 Runs + 4 Ablations)"
echo "========================================="

TASK="Stabilize-Isaac-Cartpole-v0"

# Array of all 9 experiment configurations
declare -a EXPERIMENTS=(
    "--algo Linear_Q"
    "--algo DQN"
    "--algo MC_REINFORCE"
    "--algo AC"
    "--algo A2C"
    "--algo DQN --buffer_size 100"
    "--algo DQN --buffer_size 1000"
    "--algo A2C --num_envs_a2c 4"
    "--algo A2C --num_envs_a2c 16"
)

# 1. Train all algorithms
echo -e "\n[Phase 1] Training Agents sequentially..."
for EXP in "${EXPERIMENTS[@]}"; do
    echo ">>> Training: python scripts/Function_based/train.py --task ${TASK} ${EXP}"
    python scripts/Function_based/train.py --task ${TASK} ${EXP}
    echo ">>> Finished."
done

# 2. Plot results
echo -e "\n[Phase 2] Generating Unified Learning Efficiency Plot..."
python scripts/Function_based/plot_all_algorithms.py

# 3. Deployment Prompt
echo -e "\n[Phase 3] All models are trained and plotted!"
echo "To evaluate their performance visually, run:"
for EXP in "${EXPERIMENTS[@]}"; do
    echo "  python scripts/Function_based/play.py --task ${TASK} ${EXP}"
done

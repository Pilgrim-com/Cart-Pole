#!/bin/bash

# Cart-Pole HW3 Auto-Experiment Script
echo "========================================="
echo " Starting Cart-Pole Full Automation Test"
echo "========================================="

ALGORITHMS=("Linear_Q" "DQN" "MC_REINFORCE" "AC" "A2C")
TASK="Isaac-Cartpole-v0"

# 1. Train all algorithms
echo "\n[Phase 1] Training Agents sequentially..."
for ALGO in "${ALGORITHMS[@]}"; do
    echo ">>> Training ${ALGO}..."
    python scripts/Function_based/train.py --task ${TASK} --algo ${ALGO}
    echo ">>> Finished ${ALGO}."
done

# 2. Plot results
echo "\n[Phase 2] Generating Unified Learning Efficiency Plot..."
python scripts/Function_based/plot_all_algorithms.py

# 3. Deployment Prompt
echo "\n[Phase 3] All models are trained and plotted!"
echo "To evaluate their performance visually, run:"
for ALGO in "${ALGORITHMS[@]}"; do
    echo "  python scripts/Function_based/play.py --task ${TASK} --algo ${ALGO}"
done

#!/bin/bash

# Cart-Pole HW3 — Evaluate all trained models
echo "========================================="
echo " Evaluating All Trained Models"
echo "========================================="

TASK="Stabilize-Isaac-Cartpole-v0"

declare -a EXPERIMENTS=(
    "--algo Linear_Q"
    "--algo DQN"
    "--algo MC_REINFORCE"
    "--algo AC"
    "--algo A2C"
    "--algo DQN --buffer_size 100"
    "--algo DQN --buffer_size 1000"
    "--algo A2C --num_envs 4 --num_envs_a2c 4"
    "--algo A2C --num_envs 16 --num_envs_a2c 16"
    "--algo PPO"
    "--algo TD3"
    "--algo SAC"
)

for EXP in "${EXPERIMENTS[@]}"; do
    echo ">>> Evaluating: python scripts/Function_based/play.py --task ${TASK} ${EXP}"
    python scripts/Function_based/play.py --task ${TASK} ${EXP}
    echo ">>> Finished."
done

echo -e "\nAll evaluations complete!"

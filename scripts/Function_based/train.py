"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../RL_Algorithm")))

from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--algo", type=str, default="DQN", help="Name of the RL algorithm (e.g. Linear_Q, DQN, AC, MC_REINFORCE, A2C).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# Ablation arguments (for experiments E3, E4)
parser.add_argument("--buffer_size", type=int, default=None, help="[Ablation E3] DQN replay buffer size.")
parser.add_argument("--num_envs_a2c", type=int, default=1, help="[Ablation E4] A2C number of parallel envs.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import numpy as np
import torch
import random
import matplotlib
import matplotlib.pyplot as plt

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.algo in ["A2C", "PPO"] and getattr(args_cli, "num_envs_a2c", 1) > 1:
        args_cli.num_envs = args_cli.num_envs_a2c

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    from RL_Algorithm.Function_based.Linear_Q    import Linear_QN
    from RL_Algorithm.Function_based.DQN          import DQN
    from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE
    from RL_Algorithm.Function_based.AC            import AC
    from RL_Algorithm.Function_based.A2C           import A2C
    from RL_Algorithm.Function_based.PPO           import PPO
    from RL_Algorithm.Function_based.TD3           import TD3
    from RL_Algorithm.Function_based.SAC           import SAC

    # ------------------------------------------------------------------ #
    # Device & naming
    # ------------------------------------------------------------------ #
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print("device:", device)

    task_name      = "Stabilize"
    Algorithm_name = args_cli.algo

    # ------------------------------------------------------------------ #
    # Hyperparameters — Shared by all algorithms
    # ------------------------------------------------------------------ #
    num_of_action   = 2
    action_range    = [-1.0, 1.0] if task_name == "Stabilize" else [-2.0, 2.0]
    learning_rate   = 1e-3
    discount_factor = 0.99
    n_episodes      = 2000
    max_steps       = 500         # max steps per episode (single-env algorithms)

    # Exploration — Linear_Q, DQN only
    initial_epsilon = 1.0
    epsilon_decay   = 5e-4        # FIX: was 1e-3 → reached final too fast (~950 ep)
    final_epsilon   = 0.05

    # Network — neural-network algorithms
    n_observations  = 4
    hidden_dim      = 64          # DQN, MC_REINFORCE
    hidden_dims     = [64, 64]    # AC, A2C
    dropout         = 0.0

    # Action type — MC_REINFORCE, AC, A2C
    action_type     = "discrete"  # "continuous" | "discrete"

    # Replay buffer — DQN only
    # Allow CLI override for ablation experiment E3
    buffer_size     = args_cli.buffer_size if args_cli.buffer_size is not None else 10000
    batch_size      = 64
    tau             = 0.005       # Polyak soft-update rate for target network

    # Actor-Critic shared — AC, A2C
    value_loss_coef = 0.5
    entropy_coef    = 0.01
    max_grad_norm   = 0.5

    # Rollout — A2C only
    # Allow CLI override for ablation experiment E4
    num_envs_a2c             = args_cli.num_envs_a2c   # default 1
    # Keep total transitions = 500 regardless of num_envs (fair comparison)
    total_transitions        = 500
    num_transitions_per_env  = max(1, total_transitions // num_envs_a2c)

    # ------------------------------------------------------------------ #
    # Agent construction
    # ------------------------------------------------------------------ #
    if Algorithm_name == "Linear_Q":
        agent = Linear_QN(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,   # FIX: was hardcoded 0.01, now matches others
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

    elif Algorithm_name == "DQN":
        agent = DQN(
            device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            n_observations=n_observations,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            tau=tau,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    elif Algorithm_name == "MC_REINFORCE":
        agent = MC_REINFORCE(
            device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            n_observations=n_observations,
            hidden_dim=hidden_dim,
            dropout=dropout,
            action_type=action_type,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    elif Algorithm_name == "AC":
        agent = AC(
            device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            n_observations=n_observations,
            hidden_dims=hidden_dims,
            activation="relu",
            action_type=action_type,
            init_noise_std=1.0,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )

    elif Algorithm_name == "A2C":
        agent = A2C(
            device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            n_observations=n_observations,
            hidden_dims=hidden_dims,
            activation="relu",
            action_type=action_type,
            init_noise_std=1.0,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )

    elif Algorithm_name == "PPO":
        agent = PPO(
            device=device,
            num_of_action=1,  # Continuous Cart-Pole strictly takes 1D force
            action_range=action_range,
            n_observations=n_observations,
            hidden_dims=hidden_dims,
            activation="relu",
            action_type="continuous",
            init_noise_std=1.0,
            num_learning_epochs=4,
            num_mini_batches=4,
            clip_param=0.2,
            gamma=discount_factor,
            lam=0.95,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            desired_kl=0.01,
        )

    elif Algorithm_name == "TD3":
        agent = TD3(
            device=device,
            num_of_action=1,  # Continuous Cart-Pole strictly takes 1D force
            action_range=action_range,
            n_observations=n_observations,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            tau=tau,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
            exploration_noise=0.1,
            target_noise=0.2,
            target_noise_clip=0.5,
            policy_update_freq=2,
        )

    elif Algorithm_name == "SAC":
        agent = SAC(
            device=device,
            num_of_action=1,  # Continuous Cart-Pole strictly takes 1D force
            action_range=action_range,
            n_observations=n_observations,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            alpha_lr=learning_rate,
            tau=tau,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
            init_alpha=0.2,
            auto_alpha=True,
        )

    else:
        raise ValueError(f"Unknown algorithm: {Algorithm_name}")

    # ------------------------------------------------------------------ #
    # Save path
    # ------------------------------------------------------------------ #
    # Include ablation config in folder name for E3/E4
    if Algorithm_name == "DQN" and args_cli.buffer_size is not None:
        run_label = f"{Algorithm_name}_buf{buffer_size}"
    elif Algorithm_name == "A2C" and num_envs_a2c != 1:
        run_label = f"{Algorithm_name}_envs{num_envs_a2c}"
    else:
        run_label = Algorithm_name

    save_interval = 500
    model_dir     = os.path.join("model", task_name, run_label)
    os.makedirs(model_dir, exist_ok=True)

    obs, _ = env.reset()
    timestep = 0

    while simulation_app.is_running():
        log_data = []

        for episode in tqdm(range(n_episodes), desc=f"[{run_label}]"):

            loss  = None
            steps = 0
            episode_return = 0.0

            # ---------------------------------------------------------------- #
            # Per-algorithm training step
            # ---------------------------------------------------------------- #
            if Algorithm_name == "Linear_Q":
                episode_return, steps = agent.learn(env, max_steps=max_steps)

            elif Algorithm_name == "DQN":
                episode_return, steps = agent.learn(env, max_steps=max_steps)

            elif Algorithm_name == "MC_REINFORCE":
                episode_return, loss, traj = agent.learn(env)
                steps = len(traj)

            elif Algorithm_name == "AC":
                episode_return, loss, steps = agent.learn(env, max_steps=max_steps, num_agents=1)

            elif Algorithm_name == "A2C":
                # A2C collects a fixed-length rollout (not episode-based)
                agent.learn(
                    env,
                    num_envs=num_envs_a2c,
                    num_transitions_per_env=num_transitions_per_env,
                    max_episodes=1,
                )
                # steps = total transitions collected this rollout
                steps = num_envs_a2c * num_transitions_per_env
                # episode_return = mean reward per step × steps (comparable scale)
                episode_return = agent.storage.rewards.sum().item()

            elif Algorithm_name == "PPO":
                agent.learn(
                    env,
                    num_envs=num_envs_a2c,
                    num_transitions_per_env=num_transitions_per_env,
                    max_episodes=1,
                )
                steps = num_envs_a2c * num_transitions_per_env
                episode_return = agent.storage.rewards.sum().item()
                
            elif Algorithm_name in ["TD3", "SAC"]:
                episode_return, steps = agent.learn(env, max_steps=max_steps)

            # ---------------------------------------------------------------- #
            # Logging — unified for all algorithms (FIX: was inside if-else)
            # ---------------------------------------------------------------- #
            if Algorithm_name in ["A2C", "PPO"]:
                if episode % 50 == 0:
                    agent.plot_durations(None, show_result=False)  # A2C/PPO tracks durations internally
            else:
                if episode % 50 == 0:
                    agent.plot_durations(steps, show_result=False)
                else:
                    agent.episode_durations.append(steps)  # still track, just don't plot

            log_data.append({
                "episode"  : episode,
                "reward"   : float(episode_return),
                "steps"    : steps,
                "epsilon"  : getattr(agent, "epsilon", 0.0),
                "loss"     : float(loss) if loss is not None else None,
            })

            if episode % 100 == 0:
                print(f"[{run_label}] ep {episode:4d} | steps {steps:4d} | return {episode_return:.2f}")

            if episode % save_interval == 0 and episode > 0:
                agent.save_model(model_dir, f"{run_label}_{episode}.pth")

        # ------------------------------------------------------------------ #
        # End of training — save final model, curve, log
        # ------------------------------------------------------------------ #
        agent.save_model(model_dir, f"{run_label}_final.pth")
        print("Training complete.")

        agent.plot_durations(show_result=True)
        plt.ioff()
        plt.savefig(os.path.join(model_dir, f"{run_label}_training_curve.png"))
        np.save(
            os.path.join(model_dir, f"{run_label}_durations.npy"),
            np.array(agent.episode_durations),
        )
        print(f"Training curve saved → {model_dir}/{run_label}_training_curve.png")
        plt.close("all")

        log_path = os.path.join(model_dir, "training_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "reward", "steps", "epsilon", "loss"])
            writer.writeheader()
            writer.writerows(log_data)
        print(f"Training log saved → {log_path}")

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        break
    # ==================================================================== #

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
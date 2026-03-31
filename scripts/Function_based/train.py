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

import gymnasium as gym
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

    # ------------------------------------------------------------------ #
    # Device & naming
    # ------------------------------------------------------------------ #
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print("device:", device)

    task_name      = str(args_cli.task).split("-")[0]   # "Stabilize" or "SwingUp"
    Algorithm_name = args_cli.algo

    # ------------------------------------------------------------------ #
    # Hyperparameters
    # ------------------------------------------------------------------ #

    # Shared by all algorithms
    num_of_action   = 2
    action_range    = [-1.0, 1.0] if task_name == "Stabilize" else [-2.0, 2.0]
    learning_rate   = 1e-3
    discount_factor = 0.99
    n_episodes      = 2000

    # Exploration — LinearQ, DQN only
    initial_epsilon = 1.0
    epsilon_decay   = 1e-3
    final_epsilon   = 0.05

    # Network size — neural-network algorithms (DQN, MC_REINFORCE, AC, PPO)
    n_observations  = 4

    hidden_dim      = 64      # single int for DQN / MC_REINFORCE
    hidden_dims     = [64, 64]    # list of ints for AC / PPO   

    dropout         = 0.0

    # Action Type — (MC_REINFORCE, AC, PPO)
    action_type     = "discrete"      # "continuous" | "discrete" 

    # Replay buffer — DQN only
    buffer_size     = 10000
    batch_size      = 64
    tau             = 0.005      # Polyak soft-update rate for target network

    # Rollout — PPO only
    num_transitions_per_env = None   # steps collected per env before each update
    num_learning_epochs     = None   # gradient epochs per update
    num_mini_batches        = None
    clip_param              = None
    lam                     = None   # GAE lambda
    value_loss_coef         = None
    entropy_coef            = None
    max_grad_norm           = None
    desired_kl              = None   # adaptive LR target; set 0 for discrete

    # ------------------------------------------------------------------ #
    # Agent construction
    # ------------------------------------------------------------------ #
    if Algorithm_name == "Linear_Q":
        agent = Linear_QN(num_of_action=num_of_action, action_range=action_range, learning_rate=0.01,
                          initial_epsilon=initial_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon,
                          discount_factor=discount_factor)
    elif Algorithm_name == "DQN":
        agent = DQN(device=device, num_of_action=num_of_action, action_range=action_range, n_observations=n_observations,
                    hidden_dim=hidden_dim, dropout=dropout, learning_rate=learning_rate, tau=tau,
                    initial_epsilon=initial_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon,
                    discount_factor=discount_factor, buffer_size=buffer_size, batch_size=batch_size)
    elif Algorithm_name == "MC_REINFORCE":
        agent = MC_REINFORCE(device=device, num_of_action=num_of_action, action_range=action_range, n_observations=n_observations,
                             hidden_dim=hidden_dim, dropout=dropout, action_type=action_type, learning_rate=learning_rate,
                             discount_factor=discount_factor)
    elif Algorithm_name == "AC":
        agent = AC(device=device, num_of_action=num_of_action, action_range=action_range, n_observations=n_observations,
                   hidden_dims=hidden_dims, activation="relu", action_type=action_type, init_noise_std=1.0,
                   learning_rate=learning_rate, discount_factor=discount_factor, value_loss_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5)
    elif Algorithm_name == "A2C":
        agent = A2C(device=device, num_of_action=num_of_action, action_range=action_range, n_observations=n_observations,
                   hidden_dims=hidden_dims, activation="relu", action_type=action_type, init_noise_std=1.0,
                   learning_rate=learning_rate, discount_factor=discount_factor, value_loss_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5)

    # ------------------------------------------------------------------ #
    # Save path — checkpoints saved every save_interval episodes
    # ------------------------------------------------------------------ #
    save_interval = 500
    model_dir     = os.path.join("model", task_name, Algorithm_name)
    os.makedirs(model_dir, exist_ok=True)

    obs, _ = env.reset()
    timestep = 0

    while simulation_app.is_running():
        log_data = []

        for episode in tqdm(range(n_episodes)):

            # ========= put your code here ========= #
            loss = None
            if Algorithm_name in ["DQN", "Linear_Q"]:
                episode_return, steps = agent.learn(env, max_steps=500)
            elif Algorithm_name == "MC_REINFORCE":
                episode_return, loss, traj = agent.learn(env)
                steps = len(traj)
            elif Algorithm_name == "AC":
                episode_return, loss, steps = agent.learn(env, max_steps=500, num_agents=1)
            elif Algorithm_name == "A2C":
                agent.learn(env, num_envs=1, num_transitions_per_env=500, max_episodes=1)
                steps = int(agent.episode_durations[-1]) if len(agent.episode_durations) > 0 else 0
                episode_return = float(steps)  # Reward is +1 per step
                
            # Prevent A2C from double-appending the same duration via plot_durations
            if Algorithm_name != "A2C":
                agent.plot_durations(steps, show_result=False)
            else:
                agent.plot_durations(None, show_result=False)
            log_data.append({
                "episode": episode,
                "reward": float(episode_return),
                "steps": steps,
                "epsilon": getattr(agent, "epsilon", 0.0),
                "loss": float(loss) if loss is not None else None
            })
            # ====================================== #

            # Logging & checkpointing
            if episode % 100 == 0:
                print(f"[{Algorithm_name}] episode {episode}")

            if episode % save_interval == 0 and episode > 0:
                agent.save_model(model_dir, f"{Algorithm_name}_{episode}.pth")

        # Save final model and display training curve
        agent.save_model(model_dir, f"{Algorithm_name}_final.pth")
        print("Training complete.")

        agent.plot_durations(show_result=True)
        plt.ioff()
        plt.savefig(os.path.join(model_dir, f"{Algorithm_name}_training_curve.png"))
        np.save(os.path.join(model_dir, f"{Algorithm_name}_durations.npy"), np.array(agent.episode_durations))
        print(f"Training curve saved to {os.path.join(model_dir, f'{Algorithm_name}_training_curve.png')}")
        plt.close('all')

        import csv
        log_path = os.path.join(model_dir, "training_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode","reward","steps","epsilon","loss"])
            writer.writeheader()
            writer.writerows(log_data)
        print(f"Training log saved to {log_path}")

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
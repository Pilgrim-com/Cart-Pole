"""Script to play a trained RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../RL_Algorithm")))

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
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

import gymnasium as gym
import torch
import random
import numpy as np
import csv

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
    """Play with a trained agent."""

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.algo == "A2C" and getattr(args_cli, "num_envs_a2c", 1) > 1:
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

    task_name = str(args_cli.task).split("-")[0] if "Stabilize" not in str(args_cli.task) else "Stabilize"
    Algorithm_name = args_cli.algo
    n_episodes     = 10

    num_of_action   = 2
    action_range    = [-1.0, 1.0] if task_name == "Stabilize" else [-2.0, 2.0]
    n_observations  = 4
    hidden_dim      = 64
    hidden_dims     = [64, 64]
    action_type     = "discrete"
    
    # Ablation settings
    buffer_size = args_cli.buffer_size if args_cli.buffer_size is not None else 10000
    num_envs_a2c = args_cli.num_envs_a2c
    
    if Algorithm_name == "Linear_Q":
        agent = Linear_QN(num_of_action=num_of_action, action_range=action_range)
    elif Algorithm_name == "DQN":
        agent = DQN(device=device, num_of_action=num_of_action, action_range=action_range, n_observations=n_observations, hidden_dim=hidden_dim, dropout=0.0)
    elif Algorithm_name == "MC_REINFORCE":
        agent = MC_REINFORCE(device=device, num_of_action=num_of_action, action_range=action_range, n_observations=n_observations, hidden_dim=hidden_dim, dropout=0.0, action_type=action_type)
    elif Algorithm_name == "AC":
        agent = AC(device=device, num_of_action=num_of_action, action_range=action_range, n_observations=n_observations, hidden_dims=hidden_dims, activation="relu", action_type=action_type, init_noise_std=1.0)
    elif Algorithm_name == "A2C":
        agent = A2C(device=device, num_of_action=num_of_action, action_range=action_range, n_observations=n_observations, hidden_dims=hidden_dims, activation="relu", action_type=action_type, init_noise_std=1.0)
    elif Algorithm_name == "PPO":
        agent = PPO(device=device, num_of_action=num_of_action, action_range=action_range, n_observations=n_observations, hidden_dims=hidden_dims, activation="relu", action_type="continuous", init_noise_std=1.0, num_learning_epochs=4, num_mini_batches=4, clip_param=0.2, gamma=0.99, lam=0.95, value_loss_coef=0.5, entropy_coef=0.01, learning_rate=1e-3, max_grad_norm=0.5, desired_kl=0.01)
    elif Algorithm_name == "TD3":
        agent = TD3(device=device, num_of_action=1 if action_type == "continuous" else num_of_action, action_range=action_range, n_observations=n_observations, hidden_dim=hidden_dim, learning_rate=1e-3, tau=0.005, discount_factor=0.99, buffer_size=10000, batch_size=64, exploration_noise=0.1, target_noise=0.2, target_noise_clip=0.5, policy_update_freq=2)
    elif Algorithm_name == "SAC":
        agent = SAC(device=device, num_of_action=1 if action_type == "continuous" else num_of_action, action_range=action_range, n_observations=n_observations, hidden_dim=hidden_dim, learning_rate=1e-3, alpha_lr=1e-3, tau=0.005, discount_factor=0.99, buffer_size=10000, batch_size=64, init_alpha=0.2, auto_alpha=True)

    if Algorithm_name == "DQN" and args_cli.buffer_size is not None:
        run_label = f"{Algorithm_name}_buf{buffer_size}"
    elif Algorithm_name == "A2C" and num_envs_a2c != 1:
        run_label = f"{Algorithm_name}_envs{num_envs_a2c}"
    else:
        run_label = Algorithm_name

    model_dir      = os.path.join("model", task_name, run_label)
    model_filename = f"{run_label}_final.pth"
    agent.load_model(model_dir, model_filename)
    print(f"Loaded: {os.path.join(model_dir, model_filename)}")

    # Force fully deterministic evaluation (No sampling)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    obs, _ = env.reset()
    timestep = 0
    results = []

    while simulation_app.is_running():
        with torch.inference_mode():

            for episode in range(n_episodes):
                
                # ========= put your code here ========= #
                obs, _ = env.reset()
                if isinstance(obs, torch.Tensor):
                    obs_np = obs.squeeze().cpu().numpy()
                else:
                    obs_np = np.squeeze(obs)
                    
                episode_return = 0
                for t in range(500):
                    is_multi_env = obs_np.ndim > 1
                    state_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
                    if not is_multi_env:
                        state_tensor = state_tensor.unsqueeze(0)
                    
                    in_state = obs_np if Algorithm_name == "Linear_Q" else state_tensor
                    
                    if Algorithm_name in ["AC", "A2C", "PPO"]:
                        with torch.no_grad():
                            act_out = agent.policy.act_inference(in_state)
                            if agent.action_type == "discrete":
                                a_idx = act_out.squeeze(-1).cpu().numpy()
                                scaled_a = np.array([agent.scale_action(a) for a in a_idx]) if is_multi_env else agent.scale_action(a_idx.item())
                            else:
                                a_idx = act_out.cpu().numpy()
                                scaled_a = np.array([agent.scale_action(a) for a in a_idx]) if is_multi_env else agent.scale_action(a_idx[0])
                    elif Algorithm_name == "SAC":
                        scaled_a, a_idx = agent.select_action(in_state, evaluate=True)
                    elif Algorithm_name == "TD3":
                        scaled_a, a_idx = agent.select_action(in_state, add_noise=False)
                    else:
                        scaled_a, a_idx = agent.select_action(in_state)
                    
                    if not isinstance(scaled_a, torch.Tensor):
                        scaled_a = torch.tensor(scaled_a, dtype=torch.float32)
                        
                    env_action = scaled_a.to(device) if is_multi_env else scaled_a.unsqueeze(0).to(device)
                    if env_action.dim() == 1 and env_action.size(0) == num_envs_a2c and Algorithm_name == "A2C":
                        env_action = env_action.unsqueeze(-1)

                    next_obs, reward, terminated, truncated, _ = env.step(env_action)
                    
                    if isinstance(next_obs, torch.Tensor):
                        next_obs_np = next_obs.cpu().numpy()
                    else:
                        next_obs_np = np.array(next_obs)
                    
                    if isinstance(reward, torch.Tensor):
                        rew_val = reward.mean().item()
                    else:
                        rew_val = np.mean(reward)
                        
                    if isinstance(terminated, torch.Tensor):
                        term_val = terminated.any().item() or truncated.any().item()
                    else:
                        term_val = np.any(terminated) or np.any(truncated)
                    
                    episode_return += rew_val
                    obs_np = next_obs_np
                    
                    if term_val:
                        break
                
                print(f"Episode {episode+1}: Mean Env Return = {episode_return:.2f}, Steps = {t+1}")
                results.append({"Episode": episode+1, "Return": episode_return, "Steps": t+1})
                # ====================================== #

            csv_filename = os.path.join(model_dir, f"{run_label}_evaluation.csv")
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["Episode", "Return", "Steps"])
                writer.writeheader()
                writer.writerows(results)
            print(f"Evaluation results saved to {csv_filename}")

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
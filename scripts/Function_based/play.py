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
    Algorithm_name = "DQN"                              # update this to match your import
    n_episodes     = 10

    num_of_action   = 2
    action_range    = [-1.0, 1.0] if task_name == "Stabilize" else [-2.0, 2.0]
    n_observations  = 4
    hidden_dim      = 64
    hidden_dims     = [64, 64]
    action_type     = "discrete"
    
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


    model_dir      = os.path.join("model", task_name, Algorithm_name)
    model_filename = f"{Algorithm_name}_final.pth"
    agent.load_model(model_dir, model_filename)
    print(f"Loaded: {os.path.join(model_dir, model_filename)}")

    obs, _ = env.reset()
    timestep = 0

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
                    state_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    in_state = obs_np if Algorithm_name == "Linear_Q" else state_tensor
                    scaled_a, a_idx = agent.select_action(in_state)
                    
                    if not isinstance(scaled_a, torch.Tensor):
                        scaled_a = torch.tensor([scaled_a], dtype=torch.float32)
                        
                    env_action = scaled_a.unsqueeze(0).to(device)
                    next_obs, reward, terminated, truncated, _ = env.step(env_action)
                    
                    if isinstance(next_obs, torch.Tensor):
                        next_obs_np = next_obs.squeeze().cpu().numpy()
                    else:
                        next_obs_np = np.squeeze(next_obs)
                    
                    if isinstance(reward, torch.Tensor):
                        rew_val = reward.squeeze().item()
                    else:
                        rew_val = np.squeeze(reward).item()
                        
                    if isinstance(terminated, torch.Tensor):
                        term_val = terminated.squeeze().item() or truncated.squeeze().item()
                    else:
                        term_val = bool(np.squeeze(terminated)) or bool(np.squeeze(truncated))
                    
                    episode_return += rew_val
                    obs_np = next_obs_np
                    
                    if term_val:
                        break
                
                print(f"Episode {episode+1}: Return = {episode_return}, Steps = {t+1}")
                # ====================================== #

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
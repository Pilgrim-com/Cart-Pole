from __future__ import annotations
import os
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    """
    Linear Q-Learning with function approximation.

    Args:
        num_of_action (int): Number of discrete actions.
        action_range (list): [min, max] continuous action range.
        learning_rate (float): TD weight-update step size.
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step epsilon decay.
        final_epsilon (float): Minimum exploration rate.
        discount_factor (float): Discount factor γ.
    """

    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
    ) -> None:

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

        # ===== Linear weight matrix ===== #
        # Shape: (obs_feature_dim, num_of_action)
        self.w = np.zeros((4, num_of_action))

    # ------------------------------------------------------------------ #
    # Linear Q-value estimation                                           #
    # ------------------------------------------------------------------ #

    def q(self, obs, a=None):
        """
        Return the linearly-estimated Q-value(s) for a given observation.

        Args:
            obs: State feature vector φ(s), shape (obs_dim,).
            a (int | None): Action index. If None, returns Q for all actions
                            as a 1-D array of shape (num_of_action,).

        Returns:
            float | np.ndarray: Q(s, a) scalar, or Q(s, :) array.
        """
        # ========= put your code here ========= #
        qs = obs @ self.w
        if a is None:
            return qs
        return qs[a]
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        next_action: int,
        terminated: bool,
    ):
        """
        Update the weight vector using the TD error.

        Args:
            obs: Current state feature vector φ(s).
            action (int): Action index taken in state s.
            reward (float): Reward received.
            next_obs: Next state feature vector φ(s').
            next_action (int): Next action taken (for SARSA-style update).
            terminated (bool): True if the episode ended.
        """
        # ========= put your code here ========= #
        target = reward
        if not terminated:
            target += self.discount_factor * np.max(self.q(next_obs))
        td_error = target - self.q(obs, action)
        self.w[:, action] += self.lr * td_error * obs
        # ====================================== #

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy over Q(s, :).

        Args:
            state: Current state feature vector φ(s).

        Returns:
            Tuple[Tensor, int]: Scaled continuous action tensor and action index.
        """
        # ========= put your code here ========= #
        self.decay_epsilon()
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.num_of_action)
        else:
            action_idx = int(np.argmax(self.q(state)))
        
        scaled_action = self.scale_action(action_idx)
        return scaled_action, action_idx
        # ====================================== #

    def learn(self, env, max_steps: int):
        """
        Train the agent for one episode.

        Args:
            env: The environment.
            max_steps (int): Maximum steps per episode.

        Returns:
            Tuple[float, int]: (episode_return, timestep)
        """
        # ========= put your code here ========= #
        obs, _ = env.reset()
        if isinstance(obs, dict): obs = obs.get("policy", next(iter(obs.values())))
        if isinstance(obs, torch.Tensor):
            obs_np = obs.squeeze().cpu().numpy()
        else:
            obs_np = np.squeeze(obs)
            
        episode_return = 0.0
        timestep = 0
        # Infer device loosely if needed, normally env takes same device as tensors
        device = next(env.parameters()).device if hasattr(env, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(obs, torch.Tensor):
            device = obs.device
            
        for t in range(max_steps):
            scaled_a, a_idx = self.select_action(obs_np)
            
            if not isinstance(scaled_a, torch.Tensor):
                scaled_a = torch.tensor([scaled_a], dtype=torch.float32)
            
            env_action = scaled_a.unsqueeze(0).to(device)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            if isinstance(next_obs, dict): next_obs = next_obs.get("policy", next(iter(next_obs.values())))
            
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
                
            self.update(obs_np, a_idx, rew_val, next_obs_np, None, term_val)
            
            obs_np = next_obs_np
            episode_return += rew_val
            timestep += 1
            
            if term_val:
                break
                
        return episode_return, timestep
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence — linear weights only                                    #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save the weight matrix self.w to disk as a .npy file.

        Args:
            path (str): Directory to save the file.
            filename (str): File name (e.g., 'linear_q_cartpole.npy').
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, filename), self.w)
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load the weight matrix self.w from a .npy file.

        Args:
            path (str): Directory containing the file.
            filename (str): File name (e.g., 'linear_q_cartpole.npy').
        """
        # ========= put your code here ========= #
        self.w = np.load(os.path.join(path, filename))
        # ====================================== #
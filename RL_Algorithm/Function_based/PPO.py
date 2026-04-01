from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from storage.on_policy import OnPolicyAlgorithm
from storage.buffers import RolloutBuffer
from RL_Algorithm.Function_based.AC import ActorCritic


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization (PPO) — on-policy, clipped surrogate.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or number of choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous policy.
        num_learning_epochs (int): Epochs per PPO update.
        num_mini_batches (int): Mini-batches per epoch.
        clip_param (float): PPO clipping ε.
        gamma (float): Discount factor γ.
        lam (float): GAE lambda λ.
        value_loss_coef (float): Coefficient for value loss.
        entropy_coef (float): Coefficient for entropy bonus.
        learning_rate (float): Adam learning rate.
        max_grad_norm (float): Gradient clipping norm.
        desired_kl (float): KL target for adaptive LR (0 to disable; use 0 for discrete).
        normalize_advantage_per_mini_batch (bool): Normalise advantages per mini-batch.
        use_clipped_value_loss (bool): Apply clipped value loss.
    """

    def __init__(
        self,
        device=None,
        num_of_action: int = None,
        action_range: list = [None, None],
        n_observations: int = None,
        hidden_dims: list[int] = [None],
        activation: str = None,
        action_type: str = None,
        init_noise_std: float = None,
        num_learning_epochs: int = None,
        num_mini_batches: int = None,
        clip_param: float = None,
        gamma: float = None,
        lam: float = None,
        value_loss_coef: float = None,
        entropy_coef: float = None,
        learning_rate: float = None,
        max_grad_norm: float = None,
        desired_kl: float = None,
        normalize_advantage_per_mini_batch: bool = False,
        use_clipped_value_loss: bool = True,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ===== Build ActorCritic network (imported from AC.py) ===== #
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.policy = ActorCritic(
            state_dim=n_observations,
            action_dim=num_of_action,
            hidden_dims=hidden_dims,
            activation=activation,
            action_type=action_type,
            init_noise_std=init_noise_std,
        ).to(self.device)
        # ====================================== #

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # ===== PPO hyperparameters ===== #
        self.action_type                        = action_type
        self.clip_param                         = clip_param
        self.num_learning_epochs                = num_learning_epochs
        self.num_mini_batches                   = num_mini_batches
        self.value_loss_coef                    = value_loss_coef
        self.entropy_coef                       = entropy_coef
        self.gamma                              = gamma
        self.lam                                = lam
        self.max_grad_norm                      = max_grad_norm
        self.desired_kl                         = desired_kl
        self.learning_rate                      = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.use_clipped_value_loss             = use_clipped_value_loss

        super(PPO, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
        )

    # ------------------------------------------------------------------ #
    # Rollout collection                                                   #
    # ------------------------------------------------------------------ #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample actions for all parallel envs and populate self.transition.

        Continuous: actions shape (num_envs, action_dim).
        Discrete  : actions shape (num_envs, 1).

        Args:
            obs (Tensor): shape (num_envs, obs_dim).

        Returns:
            Tensor: Sampled actions.
        """
        obs = obs.to(self.device)
        with torch.no_grad():
            self.transition.actions = self.policy.act(obs)

            val = self.policy.evaluate(obs)
            self.transition.values = val.view(-1, 1)

            if self.action_type == 'continuous':
                log_p = self.policy.distribution.log_prob(self.transition.actions).sum(dim=-1)
                self.transition.action_mean = self.policy.action_mean
                self.transition.action_sigma = self.policy.action_std
            else:
                log_p = self.policy.distribution.log_prob(self.transition.actions.squeeze(-1))
                self.transition.action_mean = self.policy.action_mean
                self.transition.action_sigma = self.policy.action_std

            self.transition.actions_log_prob = log_p.view(-1, 1)
        # ====================================== #

        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """
        Write rewards and dones into self.transition, then flush to storage.

        Args:
            rewards (Tensor): shape (num_envs,) or (num_envs, 1).
            dones (Tensor): shape (num_envs,) or (num_envs, 1).
        """
        # ========= put your code here ========= #
        self.transition.rewards = rewards.clone().detach().to(self.device).view(-1, 1)
        self.transition.dones = dones.clone().detach().to(self.device).view(-1, 1)
        # ====================================== #

        # Flush transition into RolloutBuffer via inherited add_transition()
        self.add_transition()

    # ------------------------------------------------------------------ #
    # Return & Advantage Computation                                       #
    # ------------------------------------------------------------------ #

    def compute_returns(self, last_obs: torch.Tensor) -> None:
        """
        Compute GAE returns and advantages over the collected rollout.

        Args:
            last_obs (Tensor): Observation after the final rollout step.
                               Shape: (num_envs, obs_dim).
        """
        # ========= put your code here ========= #
        last_obs = last_obs.to(self.device)
        with torch.no_grad():
            val_out = self.policy.evaluate(last_obs)
            last_values = val_out.view(-1, 1)
            
        advantage = 0
        for step in reversed(range(self.storage.num_transitions_per_env)):
            if step == self.storage.num_transitions_per_env - 1:
                next_non_terminal = 1.0 - self.transition.dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.storage.dones[step + 1]
                next_values = self.storage.values[step + 1]
                
            delta = self.storage.rewards[step] + self.gamma * next_values * next_non_terminal - self.storage.values[step]
            advantage = delta + self.gamma * self.lam * next_non_terminal * advantage
            self.storage.advantages[step] = advantage
            
        self.storage.returns = self.storage.advantages + self.storage.values
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Policy Update                                                        #
    # ------------------------------------------------------------------ #

    def update(self) -> dict:
        """
        Perform PPO updates over the collected rollout.

        Calls ``self.storage.mini_batch_generator()`` which now lives in
        ``RolloutBuffer`` (storage/buffers.py) and yields 8-tuples.

        Returns:
            dict: Mean losses {'value', 'surrogate', 'entropy'}.
        """
        mean_value_loss     = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy        = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            # ========= put your code here ========= #
            obs_batch = obs_batch.to(self.device)
            actions_batch = actions_batch.to(self.device)
            advantages_batch = advantages_batch.to(self.device)
            returns_batch = returns_batch.to(self.device)
            old_actions_log_prob_batch = old_actions_log_prob_batch.to(self.device)
            
            if self.normalize_advantage_per_mini_batch:
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                
            values = self.policy.evaluate(obs_batch).squeeze(-1)
            self.policy.act(obs_batch) # update distribution
            
            if self.action_type == 'continuous':
                actions_log_prob = self.policy.distribution.log_prob(actions_batch).sum(dim=-1)
                entropy = self.policy.distribution.entropy().sum(dim=-1).mean()
            else:
                actions_log_prob = self.policy.distribution.log_prob(actions_batch.squeeze(-1))
                entropy = self.policy.distribution.entropy().mean()
                
            ratio = torch.exp(actions_log_prob - old_actions_log_prob_batch)
            surrogate1 = ratio * advantages_batch
            surrogate2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
            surrogate_loss = -torch.min(surrogate1, surrogate2).mean()
            
            if self.use_clipped_value_loss and target_values_batch is not None:
                target_values_batch = target_values_batch.to(self.device)
                value_pred_clipped = target_values_batch + torch.clamp(values - target_values_batch, -self.clip_param, self.clip_param)
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = 0.5 * (returns_batch - values).pow(2).mean()
                
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.item()
            # ====================================== #

        num_updates          = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss     /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy        /= num_updates

        self.storage.clear()   # on-policy: discard rollout after update

        return {
            "value":     mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy":   mean_entropy,
        }

    # ------------------------------------------------------------------ #
    # Main Training Loop                                                   #
    # ------------------------------------------------------------------ #

    def learn(
        self,
        env,
        num_envs: int,
        num_transitions_per_env: int,
        max_episodes: int = 10000,
    ) -> None:
        """
        Main PPO parallel training loop.

        Calls ``_init_storage()`` (from OnPolicyAlgorithm) to create the buffer.

        Continuous: actions_shape = (num_of_action,)
        Discrete  : actions_shape = (1,)

        Args:
            env: Isaac Lab vectorised environment.
            num_envs (int): Number of parallel environments.
            num_transitions_per_env (int): Rollout horizon per env.
            max_episodes (int): Total number of training rollouts.
        """
        # ========= put your code here ========= #
        actions_shape = (self.num_of_action,) if self.action_type == "continuous" else (1,)
        if hasattr(env, "observation_space") and getattr(env.observation_space, "shape", None) is not None:
            obs_shape = env.observation_space.shape
        else:
            # Try querying Isaac properties or fall back
            obs_shape = (env.observation_manager.group_obs_dim["policy"][0],) if hasattr(env, "observation_manager") else (4,)
            
        self._init_storage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=obs_shape,
            actions_shape=actions_shape,
            device=self.device
        )

        import time
        import numpy as np
        from tqdm import tqdm
        
        obs, _ = env.reset()
        if isinstance(obs, dict): obs = obs.get("policy", next(iter(obs.values())))

        self.episode_durations = []
        self.current_steps = torch.zeros(num_envs, device=self.device)
        
        pbar = tqdm(range(max_episodes), desc="[PPO]")
        for episode in pbar:
            for _ in range(num_transitions_per_env):
                self.transition.observations = obs.clone()
                actions = self.act(obs)
                if self.action_type == "discrete":
                    env_actions = actions.squeeze(-1)
                else:
                    env_actions = []
                    for a in actions:
                        env_actions.append(self.scale_action(a.cpu().numpy()))
                    env_actions = torch.tensor(np.array(env_actions), device=self.device, dtype=torch.float32)
                    
                if env_actions.dim() == 1:
                    env_actions = env_actions.unsqueeze(-1)
                    
                next_obs, rewards, dones, _, _ = env.step(env_actions)
                if isinstance(next_obs, dict): next_obs = next_obs.get("policy", next(iter(next_obs.values())))
                
                self.process_env_step(rewards, dones)
                obs = next_obs
                
                self.current_steps += 1
                for env_idx in range(num_envs):
                    if dones[env_idx].item():
                        self.episode_durations.append(self.current_steps[env_idx].item())
                        self.current_steps[env_idx] = 0
                        
            self.compute_returns(last_obs=obs)
            losses = self.update()
            
            episode_return = float(np.mean(self.episode_durations[-1:])) if len(self.episode_durations) > 0 else 0.0
            pbar.set_postfix({
                "Ep_Ret": f"{episode_return:.2f}",
                "V_loss": f"{losses['value']:.4f}",
                "S_loss": f"{losses['surrogate']:.4f}"
            })
            
        self.plot_durations(self.storage.num_transitions_per_env, show_result=False)
        # ====================================== #


    # ------------------------------------------------------------------ #
    # Inference & Persistence                                              #
    # ------------------------------------------------------------------ #

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for evaluation.

        Continuous: actor mean. Discrete: argmax of logits.

        Args:
            obs (Tensor): shape (1, obs_dim) or (obs_dim,).
        """
        # ========= put your code here ========= #
        with torch.no_grad():
            self.policy._update_distribution(obs.to(self.device))
            if self.action_type == "discrete":
                action = torch.argmax(self.policy.distribution.probs, dim=-1)
                scaled_a = self.scale_action(action.item())
                return scaled_a, action.item()
            else:
                action = self.policy.distribution.mean
                scaled_a = self.scale_action(action.cpu().numpy()[0])
                return scaled_a, action.cpu().numpy()[0]
        # ====================================== #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save actor-critic weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'ppo_cartpole.pth').
        """
        # ========= put your code here ========= #
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor-critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'ppo_cartpole.pth').
        """
        # ========= put your code here ========= #
        import os
        self.policy.load_state_dict(torch.load(os.path.join(path, filename), map_location=self.device))
        # ====================================== #
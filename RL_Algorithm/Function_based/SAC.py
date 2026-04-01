from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from storage.off_policy import OffPolicyAlgorithm


class SAC_Actor(nn.Module):
    """
    Stochastic actor network for SAC using the reparameterisation trick.

    Args:
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        n_actions (int): Action space dimension.
        log_std_min (float): Lower bound for log standard deviation.
        log_std_max (float): Upper bound for log standard deviation.
    """

    def __init__(
        self,
        n_observations: int,
        hidden_dim: int,
        n_actions: int,
        log_std_min: float = None,
        log_std_max: float = None,
    ):
        super(SAC_Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, n_actions)
        self.log_std_layer = nn.Linear(hidden_dim, n_actions)
        # ====================================== #

    def forward(self, state: torch.Tensor):
        """
        Compute mean and log_std of the Gaussian policy.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tuple[Tensor, Tensor]: (mean, log_std) both shape (batch, n_actions).
        """
        # ========= put your code here ========= #
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        if self.log_std_min is not None and self.log_std_max is not None:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
        # ====================================== #

    def sample(self, state: torch.Tensor):
        """
        Sample an action using the reparameterisation trick and compute
        the corrected log-probability.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tuple[Tensor, Tensor]:
                - action   : Squashed action in (-1, 1), shape (batch, n_actions).
                - log_prob : Corrected log π(a|s),       shape (batch,).
        """
        # ========= put your code here ========= #
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob.squeeze(-1)
        # ====================================== #


class SAC_Critic(nn.Module):
    """
    Twin Q-value network for SAC.

    SAC uses two critics (same as TD3) to reduce overestimation.
    The soft Bellman target uses ``min(Q1, Q2) − α · log π(a'|s')``.

    Args:
        n_observations (int): Observation space dimension.
        n_actions (int): Action space dimension.
        hidden_dim (int): Hidden layer width.
    """

    def __init__(self, n_observations: int, n_actions: int, hidden_dim: int):
        super(SAC_Critic, self).__init__()

        # ===== Q1 network ===== #
        # ========= put your code here ========= #
        self.q1_fc1 = nn.Linear(n_observations + n_actions, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        # ====================================== #

        # ===== Q2 network (independent weights) ===== #
        # ========= put your code here ========= #
        self.q2_fc1 = nn.Linear(n_observations + n_actions, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
        # ====================================== #

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Compute both Q-values.

        Args:
            state (Tensor): State tensor.
            action (Tensor): Action tensor.

        Returns:
            Tuple[Tensor, Tensor]: (Q1, Q2) both shape (batch, 1).
        """
        # ========= put your code here ========= #
        xu = torch.cat([state, action], dim=-1)
        x1 = F.relu(self.q1_fc1(xu))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)

        x2 = F.relu(self.q2_fc1(xu))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)
        return q1, q2
        # ====================================== #


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC) — off-policy, maximum entropy actor-critic.

    Args:
        device: Torch device.
        num_of_action (int): Action space dimension.
        action_range (list): [min, max] for action scaling.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        learning_rate (float): Learning rate for actor and critics.
        alpha_lr (float): Learning rate for automatic temperature tuning.
        tau (float): Polyak soft-update coefficient.
        discount_factor (float): Discount factor γ.
        buffer_size (int): Replay buffer capacity.
        batch_size (int): Mini-batch size per update.
        init_alpha (float): Initial temperature α.
        auto_alpha (bool): Enable automatic α tuning.
        target_entropy (float | None): Target entropy for auto-tuning.
                                       Defaults to −action_dim if None.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            learning_rate: float = None,
            alpha_lr: float = None,
            tau: float = None,
            discount_factor: float = None,
            buffer_size: int = None,
            batch_size: int = None,
            init_alpha: float = None,
            auto_alpha: bool = None,
            target_entropy: float | None = None,
    ) -> None:

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.actor        = SAC_Actor(n_observations, hidden_dim, num_of_action).to(device)
        self.critic       = SAC_Critic(n_observations, num_of_action, hidden_dim).to(device)
        self.critic_target = SAC_Critic(n_observations, num_of_action, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.device    = device
        self.tau       = tau
        self.auto_alpha = auto_alpha

        # ===== Automatic temperature tuning ===== #
        # log_alpha is optimised instead of alpha directly to keep alpha > 0.
        # target_entropy is set to -action_dim as a heuristic (Haarnoja et al. 2018).
        self.log_alpha      = torch.tensor(
            [float(init_alpha)], requires_grad=True, device=device
        ).log()
        self.alpha          = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy if target_entropy is not None \
                              else -float(num_of_action)
        # ====================================== #

        # OffPolicyAlgorithm.__init__ creates self.memory = ReplayBuffer(buffer_size, batch_size)
        super(SAC, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def select_action(self, state: torch.Tensor, evaluate: bool = False):
        """
        Sample an action from the stochastic policy.

        Training  (evaluate=False): sample from Normal distribution.
        Inference (evaluate=True) : use the mean (deterministic).

        Args:
            state (Tensor): Current state.
            evaluate (bool): True for deterministic inference.

        Returns:
            Tensor: Scaled action tensor.
        """
        # ========= put your code here ========= #
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_tensor)
                
        action_np = action.cpu().numpy()[0]
        return self.scale_action(action_np), action_np
        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        Compute SAC losses for critics, actor, and temperature.

        Args:
            states (Tensor): Batch of current states.
            actions (Tensor): Batch of actions taken.
            rewards (Tensor): Batch of rewards.
            next_states (Tensor): Batch of next states.
            dones (Tensor): Batch of terminal flags.

        Returns:
            Tuple[Tensor, Tensor, Tensor | None]:
                (critic_loss, actor_loss, alpha_loss or None)
        """
        # ========= put your code here ========= #
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi.unsqueeze(-1)
            q_target = rewards.unsqueeze(-1) + self.discount_factor * (1 - dones.unsqueeze(-1)) * q_next

        q1_curr, q2_curr = self.critic(states, actions)
        critic_loss = F.mse_loss(q1_curr, q_target) + F.mse_loss(q2_curr, q_target)

        curr_actions, curr_log_pi = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, curr_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * curr_log_pi.unsqueeze(-1) - q_new).mean()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (curr_log_pi.unsqueeze(-1) + self.target_entropy).detach()).mean()
        else:
            alpha_loss = None

        return critic_loss, actor_loss, alpha_loss
        # ====================================== #

    def generate_sample(self, batch_size=None):
        """
        Sample a mini-batch and unpack into SAC-ready tensors.

        Returns:
            Tuple or None:
                - states (Tensor)
                - actions (Tensor)
                - rewards (Tensor)
                - next_states (Tensor)
                - dones (Tensor)
        """
        # ========= put your code here ========= #
        batch = super().generate_sample()
        if batch is None:
            return None
            
        import numpy as np
        states, actions, rewards, next_states, dones = zip(*batch)
        device = self.device
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        
        return states, actions, rewards, next_states, dones
        # ====================================== #

    def update_policy(self):
        """
        Perform one update step for critics, actor, and temperature.

        Returns:
            float | None: Critic loss, or None if buffer not ready.
        """
        sample = self.generate_sample()
        if sample is None:
            return None

        states, actions, rewards, next_states, dones = sample
        critic_loss, actor_loss, alpha_loss = self.calculate_loss(
            states, actions, rewards, next_states, dones
        )
        # ========= put your code here ========= #
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if alpha_loss is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        # ====================================== #

        self.update_target_networks()

    def update_target_networks(self):
        """
        Overrides the no-op in OffPolicyAlgorithm.
        """
        # ========= put your code here ========= #
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        # ====================================== #

    def learn(self, env, num_agents: int = 1, max_steps: int = 1000):
        """
        Train the agent for one episode (single env) or fixed-length run
        (parallel envs).

        Args:
            env: The Isaac Lab environment.
            num_agents (int): Number of parallel environments.
            max_steps (int): Steps per episode (single) or total steps (parallel).

        Returns:
            Tuple[float, int]: (episode_return, timestep)
        """
        # ========= put your code here ========= #
        import numpy as np
        obs, _ = env.reset()
        if isinstance(obs, dict): obs = obs.get("policy", next(iter(obs.values())))
        if isinstance(obs, torch.Tensor): obs_np = obs.squeeze().cpu().numpy()
        else: obs_np = np.squeeze(obs)
            
        episode_return = 0.0
        timestep = 0
            
        for t in range(max_steps):
            scaled_a, a_original = self.select_action(obs_np, evaluate=False)
            
            if not isinstance(scaled_a, torch.Tensor):
                scaled_a = torch.tensor(np.array(scaled_a), dtype=torch.float32)
                
            env_action = scaled_a.unsqueeze(0).to(self.device) if scaled_a.dim() == 1 else scaled_a.view(1, -1).to(self.device)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            
            if isinstance(next_obs, dict): next_obs = next_obs.get("policy", next(iter(next_obs.values())))
            if isinstance(next_obs, torch.Tensor): next_obs_np = next_obs.squeeze().cpu().numpy()
            else: next_obs_np = np.squeeze(next_obs)
                
            if isinstance(reward, torch.Tensor): rew_val = reward.squeeze().item()
            else: rew_val = np.squeeze(reward).item()
                
            if isinstance(terminated, torch.Tensor): term_val = terminated.squeeze().item() or truncated.squeeze().item()
            else: term_val = bool(np.squeeze(terminated)) or bool(np.squeeze(truncated))
            
            self.store_transition(obs_np, a_original, rew_val, next_obs_np, term_val)
            self.update_policy()
            
            obs_np = next_obs_np
            episode_return += rew_val
            timestep += 1
            
            if term_val:
                break
                
        return episode_return, timestep
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save actor and critic weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'sac_cartpole.pth').
        """
        # ========= put your code here ========= #
        import os
        os.makedirs(path, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'log_alpha': self.log_alpha,
        }, os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor and critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'sac_cartpole.pth').
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #
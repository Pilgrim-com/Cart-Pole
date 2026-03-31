from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from storage.on_policy import OnPolicyAlgorithm
from RL_Algorithm.networks.mlp import MLP


# ============================================================ #
# =================== Actor-Critic Network =================== #
# ============================================================ #

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network supporting continuous and discrete actions.

    ``action_type='continuous'`` → ``Normal(mean, std)``, learnable ``self.std``
    ``action_type='discrete'``   → ``Categorical(logits)``, no ``self.std``

    Args:
        state_dim (int): Observation space dimension.
        action_dim (int): Action vector dim (continuous) or # choices (discrete).
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous distribution.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [None],
        activation: str = None,
        action_type: str = None,
        init_noise_std: float = None,
    ):
        super().__init__()

        assert action_type in ("continuous", "discrete"), \
            f"action_type must be 'continuous' or 'discrete', got '{action_type}'"

        self.action_type = action_type
        self.action_dim  = action_dim

        self.actor  = MLP(state_dim, action_dim, hidden_dims, activation)
        self.critic = MLP(state_dim, 1,          hidden_dims, activation)

        if self.action_type == "continuous":
            self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))

        self.distribution: Normal | Categorical | None = None

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def action_mean(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.mean
        return self.distribution.probs

    @property
    def action_std(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.stddev
        return torch.ones_like(self.distribution.probs)

    @property
    def entropy(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.entropy().sum(dim=-1)
        return self.distribution.entropy()

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError("Use act() or evaluate().")

    def _update_distribution(self, obs: torch.Tensor) -> None:
        """
        Build the action distribution from current observations.

        Continuous: ``Normal(mean, std)``
        Discrete  : ``Categorical(logits)``
        """
        # ========= put your code here ========= #
        actor_out = self.actor(obs)
        if self.action_type == "discrete":
            self.distribution = Categorical(logits=actor_out)
        else:
            self.distribution = Normal(actor_out, self.std)
        # ====================================== #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample an action.

        Continuous: shape (batch, action_dim).
        Discrete  : shape (batch, 1).
        """
        # ========= put your code here ========= #
        self._update_distribution(obs)
        action = self.distribution.sample()
        if self.action_type == "discrete" and action.dim() == 1:
            action = action.unsqueeze(1)
        return action
        # ====================================== #

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action: actor mean (continuous) or argmax (discrete)."""
        # ========= put your code here ========= #
        self._update_distribution(obs)
        if self.action_type == "discrete":
            action = self.distribution.probs.argmax(dim=-1, keepdim=True)
        else:
            action = self.distribution.mean
        return action
        # ====================================== #

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic value estimate V(s), shape (batch, 1)."""
        # ========= put your code here ========= #
        return self.critic(obs)
        # ====================================== #

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of given actions under the current distribution.

        Continuous: sum over action dims → shape (batch,).
        Discrete  : scalar log-prob → shape (batch,).
        """
        # ========= put your code here ========= #
        log_prob = self.distribution.log_prob(actions)
        if self.action_type == "continuous":
            log_prob = log_prob.sum(dim=-1)
        elif self.action_type == "discrete" and log_prob.dim() > 1:
            log_prob = log_prob.squeeze()
        return log_prob
        # ====================================== #


# ============================================================ #
# ====================== AC Agent ============================ #
# ============================================================ #

class AC(OnPolicyAlgorithm):
    """
    Advantage Actor-Critic (A2C) — on-policy, episodic.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or # choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous policy.
        learning_rate (float): Adam learning rate.
        discount_factor (float): Discount factor γ.
        value_loss_coef (float): Coefficient for value loss.
        entropy_coef (float): Coefficient for entropy bonus.
        max_grad_norm (float): Gradient clipping norm.
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
        learning_rate: float = None,
        discount_factor: float = None,
        value_loss_coef: float = None,
        entropy_coef: float = None,
        max_grad_norm: float = None,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

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

        self.optimizer       = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.action_type     = action_type
        self.value_loss_coef = value_loss_coef
        self.entropy_coef    = entropy_coef
        self.max_grad_norm   = max_grad_norm

        super(AC, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    # ------------------------------------------------------------------ #
    # Trajectory Collection                                                #
    # ------------------------------------------------------------------ #

    def generate_trajectory(self, env) -> tuple:
        """
        Run one full episode and collect the trajectory as lists.

        Args:
            env: The environment.

        Returns:
            Tuple: (episode_return, log_prob_actions, values, rewards, timestep)
        """
        # ========= put your code here ========= #
        obs, _ = env.reset()
        if isinstance(obs, torch.Tensor):
            obs_np = obs.squeeze().cpu().numpy()
        else:
            obs_np = np.squeeze(obs)
            
        log_prob_actions_list = []
        values_list = []
        rewards = []
        timestep = 0
        device = self.device
        
        while True:
            state_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            
            self.policy._update_distribution(state_tensor)
            action = self.policy.distribution.sample()
            if self.action_type == "discrete" and action.dim() == 1:
                action = action.unsqueeze(1)
                
            log_prob = self.policy.get_actions_log_prob(action)
            value = self.policy.evaluate(state_tensor)
            
            log_prob_actions_list.append(log_prob.squeeze())
            values_list.append(value.squeeze())
            
            if self.action_type == "discrete":
                action_idx = action.item()
                scaled_a = self.scale_action(action_idx)
            else:
                action_idx = action.squeeze(0).cpu().numpy()
                scaled_a = action.squeeze(0)
                
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
            
            rewards.append(rew_val)
            obs_np = next_obs_np
            timestep += 1
            
            if term_val:
                break
                
        episode_return = sum(rewards)
        log_prob_actions = torch.stack(log_prob_actions_list)
        values = torch.stack(values_list)
        return episode_return, log_prob_actions, values, rewards, timestep
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Return & Loss                                                        #
    # ------------------------------------------------------------------ #

    def compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute discounted Monte-Carlo returns G_t = r_t + γ·r_{t+1} + ...

        Args:
            rewards (Tensor): shape (T,).

        Returns:
            Tensor: Normalised return tensor of shape (T,).
        """
        # ========= put your code here ========= #
        returns = []
        G = 0
        for r in reversed(rewards.tolist()):
            G = r + self.discount_factor * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return returns_tensor
        # ====================================== #

    def calculate_loss(self, log_prob_actions, values, returns):
        """
        Compute actor and critic losses.

        Args:
            log_prob_actions (Tensor): shape (T,).
            values (Tensor): shape (T,).
            returns (Tensor): shape (T,).

        Returns:
            Tuple[Tensor, Tensor]: (actor_loss, critic_loss)
        """
        # ========= put your code here ========= #
        advantage = returns - values.detach()
        actor_loss = -(log_prob_actions * advantage).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        return actor_loss, critic_loss
        # ====================================== #

    def update_policy(self, log_prob_actions, values, returns) -> float:
        """
        Backpropagate and update.

        Returns:
            float: Total combined loss.
        """
        # ========= put your code here ========= #
        actor_loss, critic_loss = self.calculate_loss(log_prob_actions, values, returns)
        entropy = self.policy.entropy.mean()
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Main Training Loop                                                   #
    # ------------------------------------------------------------------ #

    def learn(self, env, max_steps: int, num_agents: int) -> tuple:
        """
        Train the agent for one episode.

        Args:
            env: The environment.
            max_steps (int): Maximum steps per episode.
            num_agents (int): Number of parallel agents.

        Returns:
            Tuple: (episode_return, loss, timestep)
        """
        self.policy.train()

        # ========= put your code here ========= #
        episode_return, log_prob_actions, values, rewards_list, timestep = self.generate_trajectory(env)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        returns = self.compute_returns(rewards)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = self.update_policy(log_prob_actions, values, returns)
        return episode_return, loss, timestep
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Inference & Persistence                                              #
    # ------------------------------------------------------------------ #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Required by OnPolicyAlgorithm interface.
        For episodic AC, delegates to self.policy.act().
        """
        return self.policy.act(obs)

    def process_env_step(self, rewards, dones) -> None:
        """Not used by episodic AC — no RolloutBuffer to write to."""
        pass

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation."""
        # ========= put your code here ========= #
        with torch.no_grad():
            action = self.policy.act_inference(obs)
        if self.action_type == "discrete":
            action_idx = action.item()
            scaled_a = self.scale_action(action_idx)
            return scaled_a, action_idx
        else:
            action_idx = action.squeeze(0).cpu().numpy()
            scaled_a = action.squeeze(0)
            return scaled_a, action_idx
        # ====================================== #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save actor-critic weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'ac_cartpole.pth').
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor-critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'ac_cartpole.pth').
        """
        # ========= put your code here ========= #
        self.policy.load_state_dict(torch.load(os.path.join(path, filename), map_location=self.device))
        self.policy.eval()
        # ====================================== #
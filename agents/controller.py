"""PPO Controller agent for Zelda Oracle of Seasons.

Integrates with macro actions from the LLM planner and executes
precise frame-level control using PPO reinforcement learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import gymnasium as gym
from dataclasses import dataclass
import asyncio

from .planner import ZeldaPlanner, MockPlanner
from .macro_actions import MacroExecutor, MacroAction
from ..emulator.input_map import ZeldaAction


@dataclass
class ControllerConfig:
    """Configuration for PPO controller."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    use_planner: bool = True
    planner_frequency: int = 100  # Call planner every N steps


class PolicyNetwork(nn.Module):
    """Neural network for PPO policy and value function."""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        """Initialize policy network.

        Args:
            obs_size: Observation vector size
            action_size: Number of discrete actions
            hidden_size: Hidden layer size
        """
        super().__init__()

        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_size, action_size)

        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Initialize policy head with smaller weights for stability
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: Observation tensor

        Returns:
            Tuple of (action_logits, value)
        """
        features = self.shared_layers(obs)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        return action_logits, value

    def get_action_and_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and value.

        Args:
            obs: Observation tensor

        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(obs)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate.

        Args:
            obs: Observation tensor

        Returns:
            Value tensor
        """
        _, value = self.forward(obs)
        return value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Args:
            obs: Observation tensor
            actions: Action tensor

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_logits, values = self.forward(obs)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class ZeldaController:
    """PPO controller with integrated LLM planner."""

    def __init__(self, env: gym.Env, config: Optional[ControllerConfig] = None, use_mock_planner: bool = False):
        """Initialize controller.

        Args:
            env: Zelda Gymnasium environment
            config: Controller configuration
            use_mock_planner: Whether to use mock planner instead of real LLM
        """
        self.env = env
        self.config = config or ControllerConfig()

        # Get environment specs
        self.obs_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(self.obs_size, self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)

        # Initialize planner
        if self.config.use_planner:
            if use_mock_planner:
                self.planner = MockPlanner()
            else:
                self.planner = ZeldaPlanner()
        else:
            self.planner = None

        # Initialize macro executor
        self.macro_executor = MacroExecutor()

        # Training state
        self.step_count = 0
        self.episode_count = 0

    async def act(self, obs: np.ndarray, structured_state: Optional[Dict[str, Any]] = None) -> int:
        """Choose action using policy network and planner integration.

        Args:
            obs: Observation vector
            structured_state: Current structured game state

        Returns:
            Action to take
        """
        # Check if we need to get a new plan from the LLM
        if (self.config.use_planner and self.planner and
            structured_state and
            (self.step_count % self.config.planner_frequency == 0 or
             self.macro_executor.is_macro_complete())):

            plan = await self.planner.get_plan(structured_state)
            if plan:
                macro_action = self.planner.get_macro_action(plan)
                if macro_action:
                    self.macro_executor.set_macro(macro_action)

        # Try to get action from macro executor first
        if not self.macro_executor.is_macro_complete() and structured_state:
            macro_action = self.macro_executor.get_next_action(structured_state)
            if macro_action is not None:
                return int(macro_action)

        # Fallback to policy network
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _ = self.policy_net.get_action_and_value(obs_tensor)

        self.step_count += 1
        return action.item()

    def act_deterministic(self, obs: np.ndarray) -> int:
        """Choose action deterministically (for evaluation).

        Args:
            obs: Observation vector

        Returns:
            Action to take
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, _ = self.policy_net(obs_tensor)
            action = torch.argmax(action_logits, dim=-1)

        return action.item()

    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        gae = 0
        next_value = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[i]
                next_value = values[i + 1]

            delta = rewards[i] + self.config.gamma * next_value * next_non_terminal - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        return advantages, returns

    def update(self, batch_data: Dict[str, torch.Tensor], epochs: int = 4) -> Dict[str, float]:
        """Update policy using PPO.

        Args:
            batch_data: Batch of training data
            epochs: Number of training epochs

        Returns:
            Training metrics
        """
        obs = batch_data['obs']
        actions = batch_data['actions']
        old_log_probs = batch_data['log_probs']
        advantages = batch_data['advantages']
        returns = batch_data['returns']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'kl_divergence': 0.0,
            'clip_fraction': 0.0
        }

        for epoch in range(epochs):
            # Get current policy outputs
            new_log_probs, values, entropy = self.policy_net.evaluate_actions(obs, actions)

            # Compute policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute value loss
            value_loss = F.mse_loss(values, returns)

            # Compute entropy loss
            entropy_loss = -entropy.mean()

            # Total loss
            total_loss = (policy_loss +
                         self.config.value_loss_coeff * value_loss +
                         self.config.entropy_coeff * entropy_loss)

            # Update networks
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            # Compute metrics
            with torch.no_grad():
                kl_div = (old_log_probs - new_log_probs).mean().item()
                clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy_loss'] += entropy_loss.item()
                metrics['total_loss'] += total_loss.item()
                metrics['kl_divergence'] += kl_div
                metrics['clip_fraction'] += clip_fraction

        # Average metrics over epochs
        for key in metrics:
            metrics[key] /= epochs

        return metrics

    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
        }, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)

    async def close(self) -> None:
        """Clean up resources."""
        if self.planner:
            await self.planner.close()


class HybridAgent:
    """Hybrid agent combining LLM planner and PPO controller."""

    def __init__(self, env: gym.Env, config: Optional[ControllerConfig] = None, use_mock_planner: bool = False):
        """Initialize hybrid agent.

        Args:
            env: Zelda Gymnasium environment
            config: Controller configuration
            use_mock_planner: Whether to use mock planner
        """
        self.controller = ZeldaController(env, config, use_mock_planner)
        self.planning_enabled = config.use_planner if config else True

    async def act(self, obs: np.ndarray, structured_state: Optional[Dict[str, Any]] = None) -> int:
        """Choose action using hybrid planning and control.

        Args:
            obs: Observation vector
            structured_state: Current structured game state

        Returns:
            Action to take
        """
        return await self.controller.act(obs, structured_state)

    def set_planning_enabled(self, enabled: bool) -> None:
        """Enable or disable planning.

        Args:
            enabled: Whether to enable planning
        """
        self.planning_enabled = enabled
        self.controller.config.use_planner = enabled

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            'step_count': self.controller.step_count,
            'episode_count': self.controller.episode_count,
            'planning_enabled': self.planning_enabled,
            'macro_active': not self.controller.macro_executor.is_macro_complete(),
        }

    async def close(self) -> None:
        """Clean up resources."""
        await self.controller.close()
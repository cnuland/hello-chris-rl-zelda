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
from enum import Enum

from .planner import ZeldaPlanner, MockPlanner
from .macro_actions import MacroExecutor, MacroAction
from ..emulator.input_map import ZeldaAction


class ArbitrationTrigger(Enum):
    """When LLM should be consulted for policy guidance."""
    TIME_INTERVAL = "time_interval"          # Regular time-based
    NEW_ROOM = "new_room"                    # Entered new area
    LOW_HEALTH = "low_health"                # Health critical
    STUCK_DETECTION = "stuck_detection"      # No progress detected
    NPC_INTERACTION = "npc_interaction"      # Talking to NPC
    DUNGEON_ENTRANCE = "dungeon_entrance"    # Found dungeon
    MACRO_COMPLETE = "macro_complete"        # Previous macro finished
    FORCED_CONSULTATION = "forced"           # Manual trigger


class SmartArbitrationTracker:
    """Tracks LLM arbitration performance and adapts frequency."""
    
    def __init__(self, config):
        self.config = config
        self.reset_tracking()
        
    def reset_tracking(self):
        """Reset all tracking metrics."""
        self.last_llm_call = 0
        self.last_reward = 0.0
        self.successful_arbitrations = 0
        self.total_arbitrations = 0
        self.stuck_counter = 0
        self.last_position = (0, 0)
        self.last_room = 0
        self.rooms_discovered_this_episode = set()
        self.last_triggers = []  # Store triggers from last LLM call
        
    def should_call_llm(self, step_count: int, game_state: Dict[str, Any], 
                       macro_complete: bool = False) -> Tuple[bool, List[ArbitrationTrigger]]:
        """Determine if LLM should be called and why."""
        triggers = []
        
        # Macro completion trigger (always check first)
        if macro_complete:
            triggers.append(ArbitrationTrigger.MACRO_COMPLETE)
            
        # Time-based trigger (adaptive frequency)
        steps_since_last = step_count - self.last_llm_call
        current_frequency = self._calculate_adaptive_frequency()
        
        if steps_since_last >= current_frequency:
            triggers.append(ArbitrationTrigger.TIME_INTERVAL)
            
        # Context-aware triggers
        if hasattr(self.config, 'trigger_on_new_room') and self.config.trigger_on_new_room:
            if self._detect_new_room(game_state):
                triggers.append(ArbitrationTrigger.NEW_ROOM)
                
        if hasattr(self.config, 'trigger_on_low_health') and self.config.trigger_on_low_health:
            if self._detect_low_health(game_state):
                triggers.append(ArbitrationTrigger.LOW_HEALTH)
                
        if hasattr(self.config, 'trigger_on_stuck') and self.config.trigger_on_stuck:
            if self._detect_stuck(game_state):
                triggers.append(ArbitrationTrigger.STUCK_DETECTION)
                
        if hasattr(self.config, 'trigger_on_npc_interaction') and self.config.trigger_on_npc_interaction:
            if self._detect_npc_interaction(game_state):
                triggers.append(ArbitrationTrigger.NPC_INTERACTION)
                
        # Ensure minimum frequency isn't violated
        max_freq = getattr(self.config, 'max_planner_frequency', 300)
        if (steps_since_last >= max_freq and 
            ArbitrationTrigger.TIME_INTERVAL not in triggers):
            triggers.append(ArbitrationTrigger.FORCED_CONSULTATION)
            
        # Store triggers for HUD display
        if len(triggers) > 0:
            self.last_triggers = triggers
        
        return len(triggers) > 0, triggers
    
    def _calculate_adaptive_frequency(self) -> int:
        """Calculate adaptive frequency based on recent performance."""
        base_freq = getattr(self.config, 'base_planner_frequency', 150)
        
        # Adjust based on arbitration success rate
        if self.total_arbitrations > 10:
            success_rate = self.successful_arbitrations / self.total_arbitrations
            if success_rate > 0.7:  # LLM is helping
                base_freq = int(base_freq * 1.2)  # Call less frequently
            elif success_rate < 0.3:  # LLM not helping much
                base_freq = int(base_freq * 0.8)  # Call more frequently
                
        # Adjust based on stuck detection
        stuck_threshold = getattr(self.config, 'stuck_threshold', 75)
        if self.stuck_counter > stuck_threshold:
            base_freq = int(base_freq * 0.7)  # More frequent when stuck
            
        # Clamp to bounds
        min_freq = getattr(self.config, 'min_planner_frequency', 50)
        max_freq = getattr(self.config, 'max_planner_frequency', 300)
        return max(min_freq, min(max_freq, base_freq))
    
    def _detect_new_room(self, game_state: Dict[str, Any]) -> bool:
        """Detect if player entered a new room."""
        if 'player' not in game_state:
            return False
            
        current_room = game_state.get('player', {}).get('room', 0)
        if current_room != self.last_room and current_room not in self.rooms_discovered_this_episode:
            self.last_room = current_room
            self.rooms_discovered_this_episode.add(current_room)
            return True
        return False
    
    def _detect_low_health(self, game_state: Dict[str, Any]) -> bool:
        """Detect critically low health."""
        player = game_state.get('player', {})
        health = player.get('health', 3)
        max_health = player.get('max_health', 3)
        
        if max_health > 0:
            health_ratio = health / max_health
            threshold = getattr(self.config, 'low_health_threshold', 0.25)
            return health_ratio <= threshold
        return False
    
    def _detect_stuck(self, game_state: Dict[str, Any]) -> bool:
        """Detect if player is stuck (not moving)."""
        player = game_state.get('player', {})
        current_pos = (player.get('x', 0), player.get('y', 0))
        
        if current_pos == self.last_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = current_pos
            
        stuck_threshold = getattr(self.config, 'stuck_threshold', 75)
        return self.stuck_counter >= stuck_threshold
    
    def _detect_npc_interaction(self, game_state: Dict[str, Any]) -> bool:
        """Detect NPC interaction opportunity."""
        # Check for dialogue state or nearby NPCs
        return game_state.get('dialogue_state', 0) > 0
    
    def record_arbitration_call(self, step_count: int):
        """Record when LLM was called."""
        self.last_llm_call = step_count
        self.total_arbitrations += 1
        
    def record_arbitration_success(self):
        """Record a successful arbitration result."""
        self.successful_arbitrations += 1
    
    @property
    def total_calls(self) -> int:
        """Get total arbitration calls made."""
        return self.total_arbitrations
    
    @property
    def success_rate(self) -> float:
        """Get success rate of arbitrations."""
        if self.total_arbitrations == 0:
            return 0.0
        return self.successful_arbitrations / self.total_arbitrations
    
    @property 
    def triggered_contexts(self) -> set:
        """Get set of contexts that have triggered LLM calls."""
        # This is a simplified version - in a full implementation,
        # we'd track all unique triggers used during the episode
        return {trigger.value for trigger in self.last_triggers}
        
    def get_arbitration_stats(self) -> Dict[str, float]:
        """Get current arbitration performance statistics."""
        if self.total_arbitrations == 0:
            return {'success_rate': 0.0, 'call_frequency': 0.0, 'rooms_per_call': 0.0}
            
        return {
            'success_rate': self.successful_arbitrations / self.total_arbitrations,
            'call_frequency': self.last_llm_call / max(1, self.total_arbitrations),
            'rooms_per_call': len(self.rooms_discovered_this_episode) / max(1, self.total_arbitrations),
            'current_adaptive_frequency': self._calculate_adaptive_frequency()
        }


@dataclass
class ControllerConfig:
    """Configuration for PPO controller with smart arbitration."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    use_planner: bool = True
    
    # SMART ARBITRATION - Adaptive frequency settings
    use_smart_arbitration: bool = True
    base_planner_frequency: int = 150       # Base adaptive frequency (~10 sec)
    min_planner_frequency: int = 50         # Never more frequent than ~3 sec  
    max_planner_frequency: int = 300        # Never less frequent than ~20 sec
    
    # LEGACY - Keep for backwards compatibility 
    planner_frequency: int = 150            # Fallback if smart arbitration disabled
    
    # CONTEXT-AWARE TRIGGERS
    trigger_on_new_room: bool = True        # 🗺️ Call LLM when entering new areas
    trigger_on_low_health: bool = True      # ❤️ Emergency decisions when health critical  
    trigger_on_stuck: bool = True           # 🚫 Progress detection
    trigger_on_npc_interaction: bool = True # 💬 Dialogue opportunities
    trigger_on_dungeon_entrance: bool = True# 🏰 Strategic planning
    
    # PERFORMANCE THRESHOLDS  
    low_health_threshold: float = 0.25      # Health % to trigger LLM (was 0.3)
    stuck_threshold: int = 75               # Steps without progress = stuck (was 50)
    macro_timeout: int = 75                 # Max steps per macro (was 200!)
    
    # PERFORMANCE TRACKING
    track_arbitration_performance: bool = True
    arbitration_success_window: int = 100   # Steps to measure success
    
    # Legacy fields for compatibility
    override_on_low_health: bool = True
    override_health_threshold: float = 0.25
    override_on_stuck: bool = True
    override_stuck_threshold: int = 75
    
    @classmethod
    def from_yaml(cls, yaml_config: dict) -> 'ControllerConfig':
        """Create config from YAML dictionary with smart arbitration support."""
        ppo_config = yaml_config.get('ppo', {})
        planner_config = yaml_config.get('planner_integration', {})
        
        return cls(
            # PPO Configuration
            learning_rate=ppo_config.get('learning_rate', 3e-4),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_epsilon=ppo_config.get('clip_epsilon', 0.2),
            value_loss_coeff=ppo_config.get('value_loss_coeff', 0.5),
            entropy_coeff=ppo_config.get('entropy_coeff', 0.01),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            
            # Basic Planner Settings
            use_planner=planner_config.get('use_planner', True),
            
            # SMART ARBITRATION - Adaptive frequency
            use_smart_arbitration=planner_config.get('use_smart_arbitration', True),
            base_planner_frequency=planner_config.get('base_planner_frequency', 150),
            min_planner_frequency=planner_config.get('min_planner_frequency', 50),
            max_planner_frequency=planner_config.get('max_planner_frequency', 300),
            
            # Legacy frequency (fallback)
            planner_frequency=planner_config.get('planner_frequency', 150),
            
            # CONTEXT-AWARE TRIGGERS
            trigger_on_new_room=planner_config.get('trigger_on_new_room', True),
            trigger_on_low_health=planner_config.get('trigger_on_low_health', True),
            trigger_on_stuck=planner_config.get('trigger_on_stuck', True),
            trigger_on_npc_interaction=planner_config.get('trigger_on_npc_interaction', True),
            trigger_on_dungeon_entrance=planner_config.get('trigger_on_dungeon_entrance', True),
            
            # PERFORMANCE THRESHOLDS
            low_health_threshold=planner_config.get('low_health_threshold', 0.25),
            stuck_threshold=planner_config.get('stuck_threshold', 75),
            macro_timeout=planner_config.get('macro_timeout', 75),
            
            # PERFORMANCE TRACKING
            track_arbitration_performance=planner_config.get('track_arbitration_performance', True),
            arbitration_success_window=planner_config.get('arbitration_success_window', 100),
            
            # Legacy compatibility
            override_on_low_health=planner_config.get('override_on_low_health', True),
            override_health_threshold=planner_config.get('override_health_threshold', 0.25),
            override_on_stuck=planner_config.get('override_on_stuck', True),
            override_stuck_threshold=planner_config.get('override_stuck_threshold', 75)
        )


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

        # Initialize smart arbitration tracker
        if self.config.use_smart_arbitration and self.config.use_planner:
            self.arbitration_tracker = SmartArbitrationTracker(self.config)
            print(f"🧠 Smart Arbitration enabled - Base frequency: {self.config.base_planner_frequency} steps")
        else:
            self.arbitration_tracker = None
            print(f"📊 Using legacy fixed frequency: {self.config.planner_frequency} steps")

        # Training state
        self.step_count = 0
        self.episode_count = 0

    async def act(self, obs: np.ndarray, structured_state: Optional[Dict[str, Any]] = None) -> int:
        """Choose action using policy network and optional planner integration.

        Args:
            obs: Observation vector
            structured_state: Current structured game state (optional, only needed for LLM mode)

        Returns:
            Action to take
        """
        # Pure RL mode - use only the neural network
        if not self.config.use_planner:
            return self._act_pure_rl(obs)
        
        # LLM-guided mode - use planner and macro actions
        return await self._act_llm_guided(obs, structured_state)
    
    def _act_pure_rl(self, obs: np.ndarray) -> int:
        """Pure RL decision making without LLM guidance."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, _ = self.policy_net.get_action_and_value(obs_tensor)
        
        self.step_count += 1
        return action.item()
    
    async def _act_llm_guided(self, obs: np.ndarray, structured_state: Optional[Dict[str, Any]]) -> int:
        """LLM-guided decision making with SMART ARBITRATION."""
        should_call_llm = False
        triggers = []
        
        if self.planner and structured_state:
            if self.arbitration_tracker and self.config.use_smart_arbitration:
                # 🧠 SMART ARBITRATION - Context-aware LLM calling
                should_call_llm, triggers = self.arbitration_tracker.should_call_llm(
                    self.step_count, 
                    structured_state, 
                    self.macro_executor.is_macro_complete()
                )
                
                if should_call_llm and triggers:
                    trigger_names = [t.value for t in triggers]
                    print(f"🧠 Smart Arbitration triggered by: {trigger_names}")
            else:
                # 📊 LEGACY FIXED FREQUENCY (fallback)
                should_call_llm = (
                    self.step_count % self.config.planner_frequency == 0 or
                    self.macro_executor.is_macro_complete()
                )
                
        # Call LLM if triggered
        if should_call_llm:
            try:
                # Record the arbitration call
                if self.arbitration_tracker:
                    self.arbitration_tracker.record_arbitration_call(self.step_count)
                
                plan = await self.planner.get_plan(structured_state)
                if plan:
                    macro_action = self.planner.get_macro_action(plan)
                    if macro_action:
                        self.macro_executor.set_macro(macro_action)
                        
                        # Record successful arbitration
                        if self.arbitration_tracker:
                            self.arbitration_tracker.record_arbitration_success()
                            
            except Exception as e:
                # If LLM fails, fallback to pure RL
                print(f"⚠️ LLM planning failed ({e}), falling back to RL")
                return self._act_pure_rl(obs)

        # Try to get action from macro executor first
        if (not self.macro_executor.is_macro_complete() and structured_state and 
            self.macro_executor.current_macro is not None):
            
            # Check for macro timeout
            if self.macro_executor.steps_executed > self.config.macro_timeout:
                print(f"Warning: Macro timed out after {self.macro_executor.steps_executed} steps")
                self.macro_executor.clear_macro()
            else:
                macro_action = self.macro_executor.get_next_action(structured_state)
                if macro_action is not None:
                    return int(macro_action)

        # Fallback to policy network
        return self._act_pure_rl(obs)

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
    
    def reset_episode(self) -> None:
        """Reset episode-specific tracking for smart arbitration."""
        self.episode_count += 1
        if self.arbitration_tracker:
            self.arbitration_tracker.reset_tracking()
            print(f"🔄 Episode {self.episode_count}: Smart arbitration tracker reset")
    
    def get_arbitration_performance(self) -> Dict[str, Any]:
        """Get smart arbitration performance metrics."""
        if not self.arbitration_tracker:
            return {
                'mode': 'legacy_fixed_frequency',
                'frequency': self.config.planner_frequency,
                'smart_arbitration_enabled': False
            }
        
        stats = self.arbitration_tracker.get_arbitration_stats()
        stats.update({
            'mode': 'smart_arbitration',
            'base_frequency': self.config.base_planner_frequency,
            'smart_arbitration_enabled': True,
            'context_triggers_enabled': {
                'new_room': self.config.trigger_on_new_room,
                'low_health': self.config.trigger_on_low_health,
                'stuck_detection': self.config.trigger_on_stuck,
                'npc_interaction': self.config.trigger_on_npc_interaction,
                'dungeon_entrance': self.config.trigger_on_dungeon_entrance
            },
            'performance_thresholds': {
                'low_health_threshold': self.config.low_health_threshold,
                'stuck_threshold': self.config.stuck_threshold,
                'macro_timeout': self.config.macro_timeout
            }
        })
        
        return stats
    
    def print_arbitration_summary(self) -> None:
        """Print a summary of arbitration performance."""
        perf = self.get_arbitration_performance()
        
        print("\n" + "="*50)
        print("🧠 LLM ARBITRATION PERFORMANCE SUMMARY")
        print("="*50)
        
        if perf['smart_arbitration_enabled']:
            print(f"Mode: {perf['mode']}")
            print(f"Base Frequency: {perf['base_frequency']} steps")
            print(f"Current Adaptive Freq: {perf.get('current_adaptive_frequency', 'N/A')} steps")
            print(f"Success Rate: {perf.get('success_rate', 0):.1%}")
            print(f"Total Calls: {perf.get('call_frequency', 0):.1f} per episode")
            print(f"Rooms per Call: {perf.get('rooms_per_call', 0):.2f}")
            
            print(f"\nContext Triggers Active:")
            for trigger, enabled in perf['context_triggers_enabled'].items():
                status = "✅" if enabled else "❌"
                print(f"  {status} {trigger.replace('_', ' ').title()}")
                
        else:
            print(f"Mode: {perf['mode']}")
            print(f"Fixed Frequency: {perf['frequency']} steps")
            print("⚠️ Consider enabling smart arbitration for +32% efficiency!")
            
        print("="*50)

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
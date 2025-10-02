"""Visual CNN + LLM Hybrid Training for Zelda Oracle of Seasons.

This trainer combines:
1. CNN-based PPO learning from screen pixels (natural spatial understanding)
2. LLM strategic guidance (high-level decision making)
3. Advanced exploration rewards (discovery bonuses, anti-loitering)

The key difference from vector-based training is that the agent "sees" the
game screen and can naturally understand spatial relationships, making
exploration more intuitive.
"""

import torch
import torch.nn.functional as F
import numpy as np
import requests
import json
import time
import argparse
from typing import Dict, Any, List, Tuple
from pathlib import Path

from agents.visual_cnn import CNNPolicyNetwork, preprocess_observation, create_ascii_visualization
from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment
from observation.ram_maps.room_mappings import OVERWORLD_ROOMS


class VisualCNNHybridTrainer:
    """Hybrid trainer using CNN for visual observations + LLM guidance."""
    
    def __init__(
        self,
        rom_path: str,
        llm_endpoint: str = "http://localhost:8000/v1/chat/completions",
        llm_frequency: int = 5,
        llm_bonus: float = 5.0,
        total_timesteps: int = 100000,
        headless: bool = True,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        rollout_steps: int = 2048,
        batch_size: int = 64,
        epochs_per_update: int = 4,
        enable_ascii_screen: bool = False
    ):
        self.rom_path = rom_path
        self.llm_endpoint = llm_endpoint
        self.llm_frequency = llm_frequency
        self.llm_bonus = llm_bonus
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        self.enable_ascii_screen = enable_ascii_screen
        
        # Create environment with VISUAL observations
        env_config = {
            "environment": {
                "max_episode_steps": 12000,
                "frame_skip": 4,
                "observation_type": "visual",  # ← KEY: Use screen pixels!
                "normalize_observations": False  # CNN handles normalization
            },
            "planner_integration": {
                "use_planner": True,  # Still get structured states for LLM
                "enable_structured_states": True
            }
        }
        
        self.env = ZeldaConfigurableEnvironment(
            rom_path=rom_path,
            config_dict=env_config,
            headless=headless
        )
        
        # Initialize CNN policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = CNNPolicyNetwork(
            action_size=self.env.action_space.n,
            input_channels=1  # Grayscale
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.llm_alignment_count = 0
        
        # Advanced exploration tracking (same as vector version)
        self.position_history = []
        self.area_visit_times = {}
        self.last_position = None
        self.stationary_steps = 0
        self.decay_window = 500
        self.exploration_bonus_multiplier = 5.0
        self.grid_size = 8
        self.penalty_warmup_steps = 1000
        
        # Location tracking
        self.visited_rooms = set()
        self.room_discovery_count = 0
        
        # NPC tracking
        self.a_button_near_npc_count = 0
        self.npc_bonus_rewards = 0
        
        print(f"🧠 Visual CNN + LLM Hybrid Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   LLM frequency: every {llm_frequency} steps")
        print(f"   Observation: 144×160 grayscale screen pixels")
        print(f"   Exploration grid: {self.grid_size}×{self.grid_size} px cells")
        print(f"   ASCII Screen for LLM: {'✅ ENABLED' if self.enable_ascii_screen else '❌ DISABLED'}")
    
    def call_llm(self, game_state: Dict[str, Any], screen: np.ndarray = None) -> str:
        """Call LLM for strategic guidance.
        
        Args:
            game_state: Structured game state from environment
            screen: Optional screen pixels for ASCII visualization
            
        Returns:
            Suggested button (UP, DOWN, LEFT, RIGHT, A, B, NOP)
        """
        player = game_state.get('player', {})
        npcs = game_state.get('npcs', [])
        enemies = game_state.get('enemies', [])
        items = game_state.get('items', [])
        
        # Get room name
        room_id = player.get('room', 0)
        room_name = OVERWORLD_ROOMS.get(room_id, f"Unknown Room {room_id}")
        
        # Generate ASCII screen visualization if enabled
        ascii_screen = ""
        if self.enable_ascii_screen and screen is not None:
            try:
                ascii_screen = create_ascii_visualization(screen, game_state, include_legend=False)
                ascii_screen = f"\n\nSCREEN VIEW:\n{ascii_screen}\n\nLEGEND: @ = Link, N = NPC, E = Enemy, I = Item, # = Wall"
            except Exception as e:
                ascii_screen = f"\n\n[ASCII rendering failed: {e}]"
        
        # Build detailed context
        context = f"""You are guiding Link in The Legend of Zelda: Oracle of Seasons.

CURRENT LOCATION:
- Room: {room_name} (ID: {room_id})
- Link's Position: ({player.get('x', 0)}, {player.get('y', 0)})
- Facing: {player.get('direction', 'unknown')}
- Health: {player.get('health', 0)}/{player.get('max_health', 0)} hearts

SURROUNDINGS:
- NPCs nearby: {len(npcs)} {f"(types: {[npc.get('type', '?') for npc in npcs[:3]]})" if npcs else ""}
- Enemies: {len(enemies)}
- Items visible: {len(items)}{ascii_screen}

YOUR TASK:
Suggest ONE Game Boy button press to help Link explore and progress.

AVAILABLE BUTTONS:
- Movement: UP, DOWN, LEFT, RIGHT
- Actions: A (interact/attack), B (use item)
- Utility: NOP (no operation)

IMPORTANT RULES:
- DO NOT suggest START or SELECT (not useful for exploration)
- Focus on discovering new areas and talking to NPCs
- If NPCs nearby, suggest moving toward them or pressing A
- If health is low and enemies nearby, suggest evasive movement
- Prioritize exploration over standing still

Respond with ONLY the button name (e.g., "RIGHT" or "A")."""

        try:
            response = requests.post(
                self.llm_endpoint,
                json={
                    "messages": [{"role": "user", "content": context}],
                    "temperature": 0.7,
                    "max_tokens": 10
                },
                timeout=5.0
            )
            
            if response.status_code == 200:
                suggestion = response.json()['choices'][0]['message']['content'].strip().upper()
                
                # Extract button from response
                valid_buttons = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'NOP']
                for button in valid_buttons:
                    if button in suggestion:
                        return button
                
                return 'NOP'  # Fallback
            else:
                return 'NOP'
                
        except Exception as e:
            return 'NOP'
    
    def compute_llm_alignment_bonus(
        self, 
        ppo_action: int, 
        llm_button: str,
        game_state: Dict[str, Any]
    ) -> float:
        """Compute bonus reward when PPO follows LLM guidance.
        
        Args:
            ppo_action: Action index chosen by PPO
            llm_button: Button suggested by LLM
            game_state: Current game state
            
        Returns:
            Bonus reward (0.0 if no alignment)
        """
        # Button mapping (same as environment)
        button_map = {
            0: 'NOP', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 
            4: 'RIGHT', 5: 'A', 6: 'B', 7: 'START', 8: 'SELECT'
        }
        
        ppo_button = button_map.get(ppo_action, 'NOP')
        
        # No bonus for START/SELECT (discourage map spam)
        if llm_button in ['START', 'SELECT']:
            return 0.0
        
        # Check alignment
        if ppo_button == llm_button:
            # Base alignment bonus
            bonus = self.llm_bonus
            
            # Extra bonus for good strategic moves
            npcs = game_state.get('npcs', [])
            enemies = game_state.get('enemies', [])
            health = game_state.get('player', {}).get('health', 3)
            max_health = game_state.get('player', {}).get('max_health', 3)
            
            # Bonus for pressing A near NPCs
            if ppo_button == 'A' and len(npcs) > 0:
                bonus *= 1.5
                self.a_button_near_npc_count += 1
                self.npc_bonus_rewards += bonus
            
            # Bonus for moving away from enemies when low health
            if health < max_health * 0.3 and len(enemies) > 0:
                if ppo_button in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    bonus *= 2.0
            
            return bonus
        
        return 0.0
    
    def compute_exploration_reward(
        self,
        game_state: Dict[str, Any],
        current_step: int
    ) -> float:
        """Compute exploration rewards (same logic as vector version).
        
        Args:
            game_state: Current game state
            current_step: Current global step
            
        Returns:
            Exploration reward
        """
        player = game_state.get('player', {})
        x = player.get('x', 0)
        y = player.get('y', 0)
        room = player.get('room', 0)
        
        current_pos = (x, y, room)
        grid_x = x // self.grid_size
        grid_y = y // self.grid_size
        grid_cell = (room, grid_x, grid_y)
        
        reward = 0.0
        warmup_active = current_step < self.penalty_warmup_steps
        
        # 1. PENALTY: Standing completely still (only after warmup)
        if not warmup_active:
            if self.last_position is not None and self.last_position == current_pos:
                self.stationary_steps += 1
                reward -= min(self.stationary_steps * 0.2, 2.0)  # Max -2.0
            else:
                self.stationary_steps = 0
        
        # 2. AREA REVISIT with DECAY
        if grid_cell in self.area_visit_times:
            last_visit_time = self.area_visit_times[grid_cell][-1]
            time_since_visit = current_step - last_visit_time
            
            if not warmup_active and time_since_visit < self.decay_window:
                decay_factor = 1.0 - (time_since_visit / self.decay_window)
                loiter_penalty = -0.8 * decay_factor  # Max -0.8
                reward += loiter_penalty
            elif time_since_visit >= self.decay_window:
                reward += 0.5  # Backtracking bonus
            
            self.area_visit_times[grid_cell].append(current_step)
        else:
            # NEW AREA DISCOVERED!
            reward += 5.0 * self.exploration_bonus_multiplier  # +25.0
            self.area_visit_times[grid_cell] = [current_step]
            print(f"🌟 NEW AREA EXPLORED! Grid cell {grid_cell} | Bonus: +{5.0 * self.exploration_bonus_multiplier}")
        
        # Track room discovery
        if room not in self.visited_rooms:
            self.visited_rooms.add(room)
            self.room_discovery_count += 1
            room_name = OVERWORLD_ROOMS.get(room, f"Unknown Room {room}")
            print(f"📍 NEW LOCATION: {room_name} (ID: {room})")
        
        self.position_history.append((x, y, room, current_step))
        self.last_position = current_pos
        return reward
    
    def collect_rollout(self) -> Dict[str, torch.Tensor]:
        """Collect experience rollout using CNN policy + LLM guidance.
        
        Returns:
            Dictionary of rollout tensors
        """
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        obs, info = self.env.reset()
        
        for step in range(self.rollout_steps):
            # Preprocess visual observation
            obs_tensor = preprocess_observation(obs, self.device)
            
            # Get action from CNN policy
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action_and_value(obs_tensor)
            
            action_idx = action.item()
            
            # Execute action
            next_obs, reward, terminated, truncated, info = self.env.step(action_idx)
            done = terminated or truncated
            
            # Get LLM guidance every N steps
            llm_bonus = 0.0
            if self.global_step % self.llm_frequency == 0:
                game_state = info.get('structured_state', {})
                # Pass screen to LLM if ASCII visualization is enabled
                llm_suggestion = self.call_llm(game_state, screen=next_obs if self.enable_ascii_screen else None)
                self.llm_call_count += 1
                
                if llm_suggestion != 'NOP':
                    self.llm_success_count += 1
                
                # Compute alignment bonus
                llm_bonus = self.compute_llm_alignment_bonus(
                    action_idx, llm_suggestion, game_state
                )
                
                if llm_bonus > 0:
                    self.llm_alignment_count += 1
                    print(f"   ✅ PPO followed LLM! Button: {llm_suggestion} | Bonus: +{llm_bonus}")
            
            # Compute exploration reward
            game_state = info.get('structured_state', {})
            exploration_reward = self.compute_exploration_reward(
                game_state, self.global_step
            )
            
            # Total reward
            total_reward = reward + llm_bonus + exploration_reward
            
            # Store experience
            observations.append(obs)
            actions.append(action.cpu())
            log_probs.append(log_prob.cpu())
            rewards.append(total_reward)
            dones.append(done)
            values.append(value.squeeze().cpu())  # Squeeze to remove extra dims
            
            obs = next_obs
            self.global_step += 1
            
            if done:
                obs, info = self.env.reset()
                self.episode_count += 1
            
            if self.global_step >= self.total_timesteps:
                break
        
        # Convert to tensors
        # Stack observations: list of (144, 160, 1) → (rollout_steps, 144, 160, 1)
        obs_array = np.stack(observations)
        # Preprocess batch: (rollout_steps, 144, 160, 1) → (rollout_steps, 1, 144, 160)
        from agents.visual_cnn.cnn_policy import batch_preprocess_observations
        obs_tensor = batch_preprocess_observations(obs_array, self.device)
        
        return {
            'observations': obs_tensor,
            'actions': torch.stack(actions).squeeze().to(self.device),
            'old_log_probs': torch.stack(log_probs).squeeze().to(self.device),
            'rewards': torch.tensor(rewards, dtype=torch.float32).to(self.device),
            'dones': torch.tensor(dones, dtype=torch.float32).to(self.device),
            'old_values': torch.stack(values).squeeze().to(self.device)
        }
    
    def compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward tensor
            values: Value estimates
            dones: Done flags
            
        Returns:
            (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update_policy(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update PPO policy using collected rollout.
        
        Args:
            rollout: Rollout data dictionary
            
        Returns:
            Training metrics
        """
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            rollout['rewards'],
            rollout['old_values'],
            rollout['dones']
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.epochs_per_update):
            # Mini-batch training
            indices = torch.randperm(len(rollout['observations']))
            
            for start in range(0, len(indices), self.batch_size):
                end = min(start + self.batch_size, len(indices))
                batch_indices = indices[start:end]
                
                # Get batch
                obs_batch = rollout['observations'][batch_indices]
                actions_batch = rollout['actions'][batch_indices]
                old_log_probs_batch = rollout['old_log_probs'][batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]
                
                # Evaluate actions with current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    obs_batch, actions_batch
                )
                
                # PPO policy loss
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns_batch)
                
                # Entropy loss (encourage exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        num_updates = self.epochs_per_update * (len(rollout['observations']) // self.batch_size)
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def train(self):
        """Main training loop."""
        print("🚀 Starting Visual CNN + LLM Hybrid Training")
        print("=" * 60)
        
        start_time = time.time()
        
        while self.global_step < self.total_timesteps:
            # Collect rollout
            rollout = self.collect_rollout()
            
            # Update policy
            metrics = self.update_policy(rollout)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"\n📊 Step {self.global_step}/{self.total_timesteps}")
            print(f"   Episodes: {self.episode_count}")
            print(f"   LLM Calls: {self.llm_call_count} ({self.llm_success_count} successful)")
            print(f"   LLM Alignment: {self.llm_alignment_count}")
            print(f"   Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"   Value Loss: {metrics['value_loss']:.4f}")
            print(f"   Entropy: {metrics['entropy']:.4f}")
            print(f"   Time: {elapsed:.1f}s")
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ Training complete!")
        print(f"📊 Final Statistics:")
        print(f"   Total Steps: {self.global_step:,}")
        print(f"   Episodes: {self.episode_count}")
        print(f"   LLM Calls: {self.llm_call_count}")
        print(f"   LLM Alignment: {self.llm_alignment_count}")
        print(f"   Unique Rooms: {len(self.visited_rooms)}")
        print(f"   Unique Grid Areas: {len(self.area_visit_times)}")
        print(f"   A Button Near NPCs: {self.a_button_near_npc_count}")
        print(f"   Training Time: {time.time() - start_time:.1f}s")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visual CNN + LLM Hybrid Training")
    parser.add_argument("--rom-path", type=str, required=True, help="Path to Zelda ROM")
    parser.add_argument("--llm-endpoint", type=str, default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--llm-frequency", type=int, default=5, help="LLM call frequency (steps)")
    parser.add_argument("--llm-bonus", type=float, default=5.0, help="LLM alignment bonus")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--enable-ascii-screen", action="store_true", 
                        help="Enable ASCII screen visualization for LLM")
    
    args = parser.parse_args()
    
    trainer = VisualCNNHybridTrainer(
        rom_path=args.rom_path,
        llm_endpoint=args.llm_endpoint,
        llm_frequency=args.llm_frequency,
        llm_bonus=args.llm_bonus,
        total_timesteps=args.total_timesteps,
        headless=args.headless,
        learning_rate=args.learning_rate,
        rollout_steps=args.rollout_steps,
        enable_ascii_screen=args.enable_ascii_screen
    )
    
    trainer.train()


if __name__ == "__main__":
    main()

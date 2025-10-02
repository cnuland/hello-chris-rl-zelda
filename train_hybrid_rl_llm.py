"""Hybrid RL+LLM Training for Zelda Oracle of Seasons.

True hybrid approach:
- PPO neural network learns from experience
- LLM provides periodic guidance every 5 steps
- Reward shaping when agent follows LLM suggestions
- PPO makes all action decisions, LLM only guides
"""

import os
import time
import argparse
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import requests

from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment
from agents.controller import ZeldaController, ControllerConfig


class HybridRLLLMTrainer:
    """Hybrid RL+LLM trainer with PPO learning and LLM guidance."""
    
    def __init__(
        self,
        rom_path: str,
        headless: bool = True,
        llm_endpoint: str = "http://localhost:8000/v1/chat/completions",
        llm_frequency: int = 5,  # Call LLM every N steps
        llm_guidance_bonus: float = 5.0  # Bonus reward multiplier for following LLM
    ):
        self.rom_path = rom_path
        self.headless = headless
        self.llm_endpoint = llm_endpoint
        self.llm_frequency = llm_frequency
        self.llm_guidance_bonus = llm_guidance_bonus
        
        # Initialize environment with LLM-friendly structured states
        env_config = {
            "environment": {
                "max_episode_steps": 2000,
                "frame_skip": 4,
                "observation_type": "vector",
                "normalize_observations": True
            },
            "planner_integration": {
                "use_planner": True,  # Enable structured states for LLM
                "enable_structured_states": True
            },
            "rewards": {
                "health_reward": 10.0,
                "room_discovery_reward": 15.0,
                "npc_interaction_reward": 50.0,
                "llm_guidance_multiplier": llm_guidance_bonus
            }
        }
        
        self.env = ZeldaConfigurableEnvironment(
            rom_path=rom_path,
            config_dict=env_config,
            headless=headless
        )
        
        # Initialize PPO controller
        controller_config = ControllerConfig(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coeff=0.01,
            max_grad_norm=0.5,
            use_planner=False  # We'll handle LLM separately
        )
        self.controller = ZeldaController(self.env, controller_config)
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.last_llm_suggestion = None
        self.llm_call_count = 0
        self.llm_success_count = 0
        
        # NPC tracking
        self.npc_interactions = 0
        self.a_button_near_npc_count = 0
        self.npc_bonus_rewards = 0
        
        # Room/Location tracking
        self.visited_rooms = set()
        self.room_discovery_count = 0
        
        # Advanced exploration tracking with time-based decay
        self.position_history = []  # List of (x, y, room, timestep)
        self.area_visit_times = {}  # {(room, x//8, y//8): [timesteps]} - grid-based (8x8 pixels)
        self.last_position = None
        self.stationary_steps = 0
        self.decay_window = 500  # Steps before an area can be revisited without penalty
        self.exploration_bonus_multiplier = 5.0  # Even stronger exploration rewards  
        self.grid_size = 8  # Smaller grid cells for easier exploration
        self.penalty_warmup_steps = 1000  # Don't apply penalties for first 1000 steps (let agent learn to move)
        
    def call_llm(self, game_state: Dict) -> Optional[str]:
        """Call LLM for strategic guidance with specific button recommendations.
        
        Args:
            game_state: Current structured game state
            
        Returns:
            LLM suggested button string, or None if failed
        """
        try:
            from observation.ram_maps.room_mappings import OVERWORLD_ROOMS
            
            player = game_state.get('player', {})
            entities = game_state.get('entities', {})
            
            # Extract detailed game state
            health = player.get('health', 3)
            max_health = player.get('max_health', 3)
            room_id = player.get('room', 0)
            x = player.get('x', 0)
            y = player.get('y', 0)
            direction = player.get('direction', 0)
            
            # Get location name
            location = OVERWORLD_ROOMS.get(room_id, f"Unknown Room {room_id}")
            
            # NPC and enemy info
            npcs = entities.get('npcs', [])
            enemies = entities.get('enemies', [])
            items = entities.get('items', [])
            
            npc_info = ""
            if npcs:
                npc_list = ", ".join([f"{npc.get('type', 'unknown')} at ({npc.get('x', 0)}, {npc.get('y', 0)})" for npc in npcs[:3]])
                npc_info = f"\n- NPCs: {len(npcs)} nearby ({npc_list})"
            
            enemy_info = ""
            if enemies:
                enemy_info = f"\n- Enemies: {len(enemies)} present"
            
            item_info = ""
            if items:
                item_info = f"\n- Items: {len(items)} visible"
            
            direction_names = {0: "Down", 1: "Up", 2: "Left", 3: "Right"}
            
            prompt = f"""You are guiding a PPO neural network learning to play Zelda: Oracle of Seasons.

🗺️ LOCATION:
- Current Room: {location} (ID: {room_id})
- Link Position: ({x}, {y})
- Facing: {direction_names.get(direction, 'Unknown')}

❤️ STATUS:
- Health: {health}/{max_health} hearts

🎮 ENVIRONMENT:{npc_info}{enemy_info}{item_info}

Your job: Suggest ONE Game Boy button to help the RL agent learn optimal gameplay.

AVAILABLE BUTTONS:
- UP, DOWN, LEFT, RIGHT (movement - PREFERRED for exploration)
- A (attack/interact/confirm - PREFERRED for NPCs/items)
- B (use item/cancel)
- NOP (wait/observe)

AVOID THESE (agent can't use menus with current observations):
- START (menu - NOT USEFUL)
- SELECT (map - NOT USEFUL)

Respond with ONLY the button name, nothing else.

Strategic priorities:
1. MOVEMENT (UP/DOWN/LEFT/RIGHT) - explore new areas
2. A BUTTON near NPCs - talk and interact
3. A BUTTON - collect items and interact with environment
4. Avoid damage - retreat with directional buttons when low health

Focus on MOVEMENT and A button - these create the most learning signal."""

            response = requests.post(
                self.llm_endpoint,
                json={
                    "model": "mlx-community/Qwen2.5-14B-Instruct-4bit",
                    "messages": [
                        {"role": "system", "content": "You are a strategic advisor for an RL agent."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 20
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                button = result['choices'][0]['message']['content'].strip().upper()
                self.llm_call_count += 1
                self.llm_success_count += 1
                return button
            else:
                print(f"⚠️  LLM call failed: {response.status_code}")
                self.llm_call_count += 1
                return None
                
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
            self.llm_call_count += 1
            return None
    
    def compute_exploration_reward(
        self,
        game_state: Dict,
        current_step: int
    ) -> float:
        """Compute exploration rewards with time-based decay for revisited areas.
        
        Args:
            game_state: Current game state with player position
            current_step: Current global step count
            
        Returns:
            Exploration reward (can be negative for loitering)
        """
        player = game_state.get('player', {})
        x = player.get('x', 0)
        y = player.get('y', 0)
        room = player.get('room', 0)
        
        current_pos = (x, y, room)
        
        # Grid cell (configurable pixel regions)
        grid_x = x // self.grid_size
        grid_y = y // self.grid_size
        grid_cell = (room, grid_x, grid_y)
        
        reward = 0.0
        
        # During warmup period: ONLY give bonuses, NO penalties (let agent learn to move first)
        warmup_active = current_step < self.penalty_warmup_steps
        
        # 1. PENALTY: Standing completely still (only after warmup)
        if not warmup_active:
            if self.last_position is not None and self.last_position == current_pos:
                self.stationary_steps += 1
                # Gentler increasing penalty for staying still (max -2.0)
                reward -= min(self.stationary_steps * 0.2, 2.0)
            else:
                self.stationary_steps = 0
        
        # 2. AREA REVISIT with DECAY
        if grid_cell in self.area_visit_times:
            last_visit_time = self.area_visit_times[grid_cell][-1]
            time_since_visit = current_step - last_visit_time
            
            if not warmup_active and time_since_visit < self.decay_window:
                # Recently visited - apply gentle penalty that decreases with time (only after warmup)
                decay_factor = 1.0 - (time_since_visit / self.decay_window)
                loiter_penalty = -0.8 * decay_factor  # Reduced from -2.0
                reward += loiter_penalty
            elif time_since_visit >= self.decay_window:
                # Decayed - can revisit without penalty (backtracking allowed)
                # Small bonus for productive backtracking
                reward += 0.5
            
            self.area_visit_times[grid_cell].append(current_step)
        else:
            # NEW AREA DISCOVERED! (always rewarded, even during warmup)
            reward += 5.0 * self.exploration_bonus_multiplier
            self.area_visit_times[grid_cell] = [current_step]
        
        # Track position
        self.position_history.append((x, y, room, current_step))
        self.last_position = current_pos
        
        return reward
    
    def compute_llm_alignment_bonus(
        self, 
        action: int, 
        llm_suggestion: str, 
        game_state: Dict
    ) -> float:
        """Compute bonus reward for following LLM button suggestion.
        
        Args:
            action: Action taken by RL agent (0-8)
            llm_suggestion: LLM's suggested button string
            game_state: Current game state
            
        Returns:
            Bonus reward (0 if no alignment, positive if aligned)
        """
        if not llm_suggestion:
            return 0.0
        
        llm_suggestion = llm_suggestion.upper()
        
        # Direct button mapping: 0=NOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START, 8=SELECT
        button_map = {
            "NOP": 0,
            "UP": 1,
            "DOWN": 2,
            "LEFT": 3,
            "RIGHT": 4,
            "A": 5,
            "B": 6,
            "START": 7,
            "SELECT": 8
        }
        
        # Check for exact button match
        for button_name, button_id in button_map.items():
            if button_name in llm_suggestion and action == button_id:
                # Higher bonus for context-appropriate actions
                npcs = len(game_state.get('entities', {}).get('npcs', []))
                enemies = len(game_state.get('entities', {}).get('enemies', []))
                
                # NO BONUS for SELECT - agent can't use map with vector observations
                if button_name == "SELECT":
                    return 0.0  # Prevent reward hacking loop
                # NO BONUS for START - agent can't use menu with vector observations  
                elif button_name == "START":
                    return 0.0  # Prevent menu loop
                # Extra bonus for A button near NPCs (dialogue)
                elif button_name == "A" and npcs > 0:
                    return self.llm_guidance_bonus * 3.0
                # Extra bonus for movement away from enemies when low health
                elif button_name in ["UP", "DOWN", "LEFT", "RIGHT"] and enemies > 0:
                    health = game_state.get('player', {}).get('health', 3)
                    if health <= 1:
                        return self.llm_guidance_bonus * 2.5  # Retreat bonus
                    else:
                        return self.llm_guidance_bonus * 1.5  # Movement bonus
                # Standard bonus for exact match (movement, A, B)
                else:
                    return self.llm_guidance_bonus * 2.0
        
        return 0.0
    
    def collect_rollout(self, num_steps: int = 128) -> Dict[str, List]:
        """Collect a rollout of experience with LLM guidance.
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary containing rollout data
        """
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        llm_bonuses = []
        
        obs, info = self.env.reset()
        episode_reward = 0
        
        for step in range(num_steps):
            observations.append(obs.copy())
            
            # Get current game state for LLM
            game_state = info.get('structured_state', {})
            
            # Track room visits with full location names
            current_room = game_state.get('player', {}).get('room', None)
            if current_room is not None and current_room not in self.visited_rooms:
                from observation.ram_maps.room_mappings import OVERWORLD_ROOMS
                self.visited_rooms.add(current_room)
                self.room_discovery_count += 1
                room_name = OVERWORLD_ROOMS.get(current_room, f"Unknown Room {current_room}")
                player_x = game_state.get('player', {}).get('x', 0)
                player_y = game_state.get('player', {}).get('y', 0)
                print(f"\n   🗺️  📍 NEW LOCATION DISCOVERED!")
                print(f"   Location: {room_name}")
                print(f"   Room ID: {current_room}")
                print(f"   Coordinates: ({player_x}, {player_y})")
                print(f"   Total Locations Explored: {len(self.visited_rooms)}\n")
            
            # Call LLM periodically
            if self.global_step > 0 and self.global_step % self.llm_frequency == 0:
                llm_suggestion = self.call_llm(game_state)
                if llm_suggestion:
                    self.last_llm_suggestion = llm_suggestion
                    # Check for NPCs and highlight strategic suggestions
                    npcs = len(game_state.get('entities', {}).get('npcs', []))
                    if npcs > 0 and "A" in llm_suggestion:
                        print(f"   🧠 LLM suggests button: {llm_suggestion} 👥 [{npcs} NPCs nearby - HIGH VALUE!]")
                    elif npcs > 0:
                        print(f"   🧠 LLM suggests button: {llm_suggestion} 👥 [{npcs} NPCs detected]")
                    else:
                        print(f"   🧠 LLM suggests button: {llm_suggestion}")
            
            # Get action from PPO policy network
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.controller.device)
            
            with torch.no_grad():
                action, log_prob, value = self.controller.policy_net.get_action_and_value(obs_tensor)
            
            action_int = action.item()
            actions.append(action_int)
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action_int)
            
            # Get updated game state after step
            next_game_state = info.get('structured_state', {})
            
            # Compute exploration reward with decay (anti-loitering)
            exploration_reward = self.compute_exploration_reward(next_game_state, self.global_step)
            
            # Compute LLM alignment bonus
            llm_bonus = 0.0
            if self.last_llm_suggestion:
                llm_bonus = self.compute_llm_alignment_bonus(action_int, self.last_llm_suggestion, game_state)
                if llm_bonus > 0:
                    action_names = ["NOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
                    action_name = action_names[action_int] if action_int < len(action_names) else f"UNKNOWN({action_int})"
                    
                    # Track NPC interactions
                    npcs = len(game_state.get('entities', {}).get('npcs', []))
                    if action_name == "A" and npcs > 0:
                        self.a_button_near_npc_count += 1
                        self.npc_bonus_rewards += llm_bonus
                        print(f"   ✅ PPO followed LLM! Button: {action_name} | Bonus: +{llm_bonus:.1f} 🎯 [NPC INTERACTION #{self.a_button_near_npc_count}!]")
                    else:
                        print(f"   ✅ PPO followed LLM! Button: {action_name} | Bonus: +{llm_bonus:.1f}")
            
            # Log exploration rewards when significant
            if exploration_reward > 10.0:
                print(f"   🌟 NEW AREA EXPLORED! Bonus: +{exploration_reward:.1f}")
            elif exploration_reward < -1.0:
                print(f"   ⚠️  Loitering penalty: {exploration_reward:.1f}")
            
            llm_bonuses.append(llm_bonus)
            
            # Total reward = environment reward + LLM bonus + exploration reward
            total_reward = reward + llm_bonus + exploration_reward
            rewards.append(total_reward)
            episode_reward += total_reward
            
            done = terminated or truncated
            dones.append(done)
            
            self.global_step += 1
            
            if done:
                print(f"   📊 Episode {self.episode_count}: Reward={episode_reward:.1f}, Steps={step+1}")
                obs, info = self.env.reset()
                self.episode_count += 1
                episode_reward = 0
        
        return {
            'observations': observations,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'dones': dones,
            'llm_bonuses': llm_bonuses
        }
    
    def train(self, total_timesteps: int = 100000, save_dir: str = "checkpoints"):
        """Main training loop.
        
        Args:
            total_timesteps: Total training timesteps
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"🎯 HYBRID RL+LLM TRAINING")
        print(f"=" * 60)
        print(f"🧠 PPO Learning: Active")
        print(f"💡 LLM Guidance: Every {self.llm_frequency} steps")
        print(f"🎁 LLM Bonus Multiplier: {self.llm_guidance_bonus}x")
        print(f"📊 Total Steps: {total_timesteps}")
        print()
        
        start_time = time.time()
        
        while self.global_step < total_timesteps:
            # Collect rollout
            rollout_data = self.collect_rollout(num_steps=128)
            
            # Convert to tensors
            observations = torch.FloatTensor(rollout_data['observations']).to(self.controller.device)
            actions = torch.LongTensor(rollout_data['actions']).to(self.controller.device)
            old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.controller.device)
            
            # Compute advantages and returns
            advantages, returns = self.controller.compute_gae(
                rollout_data['rewards'], 
                rollout_data['values'], 
                rollout_data['dones']
            )
            advantages = torch.FloatTensor(advantages).to(self.controller.device)
            returns = torch.FloatTensor(returns).to(self.controller.device)
            
            # Prepare batch
            batch_data = {
                'obs': observations,
                'actions': actions,
                'log_probs': old_log_probs,
                'advantages': advantages,
                'returns': returns
            }
            
            # Update PPO policy
            metrics = self.controller.update(batch_data, epochs=4)
            
            # Log progress
            if self.global_step % 1000 == 0:
                llm_success_rate = (self.llm_success_count / self.llm_call_count * 100) if self.llm_call_count > 0 else 0
                avg_llm_bonus = np.mean(rollout_data['llm_bonuses'])
                print(f"\n📈 Step {self.global_step}/{total_timesteps}")
                print(f"   Policy Loss: {metrics.get('policy_loss', 0):.4f}")
                print(f"   Value Loss: {metrics.get('value_loss', 0):.4f}")
                print(f"   LLM Success Rate: {llm_success_rate:.1f}%")
                print(f"   Avg LLM Bonus: {avg_llm_bonus:.2f}")
            
            # Save checkpoint
            if self.global_step % 10000 == 0:
                checkpoint_path = f"{save_dir}/hybrid_checkpoint_{self.global_step}.pt"
                self.controller.save_checkpoint(checkpoint_path)
                print(f"   💾 Saved checkpoint: {checkpoint_path}")
        
        # Final save
        final_path = f"{save_dir}/hybrid_final.pt"
        self.controller.save_checkpoint(final_path)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Training complete in {elapsed:.1f}s")
        print(f"📊 Episodes: {self.episode_count}")
        print(f"🧠 LLM Calls: {self.llm_call_count} ({self.llm_success_count} successful)")
        
        print(f"\n👥 NPC INTERACTION TRACKING:")
        print(f"   A Button Near NPCs: {self.a_button_near_npc_count} times")
        print(f"   Total NPC Bonus Rewards: +{self.npc_bonus_rewards:.1f}")
        if self.a_button_near_npc_count > 0:
            print(f"   Average NPC Bonus: {self.npc_bonus_rewards / self.a_button_near_npc_count:.1f}")
        
        print(f"\n🗺️  EXPLORATION SUMMARY:")
        print(f"   Unique Rooms Visited: {len(self.visited_rooms)}")
        print(f"   Unique Grid Areas Explored: {len(self.area_visit_times)}")
        print(f"   Total Position Changes: {len(self.position_history)}")
        
        if self.visited_rooms:
            from observation.ram_maps.room_mappings import OVERWORLD_ROOMS
            print(f"\n   📍 Locations Discovered:")
            for room_id in sorted(self.visited_rooms):
                room_name = OVERWORLD_ROOMS.get(room_id, f"Unknown Room {room_id}")
                print(f"      - {room_name} (ID: {room_id})")
        
        print(f"\n🎯 EXPLORATION MECHANICS:")
        print(f"   ✓ Anti-loitering: Penalties for staying in same area")
        print(f"   ✓ Decay window: {self.decay_window} steps")
        print(f"   ✓ Backtracking allowed after decay period")
        print(f"   ✓ New area bonus: {5.0 * self.exploration_bonus_multiplier:.1f} points")
        
        print(f"\n🧠 LLM CONTEXT SUMMARY:")
        print(f"   The LLM received detailed context for EVERY suggestion:")
        print(f"   ✓ Location name (from room_mappings.py)")
        print(f"   ✓ Room ID")
        print(f"   ✓ Link's (x, y) position")
        print(f"   ✓ Facing direction")
        print(f"   ✓ Health status")
        print(f"   ✓ NPC positions and types")
        print(f"   ✓ Enemy counts")
        print(f"   ✓ Item counts")
        print(f"   This enabled strategic, context-aware guidance!")
        
        self.env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Hybrid RL+LLM training for Zelda")
    parser.add_argument("--rom-path", type=str, 
                       default="roms/zelda_oracle_of_seasons.gbc",
                       help="Path to ROM file")
    parser.add_argument("--headless", action="store_true", default=False,
                       help="Run headless")
    parser.add_argument("--total-timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--llm-endpoint", type=str,
                       default="http://localhost:8000/v1/chat/completions",
                       help="LLM endpoint URL")
    parser.add_argument("--llm-frequency", type=int, default=5,
                       help="Call LLM every N steps")
    parser.add_argument("--llm-bonus", type=float, default=5.0,
                       help="LLM guidance bonus multiplier")
    
    args = parser.parse_args()
    
    trainer = HybridRLLLMTrainer(
        rom_path=args.rom_path,
        headless=args.headless,
        llm_endpoint=args.llm_endpoint,
        llm_frequency=args.llm_frequency,
        llm_guidance_bonus=args.llm_bonus
    )
    
    trainer.train(total_timesteps=args.total_timesteps)


if __name__ == "__main__":
    main()

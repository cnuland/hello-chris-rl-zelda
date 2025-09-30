"""Hybrid RL+LLM Training for Zelda Oracle of Seasons.

True hybrid approach:
- PPO neural network learns from experience
- LLM provides periodic guidance
- Reward shaping when agent follows LLM suggestions
- Exploration mode = pure PPO policy
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
        llm_frequency: int = 30,  # Call LLM every N steps
        llm_guidance_bonus: float = 5.0,  # Bonus reward multiplier for following LLM
        exploration_steps: int = 300,  # Steps of pure PPO per new screen
    ):
        self.rom_path = rom_path
        self.headless = headless
        self.llm_endpoint = llm_endpoint
        self.llm_frequency = llm_frequency
        self.llm_guidance_bonus = llm_guidance_bonus
        self.exploration_steps = exploration_steps
        
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
        self.current_screen_id = None
        self.exploration_mode_remaining = 0
        self.last_llm_suggestion = None
        self.llm_call_count = 0
        self.llm_success_count = 0
        
    def call_llm(self, game_state: Dict) -> Optional[str]:
        """Call LLM for strategic guidance.
        
        Args:
            game_state: Current structured game state
            
        Returns:
            LLM suggested action string, or None if failed
        """
        try:
            # Extract context from game state
            player = game_state.get('player', {})
            health = player.get('health', 3)
            room = player.get('room', 0)
            npcs = len(game_state.get('entities', {}).get('npcs', []))
            
            # Create prompt
            prompt = f"""You are guiding an RL agent playing Zelda: Oracle of Seasons.

Current State:
- Health: {health} hearts
- Room: {room}
- NPCs nearby: {npcs}
- Agent is learning through PPO reinforcement learning

Provide ONE strategic suggestion to guide learning:
- GO_NORTH/GO_SOUTH/GO_EAST/GO_WEST: Navigate to new areas
- TALK_TO_NPC: Interact with NPCs (only if NPCs are nearby!)
- EXPLORE_AREA: Search current area
- AVOID_ENEMIES: Retreat from danger

Respond with ONLY the action name, nothing else."""

            response = requests.post(
                self.llm_endpoint,
                json={
                    "model": "mlx-community/Qwen2.5-14B-Instruct-4bit",
                    "messages": [
                        {"role": "system", "content": "You are a strategic advisor for an RL agent."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 50
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                action = result['choices'][0]['message']['content'].strip()
                self.llm_call_count += 1
                self.llm_success_count += 1
                return action
            else:
                print(f"⚠️  LLM call failed: {response.status_code}")
                self.llm_call_count += 1
                return None
                
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
            self.llm_call_count += 1
            return None
    
    def compute_llm_alignment_bonus(
        self, 
        action: int, 
        llm_suggestion: str, 
        game_state: Dict
    ) -> float:
        """Compute bonus reward if action aligns with LLM suggestion.
        
        Args:
            action: Action taken by RL agent (0-8)
            llm_suggestion: LLM's suggested action string
            game_state: Current game state
            
        Returns:
            Bonus reward (0 if no alignment, positive if aligned)
        """
        if not llm_suggestion:
            return 0.0
        
        llm_suggestion = llm_suggestion.upper()
        
        # Map actions to directions
        # 0=NOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START, 8=SELECT
        
        # Directional alignment
        if "NORTH" in llm_suggestion and action == 1:  # UP
            return self.llm_guidance_bonus * 2.0
        elif "SOUTH" in llm_suggestion and action == 2:  # DOWN
            return self.llm_guidance_bonus * 2.0
        elif "WEST" in llm_suggestion and action == 3:  # LEFT
            return self.llm_guidance_bonus * 2.0
        elif "EAST" in llm_suggestion and action == 4:  # RIGHT
            return self.llm_guidance_bonus * 2.0
        
        # NPC interaction alignment
        elif "TALK" in llm_suggestion and action == 5:  # A button
            npcs = len(game_state.get('entities', {}).get('npcs', []))
            if npcs > 0:
                return self.llm_guidance_bonus * 3.0  # Big bonus for correct NPC interaction
        
        # Exploration alignment (any movement or A button)
        elif "EXPLORE" in llm_suggestion and action in [1, 2, 3, 4, 5]:
            return self.llm_guidance_bonus * 0.5
        
        # Avoid enemies (no movement)
        elif "AVOID" in llm_suggestion and action == 0:  # NOP
            return self.llm_guidance_bonus * 1.0
        
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
            current_room = game_state.get('player', {}).get('room', 0)
            
            # Track screen changes for auto-exploration
            if self.current_screen_id != current_room:
                if self.current_screen_id is not None:
                    print(f"   🗺️  New screen: {current_room} → Entering exploration mode")
                self.current_screen_id = current_room
                self.exploration_mode_remaining = self.exploration_steps
            
            # Decrement exploration counter
            if self.exploration_mode_remaining > 0:
                self.exploration_mode_remaining -= 1
            
            # Call LLM periodically (but not during exploration mode)
            if (self.global_step > 0 and 
                self.global_step % self.llm_frequency == 0 and 
                self.exploration_mode_remaining == 0):
                llm_suggestion = self.call_llm(game_state)
                if llm_suggestion:
                    self.last_llm_suggestion = llm_suggestion
                    print(f"   🧠 LLM suggests: {llm_suggestion}")
            
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
            
            # Compute LLM alignment bonus
            llm_bonus = 0.0
            if self.last_llm_suggestion and self.exploration_mode_remaining == 0:
                llm_bonus = self.compute_llm_alignment_bonus(action_int, self.last_llm_suggestion, game_state)
                if llm_bonus > 0:
                    print(f"   ✨ LLM alignment bonus: +{llm_bonus:.1f}")
            
            llm_bonuses.append(llm_bonus)
            
            # Total reward = environment reward + LLM bonus
            total_reward = reward + llm_bonus
            rewards.append(total_reward)
            episode_reward += total_reward
            
            done = terminated or truncated
            dones.append(done)
            
            self.global_step += 1
            
            # Log exploration mode
            if step % 50 == 0 and self.exploration_mode_remaining > 0:
                print(f"   🔍 Exploration mode: {self.exploration_mode_remaining} steps remaining (pure PPO)")
            
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
        print(f"🔍 Auto-exploration: {self.exploration_steps} steps per new screen")
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
    parser.add_argument("--llm-frequency", type=int, default=30,
                       help="Call LLM every N steps")
    parser.add_argument("--llm-bonus", type=float, default=5.0,
                       help="LLM guidance bonus multiplier")
    parser.add_argument("--exploration-steps", type=int, default=300,
                       help="Steps of pure PPO per new screen")
    
    args = parser.parse_args()
    
    trainer = HybridRLLLMTrainer(
        rom_path=args.rom_path,
        headless=args.headless,
        llm_endpoint=args.llm_endpoint,
        llm_frequency=args.llm_frequency,
        llm_guidance_bonus=args.llm_bonus,
        exploration_steps=args.exploration_steps
    )
    
    trainer.train(total_timesteps=args.total_timesteps)


if __name__ == "__main__":
    main()

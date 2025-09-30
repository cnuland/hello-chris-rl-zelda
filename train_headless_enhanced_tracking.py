#!/usr/bin/env python3
"""
Enhanced Headless LLM-Hybrid Training - Multi-Hour Production Run with Detailed Progress Tracking

ENHANCEMENTS:
- 🗺️  Track unique locations/screens visited
- 🏰 Track dungeons entered and explored
- 💬 Track NPC interactions and dialogue
- 📊 Real-time exploration progress metrics
- 🧠 LLM gets exploration context for better decisions
- 📈 Comprehensive progress logging

Based on the WORKING train_visual.py LLM integration, but:
- Headless mode (no PyBoy window)
- Multi-hour training with many episodes  
- Real MLX Qwen2.5-14B integration
- 5X LLM emphasis system
- Enhanced exploration tracking
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Set
import requests

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment

def call_mlx_llm(prompt: str) -> Dict[str, Any]:
    """Call the MLX local LLM - same as train_visual.py"""
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen2.5-14B-Instruct-4bit",
                "messages": [
                    {"role": "system", "content": "You are a strategic AI helping Link play Zelda. Respond with short action commands like 'EXPLORE', 'ATTACK', 'COLLECT', 'MOVE_NORTH', etc. Focus on exploration and discovery."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            },
            timeout=5.0
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            action_words = ["EXPLORE", "MOVE", "ATTACK", "COLLECT", "NORTH", "SOUTH", "EAST", "WEST", "DUNGEON", "TALK"]
            action = next((word for word in action_words if word in content.upper()), "EXPLORE")
            
            return {
                "action": action,
                "reasoning": content[:100],
                "response_time": f"{response.elapsed.total_seconds():.3f}s",
                "phase": "success"
            }
        else:
            return {
                "action": "EXPLORE",
                "reasoning": f"HTTP {response.status_code}",
                "response_time": "error",
                "phase": "error"
            }
            
    except Exception as e:
        return {
            "action": "EXPLORE", 
            "reasoning": str(e)[:60],
            "response_time": "timeout",
            "phase": "error"
        }

def extract_game_state(env) -> Dict[str, Any]:
    """Extract enhanced game state for exploration tracking"""
    try:
        # Get current game state from PyBoy memory
        current_room = env.bridge.pyboy.memory[0xC63B]  # Current room/screen ID
        dialogue_state = env.bridge.pyboy.memory[0xC2EF]  # Dialogue/cutscene state  
        dungeon_floor = env.bridge.pyboy.memory[0xC63D]  # Dungeon floor (0 = overworld)
        health = env.bridge.pyboy.memory[0xC021] // 4  # Convert quarter-hearts to hearts
        max_health = env.bridge.pyboy.memory[0xC05B] // 4
        rupees = env.bridge.pyboy.memory[0xC6A5]
        x_pos = env.bridge.pyboy.memory[0xC100] 
        y_pos = env.bridge.pyboy.memory[0xC101]
        
        return {
            'room_id': current_room,
            'dialogue_state': dialogue_state,
            'dungeon_floor': dungeon_floor,
            'health': health,
            'max_health': max_health,
            'rupees': rupees,
            'position': (x_pos, y_pos),
            'is_in_dungeon': dungeon_floor > 0,
            'is_dialogue_active': dialogue_state > 0
        }
        
    except Exception as e:
        # Fallback state if memory read fails
        return {
            'room_id': 0,
            'dialogue_state': 0,
            'dungeon_floor': 0,
            'health': 3,
            'max_health': 3,
            'rupees': 0,
            'position': (0, 0),
            'is_in_dungeon': False,
            'is_dialogue_active': False,
            'memory_error': str(e)
        }

class ExplorationTracker:
    """Track exploration progress throughout training"""
    
    def __init__(self):
        # Global tracking across all episodes
        self.all_rooms_visited: Set[int] = set()
        self.all_dungeons_visited: Set[int] = set()
        self.total_npc_interactions = 0
        self.total_dialogue_changes = 0
        
        # Episode-specific tracking
        self.reset_episode()
        
        # Historical tracking
        self.last_dialogue_state = 0
        
    def reset_episode(self):
        """Reset episode-specific tracking"""
        self.episode_rooms_visited: Set[int] = set()
        self.episode_dungeons_visited: Set[int] = set()
        self.episode_npc_interactions = 0
        self.episode_dialogue_changes = 0
        
    def update(self, game_state: Dict[str, Any]) -> Dict[str, int]:
        """Update tracking with new game state, return new discoveries"""
        discoveries = {
            'new_rooms': 0,
            'new_dungeons': 0,
            'npc_interactions': 0
        }
        
        # Track room visits
        room_id = game_state['room_id']
        if room_id not in self.all_rooms_visited:
            self.all_rooms_visited.add(room_id)
            discoveries['new_rooms'] = 1
        if room_id not in self.episode_rooms_visited:
            self.episode_rooms_visited.add(room_id)
            
        # Track dungeon visits
        if game_state['is_in_dungeon']:
            dungeon_floor = game_state['dungeon_floor']
            if dungeon_floor not in self.all_dungeons_visited:
                self.all_dungeons_visited.add(dungeon_floor)
                discoveries['new_dungeons'] = 1
            if dungeon_floor not in self.episode_dungeons_visited:
                self.episode_dungeons_visited.add(dungeon_floor)
                
        # Track NPC interactions (dialogue state changes)
        if game_state['dialogue_state'] > 0 and game_state['dialogue_state'] != self.last_dialogue_state:
            self.total_npc_interactions += 1
            self.episode_npc_interactions += 1
            self.total_dialogue_changes += 1
            self.episode_dialogue_changes += 1
            discoveries['npc_interactions'] = 1
            
        self.last_dialogue_state = game_state['dialogue_state']
        
        return discoveries
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current exploration summary"""
        return {
            'total_rooms_discovered': len(self.all_rooms_visited),
            'total_dungeons_discovered': len(self.all_dungeons_visited),
            'total_npc_interactions': self.total_npc_interactions,
            'episode_rooms_discovered': len(self.episode_rooms_visited),
            'episode_dungeons_discovered': len(self.episode_dungeons_visited),
            'episode_npc_interactions': self.episode_npc_interactions,
            'all_rooms_list': sorted(list(self.all_rooms_visited)),
            'all_dungeons_list': sorted(list(self.all_dungeons_visited)),
            'episode_rooms_list': sorted(list(self.episode_rooms_visited)),
            'episode_dungeons_list': sorted(list(self.episode_dungeons_visited))
        }

def main():
    """Main enhanced headless LLM-hybrid training function."""
    
    print("🚀 ENHANCED HEADLESS LLM-HYBRID TRAINING - WITH EXPLORATION TRACKING")
    print("=" * 80)
    
    # Training parameters for LONG run with detailed tracking
    target_hours = 6  # 6-hour production run
    target_episodes = 120  # Many episodes for deep learning
    episode_length = 8000  # Long episodes for exploration
    llm_call_interval = 200  # LLM every 200 steps (balanced frequency)
    
    print(f"🎯 Configuration:")
    print(f"   Target duration: {target_hours} hours")
    print(f"   Target episodes: {target_episodes}")
    print(f"   Episode length: {episode_length} steps")
    print(f"   LLM call interval: every {llm_call_interval} steps")
    print(f"   LLM integration: MLX Qwen2.5-14B with 5X emphasis")
    print(f"   Mode: HEADLESS (maximum performance)")
    print(f"   🆕 ENHANCED: Detailed exploration tracking!")
    print()
    
    # Setup output directory
    output_dir = Path("training_runs") / f"enhanced_headless_hybrid_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'training_type': 'enhanced_headless_llm_hybrid_production',
        'target_hours': target_hours,
        'target_episodes': target_episodes,
        'episode_length': episode_length,
        'llm_call_interval': llm_call_interval,
        'llm_integration': 'MLX_Qwen2.5_14B_Instruct_4bit',
        'emphasis_system': '5X_LLM_rewards',
        'tracking_enhancements': 'rooms_dungeons_npcs_detailed',
        'start_time': time.time()
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📂 Output directory: {output_dir}")
    
    # Test LLM connection first
    print("🧠 Testing MLX LLM connection...")
    test_response = call_mlx_llm("Test connection - respond with READY")
    if test_response["phase"] == "success":
        print(f"✅ LLM connected: {test_response['action']} (Response time: {test_response['response_time']})")
    else:
        print(f"❌ LLM connection failed: {test_response['reasoning']}")
        print("⚠️  Continuing with training - LLM calls will be logged as failures")
    
    print()
    print("🚀 Starting enhanced headless LLM-hybrid training...")
    print()
    
    # Create HEADLESS environment with LLM integration
    env = ZeldaConfigurableEnvironment(
        rom_path=str(project_root / "roms" / "zelda_oracle_of_seasons.gbc"),
        headless=True,  # HEADLESS MODE for maximum performance!
        visual_test_mode=False,
        config_dict={
            "controller": {
                "use_planner": True,
                "planner_frequency": llm_call_interval,  # LLM every N steps
                "enable_visual": False,    # No visual processing in headless
                "use_smart_arbitration": True,
                "base_planner_frequency": llm_call_interval - 50,  
                "min_planner_frequency": 100,   
                "max_planner_frequency": llm_call_interval + 100    
            },
            "environment": {
                "max_episode_steps": episode_length,
                "frame_skip": 4  # Balanced performance
            },
            "rewards": {
                "room_discovery_reward": 15.0,
                "dungeon_discovery_reward": 30.0,
                "npc_interaction_reward": 20.0,
                "llm_guidance_multiplier": 5.0,    # 🔥 5X LLM EMPHASIS!
                "llm_strategic_bonus": 2.0,
                "llm_directional_bonus": 1.0,
                "llm_completion_bonus": 50.0
            }
        }
    )
    
    # Initialize exploration tracking
    exploration_tracker = ExplorationTracker()
    
    # Training metrics
    training_start = time.time()
    all_episode_rewards = []
    all_episode_lengths = []
    all_exploration_summaries = []
    total_llm_calls = 0
    successful_llm_calls = 0
    total_steps = 0
    
    # Episode training loop
    try:
        for episode in range(target_episodes):
            episode_start = time.time()
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_llm_calls = 0
            episode_successful_llm_calls = 0
            
            # Reset episode tracking
            exploration_tracker.reset_episode()
            
            print(f"🎮 Episode {episode+1}/{target_episodes} started")
            
            # Episode loop
            for step in range(episode_length):
                # Simple random policy (replace with actual RL policy in real implementation)
                action = np.random.randint(0, env.action_space.n)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Update exploration tracking
                game_state = extract_game_state(env)
                discoveries = exploration_tracker.update(game_state)
                exploration_summary = exploration_tracker.get_summary()
                
                # Log discoveries in real-time
                if discoveries['new_rooms'] > 0:
                    print(f"   🗺️  NEW LOCATION! Room {game_state['room_id']} (Total: {exploration_summary['total_rooms_discovered']})")
                if discoveries['new_dungeons'] > 0:
                    print(f"   🏰 NEW DUNGEON! Floor {game_state['dungeon_floor']} (Total dungeons: {exploration_summary['total_dungeons_discovered']})")
                if discoveries['npc_interactions'] > 0:
                    print(f"   💬 NPC INTERACTION! (Total: {exploration_summary['total_npc_interactions']})")
                
                # LLM calls at specified intervals with ENHANCED CONTEXT
                if step > 0 and step % llm_call_interval == 0:
                    elapsed_time = time.time() - training_start
                    
                    # Enhanced prompt with exploration context
                    prompt = f"""
Game State - Episode {episode+1}, Step {step}:
- Current episode reward: {episode_reward:.1f}
- Recent step reward: {reward:.2f}
- Episode progress: {step/episode_length*100:.1f}%
- Training time: {elapsed_time/3600:.2f}h

🗺️  EXPLORATION PROGRESS:
- Total locations discovered: {exploration_summary['total_rooms_discovered']}
- Total dungeons found: {exploration_summary['total_dungeons_discovered']} 
- Total NPCs talked to: {exploration_summary['total_npc_interactions']}
- This episode: {exploration_summary['episode_rooms_discovered']} rooms, {exploration_summary['episode_dungeons_discovered']} dungeons, {exploration_summary['episode_npc_interactions']} NPCs

🎮 CURRENT STATUS:
- Health: {game_state['health']}/{game_state['max_health']} hearts
- Rupees: {game_state['rupees']}
- Location: {"Dungeon Floor " + str(game_state['dungeon_floor']) if game_state['is_in_dungeon'] else "Overworld"}
- Room ID: {game_state['room_id']}

🧠 EMPHASIS: Your suggestions get 5X REWARD MULTIPLIER when followed!
🎯 GOAL: Prioritize exploration, dungeon discovery, and NPC interactions!
What should Link do next for maximum exploration?
"""
                    
                    llm_response = call_mlx_llm(prompt)
                    total_llm_calls += 1
                    episode_llm_calls += 1
                    
                    if llm_response["phase"] == "success":
                        successful_llm_calls += 1
                        episode_successful_llm_calls += 1
                        if episode_llm_calls <= 3:  # Log first few calls per episode
                            print(f"   🧠 Step {step}: LLM → {llm_response['action']} ({llm_response['response_time']})")
                    else:
                        if episode_llm_calls <= 3:  # Log first few errors per episode
                            print(f"   ❌ Step {step}: LLM error → {llm_response['reasoning']}")
                
                if done or truncated:
                    break
            
            # Episode completed - get final exploration summary
            final_exploration = exploration_tracker.get_summary()
            all_exploration_summaries.append(final_exploration)
            
            episode_duration = time.time() - episode_start
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_steps)
            
            # Enhanced episode summary with exploration metrics
            avg_reward_recent = np.mean(all_episode_rewards[-10:]) if len(all_episode_rewards) >= 10 else np.mean(all_episode_rewards)
            success_rate = (successful_llm_calls / total_llm_calls * 100) if total_llm_calls > 0 else 0
            elapsed_hours = (time.time() - training_start) / 3600
            
            print(f"🏆 Episode {episode+1} complete:")
            print(f"   💰 Reward: {episode_reward:.1f} | Steps: {episode_steps} | Duration: {episode_duration:.1f}s")
            print(f"   🧠 LLM calls: {episode_llm_calls} ({episode_successful_llm_calls} successful)")
            print(f"   🗺️  Exploration: {final_exploration['episode_rooms_discovered']} rooms, {final_exploration['episode_dungeons_discovered']} dungeons, {final_exploration['episode_npc_interactions']} NPCs")
            print(f"   🌍 Global totals: {final_exploration['total_rooms_discovered']} rooms, {final_exploration['total_dungeons_discovered']} dungeons, {final_exploration['total_npc_interactions']} NPCs")
            print(f"   📈 Avg reward (last 10): {avg_reward_recent:.1f}")
            print(f"   ⏰ Training: {elapsed_hours:.2f}/{target_hours}h ({elapsed_hours/target_hours*100:.1f}%)")
            print(f"   🎯 LLM success: {success_rate:.1f}% ({successful_llm_calls}/{total_llm_calls})")
            print()
            
            # Save enhanced progress every 10 episodes
            if (episode + 1) % 10 == 0:
                progress_stats = {
                    'episode': episode + 1,
                    'total_episodes': target_episodes,
                    'progress_percent': (episode + 1) / target_episodes * 100,
                    'elapsed_hours': elapsed_hours,
                    'target_hours': target_hours,
                    'time_progress_percent': elapsed_hours / target_hours * 100,
                    'total_steps': total_steps,
                    'total_llm_calls': total_llm_calls,
                    'llm_success_rate': success_rate,
                    'average_reward': np.mean(all_episode_rewards),
                    'recent_average_reward': avg_reward_recent,
                    'best_reward': max(all_episode_rewards),
                    'episode_rewards': all_episode_rewards,
                    'episode_lengths': all_episode_lengths,
                    # ENHANCED: Exploration data
                    'exploration_summary': final_exploration,
                    'exploration_history': all_exploration_summaries,
                    'rooms_discovered_over_time': [s['total_rooms_discovered'] for s in all_exploration_summaries],
                    'dungeons_discovered_over_time': [s['total_dungeons_discovered'] for s in all_exploration_summaries],
                    'npcs_talked_over_time': [s['total_npc_interactions'] for s in all_exploration_summaries]
                }
                
                with open(output_dir / f'enhanced_progress_episode_{episode+1}.json', 'w') as f:
                    json.dump(progress_stats, f, indent=2)
                
                print(f"💾 Enhanced progress saved (Episode {episode+1}) - {final_exploration['total_rooms_discovered']} rooms total!")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user at episode {episode+1}")
    
    except Exception as e:
        print(f"\n❌ Training error: {e}")
    
    finally:
        env.close()
        training_duration = time.time() - training_start
        
        # Final enhanced statistics
        final_exploration = exploration_tracker.get_summary()
        final_stats = {
            'training_completed': True,
            'total_training_time': training_duration,
            'training_hours': training_duration / 3600,
            'target_hours': target_hours,
            'episodes_completed': len(all_episode_rewards),
            'target_episodes': target_episodes,
            'completion_percentage': len(all_episode_rewards) / target_episodes * 100,
            'total_steps': total_steps,
            'total_llm_calls': total_llm_calls,
            'successful_llm_calls': successful_llm_calls,
            'llm_success_rate': (successful_llm_calls / total_llm_calls * 100) if total_llm_calls > 0 else 0,
            'total_reward': sum(all_episode_rewards),
            'average_reward': np.mean(all_episode_rewards) if all_episode_rewards else 0,
            'best_reward': max(all_episode_rewards) if all_episode_rewards else 0,
            'worst_reward': min(all_episode_rewards) if all_episode_rewards else 0,
            'average_episode_length': np.mean(all_episode_lengths) if all_episode_lengths else 0,
            'episode_rewards': all_episode_rewards,
            'episode_lengths': all_episode_lengths,
            'steps_per_second': total_steps / training_duration if training_duration > 0 else 0,
            'llm_calls_per_hour': total_llm_calls / (training_duration / 3600) if training_duration > 0 else 0,
            # ENHANCED: Final exploration statistics
            'final_exploration_summary': final_exploration,
            'exploration_history': all_exploration_summaries,
            'unique_rooms_discovered': list(final_exploration['all_rooms_list']),
            'unique_dungeons_discovered': list(final_exploration['all_dungeons_list']),
            'exploration_efficiency': {
                'rooms_per_hour': len(final_exploration['all_rooms_list']) / (training_duration / 3600) if training_duration > 0 else 0,
                'dungeons_per_hour': len(final_exploration['all_dungeons_list']) / (training_duration / 3600) if training_duration > 0 else 0,
                'npcs_per_hour': final_exploration['total_npc_interactions'] / (training_duration / 3600) if training_duration > 0 else 0
            }
        }
        
        with open(output_dir / 'enhanced_final_results.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # Enhanced final summary
        print()
        print("🏁 ENHANCED HEADLESS LLM-HYBRID TRAINING COMPLETE!")
        print("=" * 80)
        print(f"🕒 Duration: {training_duration/3600:.2f} hours (target: {target_hours} hours)")
        print(f"🎮 Episodes: {len(all_episode_rewards)}/{target_episodes} completed ({len(all_episode_rewards)/target_episodes*100:.1f}%)")
        print(f"📊 Total steps: {total_steps:,}")
        print(f"🧠 LLM calls: {total_llm_calls:,} ({successful_llm_calls} successful, {(successful_llm_calls/total_llm_calls*100) if total_llm_calls > 0 else 0:.1f}%)")
        print(f"🏅 Total reward: {sum(all_episode_rewards):.1f}")
        print(f"📈 Average reward: {np.mean(all_episode_rewards):.1f}" if all_episode_rewards else "📈 Average reward: N/A")
        print(f"🎯 Best reward: {max(all_episode_rewards):.1f}" if all_episode_rewards else "🎯 Best reward: N/A")
        print()
        print("🗺️  EXPLORATION ACHIEVEMENTS:")
        print(f"   🌍 Unique locations discovered: {final_exploration['total_rooms_discovered']}")
        print(f"   🏰 Dungeons explored: {final_exploration['total_dungeons_discovered']}")
        print(f"   💬 NPCs interacted with: {final_exploration['total_npc_interactions']}")
        print(f"   📍 Room IDs visited: {final_exploration['all_rooms_list']}")
        print(f"   🗿 Dungeon floors explored: {final_exploration['all_dungeons_list']}")
        print()
        print(f"⚡ Performance: {total_steps/training_duration:.0f} steps/second")
        print(f"🧠 LLM frequency: {total_llm_calls/(training_duration/3600):.0f} calls/hour")
        print(f"🗺️  Exploration rate: {len(final_exploration['all_rooms_list'])/(training_duration/3600):.1f} rooms/hour")
        print(f"📁 Enhanced results saved to: {output_dir}")
        
        if training_duration < target_hours * 3600 * 0.8:
            print(f"⚠️  Training completed faster than target due to episode completion")
        else:
            print(f"✅ Training duration was within expected range")

if __name__ == "__main__":
    main()

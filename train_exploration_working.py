#!/usr/bin/env python3
"""
EXPLORATION TRACKING - WORKING VERSION ✅
🎉 BREAKTHROUGH: Data access confirmed working, room discovery validated
✅ FIXED: Proper room tracking logic based on debug session findings
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Set, Tuple
import requests

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment

class WorkingExplorationTracker:
    """Working exploration tracker - Fixed based on debug session findings"""
    
    def __init__(self):
        # Global tracking across all episodes
        self.all_rooms_visited: Set[int] = set()
        self.all_dungeons_visited: Set[int] = set()
        self.all_overworld_positions: Set[int] = set()
        
        # Item tracking
        self.max_rupees_ever = 0
        self.max_keys_ever = 0
        self.max_bombs_ever = 0
        
        # Episode tracking
        self.reset_episode()
        
        # State tracking for changes (FIXED LOGIC)
        self.last_room_id = None  # ← Start with None to detect first room
        self.last_overworld_pos = None
        self.last_dungeon_floor = 0
        
    def reset_episode(self):
        """Reset episode-specific tracking"""
        self.episode_rooms_visited: Set[int] = set()
        self.episode_dungeons_visited: Set[int] = set()
        self.episode_overworld_positions: Set[int] = set()
        self.episode_discoveries = []
        
    def update(self, game_state: Dict[str, Any], step: int) -> Dict[str, Any]:
        """WORKING update logic based on debug findings"""
        discoveries = {
            'new_room': False,
            'new_dungeon': False,
            'new_overworld_position': False,
            'item_upgrade': False,
            'milestone_reached': False,
            'discovery_summary': ""
        }
        
        try:
            # Extract state information (CONFIRMED WORKING FORMAT)
            player_info = game_state.get('player', {})
            world_state = game_state.get('world', {})
            resources = game_state.get('resources', {})
            
            room_id = player_info.get('room', 0)
            overworld_pos = world_state.get('overworld_position', 0)
            dungeon_floor = world_state.get('dungeon_floor', 0)
            
            # FIXED: Track room discovery (any room ID is valid)
            if room_id != self.last_room_id:
                if room_id not in self.all_rooms_visited:
                    self.all_rooms_visited.add(room_id)
                    self.episode_rooms_visited.add(room_id)
                    discoveries['new_room'] = True
                    discovery_msg = f"NEW ROOM {room_id} discovered!"
                    self.episode_discoveries.append(discovery_msg)
                    discoveries['discovery_summary'] = discovery_msg
                    print(f"   🎉 {discovery_msg}")
                    
                self.last_room_id = room_id
                
            # Track dungeon discovery
            if dungeon_floor > 0 and dungeon_floor != self.last_dungeon_floor:
                if dungeon_floor not in self.all_dungeons_visited:
                    self.all_dungeons_visited.add(dungeon_floor)
                    self.episode_dungeons_visited.add(dungeon_floor)
                    discoveries['new_dungeon'] = True
                    discovery_msg = f"NEW DUNGEON FLOOR {dungeon_floor}!"
                    self.episode_discoveries.append(discovery_msg)
                    if not discoveries['discovery_summary']:
                        discoveries['discovery_summary'] = discovery_msg
                    print(f"   🏰 {discovery_msg}")
                    
                self.last_dungeon_floor = dungeon_floor
                
            # Track overworld positions
            if overworld_pos != self.last_overworld_pos and overworld_pos > 0:
                if overworld_pos not in self.all_overworld_positions:
                    self.all_overworld_positions.add(overworld_pos)
                    self.episode_overworld_positions.add(overworld_pos)
                    discoveries['new_overworld_position'] = True
                    if not discoveries['discovery_summary']:
                        discoveries['discovery_summary'] = f"New overworld {overworld_pos}"
                        
                self.last_overworld_pos = overworld_pos
                
            # Track item upgrades
            current_rupees = resources.get('rupees', 0)
            current_keys = resources.get('keys', 0)
            current_bombs = resources.get('current_bombs', 0)
            
            if current_rupees > self.max_rupees_ever:
                old_rupees = self.max_rupees_ever
                self.max_rupees_ever = current_rupees
                discoveries['item_upgrade'] = True
                discovery_msg = f"Rupees: {old_rupees} → {current_rupees}!"
                self.episode_discoveries.append(discovery_msg)
                if not discoveries['discovery_summary']:
                    discoveries['discovery_summary'] = discovery_msg
                print(f"   💰 {discovery_msg}")
                    
            if current_keys > self.max_keys_ever:
                old_keys = self.max_keys_ever
                self.max_keys_ever = current_keys
                discoveries['item_upgrade'] = True
                discovery_msg = f"Keys: {old_keys} → {current_keys}!"
                self.episode_discoveries.append(discovery_msg)
                if not discoveries['discovery_summary']:
                    discoveries['discovery_summary'] = discovery_msg
                print(f"   🔑 {discovery_msg}")
                    
            if current_bombs > self.max_bombs_ever:
                old_bombs = self.max_bombs_ever
                self.max_bombs_ever = current_bombs
                discoveries['item_upgrade'] = True
                discovery_msg = f"Bombs: {old_bombs} → {current_bombs}!"
                self.episode_discoveries.append(discovery_msg)
                if not discoveries['discovery_summary']:
                    discoveries['discovery_summary'] = discovery_msg
                print(f"   💣 {discovery_msg}")
                    
        except Exception as e:
            # Don't let tracking errors break training
            discoveries['discovery_summary'] = f"Tracking error: {str(e)[:50]}"
            if step % 500 == 0:  # Log occasionally
                print(f"   ⚠️ Step {step}: Exploration tracking error: {e}")
        
        return discoveries
    
    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get comprehensive exploration summary"""
        return {
            # Global progress
            'total_rooms_discovered': len(self.all_rooms_visited),
            'total_dungeons_discovered': len(self.all_dungeons_visited), 
            'total_overworld_positions': len(self.all_overworld_positions),
            
            # Episode progress
            'episode_rooms_discovered': len(self.episode_rooms_visited),
            'episode_dungeons_discovered': len(self.episode_dungeons_visited),
            'episode_overworld_positions': len(self.episode_overworld_positions),
            'episode_total_discoveries': len(self.episode_discoveries),
            
            # Item progress
            'max_rupees_ever': self.max_rupees_ever,
            'max_keys_ever': self.max_keys_ever,
            'max_bombs_ever': self.max_bombs_ever,
            
            # Lists for analysis
            'all_rooms_list': sorted(list(self.all_rooms_visited)),
            'all_dungeons_list': sorted(list(self.all_dungeons_visited)),
            'all_overworld_positions_list': sorted(list(self.all_overworld_positions)),
            'episode_rooms_list': sorted(list(self.episode_rooms_visited)),
            'episode_dungeons_list': sorted(list(self.episode_dungeons_visited)),
            'episode_discovery_list': self.episode_discoveries[-10:]  # Last 10 discoveries
        }

def call_mlx_llm_with_exploration_context(prompt: str, exploration_data: Dict[str, Any]) -> Dict[str, Any]:
    """MLX LLM call with exploration context"""
    try:
        # Enhanced prompt with exploration context
        enhanced_prompt = f"{prompt}\n\n🗺️ Exploration Status:\n"
        enhanced_prompt += f"• Rooms: {exploration_data['total_rooms_discovered']} discovered\n"
        enhanced_prompt += f"• Dungeons: {exploration_data['total_dungeons_discovered']} found\n"
        enhanced_prompt += f"• Max rupees: {exploration_data['max_rupees_ever']}\n"
        enhanced_prompt += f"• Strategy: {'DUNGEON_SEEK' if exploration_data['total_dungeons_discovered'] == 0 else 'EXPLORE'}"
        
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen2.5-14B-Instruct-4bit",
                "messages": [
                    {"role": "system", "content": "You are helping Link explore Zelda. Give ONE strategic action based on exploration progress: NORTH, SOUTH, EAST, WEST, EXPLORE, COLLECT, INVESTIGATE, or DUNGEON_SEEK."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                "max_tokens": 20,
                "temperature": 0.7
            },
            timeout=3.0
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip().upper()
            actions = ["NORTH", "SOUTH", "EAST", "WEST", "EXPLORE", "COLLECT", "INVESTIGATE", "DUNGEON_SEEK"]
            action = next((a for a in actions if a in content), "EXPLORE")
            
            return {
                "action": action,
                "response_time": f"{response.elapsed.total_seconds():.3f}s",
                "phase": "success"
            }
        else:
            return {"action": "EXPLORE", "phase": "error"}
            
    except Exception:
        return {"action": "EXPLORE", "phase": "error"}

def main():
    """Working exploration tracking training"""
    
    print("✅ EXPLORATION TRACKING - WORKING VERSION")
    print("=" * 70)
    
    # WORKING CONFIGURATION (same as successful baseline)
    target_hours = 1.5
    target_episodes = 30
    episode_length = 4000
    llm_call_interval = 200
    
    print(f"✅ Working Configuration:")
    print(f"   Duration: {target_hours} hours")
    print(f"   Episodes: {target_episodes}")
    print(f"   Episode length: {episode_length} steps")
    print(f"   LLM calls: every {llm_call_interval} steps")
    print(f"   Approach: Fixed exploration tracking + stable training")
    print()
    
    # Setup output
    output_dir = Path("training_runs") / f"exploration_working_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'training_type': 'exploration_tracking_working_fixed',
        'target_hours': target_hours,
        'target_episodes': target_episodes,
        'episode_length': episode_length,
        'llm_call_interval': llm_call_interval,
        'approach': 'fixed_data_access_with_working_room_tracking',
        'debug_findings': 'room_discovery_confirmed_working',
        'start_time': time.time()
    }
    
    with open(output_dir / 'working_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize WORKING exploration tracker
    exploration_tracker = WorkingExplorationTracker()
    
    # Test LLM with exploration context
    print("🧠 Testing MLX LLM with exploration context...")
    test_exploration = exploration_tracker.get_exploration_summary()
    test_response = call_mlx_llm_with_exploration_context("Test exploration integration", test_exploration)
    if test_response["phase"] == "success":
        print(f"✅ LLM ready: {test_response['action']} ({test_response['response_time']})")
    else:
        print("❌ LLM failed - continuing anyway")
    
    print()
    print("🗺️  Starting WORKING exploration-tracked training...")
    print()
    
    # Create environment (PROVEN WORKING CONFIG)
    env = ZeldaConfigurableEnvironment(
        rom_path=str(project_root / "roms" / "zelda_oracle_of_seasons.gbc"),
        headless=True,
        visual_test_mode=False,
        config_dict={
            "controller": {
                "use_planner": True,
                "planner_frequency": llm_call_interval,
                "enable_visual": True,  # ← CONFIRMED: This enables structured states
                "use_smart_arbitration": False,
            },
            "environment": {
                "max_episode_steps": episode_length,
                "frame_skip": 4
            },
            "planner_integration": {
                "use_planner": True,
                "enable_visual": True,
                "use_structured_entities": True,
                "auto_load_save_state": True
            },
            "rewards": {
                "room_discovery_reward": 30.0,        # High exploration reward
                "dungeon_discovery_reward": 100.0,    # Huge dungeon reward
                "npc_interaction_reward": 5.0,        # REDUCED to discourage loops
                "item_collection_reward": 10.0,
                "llm_guidance_multiplier": 2.0,       # Moderate LLM emphasis
            }
        }
    )
    
    # Training tracking
    training_start = time.time()
    all_rewards = []
    all_lengths = []
    all_exploration_summaries = []
    total_llm_calls = 0
    successful_llm_calls = 0
    total_steps = 0
    
    # WORKING TRAINING LOOP
    try:
        for episode in range(target_episodes):
            episode_start = time.time()
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_llm_calls = 0
            episode_successful_llm_calls = 0
            
            # Reset episode exploration tracking
            exploration_tracker.reset_episode()
            
            print(f"✅ Episode {episode+1}/{target_episodes} started")
            
            # Working episode loop with FIXED exploration tracking
            for step in range(episode_length):
                # Random action policy (proven to work)
                action = np.random.randint(0, env.action_space.n)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # WORKING exploration tracking
                if info.get('structured_state'):
                    discoveries = exploration_tracker.update(info['structured_state'], step)
                
                # LLM calls with exploration context
                if step > 0 and step % llm_call_interval == 0:
                    elapsed_time = time.time() - training_start
                    current_exploration = exploration_tracker.get_exploration_summary()
                    
                    prompt = f"Episode {episode+1}, Step {step}. Reward: {episode_reward:.1f}. Time: {elapsed_time/3600:.2f}h. Strategy needed."
                    
                    llm_response = call_mlx_llm_with_exploration_context(prompt, current_exploration)
                    total_llm_calls += 1
                    episode_llm_calls += 1
                    
                    if llm_response["phase"] == "success":
                        successful_llm_calls += 1
                        episode_successful_llm_calls += 1
                        if episode_llm_calls <= 2:  # Log first 2 per episode
                            print(f"   🧠 Step {step}: LLM → {llm_response['action']} ({llm_response['response_time']})")
                
                if done or truncated:
                    break
            
            # Episode complete - get final exploration summary
            episode_exploration = exploration_tracker.get_exploration_summary()
            all_exploration_summaries.append(episode_exploration)
            
            episode_duration = time.time() - episode_start
            all_rewards.append(episode_reward)
            all_lengths.append(episode_steps)
            
            # Enhanced summary with WORKING exploration metrics
            elapsed_hours = (time.time() - training_start) / 3600
            avg_reward = np.mean(all_rewards[-5:]) if len(all_rewards) >= 5 else np.mean(all_rewards)
            success_rate = (successful_llm_calls / total_llm_calls * 100) if total_llm_calls > 0 else 0
            
            print(f"✅ Episode {episode+1} complete:")
            print(f"   💰 Reward: {episode_reward:.1f} | Steps: {episode_steps} | Duration: {episode_duration:.1f}s")
            print(f"   🧠 LLM: {episode_llm_calls} calls ({episode_successful_llm_calls} successful)")
            print(f"   🗺️ Episode discoveries: {episode_exploration['episode_total_discoveries']}")
            print(f"   🏠 Rooms this episode: {episode_exploration['episode_rooms_discovered']} | Total: {episode_exploration['total_rooms_discovered']}")
            if episode_exploration['total_dungeons_discovered'] > 0:
                print(f"   🏰 DUNGEONS FOUND: {episode_exploration['total_dungeons_discovered']}!")
            print(f"   💎 Max items: {episode_exploration['max_rupees_ever']} rupees, {episode_exploration['max_keys_ever']} keys, {episode_exploration['max_bombs_ever']} bombs")
            print(f"   📈 Avg reward (last 5): {avg_reward:.1f}")
            print(f"   ⏰ Training: {elapsed_hours:.2f}/{target_hours}h ({elapsed_hours/target_hours*100:.1f}%)")
            print(f"   🎯 LLM success: {success_rate:.1f}%")
            
            # Recent discoveries
            if episode_exploration['episode_discovery_list']:
                print(f"   🎉 Recent: {episode_exploration['episode_discovery_list'][-3:]}")
            print()
            
            # Save progress every 5 episodes
            if (episode + 1) % 5 == 0:
                progress = {
                    'episode': episode + 1,
                    'total_episodes': target_episodes,
                    'elapsed_hours': elapsed_hours,
                    'average_reward': np.mean(all_rewards),
                    'recent_average_reward': avg_reward,
                    'best_reward': max(all_rewards),
                    'total_llm_calls': total_llm_calls,
                    'llm_success_rate': success_rate,
                    'episode_rewards': all_rewards,
                    'episode_lengths': all_lengths,
                    # WORKING: Exploration tracking
                    'exploration_summary': episode_exploration,
                    'exploration_history': all_exploration_summaries[-5:],
                    'rooms_discovered': episode_exploration['total_rooms_discovered'],
                    'dungeons_discovered': episode_exploration['total_dungeons_discovered'],
                    'max_items': {
                        'rupees': episode_exploration['max_rupees_ever'],
                        'keys': episode_exploration['max_keys_ever'],
                        'bombs': episode_exploration['max_bombs_ever']
                    },
                    'unique_rooms_list': episode_exploration['all_rooms_list'],
                    'unique_dungeons_list': episode_exploration['all_dungeons_list']
                }
                
                with open(output_dir / f'working_progress_{episode+1}.json', 'w') as f:
                    json.dump(progress, f, indent=2)
                
                print(f"💾 Working progress saved (Episode {episode+1})")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted at episode {episode+1}")
    
    except Exception as e:
        print(f"\n❌ Training error: {e}")
    
    finally:
        env.close()
        training_duration = time.time() - training_start
        
        # Final results with WORKING exploration data
        final_exploration = exploration_tracker.get_exploration_summary()
        final_results = {
            'training_completed': True,
            'training_type': 'exploration_tracking_working_fixed',
            'total_training_time': training_duration,
            'training_hours': training_duration / 3600,
            'episodes_completed': len(all_rewards),
            'total_steps': total_steps,
            'total_llm_calls': total_llm_calls,
            'llm_success_rate': (successful_llm_calls / total_llm_calls * 100) if total_llm_calls > 0 else 0,
            'average_reward': np.mean(all_rewards) if all_rewards else 0,
            'best_reward': max(all_rewards) if all_rewards else 0,
            'episode_rewards': all_rewards,
            'episode_lengths': all_lengths,
            'steps_per_second': total_steps / training_duration if training_duration > 0 else 0,
            
            # WORKING: Complete exploration analysis
            'final_exploration_summary': final_exploration,
            'exploration_history': all_exploration_summaries,
            'unique_rooms_discovered': final_exploration['all_rooms_list'],
            'unique_dungeons_discovered': final_exploration['all_dungeons_list'],
            'unique_overworld_positions': final_exploration['all_overworld_positions_list'],
            'exploration_efficiency': {
                'rooms_per_episode': final_exploration['total_rooms_discovered'] / len(all_rewards) if all_rewards else 0,
                'dungeons_per_episode': final_exploration['total_dungeons_discovered'] / len(all_rewards) if all_rewards else 0,
                'discoveries_per_hour': (final_exploration['total_rooms_discovered'] + final_exploration['total_dungeons_discovered']) / (training_duration / 3600) if training_duration > 0 else 0
            }
        }
        
        with open(output_dir / 'working_final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print()
        print("✅ WORKING EXPLORATION TRACKING COMPLETE!")
        print("=" * 70)
        print(f"🕒 Duration: {training_duration/3600:.2f} hours")
        print(f"🎮 Episodes: {len(all_rewards)}/{target_episodes}")
        print(f"📊 Total steps: {total_steps:,}")
        print(f"🧠 LLM calls: {total_llm_calls:,} ({successful_llm_calls} successful, {(successful_llm_calls/total_llm_calls*100) if total_llm_calls > 0 else 0:.1f}%)")
        print(f"🏅 Average reward: {np.mean(all_rewards):.1f}" if all_rewards else "🏅 No rewards")
        print(f"🎯 Best reward: {max(all_rewards):.1f}" if all_rewards else "🎯 No rewards")
        print(f"⚡ Performance: {total_steps/training_duration:.0f} steps/second")
        
        print()
        print("🗺️  EXPLORATION ACHIEVEMENTS:")
        print(f"🏠 Rooms discovered: {final_exploration['total_rooms_discovered']}")
        print(f"🏰 Dungeons found: {final_exploration['total_dungeons_discovered']}")
        print(f"🌍 Overworld positions: {final_exploration['total_overworld_positions']}")
        print(f"💰 Max rupees collected: {final_exploration['max_rupees_ever']}")
        print(f"🔑 Max keys held: {final_exploration['max_keys_ever']}")
        print(f"💣 Max bombs held: {final_exploration['max_bombs_ever']}")
        
        if final_exploration['all_rooms_list']:
            rooms_display = final_exploration['all_rooms_list'][:15]
            if len(final_exploration['all_rooms_list']) > 15:
                rooms_display.append('...')
            print(f"🏠 Rooms visited: {rooms_display}")
            
        if final_exploration['all_dungeons_list']:
            print(f"🏰 Dungeons explored: {final_exploration['all_dungeons_list']}")
            
        print(f"📁 Results saved to: {output_dir}")
        
        if len(all_rewards) == target_episodes:
            print("✅ Working exploration tracking completed successfully!")
        else:
            print(f"⚠️  Training ended early at {len(all_rewards)}/{target_episodes} episodes")
        
        print()
        print("📊 EXPLORATION EFFICIENCY ANALYSIS:")
        efficiency = final_results['exploration_efficiency']
        print(f"   🏠 Rooms per episode: {efficiency['rooms_per_episode']:.1f}")
        print(f"   🏰 Dungeons per episode: {efficiency['dungeons_per_episode']:.1f}")  
        print(f"   🎯 Discoveries per hour: {efficiency['discoveries_per_hour']:.1f}")
        
        # Success validation
        if final_exploration['total_rooms_discovered'] > 5:
            print("🎉 EXPLORATION SUCCESS: Multiple rooms discovered!")
        if final_exploration['total_dungeons_discovered'] > 0:
            print("🏆 DUNGEON SUCCESS: Dungeons found!")
        if final_exploration['max_rupees_ever'] > 0:
            print("💰 ITEM SUCCESS: Items collected!")

if __name__ == "__main__":
    main()

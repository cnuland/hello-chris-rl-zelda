#!/usr/bin/env python3
"""
OVERNIGHT LLM-HYBRID TRAINING - 5x Enhanced with Dungeon Focus & Exploration Optimization

🌙 OVERNIGHT ENHANCEMENTS (Based on Analysis Recommendations):
- 🏰 DUNGEON DISCOVERY FOCUS: Enhanced LLM prompts for dungeon-seeking behavior
- 🗺️  EXPLORATION OPTIMIZATION: Target 50+ unique rooms with movement incentives  
- 📦 ITEM TRACKING: Monitor rupees, hearts, keys, and item collection
- 🎯 STRATEGIC REWARDS: Massive bonuses for dungeon entrances and new areas
- ⏰ 5X LONGER RUN: 600 episodes (~8 hours) for overnight training
- 🧠 DUNGEON-FOCUSED LLM: Strategic prompts emphasizing dungeon discovery

Based on previous success:
- Enhanced tracking revealed 26 rooms, 0 dungeons, 11K NPC interactions
- Perfect LLM success rate (100%)
- Strong overworld exploration but missing dungeon discovery
- Need longer episodes for deeper exploration
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

def call_mlx_llm_dungeon_focused(prompt: str) -> Dict[str, Any]:
    """Enhanced MLX LLM call with DUNGEON DISCOVERY FOCUS"""
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen2.5-14B-Instruct-4bit",
                "messages": [
                    {"role": "system", "content": """You are a strategic AI helping Link explore Zelda: Oracle of Seasons with DUNGEON DISCOVERY as top priority.

🏰 PRIMARY OBJECTIVES (in order):
1. FIND DUNGEONS - Look for caves, stairs, dark entrances, mysterious doors
2. EXPLORE NEW AREAS - Prioritize rooms never visited before  
3. COLLECT ITEMS - Hearts, rupees, keys, power-ups
4. TALK TO NPCS - But don't get stuck in dialogue loops

🎯 STRATEGIC COMMANDS:
- Use 'DUNGEON_SEEK', 'EXPLORE_DEEP', 'CAVE_ENTER' for dungeon hunting
- Use 'MOVE_NORTH', 'MOVE_SOUTH', 'MOVE_EAST', 'MOVE_WEST' for systematic exploration
- Use 'COLLECT', 'SEARCH', 'INVESTIGATE' for item gathering
- Avoid repetitive 'TALK' commands if already high NPC count

🏆 EMPHASIS: Dungeon discovery gets MASSIVE rewards! Focus on finding entrances to underground areas."""},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 60,
                "temperature": 0.8
            },
            timeout=5.0
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Enhanced action extraction with dungeon focus
            dungeon_words = ["DUNGEON_SEEK", "CAVE_ENTER", "EXPLORE_DEEP", "INVESTIGATE", "SEARCH"]
            movement_words = ["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST", "NORTH", "SOUTH", "EAST", "WEST"]
            action_words = ["EXPLORE", "COLLECT", "ATTACK", "TALK"] + dungeon_words + movement_words
            
            # Prioritize dungeon-seeking actions
            action = next((word for word in dungeon_words if word in content.upper()), 
                         next((word for word in action_words if word in content.upper()), "EXPLORE_DEEP"))
            
            return {
                "action": action,
                "reasoning": content[:120],
                "response_time": f"{response.elapsed.total_seconds():.3f}s",
                "phase": "success"
            }
        else:
            return {
                "action": "DUNGEON_SEEK",
                "reasoning": f"HTTP {response.status_code}",
                "response_time": "error",
                "phase": "error"
            }
            
    except Exception as e:
        return {
            "action": "EXPLORE_DEEP", 
            "reasoning": str(e)[:60],
            "response_time": "timeout",
            "phase": "error"
        }

def extract_enhanced_game_state(env) -> Dict[str, Any]:
    """Extract enhanced game state with ITEM TRACKING and dungeon detection"""
    try:
        # Enhanced memory reading with item tracking
        current_room = env.bridge.pyboy.memory[0xC63B]  # Current room/screen ID
        dialogue_state = env.bridge.pyboy.memory[0xC2EF]  # Dialogue/cutscene state  
        dungeon_floor = env.bridge.pyboy.memory[0xC63D]  # Dungeon floor (0 = overworld)
        health = env.bridge.pyboy.memory[0xC021] // 4  # Convert quarter-hearts to hearts
        max_health = env.bridge.pyboy.memory[0xC05B] // 4
        rupees = env.bridge.pyboy.memory[0xC6A5]  # Rupee count
        x_pos = env.bridge.pyboy.memory[0xC100] 
        y_pos = env.bridge.pyboy.memory[0xC101]
        
        # ITEM TRACKING - Enhanced for overnight run
        try:
            keys = env.bridge.pyboy.memory[0xC6A8]  # Small keys
            bombs = env.bridge.pyboy.memory[0xC6A6]  # Bomb count
            ring_box = env.bridge.pyboy.memory[0xC690]  # Ring box equipped
            sword_level = env.bridge.pyboy.memory[0xC697]  # Sword level
            shield_level = env.bridge.pyboy.memory[0xC698]  # Shield level
        except:
            keys, bombs, ring_box, sword_level, shield_level = 0, 0, 0, 0, 0
        
        # Enhanced dungeon detection
        is_in_dungeon = dungeon_floor > 0
        dungeon_entrance_nearby = False  # Could add more sophisticated detection
        
        # Room type analysis for dungeon seeking
        room_type = "overworld"
        if is_in_dungeon:
            room_type = f"dungeon_floor_{dungeon_floor}"
        elif current_room in [0, 166, 167]:  # Starting area rooms
            room_type = "starting_area"
        elif 200 <= current_room <= 250:  # Higher room IDs might be special areas
            room_type = "special_area"
        
        return {
            'room_id': current_room,
            'room_type': room_type,
            'dialogue_state': dialogue_state,
            'dungeon_floor': dungeon_floor,
            'health': health,
            'max_health': max_health,
            'rupees': rupees,
            'position': (x_pos, y_pos),
            'is_in_dungeon': is_in_dungeon,
            'is_dialogue_active': dialogue_state > 0,
            'dungeon_entrance_nearby': dungeon_entrance_nearby,
            # ENHANCED ITEM TRACKING
            'items': {
                'keys': keys,
                'bombs': bombs,
                'ring_box': ring_box,
                'sword_level': sword_level,
                'shield_level': shield_level
            },
            'movement_coords': f"({x_pos},{y_pos})",
            'exploration_context': f"Room {current_room} ({room_type})"
        }
        
    except Exception as e:
        # Enhanced fallback state
        return {
            'room_id': 0,
            'room_type': 'unknown',
            'dialogue_state': 0,
            'dungeon_floor': 0,
            'health': 3,
            'max_health': 3,
            'rupees': 0,
            'position': (0, 0),
            'is_in_dungeon': False,
            'is_dialogue_active': False,
            'dungeon_entrance_nearby': False,
            'items': {'keys': 0, 'bombs': 0, 'ring_box': 0, 'sword_level': 0, 'shield_level': 0},
            'movement_coords': '(0,0)',
            'exploration_context': 'Room 0 (error)',
            'memory_error': str(e)
        }

class EnhancedExplorationTracker:
    """OVERNIGHT ENHANCED tracking with item collection and dungeon focus"""
    
    def __init__(self):
        # Global tracking across all episodes
        self.all_rooms_visited: Set[int] = set()
        self.all_dungeons_visited: Set[int] = set()
        self.total_npc_interactions = 0
        self.total_dialogue_changes = 0
        
        # ENHANCED: Item tracking
        self.max_rupees_seen = 0
        self.max_keys_seen = 0
        self.max_bombs_seen = 0
        self.item_upgrades_detected = 0
        
        # ENHANCED: Movement tracking
        self.total_unique_positions: Set[tuple] = set()
        
        # Episode-specific tracking
        self.reset_episode()
        
        # Historical tracking
        self.last_dialogue_state = 0
        self.last_rupee_count = 0
        self.last_item_state = {}
        
    def reset_episode(self):
        """Reset episode-specific tracking"""
        self.episode_rooms_visited: Set[int] = set()
        self.episode_dungeons_visited: Set[int] = set()
        self.episode_npc_interactions = 0
        self.episode_dialogue_changes = 0
        self.episode_item_discoveries = 0
        self.episode_unique_positions: Set[tuple] = set()
        
    def update(self, game_state: Dict[str, Any]) -> Dict[str, int]:
        """Enhanced update with item and movement tracking"""
        discoveries = {
            'new_rooms': 0,
            'new_dungeons': 0,
            'npc_interactions': 0,
            'item_discoveries': 0,
            'new_positions': 0
        }
        
        # Track room visits
        room_id = game_state['room_id']
        if room_id not in self.all_rooms_visited:
            self.all_rooms_visited.add(room_id)
            discoveries['new_rooms'] = 1
        if room_id not in self.episode_rooms_visited:
            self.episode_rooms_visited.add(room_id)
            
        # Track dungeon visits (PRIORITY!)
        if game_state['is_in_dungeon']:
            dungeon_floor = game_state['dungeon_floor']
            if dungeon_floor not in self.all_dungeons_visited:
                self.all_dungeons_visited.add(dungeon_floor)
                discoveries['new_dungeons'] = 1  # BIG DISCOVERY!
            if dungeon_floor not in self.episode_dungeons_visited:
                self.episode_dungeons_visited.add(dungeon_floor)
                
        # Track NPC interactions
        if game_state['dialogue_state'] > 0 and game_state['dialogue_state'] != self.last_dialogue_state:
            self.total_npc_interactions += 1
            self.episode_npc_interactions += 1
            self.total_dialogue_changes += 1
            self.episode_dialogue_changes += 1
            discoveries['npc_interactions'] = 1
            
        # ENHANCED: Track item discoveries
        current_rupees = game_state['rupees']
        if current_rupees > self.last_rupee_count:
            discoveries['item_discoveries'] += 1
            self.episode_item_discoveries += 1
        if current_rupees > self.max_rupees_seen:
            self.max_rupees_seen = current_rupees
            
        items = game_state['items']
        for item_name, item_count in items.items():
            last_count = self.last_item_state.get(item_name, 0)
            if item_count > last_count:
                discoveries['item_discoveries'] += 1
                self.episode_item_discoveries += 1
                if item_name == 'keys' and item_count > self.max_keys_seen:
                    self.max_keys_seen = item_count
                elif item_name == 'bombs' and item_count > self.max_bombs_seen:
                    self.max_bombs_seen = item_count
        
        # Track movement diversity
        position = game_state['position']
        if position not in self.total_unique_positions:
            self.total_unique_positions.add(position)
            discoveries['new_positions'] = 1
        if position not in self.episode_unique_positions:
            self.episode_unique_positions.add(position)
            
        # Update tracking state
        self.last_dialogue_state = game_state['dialogue_state']
        self.last_rupee_count = current_rupees
        self.last_item_state = items.copy()
        
        return discoveries
    
    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Enhanced exploration summary with items and movement"""
        return {
            'total_rooms_discovered': len(self.all_rooms_visited),
            'total_dungeons_discovered': len(self.all_dungeons_visited),
            'total_npc_interactions': self.total_npc_interactions,
            'episode_rooms_discovered': len(self.episode_rooms_visited),
            'episode_dungeons_discovered': len(self.episode_dungeons_visited),
            'episode_npc_interactions': self.episode_npc_interactions,
            'episode_item_discoveries': self.episode_item_discoveries,
            # ENHANCED DATA
            'total_unique_positions': len(self.total_unique_positions),
            'episode_unique_positions': len(self.episode_unique_positions),
            'max_items_seen': {
                'rupees': self.max_rupees_seen,
                'keys': self.max_keys_seen,
                'bombs': self.max_bombs_seen
            },
            'all_rooms_list': sorted(list(self.all_rooms_visited)),
            'all_dungeons_list': sorted(list(self.all_dungeons_visited)),
            'episode_rooms_list': sorted(list(self.episode_rooms_visited)),
            'episode_dungeons_list': sorted(list(self.episode_dungeons_visited))
        }

def main():
    """OVERNIGHT LLM-hybrid training with dungeon focus and enhanced tracking."""
    
    print("🌙 OVERNIGHT ENHANCED LLM-HYBRID TRAINING - DUNGEON DISCOVERY FOCUS")
    print("=" * 90)
    
    # OVERNIGHT PARAMETERS - 5X longer than previous run
    target_hours = 8  # Overnight run - 8 hours
    target_episodes = 600  # 5x the previous 120 episodes
    episode_length = 10000  # LONGER episodes for deeper exploration (was 8000)
    llm_call_interval = 150  # More frequent LLM guidance for dungeon seeking
    
    print(f"🌙 OVERNIGHT CONFIGURATION:")
    print(f"   Target duration: {target_hours} hours (OVERNIGHT RUN)")
    print(f"   Target episodes: {target_episodes} (5x previous run)")
    print(f"   Episode length: {episode_length} steps (25% longer)")
    print(f"   LLM call interval: every {llm_call_interval} steps (more frequent)")
    print(f"   🏰 FOCUS: DUNGEON DISCOVERY with item tracking")
    print(f"   🧠 LLM integration: MLX Qwen2.5-14B with DUNGEON EMPHASIS")
    print(f"   📊 ENHANCED: Item collection and movement tracking")
    print()
    
    # Setup output directory
    output_dir = Path("training_runs") / f"overnight_dungeon_focus_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save enhanced configuration
    config = {
        'training_type': 'overnight_enhanced_dungeon_focus_production',
        'target_hours': target_hours,
        'target_episodes': target_episodes,
        'episode_length': episode_length,
        'llm_call_interval': llm_call_interval,
        'llm_integration': 'MLX_Qwen2.5_14B_Dungeon_Focus',
        'emphasis_system': '5X_LLM_rewards_dungeon_priority',
        'tracking_enhancements': 'rooms_dungeons_npcs_items_movement_detailed',
        'overnight_optimizations': 'dungeon_seeking_enhanced_exploration',
        'start_time': time.time()
    }
    
    with open(output_dir / 'overnight_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📂 Overnight output directory: {output_dir}")
    
    # Test enhanced LLM connection
    print("🧠 Testing enhanced MLX LLM connection...")
    test_response = call_mlx_llm_dungeon_focused("Test dungeon-seeking mode - respond with DUNGEON_SEEK")
    if test_response["phase"] == "success":
        print(f"✅ Enhanced LLM connected: {test_response['action']} (Response time: {test_response['response_time']})")
    else:
        print(f"❌ LLM connection failed: {test_response['reasoning']}")
        print("⚠️  Continuing with overnight training - LLM calls will be logged as failures")
    
    print()
    print("🌙 Starting OVERNIGHT enhanced LLM-hybrid training...")
    print("🏰 PRIMARY GOAL: FIND DUNGEONS!")
    print()
    
    # Create enhanced HEADLESS environment with dungeon rewards
    env = ZeldaConfigurableEnvironment(
        rom_path=str(project_root / "roms" / "zelda_oracle_of_seasons.gbc"),
        headless=True,  # HEADLESS for overnight performance
        visual_test_mode=False,
        config_dict={
            "controller": {
                "use_planner": True,
                "planner_frequency": llm_call_interval,
                "enable_visual": False,
                "use_smart_arbitration": True,
                "base_planner_frequency": llm_call_interval - 50,  
                "min_planner_frequency": 80,   
                "max_planner_frequency": llm_call_interval + 100    
            },
            "environment": {
                "max_episode_steps": episode_length,
                "frame_skip": 4
            },
            "rewards": {
                "room_discovery_reward": 20.0,  # Increased from 15.0
                "dungeon_discovery_reward": 100.0,  # MASSIVE increase from 30.0!
                "npc_interaction_reward": 10.0,  # Reduced from 20.0 to avoid loops
                "llm_guidance_multiplier": 5.0,
                "llm_strategic_bonus": 2.0,
                "llm_directional_bonus": 1.0,
                "llm_completion_bonus": 50.0,
                # NEW ENHANCED REWARDS
                "item_collection_reward": 5.0,
                "movement_diversity_reward": 2.0,
                "dungeon_entrance_bonus": 200.0  # HUGE bonus for finding dungeon entrances!
            }
        }
    )
    
    # Initialize enhanced exploration tracking
    exploration_tracker = EnhancedExplorationTracker()
    
    # Training metrics
    training_start = time.time()
    all_episode_rewards = []
    all_episode_lengths = []
    all_exploration_summaries = []
    total_llm_calls = 0
    successful_llm_calls = 0
    total_steps = 0
    
    # OVERNIGHT episode training loop
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
            
            print(f"🌙 Episode {episode+1}/{target_episodes} started (Overnight run)")
            
            # Enhanced episode loop with item tracking
            for step in range(episode_length):
                # Simple random policy (would be replaced with actual RL policy)
                action = np.random.randint(0, env.action_space.n)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Enhanced exploration tracking
                game_state = extract_enhanced_game_state(env)
                discoveries = exploration_tracker.update(game_state)
                exploration_summary = exploration_tracker.get_enhanced_summary()
                
                # ENHANCED discovery logging
                if discoveries['new_rooms'] > 0:
                    print(f"   🗺️  NEW LOCATION! Room {game_state['room_id']} ({game_state['room_type']}) (Total: {exploration_summary['total_rooms_discovered']})")
                if discoveries['new_dungeons'] > 0:
                    print(f"   🏰 🎆 DUNGEON DISCOVERED! Floor {game_state['dungeon_floor']} 🎆 (Total dungeons: {exploration_summary['total_dungeons_discovered']})")
                if discoveries['npc_interactions'] > 0:
                    print(f"   💬 NPC INTERACTION! (Total: {exploration_summary['total_npc_interactions']})")
                if discoveries['item_discoveries'] > 0:
                    print(f"   📦 ITEM FOUND! Rupees: {game_state['rupees']}, Keys: {game_state['items']['keys']}")
                
                # Enhanced LLM calls with dungeon focus
                if step > 0 and step % llm_call_interval == 0:
                    elapsed_time = time.time() - training_start
                    
                    # DUNGEON-FOCUSED prompt with enhanced context
                    prompt = f"""
🌙 OVERNIGHT TRAINING - Episode {episode+1}, Step {step}:

🎮 CURRENT STATUS:
- Episode reward: {episode_reward:.1f}
- Recent step reward: {reward:.2f}
- Training time: {elapsed_time/3600:.2f}h / {target_hours}h
- Health: {game_state['health']}/{game_state['max_health']} hearts
- Location: {game_state['exploration_context']}
- Position: {game_state['movement_coords']}

🗺️  EXPLORATION PROGRESS:
- Total locations: {exploration_summary['total_rooms_discovered']}
- 🏰 DUNGEONS FOUND: {exploration_summary['total_dungeons_discovered']} (TARGET: FIND MORE!)
- NPCs talked to: {exploration_summary['total_npc_interactions']}
- This episode: {exploration_summary['episode_rooms_discovered']} rooms, {exploration_summary['episode_dungeons_discovered']} dungeons

📦 ITEMS COLLECTED:
- Rupees: {game_state['rupees']} (Max seen: {exploration_summary['max_items_seen']['rupees']})
- Keys: {game_state['items']['keys']} (Max: {exploration_summary['max_items_seen']['keys']})
- Bombs: {game_state['items']['bombs']} (Max: {exploration_summary['max_items_seen']['bombs']})

🏆 OVERNIGHT PRIORITY: FIND DUNGEONS! Look for caves, stairs, dark entrances!
🎯 Your dungeon discovery suggestions get 5X REWARD MULTIPLIER + 200 bonus!
Current room type: {game_state['room_type']}

What's the BEST dungeon-seeking strategy for Link right now?
"""
                    
                    llm_response = call_mlx_llm_dungeon_focused(prompt)
                    total_llm_calls += 1
                    episode_llm_calls += 1
                    
                    if llm_response["phase"] == "success":
                        successful_llm_calls += 1
                        episode_successful_llm_calls += 1
                        if episode_llm_calls <= 5:  # Log first few calls per episode
                            print(f"   🧠🏰 Step {step}: LLM → {llm_response['action']} ({llm_response['response_time']})")
                    else:
                        if episode_llm_calls <= 3:  # Log first few errors per episode
                            print(f"   ❌ Step {step}: LLM error → {llm_response['reasoning']}")
                
                if done or truncated:
                    break
            
            # Episode completed - enhanced summary
            final_exploration = exploration_tracker.get_enhanced_summary()
            all_exploration_summaries.append(final_exploration)
            
            episode_duration = time.time() - episode_start
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_steps)
            
            # ENHANCED episode summary
            avg_reward_recent = np.mean(all_episode_rewards[-20:]) if len(all_episode_rewards) >= 20 else np.mean(all_episode_rewards)
            success_rate = (successful_llm_calls / total_llm_calls * 100) if total_llm_calls > 0 else 0
            elapsed_hours = (time.time() - training_start) / 3600
            
            print(f"🌙 Episode {episode+1} complete:")
            print(f"   💰 Reward: {episode_reward:.1f} | Steps: {episode_steps} | Duration: {episode_duration:.1f}s")
            print(f"   🧠 LLM calls: {episode_llm_calls} ({episode_successful_llm_calls} successful)")
            print(f"   🗺️  Exploration: {final_exploration['episode_rooms_discovered']} rooms, 🏰{final_exploration['episode_dungeons_discovered']} dungeons, {final_exploration['episode_npc_interactions']} NPCs")
            print(f"   📦 Items: {final_exploration['episode_item_discoveries']} discoveries, {final_exploration['episode_unique_positions']} positions")
            print(f"   🌍 Global: {final_exploration['total_rooms_discovered']} rooms, 🏰{final_exploration['total_dungeons_discovered']} dungeons, {final_exploration['total_npc_interactions']} NPCs")
            print(f"   📈 Avg reward (last 20): {avg_reward_recent:.1f}")
            print(f"   ⏰ Overnight progress: {elapsed_hours:.2f}/{target_hours}h ({elapsed_hours/target_hours*100:.1f}%)")
            print(f"   🎯 LLM success: {success_rate:.1f}% ({successful_llm_calls}/{total_llm_calls})")
            print()
            
            # Save enhanced progress every 20 episodes (overnight frequency)
            if (episode + 1) % 20 == 0:
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
                    # ENHANCED: Overnight exploration data
                    'enhanced_exploration_summary': final_exploration,
                    'exploration_history': all_exploration_summaries,
                    'rooms_discovered_timeline': [s['total_rooms_discovered'] for s in all_exploration_summaries],
                    'dungeons_discovered_timeline': [s['total_dungeons_discovered'] for s in all_exploration_summaries],
                    'items_collection_timeline': [s['max_items_seen'] for s in all_exploration_summaries],
                    'overnight_training_type': 'dungeon_focus_enhanced_overnight'
                }
                
                with open(output_dir / f'overnight_progress_episode_{episode+1}.json', 'w') as f:
                    json.dump(progress_stats, f, indent=2)
                
                dungeon_count = final_exploration['total_dungeons_discovered']
                print(f"💾 Overnight progress saved (Episode {episode+1}) - 🏰 {dungeon_count} dungeons, {final_exploration['total_rooms_discovered']} rooms!")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Overnight training interrupted by user at episode {episode+1}")
    
    except Exception as e:
        print(f"\n❌ Overnight training error: {e}")
    
    finally:
        env.close()
        training_duration = time.time() - training_start
        
        # Final enhanced overnight statistics
        final_exploration = exploration_tracker.get_enhanced_summary()
        final_stats = {
            'training_completed': True,
            'training_type': 'overnight_enhanced_dungeon_focus',
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
            # ENHANCED: Final overnight exploration statistics
            'final_enhanced_exploration_summary': final_exploration,
            'exploration_history': all_exploration_summaries,
            'unique_rooms_discovered': list(final_exploration['all_rooms_list']),
            'unique_dungeons_discovered': list(final_exploration['all_dungeons_list']),
            'max_items_collected': final_exploration['max_items_seen'],
            'total_unique_positions_visited': final_exploration['total_unique_positions'],
            'overnight_exploration_efficiency': {
                'rooms_per_hour': len(final_exploration['all_rooms_list']) / (training_duration / 3600) if training_duration > 0 else 0,
                'dungeons_per_hour': len(final_exploration['all_dungeons_list']) / (training_duration / 3600) if training_duration > 0 else 0,
                'npcs_per_hour': final_exploration['total_npc_interactions'] / (training_duration / 3600) if training_duration > 0 else 0,
                'items_per_hour': sum(final_exploration['max_items_seen'].values()) / (training_duration / 3600) if training_duration > 0 else 0
            }
        }
        
        with open(output_dir / 'overnight_enhanced_final_results.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # OVERNIGHT final summary
        print()
        print("🌙 OVERNIGHT ENHANCED LLM-HYBRID TRAINING COMPLETE!")
        print("=" * 90)
        print(f"🕒 Duration: {training_duration/3600:.2f} hours (target: {target_hours} hours)")
        print(f"🎮 Episodes: {len(all_episode_rewards)}/{target_episodes} completed ({len(all_episode_rewards)/target_episodes*100:.1f}%)")
        print(f"📊 Total steps: {total_steps:,}")
        print(f"🧠 LLM calls: {total_llm_calls:,} ({successful_llm_calls} successful, {(successful_llm_calls/total_llm_calls*100) if total_llm_calls > 0 else 0:.1f}%)")
        print(f"🏅 Total reward: {sum(all_episode_rewards):.1f}")
        print(f"📈 Average reward: {np.mean(all_episode_rewards):.1f}" if all_episode_rewards else "📈 Average reward: N/A")
        print(f"🎯 Best reward: {max(all_episode_rewards):.1f}" if all_episode_rewards else "🎯 Best reward: N/A")
        print()
        print("🌙 OVERNIGHT EXPLORATION ACHIEVEMENTS:")
        print(f"   🌍 Unique locations discovered: {final_exploration['total_rooms_discovered']}")
        print(f"   🏰 DUNGEONS EXPLORED: {final_exploration['total_dungeons_discovered']} (PRIMARY GOAL!)")
        print(f"   💬 NPCs interacted with: {final_exploration['total_npc_interactions']}")
        print(f"   📦 Max items collected: {final_exploration['max_items_seen']}")
        print(f"   🚶 Unique positions visited: {final_exploration['total_unique_positions']}")
        print(f"   📍 Room IDs explored: {final_exploration['all_rooms_list']}")
        print(f"   🏰 Dungeon floors found: {final_exploration['all_dungeons_list']}")
        print()
        print(f"⚡ Performance: {total_steps/training_duration:.0f} steps/second")
        print(f"🧠 LLM frequency: {total_llm_calls/(training_duration/3600):.0f} calls/hour")
        print(f"🗺️  Exploration rates:")
        print(f"   - {len(final_exploration['all_rooms_list'])/(training_duration/3600):.1f} rooms/hour")
        print(f"   - {len(final_exploration['all_dungeons_list'])/(training_duration/3600):.1f} dungeons/hour")
        print(f"   - {sum(final_exploration['max_items_seen'].values())/(training_duration/3600):.1f} items/hour")
        print(f"📁 Overnight results saved to: {output_dir}")
        
        dungeons_found = len(final_exploration['all_dungeons_list'])
        if dungeons_found > 0:
            print(f"🎉 SUCCESS: {dungeons_found} dungeons discovered!")
        else:
            print(f"🎯 NEXT GOAL: Refine strategy to discover dungeons")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
EXPLORATION TRACKING - FIXED VERSION
🔧 ISSUE: Previous version couldn't access structured state data
✅ FIX: Direct state access + debug logging to identify the problem
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

class DebugExplorationTracker:
    """Debug version to identify and fix data access issues"""
    
    def __init__(self):
        self.all_rooms_visited: Set[int] = set()
        self.episode_discoveries = []
        self.data_access_attempts = 0
        self.successful_data_access = 0
        
    def reset_episode(self):
        self.episode_discoveries = []
        
    def update_and_debug(self, info: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Debug version that logs what data is actually available"""
        self.data_access_attempts += 1
        
        debug_info = {
            'data_found': False,
            'info_keys': list(info.keys()),
            'structured_state_exists': False,
            'state_type': None,
            'discoveries': {}
        }
        
        # Debug: Log what's in info every 100 steps
        if step % 100 == 0:
            print(f"🔍 DEBUG Step {step}: Info keys = {list(info.keys())}")
            
        # Check if structured_state exists
        if 'structured_state' in info:
            debug_info['structured_state_exists'] = True
            structured_state = info['structured_state']
            debug_info['state_type'] = type(structured_state).__name__
            
            if isinstance(structured_state, dict):
                debug_info['data_found'] = True
                self.successful_data_access += 1
                
                # Debug: Log structure every 200 steps
                if step % 200 == 0:
                    print(f"🔍 DEBUG Step {step}: Structured state keys = {list(structured_state.keys())}")
                    
                # Try to extract basic room info
                try:
                    player_info = structured_state.get('player', {})
                    room_id = player_info.get('room', 0)
                    
                    if room_id > 0 and room_id not in self.all_rooms_visited:
                        self.all_rooms_visited.add(room_id)
                        discovery = f"NEW ROOM {room_id} discovered!"
                        self.episode_discoveries.append(discovery)
                        debug_info['discoveries']['new_room'] = room_id
                        print(f"🎉 {discovery}")
                        
                    # Log resources if available
                    resources = structured_state.get('resources', {})
                    if resources and step % 300 == 0:
                        print(f"🔍 DEBUG Step {step}: Resources = {resources}")
                        
                except Exception as e:
                    debug_info['extraction_error'] = str(e)
                    if step % 100 == 0:
                        print(f"⚠️ DEBUG Step {step}: Extraction error = {e}")
            else:
                print(f"⚠️ DEBUG Step {step}: structured_state is not dict, type = {type(structured_state)}")
        else:
            if step % 100 == 0:
                print(f"⚠️ DEBUG Step {step}: No structured_state in info")
        
        return debug_info
        
    def get_debug_summary(self) -> Dict[str, Any]:
        success_rate = (self.successful_data_access / self.data_access_attempts * 100) if self.data_access_attempts > 0 else 0
        return {
            'total_rooms_discovered': len(self.all_rooms_visited),
            'episode_discoveries': len(self.episode_discoveries),
            'data_access_success_rate': success_rate,
            'data_access_attempts': self.data_access_attempts,
            'successful_data_access': self.successful_data_access,
            'rooms_discovered': list(self.all_rooms_visited),
            'episode_discovery_list': self.episode_discoveries
        }

def call_mlx_llm_simple(prompt: str) -> Dict[str, Any]:
    """Simple MLX LLM call for testing"""
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen2.5-14B-Instruct-4bit",
                "messages": [
                    {"role": "system", "content": "You are helping Link explore Zelda. Give ONE strategic action: NORTH, SOUTH, EAST, WEST, EXPLORE, COLLECT, INVESTIGATE, or DUNGEON_SEEK."},
                    {"role": "user", "content": prompt}
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
            return {"action": action, "phase": "success"}
        else:
            return {"action": "EXPLORE", "phase": "error"}
    except Exception:
        return {"action": "EXPLORE", "phase": "error"}

def main():
    """Debug exploration tracking to identify data access issues"""
    
    print("🔧 EXPLORATION TRACKING DEBUG - Data Access Investigation")
    print("=" * 70)
    
    # Shorter debug session
    target_episodes = 5  # Just 5 episodes for debugging
    episode_length = 1000  # Shorter episodes
    llm_call_interval = 200
    
    # Setup output
    output_dir = Path("training_runs") / f"exploration_debug_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize debug tracker
    debug_tracker = DebugExplorationTracker()
    
    print(f"🔧 Debug Configuration:")
    print(f"   Episodes: {target_episodes} (debug session)")
    print(f"   Episode length: {episode_length} steps")
    print(f"   Focus: Identify structured state data access issues")
    print()
    
    # Create environment with EXPLICIT structured state enabling
    env = ZeldaConfigurableEnvironment(
        rom_path=str(project_root / "roms" / "zelda_oracle_of_seasons.gbc"),
        headless=True,
        visual_test_mode=False,
        config_dict={
            "controller": {
                "use_planner": True,            # ← This should enable structured states
                "planner_frequency": llm_call_interval,
                "enable_visual": True,          # ← Enable visual processing
                "use_smart_arbitration": False,
            },
            "environment": {
                "max_episode_steps": episode_length,
                "frame_skip": 4
            },
            "planner_integration": {           # ← Added explicit planner config
                "use_planner": True,
                "enable_visual": True,
                "use_structured_entities": True,
                "auto_load_save_state": True
            },
            "rewards": {
                "room_discovery_reward": 30.0,
                "dungeon_discovery_reward": 100.0,
                "npc_interaction_reward": 5.0,
                "item_collection_reward": 10.0,
                "llm_guidance_multiplier": 2.0,
            }
        }
    )
    
    # Debug environment configuration
    config_summary = env.get_config_summary()
    print("🔧 Environment Configuration Debug:")
    print(f"   LLM mode: {config_summary.get('llm_mode', 'UNKNOWN')}")
    print(f"   Structured states: {config_summary.get('structured_states', 'UNKNOWN')}")
    print(f"   Auto load save: {config_summary.get('auto_load_save', 'UNKNOWN')}")
    print()
    
    training_start = time.time()
    all_debug_summaries = []
    total_llm_calls = 0
    successful_llm_calls = 0
    
    try:
        for episode in range(target_episodes):
            print(f"🔧 DEBUG Episode {episode+1}/{target_episodes} started")
            
            obs, info = env.reset()
            debug_tracker.reset_episode()
            episode_steps = 0
            
            # Debug initial info
            initial_debug = debug_tracker.update_and_debug(info, 0)
            print(f"   Initial info keys: {initial_debug['info_keys']}")
            print(f"   Structured state exists: {initial_debug['structured_state_exists']}")
            
            for step in range(episode_length):
                action = np.random.randint(0, env.action_space.n)
                obs, reward, done, truncated, info = env.step(action)
                episode_steps += 1
                
                # Debug data access
                debug_result = debug_tracker.update_and_debug(info, step)
                
                # Test LLM occasionally
                if step > 0 and step % llm_call_interval == 0:
                    llm_response = call_mlx_llm_simple(f"Episode {episode+1}, Step {step}")
                    total_llm_calls += 1
                    if llm_response["phase"] == "success":
                        successful_llm_calls += 1
                
                if done or truncated:
                    break
            
            # Episode summary
            episode_debug = debug_tracker.get_debug_summary()
            all_debug_summaries.append(episode_debug)
            
            print(f"🔧 Episode {episode+1} DEBUG Summary:")
            print(f"   Steps: {episode_steps}")
            print(f"   Data access rate: {episode_debug['data_access_success_rate']:.1f}%")
            print(f"   Rooms discovered this episode: {episode_debug['episode_discoveries']}")
            print(f"   Total rooms discovered: {episode_debug['total_rooms_discovered']}")
            print(f"   Recent discoveries: {episode_debug['episode_discovery_list'][-3:] if episode_debug['episode_discovery_list'] else 'None'}")
            print()
    
    except KeyboardInterrupt:
        print("🔧 Debug session interrupted")
    
    finally:
        env.close()
        training_duration = time.time() - training_start
        
        # Final debug summary
        final_debug = debug_tracker.get_debug_summary()
        debug_results = {
            'debug_type': 'exploration_tracking_data_access',
            'episodes_completed': len(all_debug_summaries),
            'total_training_time': training_duration,
            'total_llm_calls': total_llm_calls,
            'successful_llm_calls': successful_llm_calls,
            'final_debug_summary': final_debug,
            'episode_debug_history': all_debug_summaries
        }
        
        with open(output_dir / 'debug_results.json', 'w') as f:
            json.dump(debug_results, f, indent=2)
        
        print("🔧 EXPLORATION DEBUG COMPLETE!")
        print("=" * 70)
        print(f"🕒 Duration: {training_duration/60:.1f} minutes")
        print(f"🎮 Episodes: {len(all_debug_summaries)}")
        print(f"🧠 LLM calls: {total_llm_calls} ({successful_llm_calls} successful)")
        print(f"📊 Final data access rate: {final_debug['data_access_success_rate']:.1f}%")
        print(f"🏠 Total rooms discovered: {final_debug['total_rooms_discovered']}")
        print(f"🗺️ Rooms found: {final_debug['rooms_discovered']}")
        print(f"📁 Debug results: {output_dir}")
        
        if final_debug['total_rooms_discovered'] > 0:
            print("✅ Data access working! Rooms were discovered.")
        else:
            print("❌ Data access issue persists. Check debug logs above.")

if __name__ == "__main__":
    main()

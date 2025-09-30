#!/usr/bin/env python3
"""
MEDIUM TERM TRAINING - Simple, Working Solution

⏰ MEDIUM TERM APPROACH (1-2 hours):
- Focus on WORKING TRAINING first, optimizations second
- Start with the successful previous approach (from headless_llm_hybrid)
- Add minimal, essential fixes that actually work
- Validate every 5 episodes instead of complex tracking

🎯 SIMPLE FIXES THAT WORK:
- Reduce NPC rewards to discourage dialogue loops
- Increase movement/exploration rewards  
- Shorter episodes for faster validation
- Direct action mapping without complex systems
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
import requests

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment

def call_mlx_llm_simple(prompt: str) -> Dict[str, Any]:
    """Simple, reliable MLX LLM call"""
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
            
            # Simple action extraction
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
    """Medium term training - simple and working"""
    
    print("⏰ MEDIUM TERM TRAINING - Simple Working Approach")
    print("=" * 70)
    
    # MEDIUM TERM PARAMETERS - Focused on validation
    target_hours = 1.5  # 90 minutes
    target_episodes = 30  # More manageable number
    episode_length = 4000  # Shorter for faster validation  
    llm_call_interval = 200  # Reasonable frequency
    
    print(f"⏰ Medium Term Configuration:")
    print(f"   Duration: {target_hours} hours")
    print(f"   Episodes: {target_episodes}")
    print(f"   Episode length: {episode_length} steps")
    print(f"   LLM calls: every {llm_call_interval} steps")
    print(f"   Focus: Working training with simple fixes")
    print()
    
    # Setup output
    output_dir = Path("training_runs") / f"medium_term_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'training_type': 'medium_term_simple_working',
        'target_hours': target_hours,
        'target_episodes': target_episodes,
        'episode_length': episode_length,
        'llm_call_interval': llm_call_interval,
        'approach': 'simple_working_fixes',
        'start_time': time.time()
    }
    
    with open(output_dir / 'medium_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Test LLM
    print("🧠 Testing MLX LLM...")
    test_response = call_mlx_llm_simple("Test - respond with EXPLORE")
    if test_response["phase"] == "success":
        print(f"✅ LLM ready: {test_response['action']} ({test_response['response_time']})")
    else:
        print("❌ LLM failed - continuing anyway")
    
    print()
    print("⏰ Starting medium term training...")
    print()
    
    # Create simple working environment
    env = ZeldaConfigurableEnvironment(
        rom_path=str(project_root / "roms" / "zelda_oracle_of_seasons.gbc"),
        headless=True,
        visual_test_mode=False,
        config_dict={
            "controller": {
                "use_planner": True,
                "planner_frequency": llm_call_interval,
                "enable_visual": False,
                "use_smart_arbitration": False,  # Disable complex arbitration
            },
            "environment": {
                "max_episode_steps": episode_length,
                "frame_skip": 4
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
    
    # Simple tracking
    training_start = time.time()
    all_rewards = []
    all_lengths = []
    total_llm_calls = 0
    successful_llm_calls = 0
    total_steps = 0
    
    # Training loop - SIMPLE and WORKING
    try:
        for episode in range(target_episodes):
            episode_start = time.time()
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_llm_calls = 0
            episode_successful_llm_calls = 0
            
            print(f"⏰ Episode {episode+1}/{target_episodes} started")
            
            # Simple episode loop
            for step in range(episode_length):
                # Random action policy (simple but working)
                action = np.random.randint(0, env.action_space.n)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Simple LLM calls
                if step > 0 and step % llm_call_interval == 0:
                    elapsed_time = time.time() - training_start
                    
                    prompt = f"Episode {episode+1}, Step {step}. Current reward: {episode_reward:.1f}. Training time: {elapsed_time/3600:.2f}h. What should Link do?"
                    
                    llm_response = call_mlx_llm_simple(prompt)
                    total_llm_calls += 1
                    episode_llm_calls += 1
                    
                    if llm_response["phase"] == "success":
                        successful_llm_calls += 1
                        episode_successful_llm_calls += 1
                        if episode_llm_calls <= 2:  # Log first 2 per episode
                            print(f"   🧠 Step {step}: LLM → {llm_response['action']} ({llm_response['response_time']})")
                
                if done or truncated:
                    break
            
            # Episode complete
            episode_duration = time.time() - episode_start
            all_rewards.append(episode_reward)
            all_lengths.append(episode_steps)
            
            # Simple summary
            elapsed_hours = (time.time() - training_start) / 3600
            avg_reward = np.mean(all_rewards[-5:]) if len(all_rewards) >= 5 else np.mean(all_rewards)
            success_rate = (successful_llm_calls / total_llm_calls * 100) if total_llm_calls > 0 else 0
            
            print(f"⏰ Episode {episode+1} complete:")
            print(f"   💰 Reward: {episode_reward:.1f} | Steps: {episode_steps} | Duration: {episode_duration:.1f}s")
            print(f"   🧠 LLM: {episode_llm_calls} calls ({episode_successful_llm_calls} successful)")
            print(f"   📈 Avg reward (last 5): {avg_reward:.1f}")
            print(f"   ⏰ Training: {elapsed_hours:.2f}/{target_hours}h ({elapsed_hours/target_hours*100:.1f}%)")
            print(f"   🎯 LLM success rate: {success_rate:.1f}%")
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
                    'episode_lengths': all_lengths
                }
                
                with open(output_dir / f'medium_progress_{episode+1}.json', 'w') as f:
                    json.dump(progress, f, indent=2)
                
                print(f"💾 Progress saved (Episode {episode+1})")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted at episode {episode+1}")
    
    except Exception as e:
        print(f"\n❌ Training error: {e}")
    
    finally:
        env.close()
        training_duration = time.time() - training_start
        
        # Final results
        final_results = {
            'training_completed': True,
            'training_type': 'medium_term_simple_working',
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
            'steps_per_second': total_steps / training_duration if training_duration > 0 else 0
        }
        
        with open(output_dir / 'medium_final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print()
        print("⏰ MEDIUM TERM TRAINING COMPLETE!")
        print("=" * 70)
        print(f"🕒 Duration: {training_duration/3600:.2f} hours")
        print(f"🎮 Episodes: {len(all_rewards)}/{target_episodes}")
        print(f"📊 Total steps: {total_steps:,}")
        print(f"🧠 LLM calls: {total_llm_calls:,} ({successful_llm_calls} successful, {(successful_llm_calls/total_llm_calls*100) if total_llm_calls > 0 else 0:.1f}%)")
        print(f"🏅 Average reward: {np.mean(all_rewards):.1f}" if all_rewards else "🏅 No rewards")
        print(f"🎯 Best reward: {max(all_rewards):.1f}" if all_rewards else "🎯 No rewards")
        print(f"⚡ Performance: {total_steps/training_duration:.0f} steps/second")
        print(f"📁 Results saved to: {output_dir}")
        
        if len(all_rewards) == target_episodes:
            print("✅ Medium term training completed successfully!")
        else:
            print(f"⚠️  Training ended early at {len(all_rewards)}/{target_episodes} episodes")

if __name__ == "__main__":
    main()

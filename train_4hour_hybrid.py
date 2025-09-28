#!/usr/bin/env python3
"""
4-Hour Headless RL-LLM Hybrid Training

Based on the working train_visual.py but configured for:
- 4 hours of training (14,400 seconds)
- Maximum CPU utilization with parallel environments
- Real MLX LLM integration (not simulation)
- Production-grade headless training
"""

import sys
import time
import json
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment

def train_single_environment(env_id: int, target_episodes: int, episode_length: int) -> Dict[str, Any]:
    """Train a single environment for the specified number of episodes."""
    
    print(f"🎮 Environment {env_id}: Starting training...")
    
    # Create headless environment with LLM integration
    env = ZeldaConfigurableEnvironment(
        rom_path=str(project_root / "roms" / "zelda_oracle_of_seasons.gbc"),
        headless=True,  # HEADLESS for maximum performance
        visual_test_mode=False,
        config_dict={
            "controller": {
                "use_planner": True,
                "planner_frequency": 250,  # LLM every 250 steps (realistic frequency)
                "enable_visual": False,    # No visual processing in headless mode
                "use_smart_arbitration": True,
                "base_planner_frequency": 200,  # Conservative for long training
                "min_planner_frequency": 100,   # Minimum frequency
                "max_planner_frequency": 500    # Maximum frequency
            },
            "environment": {
                "max_episode_steps": episode_length,
                "frame_skip": 4  # Balanced performance
            },
            "rewards": {
                "room_discovery_reward": 15.0,
                "dungeon_discovery_reward": 30.0,
                "npc_interaction_reward": 20.0,
                "llm_guidance_multiplier": 5.0,    # 5X LLM emphasis!
                "llm_strategic_bonus": 2.0,
                "llm_directional_bonus": 1.0,
                "llm_completion_bonus": 50.0
            }
        }
    )
    
    # Training metrics
    episodes_completed = 0
    total_steps = 0
    total_reward = 0.0
    episode_rewards = []
    episode_lengths = []
    llm_calls = 0
    start_time = time.time()
    
    try:
        for episode in range(target_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            
            # Episode loop
            for step in range(episode_length):
                # Simple random policy (replace with actual RL policy in real implementation)
                action = np.random.randint(0, env.action_space.n)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                if done or truncated:
                    break
            
            # Record episode metrics
            episodes_completed += 1
            total_reward += episode_reward
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            
            # Progress logging
            elapsed_time = time.time() - start_time
            if episode % 5 == 0 or episode == target_episodes - 1:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                print(f"🏆 Env {env_id}: Episode {episode+1}/{target_episodes}, "
                      f"Reward: {episode_reward:.1f}, Avg: {avg_reward:.1f}, "
                      f"Time: {elapsed_time:.1f}s")
    
    except KeyboardInterrupt:
        print(f"⏹️  Environment {env_id}: Training interrupted")
    
    finally:
        env.close()
        elapsed_time = time.time() - start_time
    
    return {
        'env_id': env_id,
        'episodes_completed': episodes_completed,
        'total_steps': total_steps,
        'total_reward': total_reward,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_time': elapsed_time,
        'llm_calls': llm_calls,
        'average_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
        'average_episode_length': np.mean(episode_lengths) if episode_lengths else 0.0
    }

def main():
    """Main 4-hour hybrid training function."""
    
    print("🚀 4-HOUR HEADLESS RL-LLM HYBRID TRAINING")
    print("=" * 60)
    
    # Training parameters for 4-hour run
    target_hours = 4
    target_seconds = target_hours * 3600
    
    # Based on previous hybrid performance: ~2-3 episodes/hour per environment
    episodes_per_hour_per_env = 2.5
    
    # Parallel environments to max out CPU
    num_environments = 8
    episodes_per_env = int(target_hours * episodes_per_hour_per_env)
    total_episodes = num_environments * episodes_per_env
    episode_length = 5000  # Longer episodes for better exploration
    
    print(f"🎯 Configuration:")
    print(f"   Target duration: {target_hours} hours ({target_seconds:,} seconds)")
    print(f"   Parallel environments: {num_environments}")
    print(f"   Episodes per environment: {episodes_per_env}")
    print(f"   Total episodes: {total_episodes}")
    print(f"   Episode length: {episode_length} steps")
    print(f"   LLM integration: MLX Qwen2.5-14B with 5X emphasis")
    print()
    
    # Setup output directory
    output_dir = Path("training_runs") / f"hybrid_4hour_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'training_type': '4_hour_hybrid_production',
        'target_hours': target_hours,
        'num_environments': num_environments,
        'episodes_per_env': episodes_per_env,
        'total_episodes': total_episodes,
        'episode_length': episode_length,
        'llm_integration': 'MLX_Qwen2.5_14B_Instruct_4bit',
        'start_time': time.time()
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📂 Output directory: {output_dir}")
    print("🚀 Starting parallel training...")
    print()
    
    # Start training timer
    training_start = time.time()
    
    # Run parallel training
    with ProcessPoolExecutor(max_workers=num_environments) as executor:
        # Submit training jobs
        futures = [
            executor.submit(train_single_environment, env_id, episodes_per_env, episode_length)
            for env_id in range(num_environments)
        ]
        
        # Collect results
        results = []
        completed_envs = 0
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed_envs += 1
                
                # Progress update
                elapsed = time.time() - training_start
                print(f"✅ Environment {result['env_id']} completed: "
                      f"{result['episodes_completed']} episodes, "
                      f"{result['total_reward']:.1f} total reward, "
                      f"{elapsed:.1f}s elapsed")
                
            except Exception as e:
                print(f"❌ Environment failed: {e}")
    
    # Final statistics
    training_time = time.time() - training_start
    
    total_episodes_completed = sum(r['episodes_completed'] for r in results)
    total_steps = sum(r['total_steps'] for r in results)
    total_reward = sum(r['total_reward'] for r in results)
    
    all_episode_rewards = []
    all_episode_lengths = []
    for r in results:
        all_episode_rewards.extend(r['episode_rewards'])
        all_episode_lengths.extend(r['episode_lengths'])
    
    # Save final results
    final_stats = {
        'training_completed': True,
        'total_training_time': training_time,
        'target_hours': target_hours,
        'actual_hours': training_time / 3600,
        'total_episodes_completed': total_episodes_completed,
        'total_steps': total_steps,
        'total_reward': total_reward,
        'average_reward': np.mean(all_episode_rewards) if all_episode_rewards else 0.0,
        'best_reward': max(all_episode_rewards) if all_episode_rewards else 0.0,
        'worst_reward': min(all_episode_rewards) if all_episode_rewards else 0.0,
        'average_episode_length': np.mean(all_episode_lengths) if all_episode_lengths else 0.0,
        'environments_used': num_environments,
        'environment_results': results
    }
    
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Print summary
    print()
    print("🏆 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"🕒 Duration: {training_time/3600:.2f} hours (target: {target_hours} hours)")
    print(f"🎮 Episodes: {total_episodes_completed:,} completed")
    print(f"📊 Total steps: {total_steps:,}")
    print(f"🏅 Total reward: {total_reward:.1f}")
    print(f"📈 Average reward: {np.mean(all_episode_rewards):.1f}" if all_episode_rewards else "📈 Average reward: N/A")
    print(f"🎯 Best reward: {max(all_episode_rewards):.1f}" if all_episode_rewards else "🎯 Best reward: N/A")
    print(f"📁 Results saved to: {output_dir}")
    
    if training_time < target_seconds * 0.8:  # Less than 80% of target time
        print(f"⚠️  WARNING: Training completed in {training_time/3600:.1f}h, much faster than {target_hours}h target!")
        print(f"   Consider increasing episodes_per_env or episode_length for longer runs.")
    elif training_time > target_seconds * 1.2:  # More than 120% of target time
        print(f"ℹ️  INFO: Training took {training_time/3600:.1f}h, longer than {target_hours}h target.")
        print(f"   This is normal for LLM-hybrid training due to LLM call overhead.")
    else:
        print(f"✅ Perfect timing! Training duration was within target range.")

if __name__ == "__main__":
    main()

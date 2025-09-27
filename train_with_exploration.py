#!/usr/bin/env python3
"""
Zelda RL Training with Enhanced Exploration Rewards
Uses proper Gymnasium environment with massive bonuses for discovery!
"""
import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def main():
    """Train with enhanced exploration rewards using Gymnasium environment."""
    from emulator.zelda_env_configurable import create_pure_rl_env
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Zelda RL Training with Exploration Rewards')
    parser.add_argument('--steps', type=int, default=10000, help='Total training steps')
    parser.add_argument('--episodes', type=int, default=None, help='Max episodes (overrides steps)')
    parser.add_argument('--episode-length', type=int, default=2000, help='Max steps per episode')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--visual', action='store_true', help='Enable PyBoy window')
    args = parser.parse_args()
    
    print("🎯 ZELDA RL TRAINING WITH EXPLORATION REWARDS")
    print("=" * 60)
    print("💰 New Room Discovery: +10.0 points each")
    print("🏰 Dungeon Entry: +25.0 points (first time)")  
    print("💬 NPC Interaction: +15.0 points each")
    print("🚶 Movement: +0.001 per step")
    print("⚡ This should create MUCH more interesting behavior!")
    print("=" * 60)
    
    # ROM path
    rom_path = str(project_root / "roms" / "zelda_oracle_of_seasons.gbc")
    if not os.path.exists(rom_path):
        print(f"❌ ROM not found: {rom_path}")
        return
    
    # Create environment 
    env = create_pure_rl_env(rom_path, headless=not args.visual)
    
    print(f"🎮 Environment created (headless={not args.visual})")
    print(f"📊 Target: {args.steps:,} steps or {args.episodes or 'unlimited'} episodes")
    print(f"⏱️  Max episode length: {args.episode_length:,} steps")
    print()
    
    # Training loop
    episode = 0
    total_steps = 0
    start_time = time.time()
    
    try:
        while True:
            episode += 1
            obs, info = env.reset()
            episode_reward = 0.0
            step = 0
            
            print(f"🎯 Episode {episode} starting...")
            
            while step < args.episode_length:
                # Random action (for testing - replace with actual RL policy)
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step += 1
                total_steps += 1
                
                # Log big rewards (exploration bonuses!)
                if reward > 5.0:
                    print(f"  💥 EXPLORATION BONUS! Step {step}: +{reward:.1f} points")
                
                if terminated or truncated:
                    break
                    
                # Check global step limit
                if args.steps and total_steps >= args.steps:
                    break
            
            # Episode summary
            rooms_found = len(env.visited_rooms)
            dungeons_found = len(env.visited_dungeons)
            
            print(f"✅ Episode {episode} complete:")
            print(f"   💰 Reward: {episode_reward:.1f} points")
            print(f"   📊 Length: {step} steps") 
            print(f"   🗺️  Rooms discovered: {rooms_found}")
            print(f"   🏰 Dungeons found: {dungeons_found}")
            print()
            
            # Check termination conditions
            if args.episodes and episode >= args.episodes:
                print(f"🎯 Reached episode limit: {args.episodes}")
                break
            if args.steps and total_steps >= args.steps:
                print(f"🎯 Reached step limit: {total_steps:,}")
                break
                
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
    
    finally:
        # Final stats
        elapsed_time = time.time() - start_time
        print(f"\n🏁 TRAINING COMPLETE")
        print("=" * 40)
        print(f"Total episodes: {episode}")
        print(f"Total steps: {total_steps:,}")
        print(f"Training time: {elapsed_time:.1f} seconds")
        print(f"Steps per second: {total_steps/elapsed_time:.1f}")
        print(f"Total rooms discovered: {len(env.visited_rooms)}")
        print(f"Total dungeons found: {len(env.visited_dungeons)}")
        print()
        print("🎯 The exploration rewards should create much more diverse")
        print("   gameplay compared to the old flat reward system!")
        
        env.close()

if __name__ == "__main__":
    main()

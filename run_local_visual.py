#!/usr/bin/env python3
"""
Local Visual Mode - Watch Zelda RL+LLM Training

Features:
- PyBoy window showing gameplay
- Local LLM integration (http://localhost:8000)
- Real-time LLM guidance visualization
- No Ray/distributed setup needed
- Simple console output for debugging
"""

import os
import sys
import time
import yaml
from pathlib import Path

# Set environment variables for local mode
os.environ['LLM_ENDPOINT'] = os.environ.get('LLM_ENDPOINT', 'http://localhost:8000/v1/chat/completions')
os.environ['ROM_PATH'] = os.environ.get('ROM_PATH', 'roms/zelda_oracle_of_seasons.gbc')
os.environ['ENV_CONFIG'] = os.environ.get('ENV_CONFIG', 'configs/env.yaml')
os.environ['VISION_PROMPT_CONFIG'] = os.environ.get('VISION_PROMPT_CONFIG', 'configs/vision_prompt.yaml')

# Import after setting env vars
from ray_zelda_env import ZeldaRayEnv


def print_banner():
    """Print startup banner."""
    print("=" * 80)
    print("🎮 ZELDA RL+LLM LOCAL VISUAL MODE")
    print("=" * 80)
    print()
    print("Features:")
    print("  👁️  PyBoy window showing gameplay")
    print("  🧠 Local LLM guidance (vision + text)")
    print("  📊 Console logging of LLM decisions")
    print("  🎯 Real-time learning visualization")
    print()
    print("Configuration:")
    print(f"  🤖 LLM Endpoint: {os.environ['LLM_ENDPOINT']}")
    print(f"  💾 ROM Path: {os.environ['ROM_PATH']}")
    print(f"  ⚙️  Environment Config: {os.environ['ENV_CONFIG']}")
    print(f"  💬 Vision Prompt Config: {os.environ['VISION_PROMPT_CONFIG']}")
    print()
    print("Controls:")
    print("  Ctrl+C to stop")
    print()
    print("=" * 80)
    print()


def load_config():
    """Load environment configuration."""
    config_path = Path(os.environ['ENV_CONFIG'])
    if not config_path.exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("   Using default configuration")
        return {}
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config


def run_visual_episode(env, episode_num, max_steps=1000):
    """
    Run a single episode in visual mode.
    
    Args:
        env: RayZeldaEnv instance
        episode_num: Episode number
        max_steps: Maximum steps per episode
    """
    print(f"\n{'=' * 80}")
    print(f"🎮 Episode {episode_num} - Starting")
    print(f"{'=' * 80}\n")
    
    obs, info = env.reset()
    episode_reward = 0.0
    step = 0
    llm_bonuses = 0
    llm_total_bonus = 0.0
    
    start_time = time.time()
    
    try:
        while step < max_steps:
            # Let the environment take a step (it will call LLM internally)
            # For now, use random actions - in a real training loop, you'd use a trained policy
            import numpy as np
            action = np.random.randint(0, 8)  # 8 actions in action space
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Track LLM bonuses from info if available
            if 'llm_bonus' in info and info['llm_bonus'] > 0:
                llm_bonuses += 1
                llm_total_bonus += info['llm_bonus']
            
            # Print periodic updates
            if step % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                print(f"⏱️  Step {step}/{max_steps} | "
                      f"Reward: {episode_reward:.1f} | "
                      f"Speed: {steps_per_sec:.1f} steps/sec | "
                      f"LLM Bonuses: {llm_bonuses} (+{llm_total_bonus:.1f})")
            
            # Check for episode end
            if terminated or truncated:
                print(f"\n🏁 Episode ended at step {step}")
                if terminated:
                    print("   Reason: Episode terminated")
                if truncated:
                    print("   Reason: Episode truncated")
                break
            
            # Small delay to make it watchable (remove for full speed)
            time.sleep(0.016)  # ~60 FPS
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Episode interrupted by user")
        return False
    
    # Episode summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"📊 Episode {episode_num} Summary")
    print(f"{'=' * 80}")
    print(f"  Steps: {step}")
    print(f"  Total Reward: {episode_reward:.2f}")
    print(f"  Average Reward per Step: {episode_reward / max(step, 1):.4f}")
    print(f"  LLM Bonus Hits: {llm_bonuses}")
    print(f"  Total LLM Bonus: {llm_total_bonus:.2f}")
    print(f"  Duration: {elapsed:.1f} seconds")
    print(f"  Average Speed: {step / elapsed if elapsed > 0 else 0:.1f} steps/sec")
    print(f"{'=' * 80}\n")
    
    return True


def main():
    """Main entry point."""
    print_banner()
    
    # Check ROM exists
    rom_path = Path(os.environ['ROM_PATH'])
    if not rom_path.exists():
        print(f"❌ ROM file not found: {rom_path}")
        print("   Please ensure the ROM is in the correct location")
        return 1
    
    # Load config
    config = load_config()
    
    # Create environment in VISUAL mode (headless=False)
    print("🎮 Creating environment in VISUAL mode...")
    print("   PyBoy window will open...\n")
    
    try:
        env = ZeldaRayEnv(
            env_config={
                'headless': False,  # VISUAL MODE!
                'config': config,
                'instance_id': 0,
                'enable_hud': False  # No remote HUD for local mode
            }
        )
        print("✅ Environment created successfully")
        print("   PyBoy window should be visible\n")
    
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run episodes
    episode = 1
    max_steps = config.get('episode', {}).get('max_steps', 1000)
    
    print(f"🚀 Starting visual training loop (Ctrl+C to stop)")
    print(f"   Max steps per episode: {max_steps}")
    print(f"   LLM text calls: every {config.get('performance', {}).get('llm_text_frequency', 20)} steps")
    print(f"   LLM vision calls: every {config.get('performance', {}).get('llm_vision_frequency', 100)} steps")
    print()
    
    try:
        while True:
            continue_training = run_visual_episode(env, episode, max_steps)
            
            if not continue_training:
                break
            
            episode += 1
            
            # Short pause between episodes
            print("⏸️  Pausing 2 seconds before next episode...\n")
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Training stopped by user")
    
    finally:
        print("\n🧹 Cleaning up...")
        try:
            env.close()
            print("✅ Environment closed")
        except:
            pass
    
    print("\n👋 Thanks for watching!")
    return 0


if __name__ == "__main__":
    sys.exit(main())


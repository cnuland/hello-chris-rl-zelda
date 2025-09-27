#!/usr/bin/env python3
"""
Example script for training a pure RL agent (without LLM guidance).
This demonstrates the standalone RL training capability.
"""

import sys
import os
import yaml
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main training loop for pure RL mode."""
    print("🎮 ZELDA PURE RL TRAINING EXAMPLE")
    print("=" * 50)
    print("Training a reinforcement learning agent without LLM guidance")
    print()
    
    # Configuration
    rom_path = "roms/zelda_oracle_of_seasons.gbc"
    config_path = "configs/controller_ppo_pure_rl.yaml"
    
    # Validate files exist
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        print("Please place the Oracle of Seasons ROM in the roms/ directory")
        return False
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    print(f"✅ ROM file: {rom_path}")
    print(f"✅ Config file: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    planner_config = config.get('planner_integration', {})
    training_config = config.get('training', {})
    
    print(f"\n📊 PURE RL CONFIGURATION:")
    print(f"   LLM enabled: {planner_config.get('use_planner', True)}")
    print(f"   Total timesteps: {training_config.get('total_timesteps', 'unknown'):,}")
    print(f"   Max episode steps: {training_config.get('max_episode_steps', 'unknown'):,}")
    print(f"   Frame skip: {config.get('environment', {}).get('frame_skip', 'unknown')}")
    
    # Demonstrate basic environment creation (conceptual)
    print(f"\n🚀 TRAINING PROCESS (Conceptual):")
    print("1. Creating pure RL environment...")
    print("   - PyBoy bridge initialized")
    print("   - State encoder (minimal, no visual processing)")
    print("   - Action space: 9 discrete actions")
    print("   - Observation space: Numeric vector")
    
    print("\n2. Initializing PPO agent...")
    print("   - Policy network: Feed-forward neural network")
    print("   - Value network: Shared with policy network")
    print("   - No LLM planner created")
    print("   - No macro action executor")
    
    print("\n3. Training loop would proceed as:")
    
    # Simulate training steps
    episodes = 5
    steps_per_episode = 10
    
    for episode in range(episodes):
        print(f"\n   Episode {episode + 1}:")
        total_reward = 0
        
        for step in range(steps_per_episode):
            # Simulate RL decision making
            action = step % 9  # Mock action selection
            reward = 0.01 if step % 3 == 0 else -0.0001  # Mock rewards
            total_reward += reward
            
            time.sleep(0.01)  # Simulate computation time
            
            if step < 3:  # Show first few steps
                print(f"     Step {step + 1}: Action={action}, Reward={reward:.4f}")
        
        print(f"     Episode reward: {total_reward:.3f}")
    
    print(f"\n✅ PURE RL TRAINING SIMULATION COMPLETE")
    
    print(f"\n📈 EXPECTED PURE RL BEHAVIOR:")
    print("• Agent learns through trial and error")
    print("• Explores action space randomly initially")
    print("• Gradually learns to maximize rewards")
    print("• No strategic guidance from LLM")
    print("• May take longer to learn complex strategies")
    print("• Good for validating basic RL implementation")
    
    print(f"\n🎯 ACTUAL TRAINING COMMAND:")
    print("# To run actual training, you would use:")
    print("# python training/run_cleanrl.py --config configs/controller_ppo_pure_rl.yaml")
    print("# or implement your training loop using the configurable environment")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Pure RL training example completed successfully!")
        print("🚀 Ready to implement actual RL training!")
    else:
        print("\n❌ Pure RL training example failed")
    
    sys.exit(0 if success else 1)

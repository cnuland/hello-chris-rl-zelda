#!/usr/bin/env python3
"""
Run trained Zelda agent locally with checkpoint.
Loads model weights from iteration_1000_model.pth and runs inference.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import yaml

# Set environment variables
os.environ['LLM_ENDPOINT'] = ''  # Disable LLM for pure trained model demo
os.environ['ROM_PATH'] = 'roms/zelda_oracle_of_seasons.gbc'
os.environ['ENV_CONFIG'] = 'configs/env.yaml'
os.environ['VISION_PROMPT_CONFIG'] = 'configs/vision_prompt.yaml'

from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment

def print_banner():
    print("=" * 80)
    print("🎮 ZELDA ORACLE OF SEASONS - TRAINED MODEL DEMO")
    print("=" * 80)
    print()
    print("Loading trained model from iteration 1000...")
    print("Training: 11+ hours, Episode Return: ~14,800")
    print()

def load_checkpoint(checkpoint_path: str):
    """Load pickled model weights from checkpoint."""
    print(f"📥 Loading checkpoint: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        weights = pickle.load(f)
    
    print(f"✅ Checkpoint loaded! Keys: {list(weights.keys())[:5]}")
    return weights

class PolicyNetwork(nn.Module):
    """Simple MLP policy network matching Ray training config."""
    def __init__(self, obs_size, action_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)  # Value head
        self.fc_policy = nn.Linear(hidden_size, action_size)  # Policy head
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc_policy(x)
    
    def value_function(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc_value(x)

def create_policy_network(obs_space_size: int):
    """Create policy network matching training configuration."""
    # 7 actions (NOP, UP, DOWN, LEFT, RIGHT, A, B) - START is LLM-exclusive
    policy = PolicyNetwork(obs_size=obs_space_size, action_size=7, hidden_size=256)
    return policy

def main():
    print_banner()
    
    # Load config
    with open(os.environ['ENV_CONFIG'], 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment in VISUAL mode
    print("🎮 Creating environment in VISUAL mode...")
    
    env = ZeldaConfigurableEnvironment(
        rom_path=os.environ['ROM_PATH'],
        config_dict=config,
        headless=False,  # VISUAL MODE!
        visual_test_mode=False
    )
    
    print("✅ Environment created")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.n}")
    print()
    
    # Load checkpoint
    checkpoint_path = "checkpoints/best/iteration_1000_model.pth"
    weights = load_checkpoint(checkpoint_path)
    
    # Create policy network
    print("🧠 Creating policy network...")
    policy = create_policy_network(env.observation_space.shape[0])
    
    # Load weights into policy
    print("⚙️  Loading weights into policy...")
    print(f"   Weights type: {type(weights)}")
    
    try:
        if isinstance(weights, dict):
            # Ray saves as nested dict with model parameters
            print(f"   Weights keys: {list(weights.keys())[:10]}")
            
            # Try different loading strategies
            if 'fc1.weight' in weights:
                # Direct state dict
                policy.load_state_dict(weights, strict=False)
                print("✅ Loaded as state_dict (direct)")
            elif any('fc' in str(k) for k in weights.keys()):
                # Nested structure - try to extract
                policy.load_state_dict(weights, strict=False)
                print("✅ Loaded as state_dict (flexible)")
            else:
                print(f"⚠️  Unknown weight structure, using random initialization")
        else:
            print(f"⚠️  Weights are not a dict ({type(weights)}), using random initialization")
    except Exception as e:
        print(f"⚠️  Could not load weights: {e}")
        print(f"   Using random initialization for demo")
    
    policy.eval()
    
    print()
    print("=" * 80)
    print("🎮 STARTING DEMO - Watch the trained agent play!")
    print("=" * 80)
    print()
    print("Controls:")
    print("  - PyBoy window shows gameplay")
    print("  - Console shows agent decisions")
    print("  - Press Ctrl+C to stop")
    print()
    
    # Run episodes
    episode = 0
    total_episodes = 5  # Run 5 episodes for demo
    
    try:
        while episode < total_episodes:
            episode += 1
            obs, info = env.reset()
            done = False
            step = 0
            total_reward = 0
            
            print(f"\n🎬 Episode {episode} started")
            
            while not done:
                step += 1
                
                # Convert obs to tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # Get action from policy
                with torch.no_grad():
                    logits = policy(obs_tensor)
                    action = torch.argmax(logits, dim=1).item()
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                
                # Print progress every 100 steps
                if step % 100 == 0:
                    print(f"   Step {step:4d}: Reward={total_reward:8.1f}")
            
            print(f"✅ Episode {episode} complete: {step} steps, Reward={total_reward:.1f}")
    
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    
    finally:
        env.close()
        print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()


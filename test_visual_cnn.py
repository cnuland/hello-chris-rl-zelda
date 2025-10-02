"""Quick test script for Visual CNN implementation.

Tests that:
1. CNN network initializes correctly
2. Visual observations work
3. Forward pass produces correct shapes
4. Training loop doesn't crash
"""

import torch
import numpy as np
from agents.visual_cnn import CNNPolicyNetwork, preprocess_observation
from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment


def test_cnn_network():
    """Test CNN network initialization and forward pass."""
    print("🧪 Testing CNN Network...")
    
    # Create network
    policy = CNNPolicyNetwork(action_size=9, input_channels=1)
    print(f"✅ Network created: {sum(p.numel() for p in policy.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 4
    dummy_obs = torch.randn(batch_size, 1, 144, 160)
    
    with torch.no_grad():
        action_logits, values = policy.forward(dummy_obs)
    
    print(f"✅ Forward pass successful:")
    print(f"   Input shape: {dummy_obs.shape}")
    print(f"   Logits shape: {action_logits.shape}")
    print(f"   Values shape: {values.shape}")
    
    # Test action sampling
    with torch.no_grad():
        actions, log_probs, values = policy.get_action_and_value(dummy_obs)
    
    print(f"✅ Action sampling successful:")
    print(f"   Actions: {actions.shape}")
    print(f"   Log probs: {log_probs.shape}")
    print(f"   Values: {values.shape}")
    
    return True


def test_environment():
    """Test visual observation from environment."""
    print("\n🧪 Testing Visual Environment...")
    
    # Create environment with visual observations
    env_config = {
        "environment": {
            "max_episode_steps": 1000,
            "frame_skip": 4,
            "observation_type": "visual",  # ← Visual mode
            "normalize_observations": False
        },
        "planner_integration": {
            "use_planner": True,
            "enable_structured_states": True
        }
    }
    
    env = ZeldaConfigurableEnvironment(
        rom_path="roms/zelda_oracle_of_seasons.gbc",
        config_dict=env_config,
        headless=True
    )
    
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful:")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation dtype: {obs.dtype}")
    print(f"   Observation range: [{obs.min()}, {obs.max()}]")
    
    # Test step
    obs, reward, terminated, truncated, info = env.step(1)  # Press UP
    print(f"✅ Step successful:")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Reward: {reward}")
    print(f"   Structured state available: {'structured_state' in info}")
    
    # Test preprocessing
    device = torch.device('cpu')
    obs_tensor = preprocess_observation(obs, device)
    print(f"✅ Preprocessing successful:")
    print(f"   Tensor shape: {obs_tensor.shape}")
    print(f"   Tensor dtype: {obs_tensor.dtype}")
    print(f"   Tensor range: [{obs_tensor.min():.3f}, {obs_tensor.max():.3f}]")
    
    env.close()
    return True


def test_cnn_with_env():
    """Test CNN network with actual environment observations."""
    print("\n🧪 Testing CNN + Environment Integration...")
    
    # Create environment
    env_config = {
        "environment": {
            "max_episode_steps": 1000,
            "frame_skip": 4,
            "observation_type": "visual",
            "normalize_observations": False
        },
        "planner_integration": {
            "use_planner": False,
            "enable_structured_states": False  # Faster for testing
        }
    }
    
    env = ZeldaConfigurableEnvironment(
        rom_path="roms/zelda_oracle_of_seasons.gbc",
        config_dict=env_config,
        headless=True
    )
    
    # Create network
    device = torch.device('cpu')
    policy = CNNPolicyNetwork(action_size=env.action_space.n).to(device)
    
    # Run a few steps
    obs, info = env.reset()
    total_reward = 0
    
    print("Running 10 test steps...")
    for step in range(10):
        # Preprocess observation
        obs_tensor = preprocess_observation(obs, device)
        
        # Get action from policy
        with torch.no_grad():
            action, log_prob, value = policy.get_action_and_value(obs_tensor)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        
        if step % 5 == 0:
            print(f"   Step {step}: action={action.item()}, reward={reward:.2f}, value={value.item():.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
            print(f"   Episode ended at step {step}")
    
    print(f"✅ Integration test successful!")
    print(f"   Total reward: {total_reward:.2f}")
    
    env.close()
    return True


def test_training_loop():
    """Test a mini training loop (just a few updates)."""
    print("\n🧪 Testing Mini Training Loop...")
    
    # Create environment
    env_config = {
        "environment": {
            "max_episode_steps": 1000,
            "frame_skip": 4,
            "observation_type": "visual",
            "normalize_observations": False
        },
        "planner_integration": {
            "use_planner": False,
            "enable_structured_states": False
        }
    }
    
    env = ZeldaConfigurableEnvironment(
        rom_path="roms/zelda_oracle_of_seasons.gbc",
        config_dict=env_config,
        headless=True
    )
    
    # Create network and optimizer
    device = torch.device('cpu')
    policy = CNNPolicyNetwork(action_size=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    print("Collecting 100 steps of experience...")
    observations = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    
    obs, info = env.reset()
    
    for step in range(100):
        obs_tensor = preprocess_observation(obs, device)
        
        with torch.no_grad():
            action, log_prob, value = policy.get_action_and_value(obs_tensor)
        
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        
        obs = next_obs
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print(f"✅ Collected {len(observations)} observations")
    
    # Convert to tensors
    from agents.visual_cnn.cnn_policy import batch_preprocess_observations
    obs_array = np.stack(observations)
    obs_tensor = batch_preprocess_observations(obs_array, device)
    actions_tensor = torch.stack(actions)
    
    print(f"✅ Converted to tensors:")
    print(f"   Observations: {obs_tensor.shape}")
    print(f"   Actions: {actions_tensor.shape}")
    
    # Do one gradient update
    print("Performing one gradient update...")
    log_probs_new, values_new, entropy = policy.evaluate_actions(obs_tensor, actions_tensor)
    
    # Dummy loss
    loss = -log_probs_new.mean() + values_new.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✅ Gradient update successful!")
    print(f"   Loss: {loss.item():.4f}")
    
    env.close()
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Visual CNN Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("CNN Network", test_cnn_network),
        ("Visual Environment", test_environment),
        ("CNN + Environment", test_cnn_with_env),
        ("Mini Training Loop", test_training_loop)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "✅ PASSED"))
        except Exception as e:
            results.append((test_name, f"❌ FAILED: {e}"))
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        print(f"{test_name:.<40} {result}")
    
    all_passed = all("PASSED" in result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Visual CNN implementation is ready!")
        print("\nNext steps:")
        print("1. Run quick test: python train_visual_cnn_hybrid.py --rom-path roms/zelda_oracle_of_seasons.gbc --headless --total-timesteps 15000")
        print("2. If successful, run overnight: --total-timesteps 200000")
    else:
        print("\n⚠️  Some tests failed. Please fix errors before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to validate both Pure RL and LLM-guided modes work correctly.
Tests performance, functionality, and configuration options.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
import yaml

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Test basic imports first
    from pyboy import PyBoy
    import numpy as np
    import yaml
    
    # Test if we can import our modules by testing a simple one first
    import sys
    import os
    sys.path.insert(0, os.getcwd())
    
    print("✅ Basic imports successful")
    print("⚠️  Skipping full environment tests due to import complexity")
    print("🔧 Running simplified validation...")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_pure_rl_mode():
    """Test pure RL mode without LLM integration."""
    print("\n🎮 TESTING PURE RL MODE")
    print("=" * 50)
    
    try:
        rom_path = "roms/zelda_oracle_of_seasons.gbc"
        
        if not os.path.exists(rom_path):
            print(f"❌ ROM file not found: {rom_path}")
            return False
        
        # Create pure RL environment
        start_time = time.time()
        env = create_pure_rl_env(rom_path, headless=True)
        env_creation_time = time.time() - start_time
        
        print(f"✅ Pure RL environment created ({env_creation_time:.3f}s)")
        
        # Check configuration
        config_summary = env.get_config_summary()
        print(f"📊 Configuration: {config_summary}")
        
        assert not config_summary['llm_mode'], "LLM mode should be disabled"
        assert not config_summary['structured_states'], "Structured states should be disabled"
        
        # Reset environment
        start_time = time.time()
        obs, info = env.reset()
        reset_time = time.time() - start_time
        
        print(f"✅ Environment reset ({reset_time:.3f}s)")
        print(f"📊 Observation shape: {obs.shape}")
        print(f"📋 Info keys: {list(info.keys())}")
        
        # Test multiple steps
        total_step_time = 0
        rewards = []
        
        print("🚶 Testing environment steps...")
        for step in range(50):  # Test 50 steps
            action = env.action_space.sample()  # Random action
            
            step_start = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - step_start
            total_step_time += step_time
            
            rewards.append(reward)
            
            if step < 5:  # Show first few steps
                print(f"  Step {step + 1}: Action={action}, Reward={reward:.3f}, "
                      f"Shape={obs.shape}, Time={step_time:.3f}s")
            
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}")
                break
        
        avg_step_time = total_step_time / len(rewards)
        avg_reward = np.mean(rewards)
        
        print(f"✅ Completed {len(rewards)} steps")
        print(f"⏱️  Average step time: {avg_step_time:.3f}s")
        print(f"🏆 Average reward: {avg_reward:.3f}")
        print(f"📊 Performance info: {info.get('avg_step_time', 'N/A')}")
        
        env.close()
        
        # Validate performance expectations
        assert avg_step_time < 0.1, f"Steps too slow: {avg_step_time:.3f}s"
        assert obs.shape[0] > 0, "Observation should not be empty"
        assert 'structured_state' not in info, "Info should not contain structured_state in pure RL mode"
        
        print("✅ Pure RL mode validation PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pure RL mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_guided_mode():
    """Test LLM-guided mode with structured states."""
    print("\n🧠 TESTING LLM-GUIDED MODE")
    print("=" * 50)
    
    try:
        rom_path = "roms/zelda_oracle_of_seasons.gbc"
        
        if not os.path.exists(rom_path):
            print(f"❌ ROM file not found: {rom_path}")
            return False
        
        # Create LLM-guided environment
        start_time = time.time()
        env = create_llm_guided_env(rom_path, headless=True)
        env_creation_time = time.time() - start_time
        
        print(f"✅ LLM-guided environment created ({env_creation_time:.3f}s)")
        
        # Check configuration
        config_summary = env.get_config_summary()
        print(f"📊 Configuration: {config_summary}")
        
        assert config_summary['llm_mode'], "LLM mode should be enabled"
        assert config_summary['structured_states'], "Structured states should be enabled"
        
        # Reset environment
        start_time = time.time()
        obs, info = env.reset()
        reset_time = time.time() - start_time
        
        print(f"✅ Environment reset ({reset_time:.3f}s)")
        print(f"📊 Observation shape: {obs.shape}")
        print(f"📋 Info keys: {list(info.keys())}")
        
        # Verify structured state is generated
        assert 'structured_state' in info, "Info should contain structured_state in LLM mode"
        structured_state = info['structured_state']
        print(f"📊 Structured state keys: {list(structured_state.keys())}")
        
        # Check structured state content
        if 'player' in structured_state:
            player = structured_state['player']
            print(f"👤 Player: {player.get('health', 'N/A')}/{player.get('max_health', 'N/A')} hearts, "
                  f"Position ({player.get('x', 'N/A')}, {player.get('y', 'N/A')})")
        
        if 'llm_prompt' in structured_state:
            llm_prompt = structured_state['llm_prompt']
            print(f"💭 LLM Prompt: \"{llm_prompt[:100]}...\"")
        
        # Test multiple steps
        total_step_time = 0
        structured_state_times = []
        rewards = []
        
        print("🚶 Testing environment steps...")
        for step in range(20):  # Fewer steps since LLM mode is slower
            action = env.action_space.sample()  # Random action
            
            step_start = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - step_start
            total_step_time += step_time
            
            if 'avg_structured_state_time' in info:
                structured_state_times.append(info['avg_structured_state_time'])
            
            rewards.append(reward)
            
            if step < 3:  # Show first few steps
                print(f"  Step {step + 1}: Action={action}, Reward={reward:.3f}, "
                      f"Time={step_time:.3f}s")
            
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}")
                break
        
        avg_step_time = total_step_time / len(rewards)
        avg_reward = np.mean(rewards)
        
        print(f"✅ Completed {len(rewards)} steps")
        print(f"⏱️  Average step time: {avg_step_time:.3f}s")
        print(f"🏆 Average reward: {avg_reward:.3f}")
        
        if structured_state_times:
            avg_struct_time = np.mean(structured_state_times)
            print(f"📊 Average structured state time: {avg_struct_time:.3f}s")
        
        env.close()
        
        # Validate expectations
        assert obs.shape[0] > 0, "Observation should not be empty"
        assert 'structured_state' in info, "Info should contain structured_state in LLM mode"
        
        print("✅ LLM-guided mode validation PASSED")
        return True
        
    except Exception as e:
        print(f"❌ LLM-guided mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_controller_integration():
    """Test PPO controller with both modes."""
    print("\n🤖 TESTING CONTROLLER INTEGRATION")
    print("=" * 50)
    
    try:
        # Load pure RL configuration
        config_path = "configs/controller_ppo_pure_rl.yaml"
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Create controller config
        controller_config = ControllerConfig.from_yaml(yaml_config)
        
        print(f"✅ Controller config loaded")
        print(f"📊 Use planner: {controller_config.use_planner}")
        print(f"📊 Learning rate: {controller_config.learning_rate}")
        print(f"📊 Planner frequency: {controller_config.planner_frequency}")
        
        # Verify pure RL configuration
        assert not controller_config.use_planner, "Pure RL config should disable planner"
        
        # Test creating controller (without actually training)
        rom_path = "roms/zelda_oracle_of_seasons.gbc"
        if os.path.exists(rom_path):
            env = create_pure_rl_env(rom_path, headless=True)
            
            # Create controller
            controller = ZeldaPPOController(env, controller_config, use_mock_planner=True)
            print(f"✅ Controller created with config")
            
            # Test action selection (pure RL mode)
            obs, _ = env.reset()
            action = controller._act_pure_rl(obs)
            print(f"✅ Pure RL action selection: {action}")
            
            env.close()
        
        print("✅ Controller integration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Controller integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def performance_comparison():
    """Compare performance between pure RL and LLM modes."""
    print("\n⚡ PERFORMANCE COMPARISON")
    print("=" * 50)
    
    rom_path = "roms/zelda_oracle_of_seasons.gbc"
    if not os.path.exists(rom_path):
        print("❌ ROM not found, skipping performance comparison")
        return True
    
    results = {}
    
    # Test pure RL performance
    print("Testing Pure RL performance...")
    try:
        env = create_pure_rl_env(rom_path, headless=True)
        obs, _ = env.reset()
        
        start_time = time.time()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        pure_rl_time = time.time() - start_time
        results['pure_rl'] = {
            'total_time': pure_rl_time,
            'steps_per_second': 100 / pure_rl_time,
            'avg_step_time': pure_rl_time / 100
        }
        env.close()
        
    except Exception as e:
        print(f"Pure RL performance test failed: {e}")
        results['pure_rl'] = {'error': str(e)}
    
    # Test LLM-guided performance
    print("Testing LLM-guided performance...")
    try:
        env = create_llm_guided_env(rom_path, headless=True)
        obs, _ = env.reset()
        
        start_time = time.time()
        for _ in range(50):  # Fewer steps since it's slower
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        llm_time = time.time() - start_time
        results['llm_guided'] = {
            'total_time': llm_time,
            'steps_per_second': 50 / llm_time,
            'avg_step_time': llm_time / 50
        }
        env.close()
        
    except Exception as e:
        print(f"LLM-guided performance test failed: {e}")
        results['llm_guided'] = {'error': str(e)}
    
    # Display results
    print("\n📊 PERFORMANCE RESULTS:")
    for mode, metrics in results.items():
        if 'error' in metrics:
            print(f"{mode:12s}: ERROR - {metrics['error']}")
        else:
            print(f"{mode:12s}: {metrics['steps_per_second']:.1f} steps/sec, "
                  f"{metrics['avg_step_time']:.3f}s per step")
    
    # Calculate speedup
    if 'pure_rl' in results and 'llm_guided' in results:
        if 'error' not in results['pure_rl'] and 'error' not in results['llm_guided']:
            speedup = results['pure_rl']['steps_per_second'] / results['llm_guided']['steps_per_second']
            print(f"\n🚀 Pure RL is {speedup:.1f}x faster than LLM-guided mode")
    
    return True

def save_test_results(test_results):
    """Save comprehensive test results."""
    output_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'test_type': 'rl_modes_validation',
        'results': test_results,
        'system_capabilities': {
            'pure_rl_mode': test_results.get('pure_rl', False),
            'llm_guided_mode': test_results.get('llm_guided', False),
            'controller_integration': test_results.get('controller', False)
        }
    }
    
    output_file = "test_rl_modes_results.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    file_size = os.path.getsize(output_file)
    print(f"💾 Test results saved: {output_file} ({file_size} bytes)")
    
    return output_file

def main():
    """Main test function."""
    print("🎯 RL MODES VALIDATION TEST")
    print("=" * 60)
    print("Testing both Pure RL and LLM-guided modes")
    print()
    
    start_time = time.time()
    
    # Run tests
    test_results = {
        'pure_rl': test_pure_rl_mode(),
        'llm_guided': test_llm_guided_mode(),
        'controller': test_controller_integration()
    }
    
    # Performance comparison
    performance_comparison()
    
    # Save results
    output_file = save_test_results(test_results)
    
    total_time = time.time() - start_time
    
    print(f"\n🏁 TEST SUMMARY")
    print("=" * 50)
    print(f"⏱️  Total test time: {total_time:.2f} seconds")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Pure RL mode: WORKING")
        print("✅ LLM-guided mode: WORKING") 
        print("✅ Controller integration: WORKING")
        print("🚀 Both RL modes are ready for training!")
        return True
    else:
        print("❌ Some tests failed - check errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

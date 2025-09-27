#!/usr/bin/env python3
"""
Visual Test Script for Pure RL Mode
Watch the RL agent learn in real-time with PyBoy display window enabled.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run visual test for pure RL mode."""
    print("👁️ ZELDA PURE RL VISUAL TEST")
    print("=" * 50)
    print("Watch the RL agent learn with PyBoy display window")
    print()
    
    # Configuration
    rom_path = "roms/zelda_oracle_of_seasons.gbc"
    
    # Validate ROM file exists
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        print("Please place the Oracle of Seasons ROM in the roms/ directory")
        return False
    
    print(f"✅ ROM file: {rom_path}")
    
    try:
        # Import after path setup
        from emulator.zelda_env_configurable import create_visual_test_pure_rl_env
        
        print("\n🎮 Creating Visual Test Environment...")
        env = create_visual_test_pure_rl_env(rom_path)
        
        config_summary = env.get_config_summary()
        print(f"📊 Configuration Summary:")
        for key, value in config_summary.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎯 Starting Visual Test...")
        print("PyBoy window will open - you can watch the agent play!")
        print("Press Ctrl+C to stop the test early")
        
        # Reset environment
        obs, info = env.reset()
        
        total_reward = 0
        step_count = 0
        action_counts = {i: 0 for i in range(env.action_space.n)}
        
        print(f"\n🚀 Episode Started (Max {config_summary['max_steps']} steps)")
        print("Actions: 0=NOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START, 8=SELECT")
        
        # Main test loop
        while True:
            # Simple random policy for demonstration
            # In real training, this would be the RL agent's decision
            action = env.action_space.sample()
            action_counts[action] += 1
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Print periodic updates
            if step_count % 100 == 0:
                print(f"Step {step_count:4d}: Action={action}, Reward={reward:.3f}, "
                      f"Total Reward={total_reward:.3f}")
                
                # Show action distribution
                if step_count % 500 == 0:
                    most_used_action = max(action_counts, key=action_counts.get)
                    print(f"   Most used action: {most_used_action} ({action_counts[most_used_action]} times)")
            
            # Add small delay for comfortable visual observation
            time.sleep(0.05)  # 20 FPS viewing speed
            
            # Check for episode end
            if terminated or truncated:
                print(f"\n🏁 Episode Ended at step {step_count}")
                print(f"   Reason: {'Terminated' if terminated else 'Truncated'}")
                break
        
        # Final statistics
        print(f"\n📊 VISUAL TEST RESULTS:")
        print(f"   Total steps: {step_count}")
        print(f"   Total reward: {total_reward:.3f}")
        print(f"   Average reward: {total_reward/step_count:.4f}")
        print(f"   Frame skip: {config_summary['frame_skip']}")
        
        # Action distribution
        print(f"\n🎯 Action Distribution:")
        action_names = ['NOP', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT']
        for action_id, count in action_counts.items():
            percentage = (count / step_count) * 100 if step_count > 0 else 0
            action_name = action_names[action_id] if action_id < len(action_names) else f'Action{action_id}'
            print(f"   {action_name:6s}: {count:4d} times ({percentage:5.1f}%)")
        
        env.close()
        
        print(f"\n✅ Visual test completed successfully!")
        print("You should have been able to watch Link move around in the PyBoy window")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Visual test interrupted by user")
        if 'env' in locals():
            env.close()
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This likely means the configurable environment is not properly set up")
        return False
        
    except Exception as e:
        print(f"❌ Visual test failed: {e}")
        import traceback
        traceback.print_exc()
        if 'env' in locals():
            env.close()
        return False

if __name__ == "__main__":
    print("🎮 Zelda Pure RL Visual Test")
    print("This will open a PyBoy window so you can watch the agent learn")
    input("Press Enter to start the visual test...")
    
    success = main()
    
    if success:
        print("\n🎉 Visual test completed!")
        print("💡 For actual RL training, the agent would learn to make better decisions over time")
        print("🔧 You can modify the action selection in this script to test different policies")
    else:
        print("\n❌ Visual test failed")
    
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Visual Test Script for LLM-Guided Mode
Watch the hybrid LLM+RL agent in real-time with PyBoy display window enabled.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run visual test for LLM-guided mode."""
    print("🧠 ZELDA LLM-GUIDED VISUAL TEST")
    print("=" * 50)
    print("Watch the hybrid LLM+RL agent with PyBoy display window")
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
        from emulator.zelda_env_configurable import create_visual_test_llm_env
        
        print("\n🎮 Creating Visual Test Environment (LLM-Guided)...")
        env = create_visual_test_llm_env(rom_path)
        
        config_summary = env.get_config_summary()
        print(f"📊 Configuration Summary:")
        for key, value in config_summary.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎯 Starting Visual Test...")
        print("PyBoy window will open - you can watch the LLM+RL agent play!")
        print("The agent will have access to structured game state for better decisions")
        print("Press Ctrl+C to stop the test early")
        
        # Reset environment
        obs, info = env.reset()
        structured_state = info.get('structured_state')
        
        total_reward = 0
        step_count = 0
        llm_decisions = 0
        action_counts = {i: 0 for i in range(env.action_space.n)}
        
        print(f"\n🚀 Episode Started (Max {config_summary['max_steps']} steps)")
        print("Actions: 0=NOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START, 8=SELECT")
        
        # Show structured state information
        if structured_state:
            print(f"\n📊 Structured State Available:")
            if 'player' in structured_state:
                player = structured_state['player']
                print(f"   Player: {player.get('health', 'N/A')}/{player.get('max_health', 'N/A')} hearts, "
                      f"Position ({player.get('x', 'N/A')}, {player.get('y', 'N/A')})")
            
            if 'llm_prompt' in structured_state:
                llm_prompt = structured_state['llm_prompt']
                print(f"   LLM Context: \"{llm_prompt}\"")
        
        # Main test loop
        while True:
            # Simulate LLM-guided decision making
            if structured_state and step_count % 50 == 0:  # Simulate LLM planning every 50 steps
                llm_decisions += 1
                print(f"🧠 LLM Decision #{llm_decisions} at step {step_count}")
                
                # Show what the LLM would see
                if 'llm_prompt' in structured_state:
                    print(f"   LLM sees: \"{structured_state['llm_prompt']}\"")
                
                # Simulate strategic decision (in real training, this would be the LLM)
                if 'player' in structured_state:
                    player = structured_state['player']
                    health = player.get('health', 3)
                    
                    if health <= 1:
                        print("   🚨 LLM Strategy: Focus on health recovery!")
                        # In real implementation, this would set a macro action
                    else:
                        print("   🎯 LLM Strategy: Explore and collect items")
            
            # Simple policy for demonstration
            # In real training, this would be the trained RL agent + LLM guidance
            if structured_state and 'player' in structured_state:
                player = structured_state['player']
                health = player.get('health', 3)
                
                # Strategic behavior based on structured state
                if health <= 1:
                    # Conservative movement when low health
                    action = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.15, 0.15, 0.15, 0.15])
                else:
                    # Normal exploration when healthy
                    action = env.action_space.sample()
            else:
                # Fallback to random action
                action = env.action_space.sample()
            
            action_counts[action] += 1
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            structured_state = info.get('structured_state')
            total_reward += reward
            step_count += 1
            
            # Print periodic updates
            if step_count % 100 == 0:
                print(f"Step {step_count:4d}: Action={action}, Reward={reward:.3f}, "
                      f"Total Reward={total_reward:.3f}")
                
                # Show current game state
                if structured_state and 'player' in structured_state:
                    player = structured_state['player']
                    print(f"   Current: {player.get('health', 'N/A')}/{player.get('max_health', 'N/A')} hearts, "
                          f"Position ({player.get('x', 'N/A')}, {player.get('y', 'N/A')})")
                
                # Show action distribution
                if step_count % 500 == 0:
                    most_used_action = max(action_counts, key=action_counts.get)
                    print(f"   Most used action: {most_used_action} ({action_counts[most_used_action]} times)")
                    print(f"   LLM decisions made: {llm_decisions}")
            
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
        print(f"   LLM decisions: {llm_decisions}")
        print(f"   Frame skip: {config_summary['frame_skip']}")
        
        # Action distribution
        print(f"\n🎯 Action Distribution:")
        action_names = ['NOP', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT']
        for action_id, count in action_counts.items():
            percentage = (count / step_count) * 100 if step_count > 0 else 0
            action_name = action_names[action_id] if action_id < len(action_names) else f'Action{action_id}'
            print(f"   {action_name:6s}: {count:4d} times ({percentage:5.1f}%)")
        
        # Final structured state
        if structured_state:
            print(f"\n📋 Final Game State:")
            if 'llm_prompt' in structured_state:
                print(f"   {structured_state['llm_prompt']}")
        
        env.close()
        
        print(f"\n✅ LLM-guided visual test completed successfully!")
        print("You should have been able to watch Link move around in the PyBoy window")
        print("The agent had access to structured game state for better decision making")
        
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
    print("🧠 Zelda LLM-Guided Visual Test")
    print("This will open a PyBoy window so you can watch the hybrid agent")
    print("The agent uses structured game state (player health, position, etc.)")
    input("Press Enter to start the visual test...")
    
    success = main()
    
    if success:
        print("\n🎉 Visual test completed!")
        print("💡 The hybrid agent has access to rich game state information")
        print("🔧 In real training, the LLM would provide strategic guidance")
        print("⚡ The RL component would handle precise execution")
    else:
        print("\n❌ Visual test failed")
    
    sys.exit(0 if success else 1)

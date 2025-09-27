#!/usr/bin/env python3
"""
Quick Visual RL Demo - 30 seconds of watching Link learn
Perfect for demonstrations and testing visual mode.
"""

import time
import random
import os
import sys
import numpy as np

def main():
    """Quick 30-second visual RL demo."""
    print("⚡ QUICK RL VISUAL DEMO (30 seconds)")
    print("=" * 50)
    
    rom_path = "roms/zelda_oracle_of_seasons.gbc"
    
    if not os.path.exists(rom_path):
        print(f"❌ ROM not found: {rom_path}")
        return False
    
    try:
        from pyboy import PyBoy
        
        print("🎮 Opening PyBoy window - you'll see Link moving around!")
        
        # Create PyBoy with visual window
        pyboy = PyBoy(rom_path, window="SDL2")
        
        # Load save state to skip intro
        save_state_path = rom_path + ".state"
        if os.path.exists(save_state_path):
            with open(save_state_path, 'rb') as f:
                pyboy.load_state(f)
            print("✅ Starting in playable area")
        
        print("\n🤖 SIMULATING RL TRAINING:")
        print("   You should see Link moving randomly at first")
        print("   Then more purposeful movement as 'learning' progresses")
        print("   Window will close automatically after 30 seconds")
        
        # Action mappings
        actions = [None, "up", "down", "left", "right", "a", "b", "start", "select"]
        action_names = ['WAIT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT']
        
        start_time = time.time()
        total_steps = 0
        last_action = 0
        action_duration = 0
        
        # 30-second demo loop
        while time.time() - start_time < 30:
            # Choose action based on progress
            elapsed = time.time() - start_time
            
            if elapsed < 10:
                # First 10 seconds: Random exploration
                action = random.randint(0, 8)
                phase = "Random Exploration"
            elif elapsed < 20:
                # Next 10 seconds: Favor movement
                if random.random() < 0.8:
                    action = random.randint(1, 4)  # Movement only
                else:
                    action = random.randint(0, 8)
                phase = "Learning Movement"
            else:
                # Last 10 seconds: More strategic
                weights = [0.1, 0.25, 0.25, 0.25, 0.25, 0.05, 0.05, 0.05, 0.05]
                action = np.random.choice(len(actions), p=weights)
                phase = "Strategic Play"
            
            # Execute action if it's time for a new one
            if action_duration <= 0:
                # Release previous action
                if last_action != 0 and last_action < len(actions) and actions[last_action] is not None:
                    pyboy.button_release(actions[last_action])
                
                # Press new action
                if action != 0 and action < len(actions) and actions[action] is not None:
                    pyboy.button_press(actions[action])
                
                last_action = action
                action_duration = random.randint(3, 8)  # Hold action for 3-8 frames
            
            action_duration -= 1
            
            # Advance game
            pyboy.tick()
            total_steps += 1
            
            # Print status updates
            if total_steps % 60 == 0:  # Every ~2 seconds at 30fps
                action_name = action_names[action] if action < len(action_names) else "UNKNOWN"
                print(f"[{elapsed:4.1f}s] {phase:18s} | Action: {action_name:6s}")
            
            # Control speed - ~30 FPS for smooth viewing
            time.sleep(0.033)
        
        # Release any held buttons
        if last_action != 0 and last_action < len(actions) and actions[last_action] is not None:
            pyboy.button_release(actions[last_action])
        
        print(f"\n✅ Demo completed! ({total_steps} steps in 30 seconds)")
        print("🎯 Link's behavior should have evolved from random to more purposeful")
        
        # Keep window open for 2 more seconds
        print("Closing in 2 seconds...")
        time.sleep(2)
        
        pyboy.stop()
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    print("⚡ Quick Visual RL Demo")
    print("Watch Link 'learn' for 30 seconds")
    input("Press Enter to start the demo...")
    
    success = main()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("💡 You should have seen Link's movement evolve over the 30 seconds")
        print("🔧 This simulates how an RL agent gradually learns better policies")
    else:
        print("\n❌ Demo failed")
    
    sys.exit(0 if success else 1)

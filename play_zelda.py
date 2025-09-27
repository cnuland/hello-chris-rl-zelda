#!/usr/bin/env python3
"""
Simple PyBoy launcher for manual Zelda gameplay.
Use this to create save states manually.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def main():
    """Launch PyBoy with Zelda for manual gameplay."""
    try:
        from pyboy import PyBoy
    except ImportError:
        print("❌ PyBoy not installed. Run: pip install pyboy")
        return
    
    # ROM path
    rom_path = os.path.join(project_root, "roms", "zelda_oracle_of_seasons.gbc")
    
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        print("Make sure zelda_oracle_of_seasons.gbc is in the roms/ directory")
        return
    
    print("🎮 Launching PyBoy with Zelda Oracle of Seasons...")
    print("📝 Instructions:")
    print("   - Play normally with keyboard controls")
    print("   - To create save state: Press Ctrl+S or use PyBoy menu")
    print("   - To load save state: Press Ctrl+L or use PyBoy menu")
    print("   - Save states will be saved as .state files")
    print("   - Close window when done")
    print()
    print("🎯 Goal: Navigate past all cutscenes to free gameplay area")
    print("   - Skip intro/story sequences")
    print("   - Get to a point where Link can freely explore")
    print("   - Save the state when you reach normal gameplay")
    print()
    
    # Launch PyBoy with SDL2 window
    pyboy = PyBoy(rom_path, window="SDL2")
    
    try:
        # Check if previous save state exists
        save_state_path = rom_path + ".state"
        if os.path.exists(save_state_path):
            print(f"📁 Found existing save state: {save_state_path}")
            print("   Loading it now...")
            with open(save_state_path, 'rb') as f:
                pyboy.load_state(f)
        
        print("▶️  Game started! Use PyBoy controls to play.")
        print("   Arrow keys: Move")
        print("   A key: A button") 
        print("   S key: B button")
        print("   Enter: Start")
        print("   Space: Select")
        print()
        
        # Keep game running until window is closed
        while not pyboy.tick():
            pass
            
    except KeyboardInterrupt:
        print("\n⏹️  Game stopped by user")
    except Exception as e:
        print(f"❌ Error during gameplay: {e}")
    finally:
        pyboy.stop()
        print("🏁 PyBoy closed. Save state should be preserved.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert Zelda save state from DMG mode to CGB mode.

This script loads an existing save state and re-saves it in CGB mode,
fixing the "Loading state which is not CGB, but PyBoy is loaded in CGB mode" error.
"""

import sys
from pathlib import Path
from pyboy import PyBoy

def convert_save_state_to_cgb(rom_path: str, old_state_path: str, new_state_path: str):
    """
    Convert a save state from DMG to CGB mode.
    
    Args:
        rom_path: Path to the .gbc ROM file
        old_state_path: Path to the existing DMG save state
        new_state_path: Path to save the new CGB save state
    """
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🔄 Converting Save State: DMG → CGB Mode")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Verify files exist
    if not Path(rom_path).exists():
        print(f"❌ ROM file not found: {rom_path}")
        sys.exit(1)
    
    if not Path(old_state_path).exists():
        print(f"❌ Save state not found: {old_state_path}")
        sys.exit(1)
    
    print(f"📁 ROM: {rom_path}")
    print(f"📁 Old save state: {old_state_path}")
    print(f"📁 New save state: {new_state_path}")
    print()
    
    try:
        # Step 1: Initialize PyBoy in CGB mode (auto-detect from .gbc ROM)
        print("1️⃣  Initializing PyBoy in CGB mode...")
        pyboy = PyBoy(
            rom_path,
            window="null",  # Headless
            debug=False
        )
        print("   ✅ PyBoy initialized in CGB mode")
        print()
        
        # Step 2: Load the old DMG save state
        print("2️⃣  Loading existing DMG save state...")
        try:
            with open(old_state_path, 'rb') as f:
                pyboy.load_state(f)
            print("   ✅ Save state loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load save state: {e}")
            print()
            print("   This might mean the save state is already in CGB mode,")
            print("   or there's a compatibility issue.")
            pyboy.stop()
            sys.exit(1)
        print()
        
        # Step 3: Tick a few frames to ensure state is stable
        print("3️⃣  Running a few frames to stabilize...")
        for _ in range(60):  # 1 second at 60 FPS
            pyboy.tick()
        print("   ✅ State stabilized")
        print()
        
        # Step 4: Save the state in CGB mode
        print("4️⃣  Saving new CGB save state...")
        with open(new_state_path, 'wb') as f:
            pyboy.save_state(f)
        print(f"   ✅ Saved to: {new_state_path}")
        print()
        
        # Step 5: Verify the new state works
        print("5️⃣  Verifying new save state...")
        pyboy.stop()
        
        # Reinitialize and test load
        pyboy = PyBoy(
            rom_path,
            window="null",
            debug=False
        )
        
        with open(new_state_path, 'rb') as f:
            pyboy.load_state(f)
        
        print("   ✅ New save state loads successfully in CGB mode!")
        print()
        
        # Clean up
        pyboy.stop()
        
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("✅ SUCCESS! Save State Converted to CGB Mode")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print()
        print("📋 Next Steps:")
        print(f"   1. Upload to S3:")
        print(f"      aws s3 cp {new_state_path} s3://roms/ \\")
        print(f"         --endpoint-url $S3_ENDPOINT_URL")
        print()
        print(f"   2. Or replace the old file:")
        print(f"      mv {new_state_path} {old_state_path}")
        print(f"      aws s3 cp {old_state_path} s3://roms/ \\")
        print(f"         --endpoint-url $S3_ENDPOINT_URL")
        print()
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Default paths
    rom_path = "roms/zelda_oracle_of_seasons.gbc"
    old_state_path = "roms/zelda_oracle_of_seasons.gbc.state"
    new_state_path = "roms/zelda_oracle_of_seasons_CGB.gbc.state"
    
    # Allow custom paths from command line
    if len(sys.argv) > 1:
        rom_path = sys.argv[1]
    if len(sys.argv) > 2:
        old_state_path = sys.argv[2]
    if len(sys.argv) > 3:
        new_state_path = sys.argv[3]
    
    convert_save_state_to_cgb(rom_path, old_state_path, new_state_path)


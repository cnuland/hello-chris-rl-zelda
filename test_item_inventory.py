#!/usr/bin/env python3
"""
Test script to monitor item inventory in Zelda Oracle of Seasons.

Usage:
    python test_item_inventory.py

Collect items manually and watch what the game detects.
Press Ctrl+C to exit.
"""

import os
import time
from pyboy import PyBoy

# Memory addresses
A_BUTTON_ITEM = 0xC681      # Item on A button
B_BUTTON_ITEM = 0xC680      # Item on B button
INVENTORY_START = 0xC682    # 16-byte inventory storage

# CORRECTED item mapping (based on ZeldaXtreme Gameshark codes)
# Note: Verified through manual testing with actual game
ITEM_NAMES = {
    0x00: 'None',
    0x01: 'Shield L1',
    0x03: 'Bombs',
    0x04: 'Sword L2',          # Level 2 Sword (Noble Sword) - unverified
    0x05: 'Wooden Sword',      # Level 1 Sword (Wooden Sword) - VERIFIED!
    0x06: 'Boomerang',
    0x07: 'Rod of Seasons',
    0x08: 'Magnetic Gloves',
    0x0A: 'Switch Hook',
    0x0C: 'Biggoron Sword',
    0x0D: 'Bombachu',
    0x0E: 'Wood Shield',
    0x13: 'Slingshot',
    0x14: 'Gnarled Key',       # From Maku Tree (decimal 20)
    0x15: 'Shovel',
    0x16: 'Power Bracelet',
    0x17: 'Roc\'s Feather',    # Dungeon 3 item (decimal 23)
    0x19: 'Seed Satchel',
}

def read_inventory(pyboy):
    """Read all 16 inventory slots plus A/B buttons."""
    a_button = pyboy.memory[A_BUTTON_ITEM]
    b_button = pyboy.memory[B_BUTTON_ITEM]
    inventory = []
    for i in range(16):
        inventory.append(pyboy.memory[INVENTORY_START + i])
    return a_button, b_button, tuple(inventory)

def format_item(item_id):
    """Format item ID with name."""
    name = ITEM_NAMES.get(item_id, f'Unknown')
    return f"{name} (0x{item_id:02X}/dec={item_id})"

def main():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎒 ITEM INVENTORY TEST - Oracle of Seasons")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("📋 Instructions:")
    print("   1. PyBoy window will open with the game")
    print("   2. Collect items manually")
    print("   3. Script will detect and log all inventory changes")
    print("   4. Press Ctrl+C to exit")
    print()
    print("🎮 Controls:")
    print("   Arrow Keys: Move")
    print("   Z: A button (sword/interact)")
    print("   X: B button (equipped item)")
    print("   Enter: Start (open menu)")
    print()
    
    # File paths
    rom_path = "roms/zelda_oracle_of_seasons.gbc"
    save_path = "roms/zelda_oracle_of_seasons.gbc.state"
    
    if not os.path.exists(rom_path):
        print(f"❌ ROM not found: {rom_path}")
        return
    
    print(f"📁 Loading ROM: {rom_path}")
    
    # Initialize PyBoy
    pyboy = PyBoy(
        rom_path,
        window="SDL2",
    )
    
    # Load save state if it exists
    if os.path.exists(save_path):
        print(f"💾 Loading save state: {save_path}")
        with open(save_path, "rb") as f:
            pyboy.load_state(f)
        print("✅ Save state loaded!")
    else:
        print("⚠️  No save state found, starting from ROM boot")
    
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎮 GAME STARTED - Collect some items!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Initial state
    last_a, last_b, last_inv = read_inventory(pyboy)
    
    print(f"📊 Initial Inventory:")
    print(f"   A Button: {format_item(last_a)}")
    print(f"   B Button: {format_item(last_b)}")
    print(f"   Inventory Slots:")
    for i, item_id in enumerate(last_inv):
        if item_id > 0:
            print(f"      Slot {i:2d}: {format_item(item_id)}")
    print()
    
    frame_count = 0
    
    try:
        while True:
            # Run one frame
            pyboy.tick()
            frame_count += 1
            
            # Check every 10 frames (6 times per second)
            if frame_count % 10 == 0:
                current_a, current_b, current_inv = read_inventory(pyboy)
                
                # Check A button change
                if current_a != last_a:
                    print(f"🔄 A BUTTON CHANGED: {format_item(last_a)} → {format_item(current_a)}")
                    last_a = current_a
                
                # Check B button change
                if current_b != last_b:
                    print(f"🔄 B BUTTON CHANGED: {format_item(last_b)} → {format_item(current_b)}")
                    last_b = current_b
                
                # Check inventory changes
                if current_inv != last_inv:
                    print(f"\n🎁 INVENTORY CHANGED!")
                    for i, (old_id, new_id) in enumerate(zip(last_inv, current_inv)):
                        if old_id != new_id:
                            if new_id > 0 and old_id == 0:
                                # New item acquired
                                print(f"   ✨ Slot {i:2d}: ACQUIRED {format_item(new_id)}")
                            elif new_id == 0 and old_id > 0:
                                # Item removed/used up
                                print(f"   ❌ Slot {i:2d}: REMOVED {format_item(old_id)}")
                            else:
                                # Item changed
                                print(f"   🔄 Slot {i:2d}: {format_item(old_id)} → {format_item(new_id)}")
                    
                    # Show full inventory snapshot
                    print(f"   📦 Current Inventory:")
                    print(f"      A Button: {format_item(current_a)}")
                    print(f"      B Button: {format_item(current_b)}")
                    for i, item_id in enumerate(current_inv):
                        if item_id > 0:
                            print(f"      Slot {i:2d}: {format_item(item_id)}")
                    print()
                    
                    last_inv = current_inv
                
                # Print periodic status (every 600 frames = ~10 seconds)
                if frame_count % 600 == 0:
                    non_empty = sum(1 for x in current_inv if x > 0)
                    print(f"📊 Status: Frame={frame_count}, Items in inventory: {non_empty}/16, "
                          f"A={format_item(current_a)}, B={format_item(current_b)}")
    
    except KeyboardInterrupt:
        print()
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🛑 Test stopped by user")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Final inventory
        final_a, final_b, final_inv = read_inventory(pyboy)
        print()
        print(f"📦 Final Inventory:")
        print(f"   A Button: {format_item(final_a)}")
        print(f"   B Button: {format_item(final_b)}")
        print(f"   Inventory Slots:")
        for i, item_id in enumerate(final_inv):
            if item_id > 0:
                print(f"      Slot {i:2d}: {format_item(item_id)}")
        print()
    
    finally:
        pyboy.stop()
        print("✅ PyBoy stopped")

if __name__ == "__main__":
    main()


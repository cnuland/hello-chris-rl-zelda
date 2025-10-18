#!/usr/bin/env python3
"""
Test script to detect enemy kills in Zelda Oracle of Seasons.

Usage:
    python test_enemy_kill_detection.py

Play the game manually and kill enemies. The script will detect and log kills.
Press Ctrl+C to exit.
"""

import os
import time
from pyboy import PyBoy

# Memory addresses for enemy tracking
ENEMIES_KILLED = 0xC620      # Cumulative enemies killed (2 bytes, little-endian)
ENEMIES_ON_SCREEN = 0xCC30   # Count of active enemies on screen
PLAYER_HEALTH = 0xC6A2       # Current health (quarter-hearts)
PLAYER_ROOM = 0xC63B         # Current room ID

def read_u16_le(pyboy, addr):
    """Read 2-byte little-endian value from memory."""
    low = pyboy.memory[addr]
    high = pyboy.memory[addr + 1]
    return low + (high << 8)

def main():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🗡️  ENEMY KILL DETECTION TEST - Oracle of Seasons")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("📋 Instructions:")
    print("   1. PyBoy window will open with the game")
    print("   2. Play manually and kill enemies")
    print("   3. Script will detect and log kills")
    print("   4. Press Ctrl+C to exit")
    print()
    print("🎮 Controls:")
    print("   Arrow Keys: Move")
    print("   Z: A button (sword/interact)")
    print("   X: B button (equipped item)")
    print("   Enter: Start")
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
        window="SDL2",  # Use SDL2 for better window support
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
    print("🎮 GAME STARTED - Kill some enemies!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Initial state
    last_total_kills = read_u16_le(pyboy, ENEMIES_KILLED)
    last_enemies_on_screen = pyboy.memory[ENEMIES_ON_SCREEN]
    last_room = pyboy.memory[PLAYER_ROOM]
    frame_count = 0
    
    print(f"📊 Initial stats:")
    print(f"   Total Kills: {last_total_kills}")
    print(f"   Enemies on screen: {last_enemies_on_screen}")
    print(f"   Room: 0x{last_room:02X} (dec {last_room})")
    print()
    
    try:
        while True:
            # Run one frame
            pyboy.tick()
            frame_count += 1
            
            # Check every 10 frames (6 times per second)
            if frame_count % 10 == 0:
                # Read current values
                current_total_kills = read_u16_le(pyboy, ENEMIES_KILLED)
                current_enemies_on_screen = pyboy.memory[ENEMIES_ON_SCREEN]
                current_room = pyboy.memory[PLAYER_ROOM]
                current_health = pyboy.memory[PLAYER_HEALTH]
                
                # Detect total kill count increase
                if current_total_kills > last_total_kills:
                    kills_gained = current_total_kills - last_total_kills
                    print(f"⚔️  ENEMY KILLED! Total: {last_total_kills} → {current_total_kills} (+{kills_gained})")
                    print(f"   Room: 0x{current_room:02X}, Enemies left on screen: {current_enemies_on_screen}")
                    last_total_kills = current_total_kills
                
                # Detect enemies on screen change (spawn/despawn)
                if current_enemies_on_screen != last_enemies_on_screen:
                    if current_enemies_on_screen < last_enemies_on_screen:
                        print(f"💀 Enemy despawned: {last_enemies_on_screen} → {current_enemies_on_screen}")
                    else:
                        print(f"👹 Enemy spawned: {last_enemies_on_screen} → {current_enemies_on_screen}")
                    last_enemies_on_screen = current_enemies_on_screen
                
                # Detect room change
                if current_room != last_room:
                    print(f"🚪 Room changed: 0x{last_room:02X} → 0x{current_room:02X}")
                    print(f"   New room enemies: {current_enemies_on_screen}")
                    last_room = current_room
                    last_enemies_on_screen = current_enemies_on_screen
                
                # Print periodic status (every 600 frames = ~10 seconds)
                if frame_count % 600 == 0:
                    hearts = current_health / 4.0
                    print(f"📊 Status: Room=0x{current_room:02X}, HP={hearts:.1f}, "
                          f"Enemies={current_enemies_on_screen}, Total Kills={current_total_kills}")
    
    except KeyboardInterrupt:
        print()
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🛑 Test stopped by user")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Final stats
        final_total_kills = read_u16_le(pyboy, ENEMIES_KILLED)
        print()
        print(f"📊 Final Stats:")
        print(f"   Total Kills (lifetime): {final_total_kills}")
        print(f"   Kills during test: {final_total_kills - last_total_kills}")
        print()
    
    finally:
        pyboy.stop()
        print("✅ PyBoy stopped")

if __name__ == "__main__":
    main()


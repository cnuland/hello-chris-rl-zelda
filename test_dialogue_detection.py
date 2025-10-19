#!/usr/bin/env python3
"""
Test script to detect dialogue vs dialogue choice menus.
Use this to figure out memory addresses for different dialogue states.
"""

import os
from pyboy import PyBoy
import time

rom_path = "roms/zelda_oracle_of_seasons.gbc"
save_path = "roms/zelda_oracle_of_seasons.gbc.state"

# Memory addresses to monitor
CUTSCENE_INDEX = 0xC2EF  # Dialogue/cutscene state
MENU_STATE = 0xD700      # Menu state
ITEM_SELECTED = 0xD701   # Item/choice selected

pyboy = PyBoy(rom_path, window="SDL2")

if os.path.exists(save_path):
    with open(save_path, "rb") as f:
        pyboy.load_state(f)
    print("✅ Save loaded")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 DIALOGUE DETECTION TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INSTRUCTIONS:
1. Play and talk to NPCs
2. Watch the console for memory values
3. Note when you see:
   - Regular dialogue text
   - Yes/No choice menu
   - What the memory addresses show

MEMORY ADDRESSES:
  0xC2EF (CUTSCENE_INDEX) - Dialogue state
  0xD700 (MENU_STATE) - Menu state  
  0xD701 (ITEM_SELECTED) - Selection cursor

Press Ctrl+C to exit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

frame = 0
last_cutscene = 0
last_menu = 0
last_selected = 0

try:
    while True:
        pyboy.tick()
        frame += 1
        
        if frame % 10 == 0:  # Check every 10 frames
            cutscene = pyboy.memory[CUTSCENE_INDEX]
            menu = pyboy.memory[MENU_STATE]
            selected = pyboy.memory[ITEM_SELECTED]
            
            # Detect changes
            if cutscene != last_cutscene or menu != last_menu or selected != last_selected:
                print(f"[{frame:6d}] CUTSCENE=0x{cutscene:02X} MENU=0x{menu:02X} SELECT=0x{selected:02X}", end="")
                
                # Interpret state
                if cutscene > 0 and menu == 0:
                    print(" → 💬 DIALOGUE TEXT")
                elif cutscene > 0 and menu > 0:
                    print(" → 🎯 DIALOGUE CHOICE MENU")
                elif menu > 0:
                    print(" → 📋 REGULAR MENU")
                else:
                    print(" → 🎮 GAMEPLAY")
                
                last_cutscene = cutscene
                last_menu = menu
                last_selected = selected

except KeyboardInterrupt:
    print("\n✅ Test stopped")
finally:
    pyboy.stop()

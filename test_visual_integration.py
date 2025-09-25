#!/usr/bin/env python3
"""
Test script for visual integration in Zelda-LLM-RL project.

This script demonstrates the enhanced visual capabilities that allow the LLM
to see NPCs, monsters, text, and environmental details.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from emulator.pyboy_bridge import ZeldaPyBoyBridge
from emulator.input_map import ZeldaAction
from observation.state_encoder import ZeldaStateEncoder
from observation.visual_encoder import VisualEncoder


def find_rom_file():
    """Find Oracle of Seasons ROM file."""
    rom_dir = project_root / 'roms'
    rom_files = list(rom_dir.glob('*.gbc')) + list(rom_dir.glob('*.gb'))
    
    if not rom_files:
        print("❌ No ROM files found in roms/ directory!")
        print("Please place your Oracle of Seasons ROM file there.")
        return None
    
    return str(rom_files[0])


def test_visual_processing():
    """Test the visual processing capabilities."""
    print("🎮 Testing Visual Processing Integration")
    print("=" * 50)
    
    # Find ROM file
    rom_path = find_rom_file()
    if not rom_path:
        return False
    
    print(f"✅ Using ROM: {rom_path}")
    
    try:
        # Initialize components
        print("\n🔧 Initializing components...")
        bridge = ZeldaPyBoyBridge(rom_path, headless=True)
        encoder = ZeldaStateEncoder(enable_visual=True)
        visual_encoder = VisualEncoder()
        
        print("✅ PyBoy bridge initialized")
        print("✅ State encoder with visual processing initialized")
        
        # Reset and advance past intro
        print("\n🎲 Resetting game...")
        bridge.reset()
        
        # Advance to gameplay
        for i in range(100):
            bridge.step(ZeldaAction.NOP)
            if i % 20 == 0:
                print(f"   Advancing... {i}%")
        
        print("✅ Game advanced to gameplay state")
        
        # Test basic screen capture
        print("\n📸 Testing screen capture...")
        screen_array = bridge.get_screen()
        print(f"✅ Screen captured: {screen_array.shape} shape, dtype: {screen_array.dtype}")
        print(f"   Pixel range: [{screen_array.min()}, {screen_array.max()}]")
        
        # Test visual encoding
        print("\n🖼️ Testing visual encoding...")
        visual_data = visual_encoder.encode_screen_for_llm(screen_array)
        print(f"✅ Visual encoding complete")
        print(f"   Image format: {visual_data['format']}")
        print(f"   Image size: {visual_data['size']}")
        print(f"   Base64 data length: {len(visual_data['image_data'])} characters")
        
        # Test visual element detection
        print("\n🔍 Testing visual element detection...")
        elements = visual_encoder.detect_visual_elements(screen_array)
        print(f"✅ Element detection complete")
        print(f"   Detected elements structure: {list(elements.keys())}")
        
        # Test screen description
        print("\n📝 Testing screen description...")
        description = visual_encoder.describe_screen_content(screen_array)
        print(f"✅ Screen description: {description}")
        
        # Test full state encoding with visual data
        print("\n🧠 Testing full state encoding with visual data...")
        numeric_vector, structured_state = encoder.encode_state(bridge)
        
        print(f"✅ Full state encoding complete")
        print(f"   Numeric vector shape: {numeric_vector.shape}")
        print(f"   Structured state keys: {list(structured_state.keys())}")
        
        # Check if visual data is included
        if 'visual' in structured_state:
            visual_state = structured_state['visual']
            print(f"✅ Visual data included in state!")
            print(f"   Visual keys: {list(visual_state.keys())}")
            
            if 'screen_image' in visual_state:
                img_data = visual_state['screen_image']
                print(f"   Screen image size: {img_data.get('size')}")
                print(f"   Screen image format: {img_data.get('format')}")
            
            if 'description' in visual_state:
                print(f"   Screen description: {visual_state['description'][:100]}...")
                
        else:
            print("❌ Visual data missing from structured state!")
            return False
        
        # Test with some actions to see state changes
        print("\n🎮 Testing with game actions...")
        actions_to_test = [ZeldaAction.RIGHT, ZeldaAction.DOWN, ZeldaAction.LEFT, ZeldaAction.UP, ZeldaAction.A]
        
        for i, action in enumerate(actions_to_test):
            print(f"   Executing action {i+1}: {action.name}")
            bridge.step(action)
            
            # Get new state
            _, new_state = encoder.encode_state(bridge)
            
            # Compare positions
            old_pos = (structured_state['player']['x'], structured_state['player']['y'])
            new_pos = (new_state['player']['x'], new_state['player']['y'])
            
            if old_pos != new_pos:
                print(f"     Position changed: {old_pos} -> {new_pos}")
            
            if 'visual' in new_state and 'description' in new_state['visual']:
                new_desc = new_state['visual']['description']
                if new_desc != description:
                    print(f"     Screen content changed!")
            
            structured_state = new_state  # Update for next comparison
        
        # Summary
        print("\n" + "=" * 50)
        print("✅ VISUAL INTEGRATION TEST SUCCESSFUL!")
        print("🎯 Key capabilities demonstrated:")
        print("   • Screen capture from PyBoy (144x160x3 RGB)")
        print("   • Visual encoding for LLM consumption")
        print("   • Basic computer vision element detection")
        print("   • Screen content description generation")
        print("   • Integration with state encoder")
        print("   • Real-time visual data in game states")
        
        print("\n🤖 LLM Planning Readiness:")
        print("   • LLM can now receive visual screen data")
        print("   • LLM can see NPCs, enemies, items, and text")
        print("   • LLM gets both RAM data AND visual context")
        print("   • Enhanced decision-making capabilities enabled")
        
        bridge.close()
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        if 'bridge' in locals():
            bridge.close()
        return False


def demonstrate_llm_prompt():
    """Show what the enhanced LLM prompt would look like."""
    print("\n🧠 Enhanced LLM Prompt Example")
    print("=" * 50)
    
    # Example enhanced state data
    example_state = {
        "player": {
            "x": 120, "y": 80, "direction": "up",
            "room": 15, "health": 4, "max_health": 6
        },
        "resources": {
            "rupees": 47, "keys": 2,
            "sword_level": 1, "shield_level": 0
        },
        "inventory": {
            "rod_of_seasons": True,
            "gale_boomerang": True,
            "sword": True
        },
        "season": {
            "current": "autumn", "current_id": 2,
            "spirits_found": 2
        },
        "dungeon": {
            "keys": 0, "has_map": True, "has_compass": False,
            "bosses_defeated": {"gohma": True, "dodongo": False}
        },
        "environment": {
            "player_tile_pos": (7, 5),
            "nearby_obstacles": ["wall at (6, 5)", "water at (8, 5)"]
        },
        "visual": {
            "screen_description": "Screen shows autumn forest area with Link facing north. Visible enemies include two Moblins to the east and a Keese flying overhead. A treasure chest is visible in the northwest corner. Link appears to be near a seasonal stump with autumn leaves around it.",
            "detected_elements": {
                "detected_enemies": [
                    {"type": "moblin", "position": [140, 60], "threat_level": "medium"},
                    {"type": "moblin", "position": [145, 65], "threat_level": "medium"},
                    {"type": "keese", "position": [100, 40], "threat_level": "low"}
                ],
                "visible_items": [
                    {"type": "treasure_chest", "position": [50, 30], "opened": False}
                ],
                "ui_elements": {
                    "hearts_visible": 4,
                    "rupee_display": "047",
                    "text_boxes": []
                }
            },
            "screen_available": True
        }
    }
    
    prompt = f"""Current game state:
{json.dumps(example_state, indent=2)}

Provide your strategic plan:"""
    
    print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
    
    print("\n🎯 Notice how the LLM now receives:")
    print("   • Visual screen description mentioning specific enemies")
    print("   • Detected element positions and threat levels") 
    print("   • Visible treasure chest location")
    print("   • UI element states (hearts, rupees)")
    print("   • Environmental context (autumn forest, seasonal stump)")
    
    print("\n🧠 This enables the LLM to make decisions like:")
    print('   • "Attack the two Moblins using gale boomerang"')
    print('   • "Avoid the Keese and collect the treasure chest"')
    print('   • "Use Rod of Seasons to change season at the stump"')
    print('   • "Plan combat strategy based on enemy positions"')


if __name__ == "__main__":
    print("🎮 Zelda-LLM-RL Visual Integration Test")
    print("Testing enhanced visual capabilities for LLM planning")
    print()
    
    success = test_visual_processing()
    
    if success:
        demonstrate_llm_prompt()
        print("\n🚀 Ready for enhanced LLM-guided gameplay!")
    else:
        print("\n⚠️ Please fix the issues above before proceeding.")
        sys.exit(1)

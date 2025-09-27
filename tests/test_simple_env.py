#!/usr/bin/env python3
"""
Simple Local Environment Test for Zelda Oracle of Seasons
Tests basic functionality without complex imports.
"""

import os
import sys
import time
import json
from pathlib import Path

def test_imports():
    """Test basic imports to see what's working."""
    print("🧪 TESTING IMPORTS")
    print("=" * 40)
    
    # Test basic dependencies
    try:
        import numpy as np
        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    try:
        import gymnasium as gym
        print("✅ Gymnasium available")
    except ImportError:
        print("❌ Gymnasium not available")
        return False
        
    try:
        from pyboy import PyBoy
        print("✅ PyBoy available")
    except ImportError:
        print("❌ PyBoy not available")
        return False
    
    return True

def test_rom_file():
    """Test if ROM file exists."""
    print("\n📁 TESTING ROM FILE")
    print("=" * 40)
    
    rom_path = "roms/zelda_oracle_of_seasons.gbc"
    
    if os.path.exists(rom_path):
        file_size = os.path.getsize(rom_path)
        print(f"✅ ROM file found: {rom_path}")
        print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        return True
    else:
        print(f"❌ ROM file not found: {rom_path}")
        print("Please ensure the ROM file is placed in the roms/ directory")
        return False

def test_pyboy_basic():
    """Test basic PyBoy functionality."""
    print("\n🎮 TESTING PYBOY BASIC")
    print("=" * 40)
    
    try:
        from pyboy import PyBoy
        from pyboy.utils import WindowEvent
        
        rom_path = "roms/zelda_oracle_of_seasons.gbc"
        
        if not os.path.exists(rom_path):
            print("❌ Cannot test PyBoy without ROM file")
            return False
        
        print("Creating PyBoy instance...")
        pyboy = PyBoy(rom_path, window="null")
        
        print("✅ PyBoy created successfully")
        print(f"📊 Game title: {pyboy.game_wrapper.cartridge_title}")
        print(f"🎯 Window type: null (headless)")
        
        # Test basic functionality
        print("Testing basic operations...")
        
        # Get initial screen
        screen = pyboy.screen.ndarray
        print(f"📺 Screen shape: {screen.shape}")
        
        # Test memory access
        memory_value = pyboy.memory[0x0000]
        print(f"🧠 Memory test (0x0000): {memory_value}")
        
        # Test a few steps
        for i in range(10):
            pyboy.tick()
        
        print("✅ Basic PyBoy operations successful")
        
        pyboy.stop()
        return True
        
    except Exception as e:
        print(f"❌ PyBoy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_addresses():
    """Test memory address reading."""
    print("\n🧠 TESTING MEMORY ADDRESSES")
    print("=" * 40)
    
    try:
        from pyboy import PyBoy
        
        rom_path = "roms/zelda_oracle_of_seasons.gbc"
        
        if not os.path.exists(rom_path):
            print("❌ Cannot test memory without ROM file")
            return False
        
        pyboy = PyBoy(rom_path, window="null")
        
        # Skip some frames to let game initialize
        for _ in range(1000):
            pyboy.tick()
        
        # Test key memory addresses
        test_addresses = {
            "Player Health": 0xC021,
            "Player Max Health": 0xC05B, 
            "Rupees": 0xC6A5,
            "Shield Level": 0xC6A6,
            "Overworld Position": 0xC63B
        }
        
        memory_data = {}
        
        for name, address in test_addresses.items():
            try:
                value = pyboy.memory[address]
                memory_data[name] = value
                print(f"✅ {name:20s}: 0x{address:04X} = {value:3d}")
            except Exception as addr_error:
                print(f"❌ {name:20s}: 0x{address:04X} = ERROR ({addr_error})")
        
        pyboy.stop()
        
        # Save memory test results
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "memory_address_test",
            "memory_values": memory_data,
            "test_successful": len(memory_data) > 0
        }
        
        output_file = "test_memory_results.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Memory test results saved: {output_file}")
        
        return len(memory_data) > 0
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_save_state():
    """Test save state loading."""
    print("\n💾 TESTING SAVE STATE")
    print("=" * 40)
    
    try:
        from pyboy import PyBoy
        
        rom_path = "roms/zelda_oracle_of_seasons.gbc"
        save_state_path = rom_path + ".state"
        
        if not os.path.exists(rom_path):
            print("❌ Cannot test save state without ROM file")
            return False
        
        pyboy = PyBoy(rom_path, window="null")
        
        if os.path.exists(save_state_path):
            print(f"📁 Save state found: {save_state_path}")
            
            try:
                with open(save_state_path, 'rb') as f:
                    pyboy.load_state(f)
                
                print("✅ Save state loaded successfully")
                
                # Test that we can read memory after loading
                health = pyboy.memory[0xC021]
                max_health = pyboy.memory[0xC05B] 
                hearts = health // 4
                max_hearts = max_health // 4
                
                print(f"❤️  Hearts after save state load: {hearts}/{max_hearts}")
                
                pyboy.stop()
                return True
                
            except Exception as load_error:
                print(f"❌ Save state load failed: {load_error}")
                pyboy.stop()
                return False
                
        else:
            print(f"⚠️  Save state not found: {save_state_path}")
            print("Save state loading test skipped")
            pyboy.stop()
            return True  # Not a failure, just not available
            
    except Exception as e:
        print(f"❌ Save state test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🎯 SIMPLE ZELDA ENVIRONMENT TEST")
    print("=" * 50)
    print("Testing basic components without complex imports")
    print()
    
    start_time = time.time()
    
    test_results = {
        "imports": test_imports(),
        "rom_file": test_rom_file(),
        "pyboy_basic": False,
        "memory_addresses": False, 
        "save_state": False
    }
    
    if test_results["imports"] and test_results["rom_file"]:
        test_results["pyboy_basic"] = test_pyboy_basic()
        
        if test_results["pyboy_basic"]:
            test_results["memory_addresses"] = test_memory_addresses()
            test_results["save_state"] = test_save_state()
    
    total_time = time.time() - start_time
    
    print(f"\n🏁 TEST SUMMARY")
    print("=" * 50)
    print(f"⏱️  Total test time: {total_time:.2f} seconds")
    print()
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Core functionality is working.")
        return True
    elif passed_tests >= 3:  # imports, rom, pyboy basic
        print("⚠️  Core functionality working, some advanced features may need attention.")
        return True
    else:
        print("❌ Critical issues detected. Please check dependencies and ROM file.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

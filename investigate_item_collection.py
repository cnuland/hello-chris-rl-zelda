#!/usr/bin/env python3
"""
COMPREHENSIVE ITEM COLLECTION INVESTIGATION
🔍 Deep investigation into why no items are being collected
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment
from emulator.input_map import ZeldaAction

def test_save_state_starting_conditions():
    """Test the exact starting conditions in the save state"""
    print("🎮 SAVE STATE STARTING CONDITIONS TEST")
    print("=" * 60)
    
    env = ZeldaConfigurableEnvironment(
        rom_path=str(project_root / "roms" / "zelda_oracle_of_seasons.gbc"),
        headless=True,
        visual_test_mode=False,
        config_dict={
            "controller": {"use_planner": True, "enable_visual": True},
            "environment": {"max_episode_steps": 100, "frame_skip": 4}
        }
    )
    
    obs, info = env.reset()
    
    if info.get('structured_state'):
        state = info['structured_state']
        player = state.get('player', {})
        resources = state.get('resources', {})
        world = state.get('world', {})
        
        print("📍 STARTING LOCATION:")
        print(f"   Room ID: {player.get('room', 'Unknown')}")
        print(f"   Position: ({player.get('x', 'Unknown')}, {player.get('y', 'Unknown')})")
        print(f"   Direction: {player.get('direction', 'Unknown')}")
        print(f"   Overworld position: {world.get('overworld_position', 'Unknown')}")
        print()
        
        print("💰 STARTING RESOURCES:")
        print(f"   Rupees: {resources.get('rupees', 0)}")
        print(f"   Keys: {resources.get('keys', 0)}")
        print(f"   Bombs: {resources.get('current_bombs', 0)} / {resources.get('max_bombs', 0)}")
        print(f"   Health: {player.get('current_health', 0)} / {player.get('max_health', 0)} hearts")
        print()
        
        print("🌍 STARTING WORLD STATE:")
        print(f"   Level bank: {world.get('level_bank', 'Unknown')}")
        print(f"   Dungeon floor: {world.get('dungeon_floor', 0)}")
        print(f"   Season: {state.get('season', {}).get('current', 'Unknown')}")
        print()
    
    env.close()

def analyze_action_distribution():
    """Analyze what actions the agent is actually taking"""
    print("🎯 ACTION DISTRIBUTION ANALYSIS")  
    print("=" * 60)
    
    env = ZeldaConfigurableEnvironment(
        rom_path=str(project_root / "roms" / "zelda_oracle_of_seasons.gbc"),
        headless=True,
        visual_test_mode=False,
        config_dict={
            "controller": {"use_planner": True, "enable_visual": True},
            "environment": {"max_episode_steps": 500, "frame_skip": 4}
        }
    )
    
    action_counts = {}
    action_names = [action.name for action in ZeldaAction]
    
    for action_name in action_names:
        action_counts[action_name] = 0
    
    obs, info = env.reset()
    
    # Track actions for 500 steps
    for step in range(500):
        # Use random policy like the training
        action = np.random.randint(0, env.action_space.n)
        action_name = ZeldaAction(action).name
        action_counts[action_name] += 1
        
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            break
    
    print("📊 ACTION DISTRIBUTION (500 random steps):")
    total_actions = sum(action_counts.values())
    
    for action_name, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_actions * 100
        print(f"   {action_name:15}: {count:3d} ({percentage:5.1f}%)")
    
    # Check if item pickup actions are being used
    pickup_actions = ['A', 'B']  # A and B buttons for pickup/interaction
    pickup_total = sum(action_counts.get(action, 0) for action in pickup_actions)
    pickup_percentage = pickup_total / total_actions * 100
    
    print()
    print(f"🎯 ITEM PICKUP ACTIONS:")
    print(f"   Total pickup actions: {pickup_total} ({pickup_percentage:.1f}%)")
    
    if pickup_percentage < 5:
        print("   ⚠️  Very few pickup actions! This could explain lack of item collection.")
    elif pickup_percentage > 20:
        print("   ✅ Reasonable number of pickup actions.")
    else:
        print("   ❓ Moderate pickup action usage.")
    
    env.close()

def test_manual_item_search():
    """Manually explore the starting area looking for items"""
    print("🔍 MANUAL ITEM SEARCH IN STARTING AREA")
    print("=" * 60)
    
    env = ZeldaConfigurableEnvironment(
        rom_path=str(project_root / "roms" / "zelda_oracle_of_seasons.gbc"),
        headless=True,
        visual_test_mode=False,
        config_dict={
            "controller": {"use_planner": True, "enable_visual": True},
            "environment": {"max_episode_steps": 2000, "frame_skip": 4}
        }
    )
    
    obs, info = env.reset()
    
    # Systematic exploration pattern
    movement_pattern = [
        # Explore initial area thoroughly
        (ZeldaAction.UP, 20),
        (ZeldaAction.RIGHT, 20),
        (ZeldaAction.DOWN, 20),
        (ZeldaAction.LEFT, 20),
        (ZeldaAction.UP, 40),
        (ZeldaAction.RIGHT, 40),
        (ZeldaAction.DOWN, 40),
        (ZeldaAction.LEFT, 40),
        # Try to move to adjacent screens
        (ZeldaAction.RIGHT, 60),
        (ZeldaAction.UP, 60),
        (ZeldaAction.LEFT, 60),
        (ZeldaAction.DOWN, 60),
    ]
    
    step_count = 0
    rooms_visited = set()
    resource_changes = []
    
    for action, duration in movement_pattern:
        for _ in range(duration):
            # Execute movement
            obs, reward, done, truncated, info = env.step(action.value)
            step_count += 1
            
            if info.get('structured_state'):
                state = info['structured_state']
                player = state.get('player', {})
                resources = state.get('resources', {})
                
                room_id = player.get('room', 0)
                rooms_visited.add(room_id)
                
                # Check for resource changes
                current_resources = {
                    'rupees': resources.get('rupees', 0),
                    'keys': resources.get('keys', 0),
                    'bombs': resources.get('current_bombs', 0)
                }
                
                # Log any resource changes
                if step_count == 1:
                    last_resources = current_resources.copy()
                elif current_resources != last_resources:
                    change = {}
                    for key in current_resources:
                        if current_resources[key] != last_resources[key]:
                            change[key] = f"{last_resources[key]} → {current_resources[key]}"
                    if change:
                        resource_changes.append((step_count, change))
                        print(f"   💰 Step {step_count}: Resource change! {change}")
                    last_resources = current_resources.copy()
            
            # Try item pickup actions occasionally
            if step_count % 50 == 0:
                # Try A button (primary pickup)
                obs, reward, done, truncated, info = env.step(ZeldaAction.A.value)
                step_count += 1
            
            if done or truncated:
                break
        
        if done or truncated:
            break
    
    print(f"🗺️  EXPLORATION RESULTS:")
    print(f"   Steps taken: {step_count}")
    print(f"   Rooms visited: {sorted(list(rooms_visited))}")
    print(f"   Resource changes detected: {len(resource_changes)}")
    
    if resource_changes:
        print("   💰 Resource changes:")
        for step, change in resource_changes:
            print(f"      Step {step}: {change}")
    else:
        print("   ❌ No resource changes detected")
    
    # Final state check
    if info.get('structured_state'):
        final_resources = info['structured_state'].get('resources', {})
        print(f"   Final resources: {final_resources.get('rupees', 0)} rupees, {final_resources.get('keys', 0)} keys, {final_resources.get('current_bombs', 0)} bombs")
    
    env.close()

def analyze_zelda_game_mechanics():
    """Analyze Zelda game mechanics for item collection"""
    print("🎮 ZELDA GAME MECHANICS ANALYSIS")
    print("=" * 60)
    
    print("💡 ZELDA ORACLE OF SEASONS ITEM MECHANICS:")
    print("   1. 💰 Rupees: Dropped by enemies, found in grass, hidden in rocks")
    print("   2. 🔑 Keys: Found in dungeons, dropped by specific enemies")
    print("   3. 💣 Bombs: Purchased from shops, found in specific locations")
    print("   4. 🌿 Items in grass: Need to cut/destroy grass with sword")
    print("   5. 🪨 Items in rocks: Need to lift/break rocks")
    print("   6. 👹 Enemy drops: Kill enemies to get rupees/items")
    print()
    
    print("🎯 REQUIRED ACTIONS FOR ITEM COLLECTION:")
    print("   1. 🗡️  Attack enemies (SWORD button)")
    print("   2. 🤏 Pick up items (A button near item)")
    print("   3. ✂️  Cut grass (SWORD button)")
    print("   4. 🪨 Lift rocks (A button near rock)")
    print("   5. 💥 Break objects (various tools)")
    print()
    
    print("🚨 LIKELY ISSUES:")
    print("   1. ❌ Save state area may have no items")
    print("   2. ❌ Random policy doesn't generate item-revealing actions")
    print("   3. ❌ No enemy killing = no enemy drops")
    print("   4. ❌ No grass cutting = no grass items")
    print("   5. ❌ No object interaction = no hidden items")
    print()

def main():
    print("🔍 COMPREHENSIVE ITEM COLLECTION INVESTIGATION")
    print("=" * 70)
    print()
    
    # Test 1: Save state starting conditions
    test_save_state_starting_conditions()
    print()
    
    # Test 2: Action distribution analysis
    analyze_action_distribution()
    print()
    
    # Test 3: Manual item search
    test_manual_item_search()
    print()
    
    # Test 4: Game mechanics analysis
    analyze_zelda_game_mechanics()
    
    print("🎯 CONCLUSIONS & RECOMMENDATIONS")
    print("=" * 70)
    
    conclusions = [
        "1. 📍 Save state starts in specific area with limited items",
        "2. 🎲 Random policy is ineffective for Zelda gameplay", 
        "3. 🗡️  Need strategic actions: attack enemies, cut grass, lift rocks",
        "4. 🔄 Need longer exploration to find item-rich areas",
        "5. 🎮 Oracle of Seasons requires purposeful gameplay, not random actions"
    ]
    
    for conclusion in conclusions:
        print(conclusion)
    
    print()
    
    recommendations = [
        "1. 🗺️  Create exploration-focused action policy",
        "2. ⚔️  Implement enemy-seeking and combat behavior",
        "3. 🌿 Add grass-cutting and object-interaction strategies",
        "4. 📍 Start from different save states with more items",
        "5. 🎯 Replace random policy with informed exploration",
        "6. 🔍 Add visual item detection to guide agent toward items"
    ]
    
    print("💡 RECOMMENDED SOLUTIONS:")
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    main()

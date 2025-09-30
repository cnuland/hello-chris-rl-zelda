#!/usr/bin/env python3
"""
🎯 Simple Strategic System Test

Quick validation that the strategic system components are in place.
"""

import sys
import os
from pathlib import Path

def test_llm_prompt_has_strategic_macros():
    """Test that LLM prompts include new strategic macro types."""
    print("🎯 TESTING LLM PROMPT INTEGRATION")
    print("=" * 50)
    
    try:
        # Check if prompt file exists and contains strategic macros
        prompt_file = Path("configs/planner_prompt.yaml")
        if not prompt_file.exists():
            print("❌ Planner prompt file not found")
            return False
            
        with open(prompt_file, 'r') as f:
            prompt_content = f.read()
            
        # Check for strategic macro types
        strategic_macro_names = [
            "COMBAT_SWEEP", "CUT_GRASS", "SEARCH_ITEMS", 
            "ENEMY_HUNT", "ENVIRONMENTAL_SEARCH", 
            "COMBAT_RETREAT", "ROOM_CLEARING"
        ]
        
        found_macros = 0
        print("🔍 Checking for strategic macro types in prompt:")
        for macro_name in strategic_macro_names:
            if macro_name in prompt_content:
                print(f"   ✅ {macro_name}: Found")
                found_macros += 1
            else:
                print(f"   ❌ {macro_name}: Missing")
                
        # Check for critical gameplay rules
        critical_phrases = [
            "Combat and grass-cutting are ESSENTIAL",
            "ROOM_CLEARING when entering new areas", 
            "prioritize combat-based actions",
            "STRATEGIC MACRO ACTIONS FOR COMBAT"
        ]
        
        found_rules = 0
        print("\n🎯 Checking for critical gameplay rules:")
        for phrase in critical_phrases:
            if phrase in prompt_content:
                print(f"   ✅ '{phrase[:40]}...' found")
                found_rules += 1
            else:
                print(f"   ❌ '{phrase[:40]}...' missing")
                
        success = found_macros >= 5 and found_rules >= 2
        print(f"\n📊 Results: {found_macros}/{len(strategic_macro_names)} macros, {found_rules}/{len(critical_phrases)} rules")
        return success
        
    except Exception as e:
        print(f"❌ Prompt integration test failed: {e}")
        return False

def test_macro_file_has_strategic_actions():
    """Test that macro_actions.py contains the new strategic macro types."""
    print("\n🎯 TESTING MACRO ACTIONS FILE")
    print("=" * 50)
    
    try:
        macro_file = Path("agents/macro_actions.py")
        if not macro_file.exists():
            print("❌ Macro actions file not found")
            return False
            
        with open(macro_file, 'r') as f:
            macro_content = f.read()
            
        # Check for strategic macro types in enum
        strategic_types = [
            "COMBAT_SWEEP", "CUT_GRASS", "SEARCH_ITEMS",
            "ENEMY_HUNT", "ENVIRONMENTAL_SEARCH",
            "COMBAT_RETREAT", "ROOM_CLEARING"
        ]
        
        found_types = 0
        print("🔍 Checking for strategic macro types in MacroType enum:")
        for macro_type in strategic_types:
            if f'{macro_type} =' in macro_content:
                print(f"   ✅ {macro_type}: Found in enum")
                found_types += 1
            else:
                print(f"   ❌ {macro_type}: Missing from enum")
                
        # Check for expansion methods
        expansion_methods = [
            "_expand_combat_sweep", "_expand_cut_grass", "_expand_search_items",
            "_expand_enemy_hunt", "_expand_environmental_search"
        ]
        
        found_methods = 0
        print("\n🔧 Checking for strategic expansion methods:")
        for method in expansion_methods:
            if f'def {method}' in macro_content:
                print(f"   ✅ {method}: Found")
                found_methods += 1
            else:
                print(f"   ❌ {method}: Missing")
                
        success = found_types >= 5 and found_methods >= 3
        print(f"\n📊 Results: {found_types}/{len(strategic_types)} types, {found_methods}/{len(expansion_methods)} methods")
        return success
        
    except Exception as e:
        print(f"❌ Macro file test failed: {e}")
        return False

def test_environment_has_strategic_rewards():
    """Test that the environment file contains strategic reward methods."""
    print("\n🎯 TESTING ENVIRONMENT STRATEGIC REWARDS")
    print("=" * 50)
    
    try:
        env_file = Path("emulator/zelda_env_configurable.py")
        if not env_file.exists():
            print("❌ Environment file not found")
            return False
            
        with open(env_file, 'r') as f:
            env_content = f.read()
            
        # Check for strategic reward method
        if "_calculate_strategic_action_rewards" in env_content:
            print("   ✅ Strategic action rewards method found")
        else:
            print("   ❌ Strategic action rewards method missing")
            return False
            
        # Check for reward types
        reward_checks = [
            "COMBAT REWARDS", "ITEM COLLECTION REWARDS", 
            "ACTION-BASED REWARDS", "PATTERN REWARDS",
            "MILESTONE REWARDS"
        ]
        
        found_rewards = 0
        print("\n💰 Checking for reward categories:")
        for reward_type in reward_checks:
            if reward_type in env_content:
                print(f"   ✅ {reward_type}: Found")
                found_rewards += 1
            else:
                print(f"   ❌ {reward_type}: Missing")
                
        # Check for action tracking
        if "self.last_action = action" in env_content:
            print("   ✅ Action tracking: Found")
            action_tracking = True
        else:
            print("   ❌ Action tracking: Missing")
            action_tracking = False
            
        success = found_rewards >= 3 and action_tracking
        print(f"\n📊 Results: {found_rewards}/{len(reward_checks)} reward types, action tracking: {action_tracking}")
        return success
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def main():
    """Run simple strategic system validation."""
    print("🎯 SIMPLE STRATEGIC SYSTEM VALIDATION")
    print("=" * 60)
    print("Checking that strategic components are properly integrated...")
    print("=" * 60)
    
    tests = [
        test_llm_prompt_has_strategic_macros,
        test_macro_file_has_strategic_actions, 
        test_environment_has_strategic_rewards
    ]
    
    results = []
    for test in tests:
        results.append(test())
        
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL {total} VALIDATION TESTS PASSED!")
        print("=" * 60)
        print("✅ LLM prompts include strategic macro guidance")
        print("✅ Macro actions system has combat and environmental actions") 
        print("✅ Environment has strategic reward calculations")
        print("\n🚀 Strategic system is ready for RL training!")
        return 0
    else:
        print(f"❌ {passed}/{total} VALIDATION TESTS PASSED")
        print("⚠️ Some components may need attention before training")
        return 1

if __name__ == "__main__":
    exit(main())

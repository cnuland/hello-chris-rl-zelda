#!/usr/bin/env python3
"""
EXPLORATION PATTERN ANALYSIS
🔍 Deep dive into agent behavior and item collection issues
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
# import matplotlib.pyplot as plt  # Not needed for this analysis

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment

def analyze_overworld_positions(positions: List[int]) -> Dict[str, Any]:
    """Analyze overworld position patterns to understand movement"""
    if not positions:
        return {"analysis": "No positions to analyze"}
    
    # Oracle of Seasons uses a grid system for overworld positions
    # Typically positions are organized in a grid (16x16 = 256 total positions)
    rows = []
    cols = []
    
    for pos in positions:
        row = pos // 16  # Integer division to get row
        col = pos % 16   # Modulo to get column
        rows.append(row)
        cols.append(col)
    
    return {
        "total_positions": len(positions),
        "position_range": f"{min(positions)}-{max(positions)}",
        "row_range": f"{min(rows)}-{max(rows)} (span: {max(rows) - min(rows) + 1} rows)",
        "col_range": f"{min(cols)}-{max(cols)} (span: {max(cols) - min(cols) + 1} cols)",
        "area_coverage": (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1),
        "positions_by_row": {row: [pos for pos in positions if pos // 16 == row] for row in set(rows)},
        "exploration_density": len(positions) / ((max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1))
    }

def analyze_episode_patterns(rewards: List[float], lengths: List[int]) -> Dict[str, Any]:
    """Analyze episode patterns to understand behavior"""
    max_length = 4000  # Expected episode length
    
    early_endings = [i for i, length in enumerate(lengths) if length < max_length]
    full_episodes = [i for i, length in enumerate(lengths) if length == max_length]
    
    return {
        "total_episodes": len(rewards),
        "early_endings": len(early_endings),
        "full_episodes": len(full_episodes),
        "early_ending_rate": len(early_endings) / len(rewards) * 100,
        "early_ending_episodes": early_endings[:10],  # First 10 for analysis
        "average_early_length": np.mean([lengths[i] for i in early_endings]) if early_endings else 0,
        "reward_analysis": {
            "mean": np.mean(rewards),
            "std": np.std(rewards),
            "min": min(rewards),
            "max": max(rewards),
            "very_low_rewards": len([r for r in rewards if r < 50]),
            "high_rewards": len([r for r in rewards if r > 300])
        }
    }

def create_item_detection_test() -> str:
    """Create a script to test if items are available but not being detected"""
    return """#!/usr/bin/env python3
import sys
import time
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment

def test_item_availability():
    env = ZeldaConfigurableEnvironment(
        rom_path=str(project_root / "roms" / "zelda_oracle_of_seasons.gbc"),
        headless=True,
        visual_test_mode=False,
        config_dict={
            "controller": {"use_planner": True, "enable_visual": True},
            "environment": {"max_episode_steps": 1000, "frame_skip": 4},
            "rewards": {"room_discovery_reward": 30.0}
        }
    )
    
    print("🔍 ITEM DETECTION TEST")
    print("=" * 50)
    
    obs, info = env.reset()
    
    for step in range(100):
        action = np.random.randint(0, env.action_space.n)
        obs, reward, done, truncated, info = env.step(action)
        
        if info.get('structured_state'):
            state = info['structured_state']
            resources = state.get('resources', {})
            
            if step % 20 == 0:
                print(f"Step {step}:")
                print(f"  Resources: {resources}")
                print(f"  Player pos: {state.get('player', {}).get('x', 'N/A')}, {state.get('player', {}).get('y', 'N/A')}")
                print(f"  Room: {state.get('player', {}).get('room', 'N/A')}")
        
        if done or truncated:
            break
    
    env.close()

if __name__ == "__main__":
    import numpy as np
    test_item_availability()
"""

def main():
    print("🔍 EXPLORATION PATTERN ANALYSIS")
    print("=" * 70)
    
    # Load the exploration results
    results_file = "training_runs/exploration_working_1759158116/working_final_results.json"
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("📊 BASIC STATISTICS")
    print("-" * 50)
    print(f"Training duration: {results['training_hours']:.2f} hours")
    print(f"Episodes completed: {results['episodes_completed']}")
    print(f"Total steps: {results['total_steps']:,}")
    print(f"Average reward: {results['average_reward']:.1f}")
    print(f"LLM success rate: {results['llm_success_rate']:.1f}%")
    print()
    
    # Analyze exploration summary
    exploration = results['final_exploration_summary']
    
    print("🗺️  EXPLORATION SUMMARY")
    print("-" * 50)
    print(f"Rooms discovered: {exploration['total_rooms_discovered']} {exploration['all_rooms_list']}")
    print(f"Dungeons found: {exploration['total_dungeons_discovered']}")
    print(f"Overworld positions: {exploration['total_overworld_positions']}")
    print(f"Max items: {exploration['max_rupees_ever']} rupees, {exploration['max_keys_ever']} keys, {exploration['max_bombs_ever']} bombs")
    print()
    
    # Analyze overworld movement
    print("🌍 OVERWORLD MOVEMENT ANALYSIS")
    print("-" * 50)
    overworld_analysis = analyze_overworld_positions(exploration['all_overworld_positions_list'])
    print(f"Position range: {overworld_analysis['position_range']}")
    print(f"Row coverage: {overworld_analysis['row_range']}")
    print(f"Column coverage: {overworld_analysis['col_range']}")
    print(f"Area coverage: {overworld_analysis['area_coverage']} grid squares")
    print(f"Exploration density: {overworld_analysis['exploration_density']:.2f} (positions/area)")
    print()
    
    print("📊 POSITIONS BY ROW:")
    for row, positions in overworld_analysis['positions_by_row'].items():
        print(f"  Row {row}: {positions}")
    print()
    
    # Analyze episode patterns
    print("📈 EPISODE PATTERN ANALYSIS")
    print("-" * 50)
    episode_analysis = analyze_episode_patterns(results['episode_rewards'], results['episode_lengths'])
    print(f"Early endings: {episode_analysis['early_endings']}/{episode_analysis['total_episodes']} ({episode_analysis['early_ending_rate']:.1f}%)")
    print(f"Average early episode length: {episode_analysis['average_early_length']:.0f} steps")
    print(f"Early ending episodes: {episode_analysis['early_ending_episodes']}")
    print()
    
    reward_analysis = episode_analysis['reward_analysis']
    print(f"Reward statistics:")
    print(f"  Mean: {reward_analysis['mean']:.1f} ± {reward_analysis['std']:.1f}")
    print(f"  Range: {reward_analysis['min']:.1f} to {reward_analysis['max']:.1f}")
    print(f"  Very low rewards (<50): {reward_analysis['very_low_rewards']} episodes")
    print(f"  High rewards (>300): {reward_analysis['high_rewards']} episodes")
    print()
    
    # Room discovery timeline
    print("🏠 ROOM DISCOVERY TIMELINE")
    print("-" * 50)
    exploration_history = results['exploration_history']
    
    for i, episode_data in enumerate(exploration_history):
        if episode_data['episode_rooms_discovered'] > 0:
            episode_num = (i + 1) * 5  # Every 5th episode
            print(f"Episode {episode_num}: {episode_data['episode_discovery_list']}")
    
    if not any(episode['episode_rooms_discovered'] > 0 for episode in exploration_history[1:]):
        print("⚠️  NO NEW ROOMS discovered after episode 5!")
        print("🔍 This indicates the agent is stuck in a very small area.")
    print()
    
    # Critical issues identified
    print("🚨 CRITICAL ISSUES IDENTIFIED")
    print("-" * 50)
    issues = []
    
    if exploration['max_rupees_ever'] == 0:
        issues.append("❌ ZERO ITEM COLLECTION: No rupees, keys, or bombs collected")
    
    if exploration['total_rooms_discovered'] <= 3:
        issues.append("❌ MINIMAL EXPLORATION: Only 3 rooms discovered")
    
    if exploration['total_dungeons_discovered'] == 0:
        issues.append("❌ NO DUNGEON DISCOVERY: Zero dungeons found")
    
    if episode_analysis['early_ending_rate'] > 30:
        issues.append(f"❌ HIGH EARLY ENDING RATE: {episode_analysis['early_ending_rate']:.1f}% episodes end early")
    
    if overworld_analysis['area_coverage'] < 20:
        issues.append(f"❌ TINY EXPLORATION AREA: Only {overworld_analysis['area_coverage']} grid squares covered")
    
    for issue in issues:
        print(issue)
    
    if not issues:
        print("✅ No critical issues identified")
    
    print()
    
    # Hypotheses for item collection failure
    print("🤔 HYPOTHESES FOR ITEM COLLECTION FAILURE")
    print("-" * 50)
    hypotheses = [
        "1. 🎯 No items present in starting area (limited exploration)",
        "2. 🤖 Agent not performing item pickup actions",
        "3. 🔍 Item detection logic not working",
        "4. 📍 Save state starts in area without items",
        "5. ⏱️  Episodes too short to find items",
        "6. 🎲 Random policy ineffective for item collection"
    ]
    
    for hypothesis in hypotheses:
        print(hypothesis)
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDED INVESTIGATIONS")
    print("-" * 50)
    recommendations = [
        "1. 🔬 Create item detection test script",
        "2. 🗺️  Visual inspection of explored area",
        "3. 🎮 Manual gameplay to check item availability",
        "4. 📊 Analyze action distribution (are pickup actions used?)",
        "5. 🔍 Check if save state area contains items",
        "6. 📈 Increase episode length for better exploration",
        "7. 🤖 Replace random policy with exploration-focused strategy"
    ]
    
    for rec in recommendations:
        print(rec)
    
    print()
    
    # Create item detection test script
    print("🔧 Creating item detection test script...")
    test_script_content = create_item_detection_test()
    
    with open("test_item_detection.py", 'w') as f:
        f.write(test_script_content)
    
    print("✅ Created: test_item_detection.py")
    print()
    
    print("🎯 NEXT STEPS")
    print("-" * 50)
    print("1. Run: python3 test_item_detection.py")
    print("2. Check if any items are detected in the starting area")
    print("3. Consider expanding exploration area")
    print("4. Implement directed exploration instead of random policy")

if __name__ == "__main__":
    main()

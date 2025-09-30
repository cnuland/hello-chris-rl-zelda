#!/usr/bin/env python3
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

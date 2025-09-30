#!/usr/bin/env python3
"""
Strategic Headless Training - Using Unified Framework

Uses the proven Strategic Training Framework for production training:
- Headless mode for maximum performance
- Strategic action translation
- 5X LLM emphasis system
- Multi-episode training with comprehensive logging
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from strategic_training_framework import (
    StrategicConfig, StrategicTrainer, create_headless_strategic_trainer
)

def main():
    """Run strategic headless training."""
    parser = argparse.ArgumentParser(description="Strategic Headless Training")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to train")
    parser.add_argument("--llm-interval", type=int, default=50, help="LLM call interval (steps)")
    parser.add_argument("--max-steps", type=int, default=8000, help="Max steps per episode")
    args = parser.parse_args()
    
    print("🎯 STRATEGIC HEADLESS TRAINING")
    print("=" * 60)
    print("🧠 Framework: Strategic Training Framework")
    print("🖥️  Mode: Headless (maximum performance)")
    print("⚡ Features: 5X LLM emphasis + strategic actions")
    print(f"📊 Episodes: {args.episodes}")
    print(f"🎯 LLM Interval: Every {args.llm_interval} steps")
    print(f"⏱️  Max Steps/Episode: {args.max_steps}")
    print()
    
    try:
        # Create strategic trainer with headless configuration
        config = StrategicConfig(
            max_episode_steps=args.max_steps,
            llm_call_interval=args.llm_interval
        )
        trainer = create_headless_strategic_trainer(config)
        
        # Run strategic training
        rom_path = str(project_root / "roms" / "zelda_oracle_of_seasons.gbc")
        results = trainer.run_strategic_training(
            rom_path=rom_path,
            episodes=args.episodes,
            headless=True
        )
        
        # Performance analysis
        print("\n🎯 STRATEGIC PERFORMANCE ANALYSIS")
        print("=" * 60)
        print(f"📊 Episodes Completed: {results['episodes_completed']}")
        print(f"⏱️  Total Duration: {results['total_duration_minutes']:.1f} minutes")
        print(f"🏆 Average Reward: {results['average_reward']:.1f}")
        print(f"🧠 Total LLM Calls: {results['total_llm_calls']}")
        print(f"✅ LLM Success Rate: {results['llm_success_rate']:.1f}%")
        
        if len(results['all_episode_rewards']) > 1:
            import numpy as np
            std_reward = np.std(results['all_episode_rewards'])
            min_reward = min(results['all_episode_rewards'])
            max_reward = max(results['all_episode_rewards'])
            
            print(f"📈 Reward Range: {min_reward:.1f} - {max_reward:.1f}")
            print(f"📊 Reward Std Dev: {std_reward:.1f}")
            
            # Performance vs baseline (assuming baseline ~200 reward)
            baseline_reward = 200.0
            improvement = ((results['average_reward'] / baseline_reward) - 1) * 100
            print(f"🚀 Performance vs Baseline: {improvement:+.1f}%")
            
            if results['average_reward'] > baseline_reward:
                multiplier = results['average_reward'] / baseline_reward
                print(f"🔥 Strategic Multiplier: {multiplier:.1f}x improvement!")
        
        print(f"\n✅ Strategic headless training complete!")
        return results
        
    except Exception as e:
        print(f"❌ Strategic training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

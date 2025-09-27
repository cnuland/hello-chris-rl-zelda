#!/usr/bin/env python3
"""
Zelda RL Training Script
High-performance headless training with configurable LLM integration.
"""

import os
import sys
import yaml
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Add project root to path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Zelda RL Agent')
    
    # Training mode
    parser.add_argument('--mode', choices=['pure_rl', 'llm_guided'], default='pure_rl',
                       help='Training mode: pure_rl (no LLM) or llm_guided (with LLM)')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to training configuration YAML file')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=100000,
                       help='Total training steps')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Max episodes (overrides steps if provided)')
    
    # Output and checkpoints
    parser.add_argument('--output-dir', type=str, default='training_runs',
                       help='Output directory for logs and checkpoints')
    parser.add_argument('--checkpoint-freq', type=int, default=10000,
                       help='Checkpoint frequency (steps)')
    parser.add_argument('--log-freq', type=int, default=100,
                       help='Logging frequency (steps)')
    
    # Performance options
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Training device')
    parser.add_argument('--num-envs', type=int, default=1,
                       help='Number of parallel environments')
    
    # Debug options
    parser.add_argument('--dry-run', action='store_true',
                       help='Print configuration and exit without training')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    return parser.parse_args()

def get_default_config(mode: str) -> Dict[str, Any]:
    """Get default training configuration."""
    base_config = {
        'environment': {
            'frame_skip': 4,
            'observation_type': 'vector',
            'normalize_observations': True,
            'max_episode_steps': 10000
        },
        'training': {
            'total_timesteps': 100000,
            'eval_frequency': 10000,
            'save_frequency': 10000,
            'log_frequency': 100,
            'max_episode_steps': 10000,
            'early_termination': {
                'health_zero': True,
                'stuck_threshold': 1000
            }
        },
        'ppo': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_loss_coeff': 0.5,
            'entropy_coeff': 0.01,
            'max_grad_norm': 0.5,
            'num_steps': 128,
            'update_epochs': 4,
            'batch_size': 32
        },
        'rewards': {
            'rupee_reward': 0.01,
            'key_reward': 0.5,
            'death_penalty': -3.0,
            'movement_reward': 0.001,
            'time_penalty': -0.0001,
            'health_loss_penalty': -0.1,
            'health_gain_reward': 0.2
        }
    }
    
    if mode == 'pure_rl':
        base_config['planner_integration'] = {
            'use_planner': False,
            'enable_visual': False,
            'use_structured_entities': False,
            'auto_load_save_state': True
        }
        base_config['training']['total_timesteps'] = 200000  # More steps for pure RL
        base_config['exploration'] = {
            'epsilon_schedule': {
                'initial': 0.2,
                'final': 0.02,
                'decay_steps': 100000
            },
            'use_curiosity': True,
            'curiosity_coeff': 0.1
        }
    else:  # llm_guided
        base_config['planner_integration'] = {
            'use_planner': True,
            'enable_visual': True,
            'use_structured_entities': True,
            'compression_mode': 'bit_packed',
            'auto_load_save_state': True,
            'planner_frequency': 100
        }
        base_config['training']['total_timesteps'] = 100000  # Fewer steps with LLM guidance
    
    return base_config

def setup_training_environment(config: Dict[str, Any], mode: str, verbose: bool = False):
    """Setup training environment and components."""
    try:
        # Import the configurable environment
        import sys
        import os
        
        # Add current directory to path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment
        
        rom_path = "roms/zelda_oracle_of_seasons.gbc"
        
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found: {rom_path}")
        
        if verbose:
            print(f"📍 Creating {mode} environment...")
            print(f"   ROM: {rom_path}")
            print(f"   Headless: True (training mode)")
            print(f"   LLM enabled: {config.get('planner_integration', {}).get('use_planner', False)}")
        
        # Create environment
        env = ZeldaConfigurableEnvironment(
            rom_path=rom_path,
            config_dict=config,
            headless=True,  # Always headless for training
            visual_test_mode=False
        )
        
        if verbose:
            config_summary = env.get_config_summary()
            print(f"✅ Environment created successfully")
            for key, value in config_summary.items():
                print(f"   {key}: {value}")
        
        return env
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        return None
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return None

def create_training_logger(output_dir: Path, mode: str):
    """Create training logger for metrics tracking."""
    
    class TrainingLogger:
        def __init__(self, output_dir: Path, mode: str):
            self.output_dir = output_dir
            self.mode = mode
            self.metrics = {
                'steps': [],
                'episodes': [],
                'episode_rewards': [],
                'episode_lengths': [],
                'avg_reward': [],
                'exploration_rate': [],
                'training_time': []
            }
            self.start_time = time.time()
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize log file
            self.log_file = output_dir / 'training.log'
            with open(self.log_file, 'w') as f:
                f.write(f"# Zelda RL Training Log - {mode}\n")
                f.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("step,episode,episode_reward,episode_length,avg_reward,exploration_rate,training_time\n")
        
        def log_episode(self, step: int, episode: int, episode_reward: float, 
                       episode_length: int, exploration_rate: float = 0.0):
            current_time = time.time() - self.start_time
            
            # Update metrics
            self.metrics['steps'].append(step)
            self.metrics['episodes'].append(episode)
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['episode_lengths'].append(episode_length)
            self.metrics['exploration_rate'].append(exploration_rate)
            self.metrics['training_time'].append(current_time)
            
            # Calculate moving average
            recent_rewards = self.metrics['episode_rewards'][-100:]  # Last 100 episodes
            avg_reward = np.mean(recent_rewards)
            self.metrics['avg_reward'].append(avg_reward)
            
            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(f"{step},{episode},{episode_reward:.3f},{episode_length},"
                       f"{avg_reward:.3f},{exploration_rate:.3f},{current_time:.1f}\n")
        
        def print_progress(self, step: int, episode: int, episode_reward: float, 
                          episode_length: int, exploration_rate: float = 0.0):
            current_time = time.time() - self.start_time
            avg_reward = self.metrics['avg_reward'][-1] if self.metrics['avg_reward'] else 0.0
            
            print(f"Step {step:6d} | Episode {episode:4d} | "
                  f"Reward: {episode_reward:6.2f} | Length: {episode_length:4d} | "
                  f"Avg100: {avg_reward:6.2f} | Explore: {exploration_rate:.3f} | "
                  f"Time: {current_time:6.1f}s")
        
        def save_metrics(self):
            """Save metrics to JSON file."""
            metrics_file = self.output_dir / 'metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            return metrics_file
    
    return TrainingLogger(output_dir, mode)

def simulate_training_loop(env, config: Dict[str, Any], args, logger):
    """Simulate training loop (placeholder for actual RL training)."""
    
    print(f"\n🚀 Starting {args.mode.upper()} Training Simulation")
    print("=" * 60)
    print("⚠️  Note: This is a training simulation, not actual RL training")
    print("   In production, this would use a real RL algorithm (PPO/SAC/etc)")
    print()
    
    total_steps = args.steps
    max_episodes = args.episodes
    
    # Training simulation
    step = 0
    episode = 0
    
    while step < total_steps and (max_episodes is None or episode < max_episodes):
        try:
            # Reset environment
            print(f"🔄 Starting episode {episode + 1}")
            obs, info = env.reset()
            
            episode_reward = 0.0
            episode_length = 0
            episode += 1
            
            # Episode simulation
            max_episode_steps = config.get('training', {}).get('max_episode_steps', 1000)
            
            for episode_step in range(max_episode_steps):
                # Simulate RL agent decision (random for demo)
                action = env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                step += 1
                
                # Check for episode end
                if terminated or truncated:
                    if args.verbose:
                        print(f"   Episode {episode} ended: {'terminated' if terminated else 'truncated'}")
                    break
                
                # Check total step limit
                if step >= total_steps:
                    break
            
            # Calculate exploration rate (simulate decay)
            exploration_rate = max(0.01, 0.2 * (1 - step / total_steps))
            
            # Log episode
            logger.log_episode(step, episode, episode_reward, episode_length, exploration_rate)
            
            # Print progress
            if episode % args.log_freq == 0 or args.verbose:
                logger.print_progress(step, episode, episode_reward, episode_length, exploration_rate)
            
            # Checkpoint
            if step % args.checkpoint_freq == 0:
                checkpoint_file = logger.output_dir / f'checkpoint_step_{step}.json'
                checkpoint_data = {
                    'step': step,
                    'episode': episode,
                    'config': config,
                    'training_mode': args.mode,
                    'performance': {
                        'avg_reward': logger.metrics['avg_reward'][-1] if logger.metrics['avg_reward'] else 0.0,
                        'exploration_rate': exploration_rate
                    }
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                print(f"💾 Checkpoint saved: {checkpoint_file}")
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Training interrupted by user at step {step}")
            break
        except Exception as e:
            print(f"\n❌ Training error at step {step}: {e}")
            break
    
    return step, episode

def main():
    """Main training function."""
    args = parse_args()
    
    print("🎯 ZELDA RL TRAINING")
    print("=" * 50)
    print(f"Mode: {args.mode.upper()}")
    print(f"Target steps: {args.steps:,}")
    if args.episodes:
        print(f"Max episodes: {args.episodes:,}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        print(f"📋 Loading config: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"📋 Using default {args.mode} configuration")
        config = get_default_config(args.mode)
    
    # Override config with CLI arguments
    config['training']['total_timesteps'] = args.steps
    if args.episodes:
        config['training']['max_episodes'] = args.episodes
    
    # Dry run - just show configuration
    if args.dry_run:
        print("\n🔍 DRY RUN - Configuration Preview:")
        print("=" * 40)
        print(yaml.dump(config, default_flow_style=False))
        return True
    
    # Setup output directory
    output_dir = Path(args.output_dir) / f"{args.mode}_{int(time.time())}"
    
    # Save configuration
    config_file = output_dir / 'config.yaml'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"💾 Configuration saved: {config_file}")
    
    # Setup environment
    env = setup_training_environment(config, args.mode, args.verbose)
    if env is None:
        return False
    
    # Setup logger
    logger = create_training_logger(output_dir, args.mode)
    
    try:
        # Run training
        final_step, final_episode = simulate_training_loop(env, config, args, logger)
        
        # Save final metrics
        metrics_file = logger.save_metrics()
        
        # Training summary
        training_time = time.time() - logger.start_time
        avg_reward = logger.metrics['avg_reward'][-1] if logger.metrics['avg_reward'] else 0.0
        
        print(f"\n🏁 TRAINING COMPLETE")
        print("=" * 50)
        print(f"Final step: {final_step:,}")
        print(f"Total episodes: {final_episode:,}")
        print(f"Training time: {training_time:.1f} seconds")
        print(f"Average reward (last 100): {avg_reward:.3f}")
        print(f"Steps per second: {final_step/training_time:.1f}")
        print(f"📊 Metrics saved: {metrics_file}")
        print(f"📁 Output directory: {output_dir}")
        
        return True
        
    finally:
        env.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""CleanRL-style PPO training for Zelda Oracle of Seasons.

Baseline PPO training script without LLM planner integration.
"""

import os
import time
import asyncio
import argparse
from typing import Dict, List, Any
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from emulator.zelda_env import ZeldaEnvironment
from agents.controller import ZeldaController, ControllerConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PPO training for Zelda")

    # Environment args
    parser.add_argument("--rom-path", type=str, required=True,
                       help="Path to Oracle of Seasons ROM file")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run emulator headless")

    # Training args
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                       help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--num-envs", type=int, default=1,
                       help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=128,
                       help="Number of steps per rollout")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                       help="GAE lambda")
    parser.add_argument("--update-epochs", type=int, default=4,
                       help="Number of PPO update epochs")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                       help="PPO clip epsilon")
    parser.add_argument("--value-loss-coeff", type=float, default=0.5,
                       help="Value loss coefficient")
    parser.add_argument("--entropy-coeff", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                       help="Max gradient norm for clipping")

    # Logging args
    parser.add_argument("--exp-name", type=str, default="zelda_ppo",
                       help="Experiment name")
    parser.add_argument("--wandb-project", type=str, default="zelda-rl",
                       help="Wandb project name")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Logging directory")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Checkpoint save directory")
    parser.add_argument("--save-frequency", type=int, default=100000,
                       help="Save checkpoint every N steps")

    # Evaluation args
    parser.add_argument("--eval-frequency", type=int, default=50000,
                       help="Evaluate every N steps")
    parser.add_argument("--eval-episodes", type=int, default=5,
                       help="Number of evaluation episodes")

    # Misc args
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--cuda", action="store_true", default=True,
                       help="Use CUDA if available")

    return parser.parse_args()


class PPOTrainer:
    """PPO trainer for Zelda environment."""

    def __init__(self, args):
        """Initialize trainer.

        Args:
            args: Command line arguments
        """
        self.args = args

        # Set up directories
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)

        # Set random seeds
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Initialize environment
        self.env = ZeldaEnvironment(
            rom_path=args.rom_path,
            headless=args.headless
        )

        # Initialize controller (without planner for baseline)
        config = ControllerConfig(
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            value_loss_coeff=args.value_loss_coeff,
            entropy_coeff=args.entropy_coeff,
            max_grad_norm=args.max_grad_norm,
            use_planner=False  # Baseline without planner
        )
        self.controller = ZeldaController(self.env, config)

        # Initialize logging
        self.writer = SummaryWriter(log_dir=f"{args.log_dir}/{args.exp_name}")

        # Initialize wandb if available
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.exp_name,
                config=vars(args)
            )
            self.use_wandb = True
        except Exception:
            print("Warning: Could not initialize wandb")
            self.use_wandb = False

        # Training state
        self.global_step = 0
        self.episode_count = 0

    def collect_rollout(self) -> Dict[str, List]:
        """Collect a rollout of experience.

        Returns:
            Dictionary containing rollout data
        """
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []

        obs, info = self.env.reset()

        for step in range(self.args.num_steps):
            observations.append(obs.copy())

            # Get action from controller
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.controller.device)

            with torch.no_grad():
                action, log_prob, value = self.controller.policy_net.get_action_and_value(obs_tensor)

            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
            dones.append(done)

            self.global_step += 1

            if done:
                obs, info = self.env.reset()
                self.episode_count += 1

        return {
            'observations': observations,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'dones': dones
        }

    def train_step(self, rollout_data: Dict[str, List]) -> Dict[str, float]:
        """Perform one training step.

        Args:
            rollout_data: Rollout data

        Returns:
            Training metrics
        """
        # Convert to tensors
        observations = torch.FloatTensor(rollout_data['observations']).to(self.controller.device)
        actions = torch.LongTensor(rollout_data['actions']).to(self.controller.device)
        old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.controller.device)
        rewards = rollout_data['rewards']
        values = rollout_data['values']
        dones = rollout_data['dones']

        # Compute advantages and returns
        advantages, returns = self.controller.compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages).to(self.controller.device)
        returns = torch.FloatTensor(returns).to(self.controller.device)

        # Prepare batch data
        batch_data = {
            'obs': observations,
            'actions': actions,
            'log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns
        }

        # Update policy
        metrics = self.controller.update(batch_data, epochs=self.args.update_epochs)

        # Add rollout metrics
        metrics['episode_reward'] = np.sum(rewards)
        metrics['episode_length'] = len(rewards)
        metrics['value_mean'] = np.mean(values)
        metrics['advantage_mean'] = advantages.mean().item()

        return metrics

    async def evaluate(self) -> Dict[str, float]:
        """Evaluate current policy.

        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []

        for episode in range(self.args.eval_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                # Use deterministic policy for evaluation
                action = self.controller.act_deterministic(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            'eval_reward_mean': np.mean(episode_rewards),
            'eval_reward_std': np.std(episode_rewards),
            'eval_length_mean': np.mean(episode_lengths),
            'eval_length_std': np.std(episode_lengths)
        }

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log training metrics.

        Args:
            metrics: Metrics dictionary
            step: Current step
        """
        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)

        # Print key metrics
        if step % 10000 == 0:
            print(f"Step {step}: "
                  f"Reward={metrics.get('episode_reward', 0):.2f}, "
                  f"Length={metrics.get('episode_length', 0):.0f}, "
                  f"Policy Loss={metrics.get('policy_loss', 0):.4f}")

    def save_checkpoint(self, step: int) -> None:
        """Save training checkpoint.

        Args:
            step: Current training step
        """
        checkpoint_path = f"{self.args.save_dir}/checkpoint_{step}.pt"
        self.controller.save_checkpoint(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    async def train(self) -> None:
        """Main training loop."""
        print(f"Starting training for {self.args.total_timesteps} timesteps")
        print(f"ROM path: {self.args.rom_path}")
        print(f"Device: {self.controller.device}")

        start_time = time.time()

        while self.global_step < self.args.total_timesteps:
            # Collect rollout
            rollout_data = self.collect_rollout()

            # Training step
            metrics = self.train_step(rollout_data)

            # Log metrics
            self.log_metrics(metrics, self.global_step)

            # Evaluation
            if self.global_step % self.args.eval_frequency == 0:
                eval_metrics = await self.evaluate()
                self.log_metrics(eval_metrics, self.global_step)

            # Save checkpoint
            if self.global_step % self.args.save_frequency == 0:
                self.save_checkpoint(self.global_step)

        # Final save
        self.save_checkpoint(self.global_step)

        # Cleanup
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")

        self.env.close()
        await self.controller.close()
        self.writer.close()

        if self.use_wandb:
            wandb.finish()


async def main():
    """Main function."""
    args = parse_args()
    trainer = PPOTrainer(args)
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
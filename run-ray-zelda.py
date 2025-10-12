"""
Ray RLlib Training Script for Zelda Oracle of Seasons
Uses vector observations + vision LLM (matches existing hybrid approach)
"""

from pathlib import Path
import uuid
import os
from ray_zelda_env import ZeldaRayEnv
from ray_hud_callback import ZeldaHUDCallback
import warnings
warnings.filterwarnings("ignore")

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.models import ModelCatalog
from ray_zelda_model import ZeldaMLPModel

# Register custom model
ModelCatalog.register_custom_model("zelda_mlp", ZeldaMLPModel)

# Define parallel rollout configuration
num_rollout_workers = 3
num_envs_per_worker = 3

# Set up session
ep_length = 2048 * 15  # ~30,000 steps per episode
sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

# Environment configuration
# This matches your existing ZeldaConfigurableEnvironment setup
env_config = {
    'gb_path': 'roms/zelda_oracle_of_seasons.gbc',
    'env_config_path': 'configs/env.yaml',  # Uses your existing env.yaml
    'headless': True,
    'worker_index': 0,  # Will be set by Ray per worker
}

# Register custom environment
register_env("zelda_env", lambda config: ZeldaRayEnv(config))

# Initialize Ray
ray.init()

# Print HUD configuration
hud_url = os.environ.get('HUD_URL')
if hud_url:
    print(f"🖥️  HUD Dashboard enabled: {hud_url}")
else:
    print("⚠️  HUD_URL not set - dashboard disabled")

# Configure PPO algorithm
# Hyperparameters match your existing hybrid vision approach
config = (
    PPOConfig()
    .environment(env="zelda_env", env_config=env_config)
    .framework("torch")
    .env_runners(
        num_env_runners=num_rollout_workers,
        num_envs_per_env_runner=num_envs_per_worker,
    )
    .callbacks(ZeldaHUDCallback)
    .training(
        model={
            "custom_model": "zelda_mlp",
            "fcnet_hiddens": [256],  # Hidden layer size (matches PolicyNetwork)
        },
        # PPO hyperparameters (matching your existing implementation)
        lr=3e-4,            # Learning rate
        gamma=0.99,         # Discount factor
        lambda_=0.95,       # GAE lambda
        clip_param=0.2,     # PPO clip parameter
        vf_clip_param=10.0, # Value function clip
        entropy_coeff=0.01, # Entropy bonus
        train_batch_size=4096,      # Training batch size
        sgd_minibatch_size=512,     # SGD minibatch size
        num_sgd_iter=10,            # SGD iterations per update
    )
    .resources(
        num_gpus=1,  # Use GPU if available
    )
)

# Run training
print("🎮 Starting Zelda Oracle of Seasons Ray RLlib Training")
print(f"   Env Runners (Workers): {num_rollout_workers}")
print(f"   Envs per runner: {num_envs_per_worker}")
print(f"   Total parallel environments: {num_rollout_workers * num_envs_per_worker}")
print(f"   Episode length: {ep_length}")
print("="*60)

tune.run(
    "PPO",
    name="PPO_ZeldaOracleSeasons",
    stop={"timesteps_total": ep_length * 10000},  # 300M timesteps total
    checkpoint_freq=10,  # Checkpoint every 10 iterations
    storage_path=str(Path("~/ray_results/zelda").expanduser()),
    config=config.to_dict()
)

print("\n✅ Zelda Oracle of Seasons training complete!")
print(f"Results saved to: ~/ray_results/zelda/")

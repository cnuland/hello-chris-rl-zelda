"""
Ray RLlib Training Script for Zelda Oracle of Seasons
Uses vector observations + vision LLM (matches existing hybrid approach)

Cache bust: 2025-10-15 04:45 - LLM now describes what it sees + suggests action
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

# Define parallel rollout configuration (overridable via env vars)
# Defaults match Ray cluster capacity (8 CPUs total)
num_rollout_workers = int(os.getenv("RAY_WORKERS", "6"))
num_envs_per_worker = int(os.getenv("ENVS_PER_WORKER", "6"))

# Set up session (episode length overridable via env var)
ep_length = int(os.getenv("EPISODE_LENGTH", str(2048 * 15)))  # default ~30,000 steps per episode
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
# Allow batch size override via env var to scale with cluster size
batch_size = int(os.getenv("BATCH_SIZE", "4096"))
config = (
    PPOConfig()
    .environment(env="zelda_env", env_config=env_config)
    .framework("torch")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
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
        lr=3e-4,                    # Learning rate
        gamma=0.99,                 # Discount factor
        lambda_=0.95,               # GAE lambda
        clip_param=0.2,             # PPO clip parameter
        vf_clip_param=10.0,         # Value function clip
        entropy_coeff=0.01,         # Entropy bonus
        train_batch_size_per_learner=batch_size,  # Training batch size per learner
        minibatch_size=512,         # Mini-batch size for SGD
        num_epochs=10,              # Training epochs per update (was num_sgd_iter)
    )
    .resources(
        num_gpus=0,  # CPU-only training (do not consume LLM GPUs)
    )
)

# Run training
print("🎮 Starting Zelda Oracle of Seasons Ray RLlib Training")
print(f"   Env Runners (Workers): {num_rollout_workers}")
print(f"   Envs per runner: {num_envs_per_worker}")
print(f"   Total parallel environments: {num_rollout_workers * num_envs_per_worker}")
print(f"   Episode length: {ep_length}")
print(f"   Train batch size per learner: {config.to_dict().get('train_batch_size_per_learner', 'N/A')}")
print("="*60)

tune.run(
    "PPO",
    name="PPO_ZeldaOracleSeasons",
    stop={"timesteps_total": ep_length * 10000},  # 300M timesteps total
    checkpoint_freq=0,  # Disable checkpointing (local storage not accessible on workers)
    # TODO: Enable S3 checkpointing for production:
    # storage_path="s3://your-bucket/zelda-checkpoints",
    config=config.to_dict()
)

print("\n✅ Zelda Oracle of Seasons training complete!")
print(f"Results saved to: ~/ray_results/zelda/")

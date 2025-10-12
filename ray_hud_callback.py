"""
Ray RLlib Callback for HUD integration
Sends training metrics to the HUD dashboard in real-time

IMPORTANT: Single Episode Reporting Strategy
============================================

With 9 parallel environments (3 workers × 3 envs each), having ALL of them send
data to the HUD would be confusing - the viewer would see data rapidly switching
between different episodes.

Solution: Only ONE designated environment sends episode-level data to the HUD.

Designated Reporter:
- Worker Index: 1 (first rollout worker, 0 is local/driver)
- Environment ID: 0 (first environment on that worker)

This means the viewer sees ONE consistent episode throughout training, while all
9 environments still contribute to the training process.

To change which environment is the "viewable" instance, modify:
- ZeldaHUDCallback.HUD_WORKER_INDEX
- ZeldaHUDCallback.HUD_ENV_ID
"""

import os
import sys
from pathlib import Path
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2

# Add HUD directory to path
sys.path.append(str(Path(__file__).parent / 'HUD'))

try:
    from HUD.hud_client import HUDClient
    HUD_AVAILABLE = True
except ImportError:
    HUD_AVAILABLE = False
    print("⚠️  HUD client not available")


class ZeldaHUDCallback(DefaultCallbacks):
    """
    Callback to send training metrics and vision data to HUD dashboard.
    
    IMPORTANT: Only ONE environment should send data to avoid confusing the viewer.
    We designate worker_index=1 (first rollout worker), env_id=0 as the "viewable" instance.
    """
    
    # Which worker/env should send data to HUD
    HUD_WORKER_INDEX = 1  # First rollout worker (0 is local/driver)
    HUD_ENV_ID = 0        # First environment on that worker
    
    def __init__(self):
        super().__init__()
        self.hud_client = None
        self.is_hud_reporter = False  # Will be set based on worker/env
        
        if HUD_AVAILABLE:
            hud_url = os.environ.get('HUD_URL')
            if hud_url:
                self.hud_client = HUDClient(hud_url=hud_url)
            else:
                print("⚠️  HUD_URL not set in environment, dashboard disabled")
    
    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode: EpisodeV2,
        env_index=None,
        **kwargs
    ):
        """
        Called when an episode ends.
        Update HUD with episode statistics.
        
        ONLY sends data if this is the designated HUD reporter
        (worker_index=1, env_id=0) to avoid confusion from multiple parallel episodes.
        """
        if not self.hud_client or not self.hud_client.enabled:
            return
        
        # Check if this is the designated HUD reporter
        worker_index = worker.worker_index if hasattr(worker, 'worker_index') else 0
        env_id = env_index if env_index is not None else 0
        
        # Only designated worker/env sends to HUD
        if worker_index != self.HUD_WORKER_INDEX or env_id != self.HUD_ENV_ID:
            return
        
        # Log first time this worker/env reports to HUD
        if not self.is_hud_reporter:
            self.is_hud_reporter = True
            print(f"🖥️  HUD Reporter: Worker {worker_index}, Env {env_id}")
        
        # Get episode metrics
        episode_reward = episode.total_reward
        episode_length = episode.length
        
        # Get custom metrics from the episode
        custom_metrics = episode.custom_metrics
        
        # Prepare training data for HUD
        training_data = {
            'episode_reward': float(episode_reward),
            'episode_length': int(episode_length),
            'avg_reward_per_step': float(episode_reward / max(episode_length, 1)),
            'worker_index': worker_index,
            'env_id': env_id,
        }
        
        # Add custom metrics if available
        for key, value in custom_metrics.items():
            if isinstance(value, (int, float)):
                training_data[key] = float(value)
        
        # Send to HUD
        self.hud_client.update_training_data(training_data)
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """
        Called after each training iteration (on driver/trainer).
        Update HUD with overall training metrics.
        
        This runs once per iteration, not per episode, so it's safe to always send.
        """
        if not self.hud_client or not self.hud_client.enabled:
            return
        
        # Extract key metrics
        training_data = {
            'timesteps_total': result.get('timesteps_total', 0),
            'episodes_total': result.get('episodes_total', 0),
            'mean_reward': result.get('episode_reward_mean', 0.0),
            'mean_episode_length': result.get('episode_len_mean', 0.0),
            'learning_rate': result.get('info', {}).get('learner', {}).get('default_policy', {}).get('cur_lr', 0.0),
            'iteration': result.get('training_iteration', 0),
        }
        
        # Add policy loss info if available
        learner_info = result.get('info', {}).get('learner', {}).get('default_policy', {})
        if learner_info:
            training_data['policy_loss'] = learner_info.get('learner_stats', {}).get('policy_loss', 0.0)
            training_data['vf_loss'] = learner_info.get('learner_stats', {}).get('vf_loss', 0.0)
            training_data['entropy'] = learner_info.get('learner_stats', {}).get('entropy', 0.0)
        
        # Send to HUD
        self.hud_client.update_training_data(training_data)
        
        # Try to get a screenshot from THE DESIGNATED environment only
        # We query the specific worker that we've designated as the HUD reporter
        try:
            # Try new API first (env_runners), fallback to old API (workers)
            env_runners = getattr(algorithm, 'env_runners', None) or getattr(algorithm, 'workers', None)
            if env_runners:
                # Get the designated worker (worker_index=1)
                remote_runners = getattr(env_runners, 'remote_env_runners', lambda: env_runners.remote_workers())()
                if len(remote_runners) >= self.HUD_WORKER_INDEX:
                    designated_worker = remote_runners[self.HUD_WORKER_INDEX - 1]  # -1 because list is 0-indexed
                    
                    # Try to get screenshot from that worker's env 0
                    # This is environment-specific and may not always work
                    try:
                        # Remote call to get screenshot from designated env
                        import ray
                        screenshot = ray.get(designated_worker.apply.remote(
                            lambda w: (
                                getattr(w.env, 'get_screenshot_base64', lambda: None)() 
                                if hasattr(w.env, 'envs') and len(w.env.envs) > self.HUD_ENV_ID 
                                else None
                            )
                        ))
                        if screenshot:
                            self.hud_client.update_vision_data(screenshot)
                    except Exception:
                        pass  # Silently ignore screenshot errors
        except Exception as e:
            # Silently ignore screenshot errors
            pass


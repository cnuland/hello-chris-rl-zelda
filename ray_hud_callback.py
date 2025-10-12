"""
Ray RLlib Callback for HUD integration
Sends training metrics to the HUD dashboard in real-time
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
    Callback to send training metrics and vision data to HUD dashboard
    """
    
    def __init__(self):
        super().__init__()
        self.hud_client = None
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
        **kwargs
    ):
        """
        Called when an episode ends.
        Update HUD with episode statistics.
        """
        if not self.hud_client or not self.hud_client.enabled:
            return
        
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
        }
        
        # Add custom metrics if available
        for key, value in custom_metrics.items():
            if isinstance(value, (int, float)):
                training_data[key] = float(value)
        
        # Send to HUD
        self.hud_client.update_training_data(training_data)
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """
        Called after each training iteration.
        Update HUD with overall training metrics.
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
        }
        
        # Add policy loss info if available
        learner_info = result.get('info', {}).get('learner', {}).get('default_policy', {})
        if learner_info:
            training_data['policy_loss'] = learner_info.get('learner_stats', {}).get('policy_loss', 0.0)
            training_data['vf_loss'] = learner_info.get('learner_stats', {}).get('vf_loss', 0.0)
            training_data['entropy'] = learner_info.get('learner_stats', {}).get('entropy', 0.0)
        
        # Send to HUD
        self.hud_client.update_training_data(training_data)
        
        # Try to get a screenshot from one of the environments
        # This is optional and will only work if the environment exposes it
        try:
            workers = algorithm.workers
            if workers and hasattr(workers, 'foreach_worker'):
                # Get screenshot from first worker
                screenshots = workers.foreach_worker(
                    lambda w: getattr(w.env, 'get_screenshot_base64', lambda: None)(),
                    local_worker=False
                )
                if screenshots and screenshots[0]:
                    self.hud_client.update_vision_data(screenshots[0])
        except Exception as e:
            # Silently ignore screenshot errors
            pass


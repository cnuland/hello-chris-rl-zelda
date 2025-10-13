"""
Ray RLlib Callback for HUD integration
Sends training metrics to the HUD dashboard in real-time

IMPORTANT: Driver-Only HUD Updates
===================================

To avoid session conflicts with the HUD server (which only allows one active
session at a time), the HUD client is initialized ONLY on the driver process,
not on worker processes.

All HUD updates happen in on_train_result(), which runs on the driver after
each training iteration. This ensures:
- No duplicate session registration attempts
- No "HUD already in use" errors
- Clean, centralized HUD updates

Workers do NOT connect to the HUD - only the driver does.
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
    
    HUD client is initialized lazily on the driver only to avoid session conflicts.
    """
    
    def __init__(self):
        super().__init__()
        self.hud_client = None
        self._hud_url = os.environ.get('HUD_URL') if HUD_AVAILABLE else None
        self._hud_initialized = False
        
        # Don't initialize HUD client here - wait until we're on the driver
        # This prevents all workers from trying to register sessions
    
    def _ensure_hud_client(self):
        """
        Lazy initialization of HUD client.
        Only called from driver (on_train_result), not from workers.
        """
        if not self._hud_initialized and HUD_AVAILABLE and self._hud_url:
            self._hud_initialized = True
            try:
                self.hud_client = HUDClient(hud_url=self._hud_url)
                if not self.hud_client.enabled:
                    print("⚠️  HUD connection failed, dashboard disabled")
                    self.hud_client = None
            except Exception as e:
                print(f"❌ Failed to initialize HUD client: {e}")
                self.hud_client = None
    
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
        
        NOTE: We don't update HUD from workers to avoid session conflicts.
        All HUD updates happen in on_train_result() on the driver.
        """
        # Skip - all HUD updates are done from the driver in on_train_result
        pass
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """
        Called after each training iteration (on driver/trainer).
        Update HUD with overall training metrics.
        
        This runs ONLY on the driver, so it's safe to initialize HUD here
        without causing session conflicts across workers.
        """
        # Lazy initialization - only happens once on the driver
        self._ensure_hud_client()
        
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
        
        # TODO: Vision data (screenshots) not yet implemented for Ray distributed training
        # The environment would need to implement get_screenshot_base64() method


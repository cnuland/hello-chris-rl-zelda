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
    
    # Designate which worker can send vision data to HUD
    HUD_WORKER_INDEX = 1  # Worker 1, Env 0
    
    def __init__(self):
        super().__init__()
        self.hud_client = None
        self._hud_url = os.environ.get('HUD_URL') if HUD_AVAILABLE else None
        self._hud_initialized = False
        self._session_manager = None
        self._last_checkpoint_iter = 0
        self.CHECKPOINT_FREQUENCY = 50  # Save model weights every 50 iterations
        
        # Don't initialize HUD client here - wait until we're on the driver
        # This prevents all workers from trying to register sessions
    
    def _ensure_hud_client(self):
        """
        Lazy initialization of HUD client.
        Only called from driver (on_train_result), not from workers.
        """
        if not self._hud_initialized:
            self._hud_initialized = True
            
            if not HUD_AVAILABLE:
                print("⚠️  HUD: HUD client module not available")
                return
            
            if not self._hud_url:
                print("⚠️  HUD: HUD_URL not set in environment")
                return
            
            try:
                print(f"🖥️  Initializing HUD client (from driver/callback)...")
                print(f"   HUD URL: {self._hud_url}")
                
                # IMPORTANT: Callback should NOT register a session!
                # Workers already registered. Callback just sends data.
                # Use a simple HTTP client instead of HUDClient to avoid registration.
                import requests
                
                class SimpleHUDClient:
                    """Lightweight HUD client that doesn't register sessions."""
                    def __init__(self, url):
                        self.hud_url = url
                        self.enabled = True
                        self.session_id = None  # Not used, but needed for compatibility
                    
                    def update_training_data(self, data):
                        """Send training data without session validation."""
                        try:
                            response = requests.post(
                                f"{self.hud_url}/api/update_training",
                                json={'session_id': None, 'data': data},
                                timeout=2
                            )
                            return response.status_code == 200
                        except:
                            return False
                
                self.hud_client = SimpleHUDClient(self._hud_url)
                print("✅ HUD callback initialized (no session registration)")
                
            except Exception as e:
                print(f"❌ Failed to initialize HUD client: {e}")
                import traceback
                traceback.print_exc()
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
        
        # Extract key metrics and map to HUD field names
        timesteps_total = result.get('timesteps_total', 0)
        episodes_total = result.get('episodes_total', 0)
        iteration = result.get('training_iteration', 0)
        
        # DEBUG: Log available keys to find episodes count
        if iteration % 10 == 0:  # Only log every 10 iterations
            print(f"🔍 DEBUG Ray result keys: {list(result.keys())[:20]}")
            print(f"   episodes_total: {episodes_total}")
            print(f"   episodes_this_iter: {result.get('episodes_this_iter', 'N/A')}")
            print(f"   num_episodes: {result.get('num_episodes', 'N/A')}")
            print(f"   env_runners: {result.get('env_runners', {}).keys() if 'env_runners' in result else 'N/A'}")
        
        training_data = {
            # HUD expects these exact field names
            # ONLY send GLOBAL metrics (not worker-specific!)
            'global_step': timesteps_total,  # Global step count
            'epoch': iteration,  # Training iteration (epoch)
            # NOTE: Don't send 'episode' or 'episode_reward' - workers handle these!
            # Sending them here overwrites worker data with 0 values
            'episode_len_mean': result.get('episode_len_mean', 0.0),  # Mean across all workers
            'episode_return_mean': result.get('episode_return_mean', 0.0),  # Mean across all workers
            'learning_rate': result.get('info', {}).get('learner', {}).get('default_policy', {}).get('cur_lr', 0.0),
            'iteration': iteration,  # keep for backward compatibility
        }
        
        # Add policy loss info if available
        learner_info = result.get('info', {}).get('learner', {}).get('default_policy', {})
        if learner_info:
            training_data['policy_loss'] = learner_info.get('learner_stats', {}).get('policy_loss', 0.0)
            training_data['vf_loss'] = learner_info.get('learner_stats', {}).get('vf_loss', 0.0)
            training_data['entropy'] = learner_info.get('learner_stats', {}).get('entropy', 0.0)
        
        # Send training metrics from callback
        # Workers will merge their data (vision, game state) with this
        try:
            if self.hud_client and self.hud_client.enabled:
                success = self.hud_client.update_training_data(training_data)
                if success:
                    print(f"📊 HUD updated (callback): epoch={training_data['epoch']}, "
                          f"global_steps={training_data['global_step']}, "
                          f"mean_return={training_data.get('episode_return_mean', 0):.1f}, "
                          f"mean_length={training_data.get('episode_len_mean', 0):.1f}")
                else:
                    print(f"⚠️  HUD update failed")
        except Exception as e:
            print(f"❌ Error sending data to HUD: {e}")
        
        # Save model checkpoint to S3 every N iterations
        if iteration > 0 and iteration % self.CHECKPOINT_FREQUENCY == 0:
            if iteration > self._last_checkpoint_iter:  # Avoid duplicate saves
                self._save_model_checkpoint(algorithm, iteration, result)
                self._last_checkpoint_iter = iteration
        
        # Note: Workers will merge vision data + game state with these training metrics
    
    def _save_model_checkpoint(self, algorithm, iteration, result):
        """Save model weights to S3/MinIO via SessionManager."""
        try:
            # Initialize SessionManager if not already done
            if self._session_manager is None:
                from session_manager import SessionManager
                import time
                session_id = f"ray_training_{int(time.time())}"
                self._session_manager = SessionManager(session_id=session_id)
                
                if not self._session_manager.enabled:
                    print(f"⚠️  SessionManager not enabled, skipping checkpoint save")
                    return
            
            # Get model weights from algorithm
            print(f"💾 Saving checkpoint at iteration {iteration}...")
            
            # Get model state dict
            model_weights = algorithm.get_policy().get_weights()
            
            # Serialize to bytes
            import pickle
            model_bytes = pickle.dumps(model_weights)
            
            # Save checkpoint with model weights
            checkpoint_data = {
                'iteration': iteration,
                'timesteps_total': result.get('timesteps_total', 0),
                'episode_return_mean': result.get('episode_return_mean', 0.0),
                'episode_len_mean': result.get('episode_len_mean', 0.0),
                'timestamp': result.get('time_this_iter_s', 0.0),
            }
            
            # Use iteration as "episode" number for consistency
            success = self._session_manager.save_checkpoint(
                worker_id=0,  # Driver/trainer
                episode_num=iteration,
                checkpoint_data=checkpoint_data,
                model_state=model_bytes
            )
            
            if success:
                print(f"✅ Checkpoint {iteration} saved to S3! Size: {len(model_bytes):,} bytes")
            else:
                print(f"❌ Failed to save checkpoint {iteration}")
                
        except Exception as e:
            print(f"❌ Error saving model checkpoint: {e}")
            import traceback
            traceback.print_exc()


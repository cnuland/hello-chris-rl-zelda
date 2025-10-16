"""
HUD Client for distributed training environments
Sends updates to remote HUD server via HTTP requests
Optimized with connection pooling for speed
"""

import os
import requests
import json
from typing import Optional, Dict


class HUDClient:
    """
    Client for sending updates to the HUD server.
    Works in both local (imported) and remote (HTTP) modes.
    """
    
    def __init__(self, hud_url: Optional[str] = None):
        """
        Initialize HUD client with async threading support.
        
        Args:
            hud_url: URL of HUD server (e.g., http://zelda-hud-service:5000)
                     If None, will check HUD_URL environment variable
        """
        self.hud_url = hud_url or os.environ.get('HUD_URL')
        self.session_id = None
        self.enabled = False
        
        # Connection pooling for faster HTTP requests (keep persistent connections)
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=0  # Don't retry, just drop if fails
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        if self.hud_url:
            print(f"🖥️  HUD Client connecting to: {self.hud_url}")
            self.register_session()
        else:
            print("⚠️  HUD_URL not set, dashboard disabled")
    
    def register_session(self) -> bool:
        """
        Register this training session with the HUD server.
        Single-session model: Only one worker can control HUD at a time.
        If another session is active (409), this client stays disabled.
        
        Returns:
            bool: True if registration successful
        """
        if not self.hud_url:
            return False
        
        try:
            response = self.session.post(
                f"{self.hud_url}/api/register_session",
                timeout=2
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get('session_id')
                self.enabled = True
                print(f"✅ HUD session registered: {self.session_id[:8]}... (this worker controls HUD)")
                return True
            elif response.status_code == 409:
                # Another session active - this worker will NOT control HUD
                self.enabled = False
                if not hasattr(self, '_registration_rejected_logged'):
                    print(f"ℹ️  HUD already in use by another worker (this is normal)")
                    self._registration_rejected_logged = True
                return False
            else:
                return False
        except requests.exceptions.Timeout as e:
            print(f"❌ HUD connection TIMEOUT after 10s: {e}")
            print(f"   Is the HUD server running at {self.hud_url}?")
            import sys
            sys.stdout.flush()
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"❌ HUD connection ERROR: {e}")
            print(f"   Cannot reach HUD server at {self.hud_url}")
            import sys
            sys.stdout.flush()
            return False
        except Exception as e:
            print(f"❌ Unexpected error connecting to HUD: {type(e).__name__}: {e}")
            import sys
            sys.stdout.flush()
            return False
    
    def update_training_data(self, data: Dict) -> bool:
        """
        Update training metrics on the HUD (fast, simple).
        
        Args:
            data: Dictionary containing training metrics
            
        Returns:
            bool: True if update successful
        """
        if not self.enabled or not self.session_id:
            return False
        
        try:
            # Debug: Log occasional training updates
            import random
            if random.random() < 0.1:  # 10% of updates
                print(f"📊 Sending training update (step={data.get('global_step', '?')}, episode={data.get('episode', '?')})")
            
            response = self.session.post(
                f"{self.hud_url}/api/update_training",
                json={
                    'session_id': self.session_id,
                    'data': data
                },
                timeout=2  # Reasonable timeout
            )
            success = response.status_code == 200
            if not success and response.status_code != 403:
                print(f"⚠️ Training update failed: {response.status_code}")
            return success
        except Exception as e:
            print(f"⚠️ Training update exception: {e}")
            return False
    
    def update_vision_data(self, image_base64: str, response_time: Optional[float] = None) -> bool:
        """
        Update vision image on the HUD (fast, simple).
        
        Args:
            image_base64: Base64 encoded image
            response_time: LLM response time in milliseconds
            
        Returns:
            bool: True if update successful
        """
        if not self.enabled or not self.session_id:
            return False
        
        try:
            payload = {
                'session_id': self.session_id,
                'image_base64': image_base64,
            }
            if response_time is not None:
                payload['response_time'] = response_time
            
            # Debug: Log occasional updates
            import random
            if random.random() < 0.05:  # 5% of updates
                print(f"📤 Sending vision update (image size: {len(image_base64)} chars)")
            
            response = self.session.post(
                f"{self.hud_url}/api/update_vision",
                json=payload,
                timeout=2  # Reasonable timeout
            )
            success = response.status_code == 200
            if not success and response.status_code != 403:
                print(f"⚠️ Vision update failed: {response.status_code}")
            return success
        except Exception as e:
            print(f"⚠️ Vision update exception: {e}")
            return False
    
    def close(self):
        """Close the session and cleanup resources"""
        self.enabled = False
        self.session_id = None
        self.session.close()


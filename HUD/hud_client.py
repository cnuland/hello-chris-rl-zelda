"""
HUD Client for distributed training environments
Sends updates to remote HUD server via HTTP requests
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
        Initialize HUD client.
        
        Args:
            hud_url: URL of HUD server (e.g., http://zelda-hud-service:5000)
                     If None, will check HUD_URL environment variable
        """
        self.hud_url = hud_url or os.environ.get('HUD_URL')
        self.session_id = None
        self.enabled = False
        self.retry_counter = 0  # Track failed registration attempts
        self.retry_delay = 10   # Wait 10 resets before retrying after 409
        
        if self.hud_url:
            print(f"🖥️  HUD Client connecting to: {self.hud_url}")
            self.register_session()
        else:
            print("⚠️  HUD_URL not set, dashboard disabled")
    
    def register_session(self) -> bool:
        """
        Register this training session with the HUD server.
        Implements backoff to avoid DDOSing server when slot is taken.
        
        Returns:
            bool: True if registration successful
        """
        if not self.hud_url:
            print("⚠️  HUD URL not set, skipping registration")
            return False
        
        # Backoff mechanism: if we were rejected before, wait before retrying
        if self.retry_counter > 0:
            self.retry_counter -= 1
            if self.retry_counter % 5 == 0:  # Only log every 5 resets
                print(f"⏳ Waiting to retry HUD registration ({self.retry_counter} resets remaining)")
            return False
        
        print(f"📡 Attempting HUD registration at: {self.hud_url}/api/register_session")
        print(f"   Timeout: 10 seconds")
        
        try:
            import sys
            sys.stdout.flush()  # Force output to appear immediately
            
            response = requests.post(
                f"{self.hud_url}/api/register_session",
                timeout=10  # Increased timeout
            )
            
            print(f"📡 HUD server responded with status: {response.status_code}")
            sys.stdout.flush()
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get('session_id')
                self.enabled = True
                self.retry_counter = 0  # Reset retry counter on success
                print(f"✅ HUD session registered: {self.session_id[:8]}...")
                sys.stdout.flush()
                return True
            elif response.status_code == 409:
                # HUD is in use - back off to avoid DDOSing the server
                self.retry_counter = self.retry_delay
                print(f"⚠️  HUD already in use (backing off for {self.retry_delay} resets)")
                sys.stdout.flush()
                self.enabled = False
                return False
            else:
                print(f"❌ HUD registration failed: {response.status_code} - {response.text}")
                sys.stdout.flush()
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
        Update training metrics on the HUD.
        
        Args:
            data: Dictionary containing training metrics
            
        Returns:
            bool: True if update successful
        """
        if not self.enabled or not self.session_id:
            return False
        
        try:
            response = requests.post(
                f"{self.hud_url}/api/update_training",
                json={
                    'session_id': self.session_id,
                    'data': data
                },
                timeout=2
            )
            
            if response.status_code == 200:
                return True
            elif response.status_code == 403:
                # Session validation issue (should not happen with shared session mode)
                print(f"⚠️  HUD update rejected (403) - retrying next time")
                return False  # Don't disable, just retry next time
            else:
                return False
        except Exception as e:
            # Don't spam errors, just fail silently
            return False
    
    def update_vision_data(self, image_base64: str, response_time: Optional[float] = None) -> bool:
        """
        Update vision image on the HUD.
        
        Args:
            image_base64: Base64 encoded JPEG image
            response_time: LLM response time in milliseconds
            
        Returns:
            bool: True if update successful
        """
        if not self.enabled or not self.session_id:
            return False
        
        try:
            # Build payload - only include response_time if it's a real LLM call
            payload = {
                'session_id': self.session_id,
                'image_base64': image_base64,
            }
            
            # Only include response_time for actual LLM calls (not streaming)
            if response_time is not None:
                payload['response_time'] = response_time
            
            response = requests.post(
                f"{self.hud_url}/api/update_vision",
                json=payload,
                timeout=2
            )
            
            if response.status_code == 200:
                return True
            elif response.status_code == 403:
                # Session validation issue (should not happen with shared session mode)
                print(f"⚠️  HUD update rejected (403) - retrying next time")
                return False  # Don't disable, just retry next time
            else:
                return False
        except Exception as e:
            # Don't spam errors, just fail silently
            return False
    
    def close(self):
        """Close the session (optional cleanup)"""
        self.enabled = False
        self.session_id = None


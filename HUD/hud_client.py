"""
HUD Client for distributed training environments
Sends updates to remote HUD server via HTTP requests
Uses threading for non-blocking async updates
"""

import os
import requests
import json
import threading
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty


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
        self.retry_counter = 0  # Track failed registration attempts
        self.retry_delay = 10   # Wait 10 resets before retrying after 409
        
        # Threading support for non-blocking updates
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hud_client")
        self.update_in_progress = False
        self.update_lock = threading.Lock()
        
        # Connection pooling for faster HTTP requests
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=2,
            pool_maxsize=2,
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
            
            response = self.session.post(
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
    
    def _send_training_data(self, data: Dict) -> bool:
        """Internal method to actually send training data (runs in thread)."""
        try:
            response = self.session.post(
                f"{self.hud_url}/api/update_training",
                json={
                    'session_id': self.session_id,
                    'data': data
                },
                timeout=1  # Fast timeout for non-blocking
            )
            return response.status_code == 200
        except Exception:
            return False
        finally:
            with self.update_lock:
                self.update_in_progress = False
    
    def update_training_data(self, data: Dict) -> bool:
        """
        Update training metrics on the HUD (async, non-blocking).
        
        Args:
            data: Dictionary containing training metrics
            
        Returns:
            bool: True if update queued successfully
        """
        if not self.enabled or not self.session_id:
            return False
        
        # Skip if previous update still in progress (drop frame)
        with self.update_lock:
            if self.update_in_progress:
                return False  # Drop this update, don't block
            self.update_in_progress = True
        
        # Send in background thread
        self.executor.submit(self._send_training_data, data)
        return True
    
    def _send_vision_data(self, image_base64: str, response_time: Optional[float]) -> bool:
        """Internal method to actually send vision data (runs in thread)."""
        try:
            payload = {
                'session_id': self.session_id,
                'image_base64': image_base64,
            }
            if response_time is not None:
                payload['response_time'] = response_time
            
            response = self.session.post(
                f"{self.hud_url}/api/update_vision",
                json=payload,
                timeout=1  # Fast timeout for non-blocking
            )
            return response.status_code == 200
        except Exception:
            return False
        finally:
            with self.update_lock:
                self.update_in_progress = False
    
    def update_vision_data(self, image_base64: str, response_time: Optional[float] = None) -> bool:
        """
        Update vision image on the HUD (async, non-blocking).
        
        Args:
            image_base64: Base64 encoded JPEG image
            response_time: LLM response time in milliseconds
            
        Returns:
            bool: True if update queued successfully
        """
        if not self.enabled or not self.session_id:
            return False
        
        # Skip if previous update still in progress (drop frame)
        with self.update_lock:
            if self.update_in_progress:
                return False  # Drop this update, don't block
            self.update_in_progress = True
        
        # Send in background thread
        self.executor.submit(self._send_vision_data, image_base64, response_time)
        return True
    
    def close(self):
        """Close the session and cleanup resources"""
        self.enabled = False
        self.session_id = None
        self.executor.shutdown(wait=False)  # Don't wait for pending updates
        self.session.close()


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
        
        if self.hud_url:
            print(f"🖥️  HUD Client connecting to: {self.hud_url}")
            self.register_session()
        else:
            print("⚠️  HUD_URL not set, dashboard disabled")
    
    def register_session(self) -> bool:
        """
        Register this training session with the HUD server.
        
        Returns:
            bool: True if registration successful
        """
        if not self.hud_url:
            return False
        
        try:
            response = requests.post(
                f"{self.hud_url}/api/register_session",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get('session_id')
                self.enabled = True
                print(f"✅ HUD session registered: {self.session_id[:8]}...")
                return True
            elif response.status_code == 409:
                print(f"⚠️  HUD already in use by another session")
                return False
            else:
                print(f"❌ HUD registration failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to HUD: {e}")
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
                # Session no longer active
                print(f"⚠️  Lost HUD connection - another session took over")
                self.enabled = False
                return False
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
            response = requests.post(
                f"{self.hud_url}/api/update_vision",
                json={
                    'session_id': self.session_id,
                    'image_base64': image_base64,
                    'response_time': response_time
                },
                timeout=2
            )
            
            if response.status_code == 200:
                return True
            elif response.status_code == 403:
                # Session no longer active
                print(f"⚠️  Lost HUD connection - another session took over")
                self.enabled = False
                return False
            else:
                return False
        except Exception as e:
            # Don't spam errors, just fail silently
            return False
    
    def close(self):
        """Close the session (optional cleanup)"""
        self.enabled = False
        self.session_id = None


"""
Ray RLlib-compatible wrapper for Zelda Oracle of Seasons environment.
This wraps the existing ZeldaConfigurableEnvironment to make it Ray-compatible
while preserving all existing functionality (vision LLM, rewards, tracking, etc.).
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path
import os
import threading
import time
import yaml
import requests
import base64
import io
from PIL import Image

# Import our existing Zelda environment
from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment

# Import ROM download utility
from init_rom import init_rom_from_s3

# Global lock to ensure only one download per pod
_rom_download_lock = threading.Lock()
_rom_downloaded = False


class ZeldaRayEnv(ZeldaConfigurableEnvironment):
    """
    Ray RLlib-compatible wrapper for Zelda environment.
    
    This is a thin wrapper around ZeldaConfigurableEnvironment that ensures
    Ray compatibility while preserving all existing functionality:
    - Vector observations for PPO
    - Vision LLM integration
    - Structured states
    - Reward system
    - NPC/room tracking
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Ray-compatible Zelda environment.
        
        Args:
            config: Ray environment configuration dict
        """
        config = config if config is not None else {}
        
        # Initialize step counter for logging
        self._step_count = 0
        self._episode_count = 0
        self._total_reward = 0.0
        
        # DEBUG: Print environment info
        print("\n" + "="*70)
        print("🔍 DEBUG: ZeldaRayEnv Path Resolution")
        print("="*70)
        print(f"CWD: {Path.cwd()}")
        print(f"__file__: {__file__ if '__file__' in globals() else 'N/A'}")
        print(f"Script dir: {Path(__file__).parent.resolve()}")
        print(f"\nCWD contents:")
        try:
            for item in sorted(Path.cwd().iterdir())[:20]:
                print(f"  - {item.name}{'/' if item.is_dir() else ''}")
        except Exception as e:
            print(f"  Error listing CWD: {e}")
        
        print(f"\nScript dir contents:")
        try:
            script_dir = Path(__file__).parent.resolve()
            for item in sorted(script_dir.iterdir())[:20]:
                print(f"  - {item.name}{'/' if item.is_dir() else ''}")
        except Exception as e:
            print(f"  Error listing script dir: {e}")
        
        print(f"\nLooking for 'roms' directory:")
        cwd_roms = Path.cwd() / 'roms'
        script_roms = Path(__file__).parent.resolve() / 'roms'
        print(f"  - CWD/roms exists: {cwd_roms.exists()}")
        if cwd_roms.exists():
            print(f"    Contents: {list(cwd_roms.iterdir())[:5]}")
        print(f"  - Script/roms exists: {script_roms.exists()}")
        if script_roms.exists():
            print(f"    Contents: {list(script_roms.iterdir())[:5]}")
        print("="*70 + "\n")
        
        # Extract paths and configuration
        rom_path = config.get('gb_path', 'roms/zelda_oracle_of_seasons.gbc')
        config_path = config.get('env_config_path', 'configs/env.yaml')
        headless = config.get('headless', True)
        
        # Ensure ROM files are downloaded (once per pod, thread-safe)
        self._ensure_rom_files(rom_path)
        
        # Resolve paths robustly
        rom_path = self._resolve_path(rom_path)
        config_path = self._resolve_path(config_path)
        
        # Initialize the base Zelda environment
        super().__init__(
            rom_path=rom_path,
            config_path=config_path if config_path and Path(config_path).exists() else None,
            headless=headless,
            render_mode=None,
            visual_test_mode=False
        )
        
        # Store Ray-specific config
        self.ray_config = config
        self.instance_id = config.get('worker_index', 0)
        
        # Initialize Vision LLM integration if enabled
        self._init_vision_llm()
        
        # Initialize HUD client for vision updates
        self._init_hud_client()
        
        print(f"✅ ZeldaRayEnv initialized (Instance: {self.instance_id})")
        print(f"   ROM: {rom_path}")
        print(f"   Config: {config_path}")
        print(f"   Observation space: {self.observation_space.shape}")
        print(f"   Action space: {self.action_space}")
        if self.llm_enabled:
            print(f"   🧠 Vision LLM: ENABLED (every {self.llm_frequency} steps)")
            print(f"   📸 Image: {160*self.image_scale}×{144*self.image_scale}, {self.image_quality}% JPEG")
        if hasattr(self, 'hud_client') and self.hud_client and self.hud_client.enabled:
            print(f"   🖥️  HUD Client: ENABLED")
    
    def _init_vision_llm(self):
        """Initialize vision LLM integration if enabled in config."""
        # Initialize as disabled by default
        self.llm_enabled = False
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.last_llm_suggestion = None
        self.last_llm_action = None
        
        # Check if LLM is enabled from environment config
        try:
            if not hasattr(self, 'config') or not self.config:
                print("   🧠 Vision LLM: DISABLED (no config)")
                return
            
            # planner_integration is nested under 'performance' in env.yaml
            perf_config = self.config.get('performance', {})
            planner_config = perf_config.get('planner_integration', {})
            if not planner_config:
                print("   🧠 Vision LLM: DISABLED (no planner_integration in config)")
                print(f"   Debug: performance keys: {list(perf_config.keys()) if perf_config else 'none'}")
                return
            
            use_planner = planner_config.get('use_planner', False)
            enable_visual = planner_config.get('enable_visual', False)
            
            if not (use_planner and enable_visual):
                print(f"   🧠 Vision LLM: DISABLED (use_planner={use_planner}, enable_visual={enable_visual})")
                return
            
            self.llm_enabled = True
            
        except Exception as e:
            print(f"   ❌ Error checking LLM config: {e}")
            return
        
        # Load vision prompt configuration (check performance section first)
        try:
            vision_config_path = perf_config.get('vision_prompt_config', planner_config.get('vision_prompt_config', 'configs/vision_prompt.yaml'))
            vision_config_path = self._resolve_path(vision_config_path)
            
            with open(vision_config_path, 'r') as f:
                vision_config = yaml.safe_load(f)
            
            # Extract LLM settings (from performance section or planner_config)
            self.llm_frequency = perf_config.get('llm_frequency', planner_config.get('llm_frequency', vision_config.get('behavior', {}).get('call_frequency', 5)))
            self.alignment_bonus_multiplier = perf_config.get('alignment_bonus_multiplier', planner_config.get('alignment_bonus_multiplier', vision_config.get('behavior', {}).get('alignment_bonus_multiplier', 2.0)))
            
            # Extract vision settings (from performance section or planner_config)
            self.image_scale = perf_config.get('image_scale', planner_config.get('image_scale', vision_config.get('vision_config', {}).get('image_scale', 2)))
            self.image_quality = perf_config.get('image_quality', planner_config.get('image_quality', vision_config.get('vision_config', {}).get('image_quality', 75)))
            self.image_format = perf_config.get('image_format', planner_config.get('image_format', vision_config.get('vision_config', {}).get('image_format', 'jpeg')))
            
            # Store prompts
            self.system_prompt = vision_config.get('system_prompt', '')
            self.user_prompt_template = vision_config.get('vision_user_prompt_template', '')
            
            # LLM endpoint from environment variable (check performance section first)
            llm_endpoint_var = perf_config.get('llm_endpoint_env_var', planner_config.get('llm_endpoint_env_var', 'LLM_ENDPOINT'))
            self.llm_endpoint = os.environ.get(llm_endpoint_var, '')
            
            if not self.llm_endpoint:
                print(f"   ⚠️  {llm_endpoint_var} not set, Vision LLM disabled")
                self.llm_enabled = False
                return
            
            # LLM model name (optional - if empty, don't send model parameter)
            self.llm_model_name = perf_config.get('llm_model_name', planner_config.get('llm_model_name', ''))
            
            # LLM Host header (required for some load balancers/ingress controllers)
            self.llm_host_header = os.environ.get('LLM_HOST_HEADER', '')
            
            print(f"   ✅ Vision LLM config loaded from: {vision_config_path}")
            print(f"   📡 LLM Endpoint: {self.llm_endpoint}")
            if self.llm_host_header:
                print(f"   🌐 Host Header: {self.llm_host_header}")
            if self.llm_model_name:
                print(f"   🤖 Model: {self.llm_model_name}")
            else:
                print(f"   🤖 Model: (using server default)")
            
        except FileNotFoundError as e:
            print(f"   ❌ Vision config file not found: {e}")
            print(f"   Vision LLM disabled")
            self.llm_enabled = False
        except Exception as e:
            print(f"   ❌ Failed to load vision config: {e}")
            print(f"   Vision LLM disabled")
            self.llm_enabled = False
    
    def _init_hud_client(self):
        """Initialize HUD client for sending vision updates."""
        self.hud_client = None
        
        print(f"   🖥️  HUD init check: instance_id={self.instance_id}")
        
        # Only initialize HUD on worker 1, env 0 to avoid multiple sessions
        if self.instance_id != 1:
            print(f"   ⏭️  Skipping HUD init (instance_id={self.instance_id}, need 1)")
            return
        
        print(f"   🎯 This is worker 1! Initializing HUD client...")
        
        try:
            hud_url = os.environ.get('HUD_URL')
            if not hud_url:
                print("   🖥️  HUD: No HUD_URL set")
                return
            
            # Import HUD client
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent / 'HUD'))
            
            from hud_client import HUDClient
            
            self.hud_client = HUDClient(hud_url=hud_url)
            
            if self.hud_client.enabled:
                print(f"   ✅ HUD client initialized for worker {self.instance_id}")
            else:
                print(f"   ⚠️  HUD client failed to connect")
                self.hud_client = None
                
        except ImportError as e:
            print(f"   ⚠️  HUD client not available: {e}")
            self.hud_client = None
        except Exception as e:
            print(f"   ❌ Error initializing HUD client: {e}")
            self.hud_client = None
    
    @staticmethod
    def _ensure_rom_files(rom_path: str):
        """
        Ensure ROM files are downloaded on this worker pod.
        Uses a global lock to ensure only one download per pod, even with multiple environments.
        """
        global _rom_downloaded
        
        # Quick check without lock (most common case: ROM already downloaded)
        if _rom_downloaded:
            return
        
        # Acquire lock for thread-safe download check/execution
        with _rom_download_lock:
            # Double-check inside lock (another thread may have downloaded while we waited)
            if _rom_downloaded:
                return
            
            # Check if ROM file exists
            rom_file = Path(rom_path)
            # Also check in CWD (Ray's working_dir extraction location)
            cwd_rom = Path.cwd() / rom_path
            
            if rom_file.exists() or cwd_rom.exists():
                print(f"✅ ROM file found, no download needed")
                _rom_downloaded = True
                return
            
            # ROM doesn't exist - download from S3
            print(f"\n{'='*70}")
            print(f"📥 ROM files not found on this worker pod - downloading from S3...")
            print(f"{'='*70}")
            
            try:
                success = init_rom_from_s3()
                if success:
                    print(f"✅ ROM download successful!")
                    _rom_downloaded = True
                else:
                    print(f"⚠️  ROM download reported failure, but continuing...")
                    # Don't set _rom_downloaded=True so next env will try again
            except Exception as e:
                print(f"❌ Error downloading ROM: {e}")
                print(f"   Training will likely fail, but letting it try...")
                # Don't set _rom_downloaded=True so next env will try again
            
            print(f"{'='*70}\n")
    
    @staticmethod
    def _resolve_path(path_str: str) -> str:
        """Resolve path relative to script directory or CWD."""
        print(f"\n🔍 Resolving path: {path_str}")
        path = Path(path_str)
        
        # Try as-is (absolute or correct relative)
        print(f"  Trying as-is: {path}")
        if path.exists():
            print(f"    ✅ Found!")
            return str(path)
        print(f"    ❌ Not found")
        
        # Try relative to script directory (Ray working_dir extraction location)
        # This is the most reliable for Ray, as __file__ will be in the extracted working_dir
        try:
            script_dir = Path(__file__).parent.resolve()
            script_path = script_dir / path
            print(f"  Trying script dir: {script_path}")
            if script_path.exists():
                print(f"    ✅ Found!")
                return str(script_path)
            print(f"    ❌ Not found")
        except NameError:
            print(f"  Script dir: N/A (__file__ not available)")
        
        # Try relative to CWD
        cwd_path = Path.cwd() / path
        print(f"  Trying CWD: {cwd_path}")
        if cwd_path.exists():
            print(f"    ✅ Found!")
            return str(cwd_path)
        print(f"    ❌ Not found")
        
        # Last resort: check WORKING_DIR env var (set by Ray)
        working_dir = os.environ.get('RAY_WORKING_DIR')
        if working_dir:
            working_dir_path = Path(working_dir) / path
            print(f"  Trying RAY_WORKING_DIR: {working_dir_path}")
            if working_dir_path.exists():
                print(f"    ✅ Found!")
                return str(working_dir_path)
            print(f"    ❌ Not found")
        else:
            print(f"  RAY_WORKING_DIR env var: Not set")
        
        # Return original if not found (will error later with clear message)
        print(f"  ⚠️  ALL ATTEMPTS FAILED - returning original path")
        return str(path)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment.
        Ray RLlib requires the (observation, info) return format.
        """
        import time
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"🔄 RESET called on Worker {self.instance_id}")
        print(f"{'='*70}")
        
        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)
        
        # Ensure observation is the correct type
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        
        # Reset counters
        self._step_count = 0
        self._episode_count += 1
        self._total_reward = 0.0
        
        elapsed = time.time() - start_time
        print(f"✅ RESET complete in {elapsed:.2f}s")
        print(f"   Episode: {self._episode_count}")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Info keys: {list(info.keys())}")
        print(f"{'='*70}\n")
        
        return obs, info
    
    def capture_screenshot_base64(self) -> Optional[str]:
        """Capture Game Boy screen as base64-encoded JPEG for LLM."""
        if not self.llm_enabled:
            return None
        
        try:
            # Get screen data from PyBoy
            screen_array = self.bridge.pyboy.screen.ndarray.copy()
            
            # Convert to PIL Image
            image = Image.fromarray(screen_array)
            
            # Convert RGBA to RGB (JPEG doesn't support alpha channel)
            if image.mode == 'RGBA':
                # Create white background and paste RGBA image on it
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Upscale for better LLM understanding
            if self.image_scale > 1:
                new_size = (image.width * self.image_scale, image.height * self.image_scale)
                image = image.resize(new_size, Image.Resampling.NEAREST)
            
            # Convert to JPEG and encode as base64
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self.image_quality)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            print(f"⚠️  Screenshot capture failed: {e}")
            return None
    
    def call_llm_vision(self, game_state: Dict, screenshot_base64: Optional[str] = None) -> Optional[str]:
        """Call vision LLM with screenshot and game state."""
        if not self.llm_enabled or not screenshot_base64:
            return None
        
        try:
            # Format prompt with game state
            user_prompt = self.user_prompt_template.format(
                location=game_state.get('location', {}).get('name', 'Unknown'),
                cave_hint=game_state.get('location', {}).get('cave_hint', ''),
                health=game_state.get('stats', {}).get('health', 0),
                max_health=game_state.get('stats', {}).get('max_health', 0),
                x=game_state.get('location', {}).get('x', 0),
                y=game_state.get('location', {}).get('y', 0),
                npc_count=len(game_state.get('entities', {}).get('npcs', [])),
                enemy_count=len(game_state.get('entities', {}).get('enemies', [])),
                item_count=len(game_state.get('entities', {}).get('items', []))
            )
            
            # Prepare API request
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{screenshot_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            # Only include model parameter if specified in config
            if self.llm_model_name:
                payload["model"] = self.llm_model_name
            
            # Prepare headers (including Host header if specified)
            headers = {"Content-Type": "application/json"}
            if self.llm_host_header:
                headers["Host"] = self.llm_host_header
            
            # Call LLM API
            response = requests.post(
                self.llm_endpoint,
                json=payload,
                headers=headers,
                timeout=5  # 5 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                suggestion = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                return suggestion if suggestion else None
            else:
                # Log detailed error for debugging
                error_msg = f"⚠️  LLM returned status {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text[:200]}"
                
                # Only print first error to avoid spam
                if not hasattr(self, '_llm_error_logged'):
                    print(error_msg)
                    print(f"   Endpoint: {self.llm_endpoint}")
                    print(f"   Model: {self.llm_model_name if self.llm_model_name else '(server default)'}")
                    print(f"   Tip: Check if endpoint supports vision API (image_url content type)")
                    print(f"   Tip: Or set llm_model_name in env.yaml if specific model needed")
                    self._llm_error_logged = True
                return None
                
        except requests.exceptions.Timeout:
            print("⚠️  LLM request timed out")
            return None
        except Exception as e:
            print(f"⚠️  LLM call failed: {e}")
            return None
    
    def compute_llm_alignment_bonus(self, action: int, llm_suggestion: Optional[str]) -> float:
        """Compute reward bonus if PPO action aligns with LLM suggestion."""
        if not llm_suggestion:
            return 0.0
        
        # Map actions to button names
        action_names = ["NOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
        if action >= len(action_names):
            return 0.0
        
        action_name = action_names[action]
        suggestion_upper = llm_suggestion.upper()
        
        # Check if LLM suggested this action
        if action_name in suggestion_upper or (action_name == "A" and "TALK" in suggestion_upper):
            return self.alignment_bonus_multiplier
        
        return 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        Ray RLlib requires (obs, reward, terminated, truncated, info) return format.
        """
        import time
        start_time = time.time()
        
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Call Vision LLM for guidance (every N steps)
        llm_bonus = 0.0
        if self.llm_enabled and self._step_count % self.llm_frequency == 0:
            # Get structured game state
            if hasattr(self, 'state_encoder') and self.state_encoder:
                # encode_state returns (vector_obs, structured_state)
                _, game_state = self.state_encoder.encode_state(self.bridge)
            else:
                game_state = {}
            
            # Capture screenshot
            screenshot = self.capture_screenshot_base64()
            
            if screenshot:
                # Call vision LLM
                llm_start_time = time.time()
                llm_suggestion = self.call_llm_vision(game_state, screenshot)
                llm_response_time = (time.time() - llm_start_time) * 1000  # Convert to ms
                
                if llm_suggestion:
                    self.llm_call_count += 1
                    self.llm_success_count += 1
                    self.last_llm_suggestion = llm_suggestion
                    
                    # Compute alignment bonus
                    llm_bonus = self.compute_llm_alignment_bonus(action, llm_suggestion)
                    
                    if llm_bonus > 0:
                        print(f"📸 LLM suggested: {llm_suggestion[:50]}... → Action {action} = +{llm_bonus:.1f} bonus!")
                    
                    # Send to HUD (same snapshot LLM sees!)
                    if self.hud_client and self.hud_client.enabled:
                        try:
                            # Send vision data (screenshot)
                            self.hud_client.update_vision_data(screenshot, llm_response_time)
                            
                            # Send training data (game state)
                            hud_training_data = {
                                'llm_suggestion': llm_suggestion,
                                'llm_calls': self.llm_call_count,
                                'llm_success_rate': self.llm_success_count / max(self.llm_call_count, 1),
                                'alignment_bonus': llm_bonus,
                                'step': self._step_count,
                                'episode': self._episode_count,
                                'location': game_state.get('location', {}).get('name', 'Unknown'),
                                'room_id': game_state.get('location', {}).get('room', 0),
                                'health': game_state.get('stats', {}).get('health', 0),
                                'max_health': game_state.get('stats', {}).get('max_health', 0),
                            }
                            self.hud_client.update_training_data(hud_training_data)
                        except Exception as e:
                            # Don't crash training if HUD fails
                            pass
                else:
                    self.llm_call_count += 1
            
            # Add LLM stats to info
            info['llm_calls'] = self.llm_call_count
            info['llm_success_rate'] = self.llm_success_count / max(self.llm_call_count, 1)
        
        # Ensure observation is the correct type
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        
        # Add LLM alignment bonus to reward
        reward = float(reward) + llm_bonus
        
        # Update counters and logging
        self._step_count += 1
        self._total_reward += reward
        elapsed = time.time() - start_time
        
        # Log every 100 steps or at episode end
        if self._step_count % 100 == 0 or terminated or truncated:
            done_str = " [DONE]" if (terminated or truncated) else ""
            print(f"🎮 Step {self._step_count:4d}: reward={reward:+.2f}, total={self._total_reward:+.1f}, time={elapsed:.3f}s{done_str}")
            
            # Extra logging at episode end
            if terminated or truncated:
                print(f"\n{'='*70}")
                print(f"🏁 EPISODE {self._episode_count} COMPLETE")
                print(f"   Steps: {self._step_count}")
                print(f"   Total Reward: {self._total_reward:.2f}")
                print(f"   Avg Reward/Step: {self._total_reward/max(self._step_count,1):.3f}")
                if 'llm_calls' in info:
                    print(f"   LLM Calls: {info.get('llm_calls', 0)}")
                if 'rooms_discovered' in info:
                    print(f"   Rooms Discovered: {info.get('rooms_discovered', 0)}")
                print(f"{'='*70}\n")
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Clean up environment resources."""
        super().close()
        print(f"ZeldaRayEnv (Instance: {self.instance_id}) closed.")


def create_zelda_ray_env(config: dict):
    """
    Factory function for Ray environment registration.
    
    Args:
        config: Ray environment configuration
    
    Returns:
        ZeldaRayEnv instance
    """
    return ZeldaRayEnv(config)

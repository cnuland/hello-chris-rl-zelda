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
        
        # Initialize exploration tracking
        self.rooms_discovered = set()
        self.grid_areas_explored = set()
        self.buildings_entered = set()
        
        # Initialize milestone tracking
        self.milestones = {
            'maku_tree_entered': False,
            'dungeon_entered': False,
            'sword_usage': 0
        }
        
        # DEBUG: Print environment info
        print("\n" + "="*70)
        print("🔍 DEBUG: ZeldaRayEnv Path Resolution")
        print("🆕 CODE VERSION: 2025-10-15-07:00 [FIX: HUD data format]")
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
        
        # Get worker ID from Ray context (for distributed training)
        # Simplified approach: Let HUD registration determine which worker sends data
        try:
            import ray
            if ray.is_initialized():
                # Get worker info from Ray
                runtime_context = ray.get_runtime_context()
                worker_id = runtime_context.get_worker_id()
                
                # Generate a unique instance ID from worker ID
                import hashlib
                if worker_id:
                    # Hash to get consistent ID
                    id_hash = hashlib.md5(worker_id.encode() if isinstance(worker_id, str) else worker_id).hexdigest()
                    self.instance_id = int(id_hash[:8], 16) % 10000
                else:
                    self.instance_id = 0
                
                # Don't pre-designate - let HUD client registration decide
                # The first worker to successfully register will send data
                self.is_hud_designated = True  # All workers try, HUD server picks one
                
                print(f"   🔍 Worker ID: {worker_id}, Instance: {self.instance_id}")
            else:
                self.instance_id = 0
                self.is_hud_designated = True
                print(f"   🔍 Ray not initialized, but will try HUD")
        except Exception as e:
            self.instance_id = 0
            self.is_hud_designated = True
            print(f"   ⚠️  Could not get Ray context: {e}, but will try HUD")
        
        # Initialize Vision LLM integration if enabled
        self._init_vision_llm()
        
        # Initialize HUD client for vision updates
        self._init_hud_client()
        
        # Initialize Session Manager for saving checkpoints and summaries
        self._init_session_manager()
        
        print(f"✅ ZeldaRayEnv initialized (Instance: {self.instance_id})")
        print(f"   ROM: {rom_path}")
        print(f"   Config: {config_path}")
        print(f"   Observation space: {self.observation_space.shape}")
        print(f"   Action space: {self.action_space}")
        if self.llm_enabled:
            print(f"   🧠 LLM Guidance: ENABLED (probability-based sampling)")
            print(f"      💬 Text-only calls: {self.llm_text_probability*100:.1f}% chance per step → +{self.text_alignment_bonus} reward")
            print(f"      📸 Vision calls: {self.llm_vision_probability*100:.1f}% chance per step → +{self.vision_alignment_bonus} reward")
            print(f"      🖼️  Image: {160*self.image_scale}×{144*self.image_scale}, {self.image_quality}% JPEG")
        if hasattr(self, 'hud_client') and self.hud_client and self.hud_client.enabled:
            print(f"   🖥️  HUD Client: ENABLED")
        if hasattr(self, 'session_manager') and self.session_manager and self.session_manager.enabled:
            print(f"   💾 Session Manager: ENABLED → s3://sessions/{self.session_manager.session_id}")
    
    def _init_vision_llm(self):
        """Initialize vision LLM integration if enabled in config."""
        print("🧠 === _init_vision_llm() CALLED ===")
        
        # Initialize as disabled by default
        self.llm_enabled = False
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.last_llm_suggestion = None
        self.last_llm_action = None
        
        # Check if LLM is enabled from environment config
        try:
            print(f"🔍 Checking config... hasattr={hasattr(self, 'config')}, config={'exists' if hasattr(self, 'config') and self.config else 'missing'}")
            if hasattr(self, 'config') and self.config:
                print(f"🔍 Config keys: {list(self.config.keys())}")
            
            if not hasattr(self, 'config') or not self.config:
                print("   🧠 Vision LLM: DISABLED (no config)")
                return
            
            # planner_integration is nested under 'performance' in env.yaml
            perf_config = self.config.get('performance', {})
            print(f"🔍 Performance config keys: {list(perf_config.keys()) if perf_config else 'none'}")
            
            planner_config = perf_config.get('planner_integration', {})
            print(f"🔍 Planner config: {planner_config}")
            
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
            
            # Extract LLM settings - Support both probability (new) and frequency (legacy)
            # NEW: Probability-based sampling (better distribution, less predictable)
            self.llm_text_probability = perf_config.get('llm_text_probability', planner_config.get('llm_text_probability', None))
            self.llm_vision_probability = perf_config.get('llm_vision_probability', planner_config.get('llm_vision_probability', None))
            
            # LEGACY: Fixed frequency (for backwards compatibility)
            self.llm_text_frequency = perf_config.get('llm_text_frequency', planner_config.get('llm_text_frequency', None))
            self.llm_vision_frequency = perf_config.get('llm_vision_frequency', planner_config.get('llm_vision_frequency', None))
            
            # If probability not set, convert frequency to probability
            if self.llm_text_probability is None and self.llm_text_frequency:
                self.llm_text_probability = 1.0 / self.llm_text_frequency
            elif self.llm_text_probability is None:
                self.llm_text_probability = 0.05  # Default 5% (1/20 steps)
                
            if self.llm_vision_probability is None and self.llm_vision_frequency:
                self.llm_vision_probability = 1.0 / self.llm_vision_frequency
            elif self.llm_vision_probability is None:
                self.llm_vision_probability = 0.01  # Default 1% (1/100 steps)
            
            # Legacy llm_frequency support
            if 'llm_frequency' in perf_config or 'llm_frequency' in planner_config:
                legacy_freq = perf_config.get('llm_frequency', planner_config.get('llm_frequency', 10))
                self.llm_text_probability = 1.0 / legacy_freq
                self.llm_vision_probability = 1.0 / (legacy_freq * 10)
            
            self.hud_update_frequency = perf_config.get('hud_update_frequency', planner_config.get('hud_update_frequency', 3))  # HUD updates more frequently than LLM
            
            # Alignment rewards (separate for text vs vision)
            self.text_alignment_bonus = perf_config.get('text_alignment_bonus', planner_config.get('text_alignment_bonus', 5.0))
            self.vision_alignment_bonus = perf_config.get('vision_alignment_bonus', planner_config.get('vision_alignment_bonus', 50.0))
            
            # Legacy support: if old alignment_bonus_multiplier exists, use it for both
            if 'alignment_bonus_multiplier' in perf_config or 'alignment_bonus_multiplier' in planner_config:
                legacy_bonus = perf_config.get('alignment_bonus_multiplier', planner_config.get('alignment_bonus_multiplier', 2.0))
                self.text_alignment_bonus = legacy_bonus
                self.vision_alignment_bonus = legacy_bonus
            
            # Extract vision settings for LLM (high quality)
            self.image_scale = perf_config.get('image_scale', planner_config.get('image_scale', vision_config.get('vision_config', {}).get('image_scale', 2)))
            self.image_quality = perf_config.get('image_quality', planner_config.get('image_quality', vision_config.get('vision_config', {}).get('image_quality', 75)))
            self.image_format = perf_config.get('image_format', planner_config.get('image_format', vision_config.get('vision_config', {}).get('image_format', 'jpeg')))
            
            # Extract HUD streaming settings (optimized for speed)
            self.hud_image_scale = perf_config.get('hud_image_scale', planner_config.get('hud_image_scale', 1))  # Default: native resolution
            self.hud_image_quality = perf_config.get('hud_image_quality', planner_config.get('hud_image_quality', 40))  # Default: 40% quality
            
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
        
        # All workers try to initialize HUD client
        # HUD server's session management will pick the first one
        print(f"   🖥️  Attempting HUD client initialization (instance {self.instance_id})...")
        
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
    
    def _init_session_manager(self):
        """Initialize Session Manager for saving checkpoints and summaries."""
        self.session_manager = None
        
        # Episode tracking for saves
        self.episode_data = {
            'steps': [],
            'rewards': [],
            'actions': [],
            'llm_suggestions': [],
            'locations': []
        }
        
        try:
            from session_manager import SessionManager
            
            # Create unique session ID from training run
            import time
            session_id = f"ray_training_{int(time.time())}"
            
            self.session_manager = SessionManager(session_id=session_id)
            
            if self.session_manager.enabled:
                print(f"   ✅ Session Manager initialized")
                print(f"      Session ID: {session_id}")
                print(f"      Bucket: s3://sessions/")
            else:
                print(f"   ⚠️  Session Manager disabled (S3 not configured)")
                
        except ImportError as e:
            print(f"   ⚠️  Session Manager not available: {e}")
            self.session_manager = None
        except Exception as e:
            print(f"   ❌ Error initializing Session Manager: {e}")
            self.session_manager = None
    
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
        
        # Try to claim HUD session if not already registered
        # This creates a "hot handoff" - when one episode ends, another worker can take over
        if hasattr(self, 'hud_client') and self.hud_client and not self.hud_client.enabled:
            print(f"🔄 Attempting to claim HUD session (episode {self._episode_count})...")
            if self.hud_client.register_session():
                print(f"✅ HUD session claimed by worker {self.instance_id}!")
            # If registration fails, that's OK - another worker has it
        
        elapsed = time.time() - start_time
        print(f"✅ RESET complete in {elapsed:.2f}s")
        print(f"   Episode: {self._episode_count}")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Info keys: {list(info.keys())}")
        print(f"{'='*70}\n")
        
        return obs, info
    
    def capture_screenshot_base64(self, for_hud: bool = False) -> Optional[str]:
        """
        Capture Game Boy screen as base64-encoded JPEG.
        
        Args:
            for_hud: If True, use fast HUD settings (lower quality, no upscaling)
                     If False, use high-quality LLM settings
        """
        if not for_hud and not self.llm_enabled:
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
            
            # Choose settings based on purpose
            if for_hud:
                # Fast HUD streaming: native resolution + lower quality
                scale = getattr(self, 'hud_image_scale', 1)
                quality = getattr(self, 'hud_image_quality', 40)
            else:
                # High-quality LLM: upscaled + better quality
                scale = self.image_scale
                quality = self.image_quality
            
            # Upscale if needed
            if scale > 1:
                new_size = (image.width * scale, image.height * scale)
                image = image.resize(new_size, Image.Resampling.NEAREST)
            
            # Convert to JPEG and encode as base64
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            print(f"⚠️  Screenshot capture failed: {e}")
            return None
    
    def call_llm_vision(self, game_state: Dict, screenshot_base64: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Call LLM with game state and optional screenshot.
        
        Supports both:
        - Vision calls: with screenshot (multimodal)
        - Text calls: without screenshot (text-only, faster)
        
        Returns:
            Dict with 'scene' (description) and 'action' (button), or None if failed
        """
        if not self.llm_enabled:
            return None
        
        try:
            # Extract values from game_state [FIXED 2025-10-15 05:30]
            # Structure: game_state['player'] has x, y, room, health, max_health
            # NO 'location' or 'entities' keys exist!
            player = game_state.get('player', {})
            health = player.get('health', 0)
            max_health = player.get('max_health', 0)
            x = player.get('x', 0)
            y = player.get('y', 0)
            room_id = player.get('room', 0)
            
            # Get room name from room_mappings (if available)
            location_name = 'Unknown'
            try:
                from observation.ram_maps.room_mappings import OVERWORLD_ROOMS
                location_name = OVERWORLD_ROOMS.get(room_id, f'Room {room_id}')
            except:
                location_name = f'Room {room_id}'
            
            # Entities might not exist (depends on use_structured_entities config)
            entities = game_state.get('entities', {})
            npc_count = len(entities.get('npcs', []))
            enemy_count = len(entities.get('enemies', []))
            item_count = len(entities.get('items', []))
            
            print(f"📤 SENDING TO LLM: {location_name}, health={health}/{max_health}, pos=({x},{y})")
            
            # Format prompt with game state
            user_prompt = self.user_prompt_template.format(
                location=location_name,
                cave_hint='',  # No cave_hint in current structure
                health=health,
                max_health=max_health,
                x=x,
                y=y,
                npc_count=npc_count,
                enemy_count=enemy_count,
                item_count=item_count
            )
            
            # Prepare API request (different format for vision vs text-only)
            if screenshot_base64:
                # Vision call: multimodal format with image
                user_content = [
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
            else:
                # Text-only call: simple string format
                user_content = user_prompt
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                "max_tokens": 150,  # Increased for scene description + action
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
            # Vision calls can take 10-30s, text-only ~1-5s
            timeout_seconds = 60 if screenshot_base64 else 15
            response = requests.post(
                self.llm_endpoint,
                json=payload,
                headers=headers,
                timeout=timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                
                if not raw_response:
                    return None
                
                # Parse response (expecting two lines: SCENE: ... and ACTION: ...)
                scene_desc = ""
                action = ""
                
                lines = raw_response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('SCENE:'):
                        scene_desc = line.replace('SCENE:', '').strip()
                    elif line.startswith('ACTION:'):
                        action = line.replace('ACTION:', '').strip()
                
                # Fallback: if no structured format, try to extract action from anywhere
                if not action:
                    # Look for button names in the response
                    for button in ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "NOP"]:
                        if button in raw_response.upper():
                            action = button
                            break
                
                # If we have at least an action, return it
                if action:
                    return {
                        'scene': scene_desc if scene_desc else raw_response[:100],  # Use full response as scene if not parsed
                        'action': action
                    }
                
                return None
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
                
        except requests.exceptions.Timeout as e:
            call_type = "VISION" if screenshot_base64 else "TEXT"
            print(f"⚠️  LLM request timed out ({call_type} call, {timeout_seconds}s limit)")
            print(f"   Endpoint: {self.llm_endpoint}")
            print(f"   Error: {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"⚠️  LLM connection error (cannot reach service)")
            print(f"   Endpoint: {self.llm_endpoint}")
            print(f"   Error: {e}")
            return None
        except Exception as e:
            print(f"⚠️  LLM call failed: {type(e).__name__}: {e}")
            print(f"   Endpoint: {self.llm_endpoint}")
            return None
    
    def compute_llm_alignment_bonus(self, action: int, llm_suggestion: Optional[str], is_vision: bool = False) -> float:
        """Compute reward bonus if PPO action aligns with LLM suggestion.
        
        Args:
            action: PPO action taken
            llm_suggestion: LLM's suggested action
            is_vision: True if this was a vision call (higher reward), False for text-only
        """
        if not llm_suggestion:
            return 0.0
        
        # Map actions to button names (SELECT removed)
        action_names = ["NOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]
        if action >= len(action_names):
            return 0.0
        
        action_name = action_names[action]
        suggestion_upper = llm_suggestion.upper()
        
        # Check if LLM suggested this action
        if action_name in suggestion_upper or (action_name == "A" and "TALK" in suggestion_upper):
            # Vision alignment is worth more (10x) because it's the gold standard
            return self.vision_alignment_bonus if is_vision else self.text_alignment_bonus
        
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
        
        # Track episode data for session saving
        if hasattr(self, 'session_manager') and self.session_manager and self.session_manager.enabled:
            try:
                # Get current game state for tracking
                game_state = self.get_structured_state() if hasattr(self, 'get_structured_state') else {}
                player_data = game_state.get('player', {})
                room_id = player_data.get('room', 0)
                
                self.episode_data['steps'].append(self._step_count)
                self.episode_data['rewards'].append(reward)
                self.episode_data['actions'].append(action)
                self.episode_data['llm_suggestions'].append(getattr(self, 'last_llm_suggestion', None))
                self.episode_data['locations'].append(room_id)
            except Exception as e:
                # Don't crash training if tracking fails
                pass
        
        # Call LLM for guidance (probability-based sampling for better distribution)
        import random
        llm_bonus = 0.0
        
        # NEW: Random probability-based sampling (better than fixed intervals!)
        # Vision and text are mutually exclusive (vision takes precedence if both trigger)
        is_vision_step = self.llm_enabled and (random.random() < self.llm_vision_probability)
        is_text_step = self.llm_enabled and (not is_vision_step) and (random.random() < self.llm_text_probability)
        
        if is_vision_step or is_text_step:
            # Get structured game state
            if hasattr(self, 'state_encoder') and self.state_encoder:
                # encode_state returns (vector_obs, structured_state)
                _, game_state = self.state_encoder.encode_state(self.bridge)
                
                # DEBUG: Log game_state structure to see what keys exist
                print(f"🔑 game_state KEYS: {list(game_state.keys())}")
                print(f"📊 PLAYER DATA: {game_state.get('player', 'MISSING')}")
                print(f"📍 LOCATION DATA: {game_state.get('location', 'MISSING')}")
                print(f"👥 ENTITIES DATA: {game_state.get('entities', 'MISSING')}")
            else:
                game_state = {}
                print(f"⚠️  No state_encoder available!")
            
            # Capture screenshot only for vision calls
            screenshot = None
            if is_vision_step:
                screenshot = self.capture_screenshot_base64()
                print(f"📸 Vision LLM call (step {self._step_count}): with screenshot")
            else:
                print(f"💬 Text LLM call (step {self._step_count}): game state only")
            
            if screenshot or is_text_step:
                # Initialize LLM result variables (in case LLM fails)
                llm_action = self.last_llm_suggestion or 'N/A'  # Use last successful suggestion
                scene_desc = 'LLM unavailable'
                llm_bonus = 0.0
                
                # Call vision LLM
                llm_start_time = time.time()
                llm_result = self.call_llm_vision(game_state, screenshot)
                llm_response_time = (time.time() - llm_start_time) * 1000  # Convert to ms
                
                if llm_result:
                    self.llm_call_count += 1
                    self.llm_success_count += 1
                    
                    # Extract scene description and action
                    scene_desc = llm_result.get('scene', '')
                    llm_action = llm_result.get('action', '')
                    
                    self.last_llm_suggestion = llm_action  # Store action for HUD
                    
                    # Log what the LLM sees and suggests
                    print(f"👁️  LLM SEES: {scene_desc}")
                    print(f"💡 LLM SUGGESTS: {llm_action}")
                    
                    # Compute alignment bonus (vision worth 10x more than text)
                    llm_bonus = self.compute_llm_alignment_bonus(action, llm_action, is_vision=is_vision_step)
                    
                    if llm_bonus > 0:
                        bonus_type = "VISION" if is_vision_step else "TEXT"
                        print(f"✅ PPO action {action} MATCHES {bonus_type} LLM → +{llm_bonus:.1f} bonus!")
                else:
                    # LLM failed, log it
                    self.llm_call_count += 1  # Count failed attempts too
                    print(f"⚠️  LLM call failed (404 or error), but continuing to update HUD with screenshot...")
                
                # Send to HUD (vision calls send screenshot, text calls skip)
                if self.hud_client and self.hud_client.enabled:
                    print(f"📤 Sending data to HUD (worker {self.instance_id})...")
                    try:
                        # Send vision data (screenshot) only for vision calls
                        vision_success = True  # Default to True for text-only calls
                        if screenshot:
                            print(f"   📸 Sending screenshot ({len(screenshot)} chars)...")
                            vision_success = self.hud_client.update_vision_data(screenshot, llm_response_time)
                            print(f"   📸 Vision data sent: {vision_success}")
                        else:
                            print(f"   💬 Text-only call - skipping screenshot send")
                        
                        # Send training data (game state)
                        # Extract data from correct keys (no 'location' key exists!)
                        player_data = game_state.get('player', {})
                        room_id = player_data.get('room', 0)
                        
                        # Track exploration
                        self.rooms_discovered.add(room_id)
                        grid_x = (room_id % 16) // 4
                        grid_y = (room_id // 16) // 4
                        grid_area = grid_y * 4 + grid_x
                        self.grid_areas_explored.add(grid_area)
                        if 0x30 <= room_id <= 0x5F:
                            self.buildings_entered.add(room_id)
                        
                        # Get room name
                        location_name = 'Unknown'
                        try:
                            from observation.ram_maps.room_mappings import OVERWORLD_ROOMS
                            location_name = OVERWORLD_ROOMS.get(room_id, f'Room {room_id}')
                        except:
                            location_name = f'Room {room_id}'
                        
                        # Update milestones
                        if 'Maku' in location_name and not self.milestones['maku_tree_entered']:
                            self.milestones['maku_tree_entered'] = True
                            print(f"🌳 MILESTONE: Maku Tree Entered!")
                        if 0x50 <= room_id <= 0x5F and not self.milestones['dungeon_entered']:
                            self.milestones['dungeon_entered'] = True
                            print(f"🏰 MILESTONE: Dungeon Entered!")
                        
                        # Extract entity counts
                        entities_data = game_state.get('entities', {})
                        npc_count = len(entities_data.get('npcs', []))
                        enemy_count = len(entities_data.get('enemies', []))
                        item_count = len(entities_data.get('items', []))
                        
                        # Format data to match HUD JavaScript expectations
                        hud_training_data = {
                            # Training Progress
                            'global_step': self._step_count,  # HUD expects 'global_step' not 'step'
                            'episode': self._episode_count,
                            'episode_id': f"E{self.instance_id:04d}-{self._episode_count:04d}",  # Format: E0001-0005
                            'epoch': 0,  # TODO: Get from Ray result
                            'episode_reward': self._total_reward,
                            'episode_length': self._step_count,
                            
                            # Game State (with correct object formats)
                            'location': location_name,
                            'room_id': room_id,
                            'position': {  # HUD expects {x, y} object (pixel coordinates within current screen)
                                'x': player_data.get('x', 0),
                                'y': player_data.get('y', 0)
                            },
                            'health': {  # HUD expects {current, max} object
                                'current': player_data.get('health', 0),
                                'max': player_data.get('max_health', 0)
                            },
                            'entities': {  # HUD expects {npcs, enemies, items} object
                                'npcs': npc_count,
                                'enemies': enemy_count,
                                'items': item_count
                            },
                            
                            # LLM Guidance
                            'llm_suggestion': llm_action,
                            'llm_scene_description': scene_desc,
                            'llm_calls': self.llm_call_count,
                            'llm_success_rate': self.llm_success_count / max(self.llm_call_count, 1),
                            'alignment_bonus': llm_bonus,
                            
                            # Exploration Statistics
                            'exploration': {
                                'rooms_discovered': len(self.rooms_discovered),
                                'grid_areas': len(self.grid_areas_explored),
                                'buildings_entered': len(self.buildings_entered)
                            },
                            
                            # Milestones
                            'milestones': self.milestones.copy()
                        }
                        print(f"   📊 Sending training data: step={self._step_count}, episode={self._episode_count}, location={location_name}...")
                        training_success = self.hud_client.update_training_data(hud_training_data)
                        print(f"   📊 Training data sent: {training_success}")
                        print(f"✅ HUD update complete!")
                    except Exception as e:
                        # Don't crash training if HUD fails
                        print(f"❌ HUD update failed: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Add LLM stats to info
            info['llm_calls'] = self.llm_call_count
            info['llm_success_rate'] = self.llm_success_count / max(self.llm_call_count, 1)
        
        # Stream screenshots to HUD more frequently (independent of LLM calls)
        # This provides smooth visual updates even when LLM isn't being called
        if hasattr(self, 'hud_update_frequency') and self._step_count % self.hud_update_frequency == 0:
            # Only send if HUD is enabled and we're not already sending via LLM block
            if self.hud_client and self.hud_client.enabled:
                # Skip if we just sent via LLM (avoid duplicate)
                llm_vision_just_ran = self.llm_enabled and (self._step_count % self.llm_vision_frequency == 0)
                llm_text_just_ran = self.llm_enabled and (self._step_count % self.llm_text_frequency == 0)
                if not (llm_vision_just_ran or llm_text_just_ran):
                    try:
                        # Capture fresh screenshot (optimized for HUD - fast!)
                        screenshot = self.capture_screenshot_base64(for_hud=True)
                        if screenshot:
                            # Get game state for HUD data
                            if hasattr(self, 'state_encoder') and self.state_encoder:
                                _, game_state = self.state_encoder.encode_state(self.bridge)
                            else:
                                game_state = {}
                            
                            # Send vision data (screenshot only, no LLM response time for streaming)
                            vision_success = self.hud_client.update_vision_data(screenshot, None)
                            
                        # Send training data (game state)
                        player_data = game_state.get('player', {})
                        room_id = player_data.get('room', 0)
                        
                        # Debug: Log position data every 100 steps
                        if self._step_count % 100 == 0:
                            print(f"🔍 Position debug (step {self._step_count}): x={player_data.get('x', 'MISSING')}, y={player_data.get('y', 'MISSING')}, room={room_id}")
                        
                        # Track exploration (rooms, grid areas, buildings)
                        self.rooms_discovered.add(room_id)
                        
                        # Grid area (divide 256 rooms into 16x16 grid -> 16 areas of 4x4 rooms)
                        grid_x = (room_id % 16) // 4
                        grid_y = (room_id // 16) // 4
                        grid_area = grid_y * 4 + grid_x
                        self.grid_areas_explored.add(grid_area)
                        
                        # Check for buildings/special locations (dungeons, shops, houses)
                        # Rooms 0x50-0x5F are typically dungeons, 0x30-0x3F shops/houses
                        if 0x30 <= room_id <= 0x5F:
                            self.buildings_entered.add(room_id)
                        
                        # Get room name
                        location_name = 'Unknown'
                        try:
                            from observation.ram_maps.room_mappings import OVERWORLD_ROOMS
                            location_name = OVERWORLD_ROOMS.get(room_id, f'Room {room_id}')
                        except:
                            location_name = f'Room {room_id}'
                        
                        # Update milestones
                        if 'Maku' in location_name and not self.milestones['maku_tree_entered']:
                            self.milestones['maku_tree_entered'] = True
                            print(f"🌳 MILESTONE: Maku Tree Entered!")
                        
                        if 0x50 <= room_id <= 0x5F and not self.milestones['dungeon_entered']:
                            self.milestones['dungeon_entered'] = True
                            print(f"🏰 MILESTONE: Dungeon Entered!")
                        
                        # Extract entity counts
                        entities_data = game_state.get('entities', {})
                        npc_count = len(entities_data.get('npcs', []))
                        enemy_count = len(entities_data.get('enemies', []))
                        item_count = len(entities_data.get('items', []))
                        
                        # Debug: Log entity detection every 100 steps
                        if self._step_count % 100 == 0:
                            print(f"👾 Entity debug (step {self._step_count}): NPCs={npc_count}, Enemies={enemy_count}, Items={item_count}")
                            if entities_data:
                                print(f"   Raw entities data: {entities_data.keys()}")
                        
                        # Format data for HUD (minimal update, preserve LLM data)
                        hud_training_data = {
                            'global_step': self._step_count,
                            'episode': self._episode_count,
                            'episode_id': f"E{self.instance_id:04d}-{self._episode_count:04d}",
                            'episode_reward': self._total_reward,
                            'location': location_name,
                            'room_id': room_id,
                            'position': {
                                'x': player_data.get('x', 0),
                                'y': player_data.get('y', 0)
                            },
                            'health': {
                                'current': player_data.get('health', 0),
                                'max': player_data.get('max_health', 0)
                            },
                            'entities': {
                                'npcs': npc_count,
                                'enemies': enemy_count,
                                'items': item_count
                            },
                            'exploration': {
                                'rooms_discovered': len(self.rooms_discovered),
                                'grid_areas': len(self.grid_areas_explored),
                                'buildings_entered': len(self.buildings_entered)
                            },
                            'milestones': self.milestones.copy()
                        }
                        
                        training_success = self.hud_client.update_training_data(hud_training_data)
                        
                        if vision_success and training_success:
                            print(f"🎬 HUD stream update: step={self._step_count}, location={location_name}")
                    except Exception as e:
                        # Don't crash training if HUD streaming fails
                        print(f"⚠️  HUD stream update failed: {e}")
        
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
                
                # Save episode summary and checkpoint to S3
                if hasattr(self, 'session_manager') and self.session_manager and self.session_manager.enabled:
                    try:
                        # Prepare episode summary
                        episode_summary = {
                            'episode_num': self._episode_count,
                            'total_steps': self._step_count,
                            'total_reward': self._total_reward,
                            'avg_reward_per_step': self._total_reward / max(self._step_count, 1),
                            'llm_calls': info.get('llm_calls', 0),
                            'llm_success_rate': info.get('llm_success_rate', 0.0),
                            'rooms_discovered': info.get('rooms_discovered', 0),
                            'unique_locations': len(set(self.episode_data['locations'])),
                            'actions_taken': self.episode_data['actions'],
                            'rewards_per_step': self.episode_data['rewards'],
                            'llm_suggestions': self.episode_data['llm_suggestions'],
                            'terminated': terminated,
                            'truncated': truncated,
                        }
                        
                        # Save episode summary
                        self.session_manager.save_episode_summary(
                            worker_id=self.instance_id,
                            episode_num=self._episode_count,
                            summary_data=episode_summary
                        )
                        
                        # Optional: Save checkpoint metadata (not full model - that's handled by Ray)
                        checkpoint_meta = {
                            'episode_num': self._episode_count,
                            'total_steps': self._step_count,
                            'total_reward': self._total_reward,
                            'timestamp': time.time()
                        }
                        self.session_manager.save_checkpoint(
                            worker_id=self.instance_id,
                            episode_num=self._episode_count,
                            checkpoint_data=checkpoint_meta,
                            model_state=None  # Ray handles model checkpointing
                        )
                        
                    except Exception as e:
                        print(f"   ⚠️  Failed to save episode data: {e}")
                    
                    # Reset episode data for next episode
                    self.episode_data = {
                        'steps': [],
                        'rewards': [],
                        'actions': [],
                        'llm_suggestions': [],
                        'locations': []
                    }
        
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

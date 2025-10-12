"""
Ray RLlib-compatible wrapper for Zelda Oracle of Seasons environment.
This wraps the existing ZeldaConfigurableEnvironment to make it Ray-compatible
while preserving all existing functionality (vision LLM, rewards, tracking, etc.).
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path
import os

# Import our existing Zelda environment
from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment


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
        
        print(f"✅ ZeldaRayEnv initialized (Instance: {self.instance_id})")
        print(f"   ROM: {rom_path}")
        print(f"   Config: {config_path}")
        print(f"   Observation space: {self.observation_space.shape}")
        print(f"   Action space: {self.action_space}")
    
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
        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)
        
        # Ensure observation is the correct type
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        Ray RLlib requires (obs, reward, terminated, truncated, info) return format.
        """
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Ensure observation is the correct type
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        
        # Ensure reward is a float scalar
        reward = float(reward)
        
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

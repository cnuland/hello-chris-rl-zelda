"""Configurable Gymnasium environment for The Legend of Zelda: Oracle of Seasons.

This version supports both LLM-guided and pure RL modes with optimized performance.
"""

import gymnasium as gym
import numpy as np
import yaml
from typing import Dict, Any, Tuple, Optional
from gymnasium import spaces
from pathlib import Path

try:
    # Try relative imports first (for module usage)
    from .pyboy_bridge import ZeldaPyBoyBridge
    from .input_map import ZeldaAction
    from ..observation.state_encoder import ZeldaStateEncoder
except ImportError:
    # Fall back to absolute imports (for direct script execution)
    from emulator.pyboy_bridge import ZeldaPyBoyBridge
    from emulator.input_map import ZeldaAction  
    from observation.state_encoder import ZeldaStateEncoder


class ZeldaConfigurableEnvironment(gym.Env):
    """Configurable Gymnasium environment for Zelda Oracle of Seasons."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, 
                 rom_path: str,
                 config_path: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None,
                 headless: bool = True, 
                 render_mode: Optional[str] = None,
                 visual_test_mode: bool = False):
        """Initialize Zelda environment with configurable LLM integration.

        Args:
            rom_path: Path to Oracle of Seasons ROM file
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to config_path)
            headless: Whether to run emulator without display
            render_mode: Rendering mode for gymnasium
            visual_test_mode: Enable visual test mode (single episode, non-headless)
        """
        super().__init__()

        self.rom_path = rom_path
        self.visual_test_mode = visual_test_mode
        
        # Override headless mode for visual testing
        if visual_test_mode:
            self.headless = False
            print("🎮 Visual Test Mode Enabled - PyBoy window will be displayed")
        else:
            self.headless = headless
            
        self.render_mode = render_mode

        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            # Default configuration for pure RL mode
            self.config = self._get_default_config()
            
        # Override configuration for visual test mode
        if visual_test_mode:
            self._apply_visual_test_overrides()

        # Extract key configuration options
        env_config = self.config.get('environment', {})
        planner_config = self.config.get('planner_integration', {})
        
        self.use_llm = planner_config.get('use_planner', False)
        self.enable_structured_states = self.use_llm  # Only generate structured states if using LLM
        self.observation_type = env_config.get('observation_type', 'vector')
        self.normalize_observations = env_config.get('normalize_observations', True)
        self.frame_skip = env_config.get('frame_skip', 4)
        
        # Initialize components
        auto_load_save = planner_config.get('auto_load_save_state', True)
        self.bridge = ZeldaPyBoyBridge(
            rom_path=rom_path, 
            headless=headless,
            auto_load_save_state=auto_load_save
        )
        
        # Initialize state encoder with appropriate settings
        if self.enable_structured_states:
            self.state_encoder = ZeldaStateEncoder(
                enable_visual=planner_config.get('enable_visual', True),
                compression_mode=planner_config.get('compression_mode', 'bit_packed'),
                use_structured_entities=planner_config.get('use_structured_entities', True)
            )
        else:
            # Minimal state encoder for pure RL
            self.state_encoder = ZeldaStateEncoder(
                enable_visual=False,
                use_structured_entities=False
            )

        # Gymnasium spaces
        self.action_space = spaces.Discrete(len(ZeldaAction))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_encoder.state_vector_size,),
            dtype=np.float32
        )

        # Environment state
        self.current_structured_state: Optional[Dict[str, Any]] = None
        self.previous_structured_state: Optional[Dict[str, Any]] = None
        self.step_count = 0
        self.episode_count = 0
        self.max_steps = env_config.get('max_episode_steps', 10000)
        
        # Performance tracking
        self.structured_state_generation_time = 0.0
        self.total_step_time = 0.0
        
        # Exploration tracking for enhanced rewards
        self.visited_rooms = set()  # Track all rooms/screens visited
        self.visited_dungeons = set()  # Track dungeons discovered
        self.last_dialogue_state = 0  # Track NPC dialogue interactions
        self.last_room_id = None  # Track room transitions

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for pure RL mode."""
        return {
            'environment': {
                'frame_skip': 4,
                'action_repeat': 1,
                'observation_type': 'vector',
                'normalize_observations': True,
                'max_episode_steps': 10000
            },
            'planner_integration': {
                'use_planner': False,
                'enable_visual': False,
                'use_structured_entities': False,
                'auto_load_save_state': True
            },
            'rewards': {
                'rupee_reward': 0.01,
                'key_reward': 0.5,
                'death_penalty': -3.0,
                'movement_reward': 0.001,
                'time_penalty': -0.0001,
                # EXPLORATION BONUSES - HUGE rewards for meaningful gameplay!
                'room_discovery_reward': 10.0,      # Big bonus for new areas
                'dungeon_discovery_reward': 25.0,   # Massive bonus for dungeon entry
                'dungeon_bonus': 5.0,               # Continuous bonus while in dungeon
                'npc_interaction_reward': 15.0      # Big bonus for talking to NPCs
            }
        }
    
    def _apply_visual_test_overrides(self) -> None:
        """Apply configuration overrides for visual test mode."""
        # Override environment settings for visual testing
        if 'environment' not in self.config:
            self.config['environment'] = {}
            
        # Shorter episodes for visual testing
        self.config['environment']['max_episode_steps'] = 1000
        
        # Slower frame skip for better visual observation
        self.config['environment']['frame_skip'] = 2
        
        # Override training settings if present
        if 'training' not in self.config:
            self.config['training'] = {}
            
        # Single episode for visual test
        self.config['training']['max_episode_steps'] = 1000
        self.config['training']['total_timesteps'] = 1000  # Single episode
        
        print("🎯 Visual Test Mode Configuration:")
        print(f"   Max episode steps: {self.config['environment']['max_episode_steps']}")
        print(f"   Frame skip: {self.config['environment']['frame_skip']}")
        print(f"   Total timesteps: {self.config['training']['total_timesteps']}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset bridge
        self.bridge.reset()
        
        # Reset environment state
        self.step_count = 0
        self.episode_count += 1
        self.current_structured_state = None
        self.previous_structured_state = None
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        import time
        step_start_time = time.time()
        
        # Convert action and execute with frame skip
        zelda_action = ZeldaAction(action)
        
        # Execute action with frame skip
        total_reward = 0.0
        for _ in range(self.frame_skip):
            self.bridge.step(zelda_action)
            # Bridge doesn't return reward, we calculate it in _calculate_reward
            total_reward += 0.0
            
        # Update step count
        self.step_count += 1
        
        # Get observation and info
        obs = self._get_observation()
        reward = self._calculate_reward(total_reward)
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        info = self._get_info()
        
        # Performance tracking
        self.total_step_time += time.time() - step_start_time
        
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        import time
        
        # Get raw game state
        raw_state = self.bridge.get_game_state()
        
        if self.enable_structured_states:
            # Generate structured state (slower but needed for LLM)
            struct_start_time = time.time()
            numeric_vector, structured_state = self.state_encoder.encode_state(self.bridge)
            self.previous_structured_state = self.current_structured_state
            self.current_structured_state = structured_state
            self.structured_state_generation_time += time.time() - struct_start_time
            
            # Return numeric vector for RL
            obs = numeric_vector
        else:
            # Fast path for pure RL - no structured state generation
            obs, _ = self.state_encoder.encode_state(self.bridge)
            
        # Normalize if configured
        if self.normalize_observations:
            obs = np.clip(obs, 0.0, 1.0)
            
        return obs.astype(np.float32)

    def _calculate_reward(self, base_reward: float) -> float:
        """Calculate reward with enhanced exploration bonuses."""
        reward_config = self.config.get('rewards', {})
        
        total_reward = base_reward
        
        # BASE REWARDS
        # Time penalty (encourages efficiency)
        total_reward += reward_config.get('time_penalty', -0.0001)
        
        # Movement reward (anti-idle)
        total_reward += reward_config.get('movement_reward', 0.001)
        
        # EXPLORATION BONUSES - Get current game state for analysis
        try:
            if hasattr(self.bridge, 'get_memory'):
                current_room = self.bridge.get_memory(0xC63B)  # Current room/screen ID
                dialogue_state = self.bridge.get_memory(0xC2EF)  # Dialogue/cutscene state
                dungeon_floor = self.bridge.get_memory(0xC63D)  # Dungeon floor (0 = overworld)
                
                # A) NEW ROOM EXPLORATION REWARD - HUGE BONUS!
                if current_room not in self.visited_rooms:
                    self.visited_rooms.add(current_room)
                    exploration_reward = reward_config.get('room_discovery_reward', 10.0)
                    total_reward += exploration_reward
                    if self.episode_count % 10 == 0:  # Log occasionally
                        print(f"🗺️  NEW ROOM DISCOVERED! Room {current_room} (+{exploration_reward:.1f} reward)")
                
                # B) DUNGEON REWARDS - MASSIVE BONUS!
                if dungeon_floor > 0:  # In a dungeon
                    # Continuous dungeon bonus
                    dungeon_bonus = reward_config.get('dungeon_bonus', 5.0)
                    total_reward += dungeon_bonus
                    
                    # First-time dungeon discovery
                    if dungeon_floor not in self.visited_dungeons:
                        self.visited_dungeons.add(dungeon_floor)
                        dungeon_discovery_bonus = reward_config.get('dungeon_discovery_reward', 25.0)
                        total_reward += dungeon_discovery_bonus
                        print(f"🏰 DUNGEON DISCOVERED! Floor {dungeon_floor} (+{dungeon_discovery_bonus:.1f} bonus)")
                
                # C) NPC INTERACTION REWARDS - DIALOGUE DETECTION
                if dialogue_state > 0 and dialogue_state != self.last_dialogue_state:
                    npc_bonus = reward_config.get('npc_interaction_reward', 15.0)
                    total_reward += npc_bonus
                    if self.episode_count % 5 == 0:  # Log occasionally
                        print(f"💬 NPC INTERACTION! Dialogue state: {dialogue_state} (+{npc_bonus:.1f} reward)")
                
                self.last_dialogue_state = dialogue_state
                self.last_room_id = current_room
                
        except Exception as e:
            # If memory access fails, just use base rewards
            pass
        
        return total_reward

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Check for death or other terminal conditions
        if self.current_structured_state:
            player = self.current_structured_state.get('player', {})
            if player.get('health', 3) <= 0:
                return True
                
        return False

    def _check_truncated(self) -> bool:
        """Check if episode should be truncated."""
        return self.step_count >= self.max_steps

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        info = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'llm_mode_enabled': self.use_llm,
            'structured_states_enabled': self.enable_structured_states
        }
        
        # Add structured state for LLM mode
        if self.enable_structured_states and self.current_structured_state:
            info['structured_state'] = self.current_structured_state
            
        # Performance metrics
        if self.step_count > 0:
            info['avg_step_time'] = self.total_step_time / self.step_count
            if self.enable_structured_states:
                info['avg_structured_state_time'] = self.structured_state_generation_time / self.step_count
                
        return info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self.bridge.get_screen_array()
        return None

    def close(self):
        """Close the environment."""
        if hasattr(self, 'bridge'):
            self.bridge.close()

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            'llm_mode': self.use_llm,
            'structured_states': self.enable_structured_states,
            'observation_type': self.observation_type,
            'frame_skip': self.frame_skip,
            'max_steps': self.max_steps,
            'normalize_obs': self.normalize_observations,
            'auto_load_save': getattr(self.bridge, 'auto_load_save_state', False),
            'visual_test_mode': self.visual_test_mode,
            'headless': self.headless
        }


# Factory functions for easy creation
def create_pure_rl_env(rom_path: str, headless: bool = True, visual_test_mode: bool = False) -> ZeldaConfigurableEnvironment:
    """Create environment configured for pure RL training."""
    config = {
        'environment': {
            'frame_skip': 4,
            'observation_type': 'vector',
            'normalize_observations': True,
            'max_episode_steps': 10000
        },
        'planner_integration': {
            'use_planner': False,
            'enable_visual': False,
            'use_structured_entities': False,
            'auto_load_save_state': True
        },
        'rewards': {
            'time_penalty': -0.0001,
            'movement_reward': 0.001,
            'death_penalty': -3.0,
            # EXPLORATION BONUSES - HUGE rewards for meaningful gameplay!
            'room_discovery_reward': 10.0,      # Big bonus for new areas
            'dungeon_discovery_reward': 25.0,   # Massive bonus for dungeon entry
            'dungeon_bonus': 5.0,               # Continuous bonus while in dungeon
            'npc_interaction_reward': 15.0      # Big bonus for talking to NPCs
        }
    }
    
    return ZeldaConfigurableEnvironment(
        rom_path=rom_path,
        config_dict=config,
        headless=headless,
        visual_test_mode=visual_test_mode
    )


def create_llm_guided_env(rom_path: str, headless: bool = True, visual_test_mode: bool = False) -> ZeldaConfigurableEnvironment:
    """Create environment configured for LLM-guided training."""
    config = {
        'environment': {
            'frame_skip': 4,
            'observation_type': 'vector',
            'normalize_observations': True,
            'max_episode_steps': 10000
        },
        'planner_integration': {
            'use_planner': True,
            'enable_visual': True,
            'use_structured_entities': True,
            'compression_mode': 'bit_packed',
            'auto_load_save_state': True,
            'planner_frequency': 100
        },
        'rewards': {
            'time_penalty': -0.0001,
            'movement_reward': 0.001,
            'death_penalty': -3.0,
            'key_reward': 0.5,
            'rupee_reward': 0.01,
            # EXPLORATION BONUSES - HUGE rewards for meaningful gameplay!
            'room_discovery_reward': 10.0,      # Big bonus for new areas
            'dungeon_discovery_reward': 25.0,   # Massive bonus for dungeon entry
            'dungeon_bonus': 5.0,               # Continuous bonus while in dungeon
            'npc_interaction_reward': 15.0      # Big bonus for talking to NPCs
        }
    }
    
    return ZeldaConfigurableEnvironment(
        rom_path=rom_path,
        config_dict=config,
        headless=headless,
        visual_test_mode=visual_test_mode
    )


# Visual test mode factory functions
def create_visual_test_pure_rl_env(rom_path: str) -> ZeldaConfigurableEnvironment:
    """Create environment for visual testing of pure RL mode."""
    return create_pure_rl_env(rom_path, headless=False, visual_test_mode=True)


def create_visual_test_llm_env(rom_path: str) -> ZeldaConfigurableEnvironment:
    """Create environment for visual testing of LLM-guided mode."""
    return create_llm_guided_env(rom_path, headless=False, visual_test_mode=True)

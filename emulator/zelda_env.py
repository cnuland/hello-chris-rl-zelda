"""Gymnasium environment for The Legend of Zelda: Oracle of Seasons."""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium import spaces

from .pyboy_bridge import ZeldaPyBoyBridge
from .input_map import ZeldaAction
from ..observation.state_encoder import ZeldaStateEncoder


class ZeldaEnvironment(gym.Env):
    """Gymnasium environment for Zelda Oracle of Seasons."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, rom_path: str, headless: bool = True, render_mode: Optional[str] = None):
        """Initialize Zelda environment.

        Args:
            rom_path: Path to Oracle of Seasons ROM file
            headless: Whether to run emulator without display
            render_mode: Rendering mode for gymnasium
        """
        super().__init__()

        self.rom_path = rom_path
        self.headless = headless
        self.render_mode = render_mode

        # Initialize components
        self.pyboy_bridge = ZeldaPyBoyBridge(rom_path, headless)
        self.state_encoder = ZeldaStateEncoder()

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
        self.max_steps = 100000  # Episode length limit

        # Reward tracking
        self.previous_rupees = 0
        self.previous_keys = 0
        self.previous_health = 0
        self.previous_bosses_defeated = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed (unused for deterministic emulator)
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset emulator
        self.pyboy_bridge.reset()

        # Advance past intro/loading
        for _ in range(100):
            self.pyboy_bridge.step(ZeldaAction.NOP)

        # Get initial state
        observation, structured_state = self.state_encoder.encode_state(self.pyboy_bridge)
        self.current_structured_state = structured_state
        self.previous_structured_state = None

        # Initialize reward tracking
        self._update_reward_tracking(structured_state)

        self.step_count = 0

        info = {
            'structured_state': structured_state,
            'state_summary': self.state_encoder.get_state_summary(structured_state)
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return next state.

        Args:
            action: Action from action space

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not (0 <= action < len(ZeldaAction)):
            raise ValueError(f"Invalid action {action}. Must be 0-{len(ZeldaAction)-1}")

        # Store previous state
        self.previous_structured_state = self.current_structured_state.copy() if self.current_structured_state else None

        # Execute action in emulator
        self.pyboy_bridge.step(action)

        # Get new state
        observation, structured_state = self.state_encoder.encode_state(self.pyboy_bridge)
        self.current_structured_state = structured_state

        # Calculate reward
        reward = self._calculate_reward(structured_state)

        # Check termination conditions
        terminated = self._is_terminated(structured_state)
        truncated = self.step_count >= self.max_steps

        self.step_count += 1

        info = {
            'structured_state': structured_state,
            'state_summary': self.state_encoder.get_state_summary(structured_state),
            'action_name': ZeldaAction(action).name,
            'step_count': self.step_count
        }

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, structured_state: Dict[str, Any]) -> float:
        """Calculate reward based on state changes.

        Args:
            structured_state: Current structured state

        Returns:
            Reward value
        """
        reward = 0.0

        try:
            current_rupees = structured_state['resources']['rupees']
            current_keys = structured_state['resources']['keys']
            current_health = structured_state['player']['health']

            # Reward for collecting rupees
            rupee_gain = current_rupees - self.previous_rupees
            if rupee_gain > 0:
                reward += rupee_gain * 0.01

            # Reward for collecting keys
            key_gain = current_keys - self.previous_keys
            if key_gain > 0:
                reward += key_gain * 0.5

            # Penalty for losing health
            health_loss = self.previous_health - current_health
            if health_loss > 0:
                reward -= health_loss * 0.1

            # Large penalty for death
            if current_health <= 0:
                reward -= 3.0

            # Reward for defeating bosses
            current_bosses = sum(structured_state['dungeon']['bosses_defeated'].values())
            boss_gain = current_bosses - self.previous_bosses_defeated
            if boss_gain > 0:
                reward += boss_gain * 2.0

            # Small positive reward for movement (anti-idle)
            if self.previous_structured_state:
                prev_x = self.previous_structured_state['player']['x']
                prev_y = self.previous_structured_state['player']['y']
                curr_x = structured_state['player']['x']
                curr_y = structured_state['player']['y']

                if prev_x != curr_x or prev_y != curr_y:
                    reward += 0.001

            # Small negative reward for time (encourages efficiency)
            reward -= 0.0001

            # Update tracking variables
            self.previous_rupees = current_rupees
            self.previous_keys = current_keys
            self.previous_health = current_health
            self.previous_bosses_defeated = current_bosses

        except (KeyError, TypeError) as e:
            print(f"Warning: Error calculating reward: {e}")
            reward = -0.0001  # Small negative reward for malformed state

        return float(reward)

    def _update_reward_tracking(self, structured_state: Dict[str, Any]) -> None:
        """Update reward tracking variables.

        Args:
            structured_state: Current structured state
        """
        try:
            self.previous_rupees = structured_state['resources']['rupees']
            self.previous_keys = structured_state['resources']['keys']
            self.previous_health = structured_state['player']['health']
            self.previous_bosses_defeated = sum(structured_state['dungeon']['bosses_defeated'].values())
        except (KeyError, TypeError):
            # Set defaults if state is malformed
            self.previous_rupees = 0
            self.previous_keys = 0
            self.previous_health = 3
            self.previous_bosses_defeated = 0

    def _is_terminated(self, structured_state: Dict[str, Any]) -> bool:
        """Check if episode should terminate.

        Args:
            structured_state: Current structured state

        Returns:
            True if episode should end
        """
        try:
            # Terminate if Link dies
            if structured_state['player']['health'] <= 0:
                return True

            # Terminate if all bosses defeated (game complete)
            if sum(structured_state['dungeon']['bosses_defeated'].values()) >= 8:
                return True

        except (KeyError, TypeError):
            pass

        return False

    def render(self) -> Optional[np.ndarray]:
        """Render current state.

        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        if self.render_mode == "rgb_array":
            try:
                return self.pyboy_bridge.get_screen()
            except RuntimeError:
                # Return black screen if emulator not ready
                return np.zeros((144, 160, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        """Clean up environment."""
        self.pyboy_bridge.close()

    def get_structured_state(self) -> Optional[Dict[str, Any]]:
        """Get current structured state for planner.

        Returns:
            Structured state dictionary or None if not available
        """
        return self.current_structured_state

    def save_state(self) -> bytes:
        """Save current emulator state.

        Returns:
            Serialized state data
        """
        return self.pyboy_bridge.save_state()

    def load_state(self, state_data: bytes) -> None:
        """Load emulator state.

        Args:
            state_data: Serialized state data
        """
        self.pyboy_bridge.load_state(state_data)

        # Update state after loading
        observation, structured_state = self.state_encoder.encode_state(self.pyboy_bridge)
        self.current_structured_state = structured_state
        self._update_reward_tracking(structured_state)
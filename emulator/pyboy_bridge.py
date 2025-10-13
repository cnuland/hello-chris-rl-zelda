"""PyBoy bridge for Zelda Oracle of Seasons emulation."""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from .input_map import ZeldaAction, ACTION_TO_EVENT, ACTION_TO_RELEASE


class ZeldaPyBoyBridge:
    """Bridge between PyBoy emulator and Zelda environment."""

    def __init__(self, rom_path: str, headless: bool = True, auto_load_save_state: bool = True):
        """Initialize PyBoy emulator.

        Args:
            rom_path: Path to Oracle of Seasons ROM file
            headless: Whether to run without graphics display
            auto_load_save_state: Whether to automatically load save state on reset
        """
        self.rom_path = rom_path
        self.headless = headless
        self.auto_load_save_state = auto_load_save_state
        self.pyboy: Optional[PyBoy] = None
        self.last_action: Optional[ZeldaAction] = None

        # Frame skip for faster training
        self.frame_skip = 4
        
        # Determine save state path (same directory as ROM, same name + .state)
        self.save_state_path = self.rom_path + ".state"

    def reset(self) -> None:
        """Reset the emulator to initial state."""
        if self.pyboy:
            self.pyboy.stop()

        # Use new PyBoy v2.x API
        # Let PyBoy auto-detect mode from ROM (CGB for .gbc files)
        self.pyboy = PyBoy(
            self.rom_path,
            window="null" if self.headless else "SDL2",
            debug=False
            # Note: PyBoy auto-detects CGB mode from .gbc ROM extension
        )

        # Load save state if available and auto_load is enabled
        # NOTE: Save state MUST be created in CGB mode to match PyBoy's auto-detection
        # Use convert_save_state_to_cgb.py to convert DMG states to CGB
        if self.auto_load_save_state:
            try:
                import os
                if os.path.exists(self.save_state_path):
                    print(f"🎮 Loading CGB save state: {self.save_state_path}")
                    # PyBoy expects a file-like object
                    with open(self.save_state_path, 'rb') as state_file:
                        self.pyboy.load_state(state_file)
                    print("✅ Save state loaded successfully - skipping intro!")
                else:
                    print(f"⚠️  Save state not found: {self.save_state_path}")
                    print("   Running from ROM start (will include intro/cutscenes)")
                    # Fallback: Skip intro manually if no save state
                    for _ in range(1000):
                        self.pyboy.tick()
            except Exception as e:
                print(f"❌ Failed to load save state: {e}")
                print("   Is this a DMG save state? Use convert_save_state_to_cgb.py")
                print("   Falling back to manual intro skip")
                # Fallback: Skip intro manually if save state loading fails
                for _ in range(1000):
                    self.pyboy.tick()
        else:
            # Manual intro skip if auto_load is disabled
            print("🎮 Auto-load disabled, running from ROM start")
            for _ in range(1000):
                self.pyboy.tick()

        self.last_action = None

    def step(self, action: int) -> None:
        """Execute action in emulator.

        Args:
            action: Action from ZeldaAction enum
        """
        if not self.pyboy:
            raise RuntimeError("Emulator not initialized. Call reset() first.")

        # Release previous action if any
        if self.last_action is not None and self.last_action != ZeldaAction.NOP:
            release_event = ACTION_TO_RELEASE.get(self.last_action)
            if release_event:
                self.pyboy.send_input(release_event)

        # Execute new action
        zelda_action = ZeldaAction(action)
        press_event = ACTION_TO_EVENT.get(zelda_action)

        if press_event:
            self.pyboy.send_input(press_event)

        # Advance frames with frame skip
        for _ in range(self.frame_skip):
            self.pyboy.tick()

        self.last_action = zelda_action

    def get_screen(self) -> np.ndarray:
        """Get current screen as numpy array.

        Returns:
            Screen pixels as (144, 160, 3) RGB array
        """
        if not self.pyboy:
            raise RuntimeError("Emulator not initialized")

        # Use new PyBoy v2.x API - screen has RGBA format, convert to RGB
        screen_rgba = self.pyboy.screen.ndarray
        return screen_rgba[:, :, :3]  # Remove alpha channel

    def get_memory(self, address: int) -> int:
        """Read byte from memory.

        Args:
            address: Memory address (0x0000-0xFFFF)

        Returns:
            Byte value at address
        """
        if not self.pyboy:
            raise RuntimeError("Emulator not initialized")

        # Use PyBoy API - compatible with both old and new versions
        try:
            # Try new API first (PyBoy 2.x)
            return self.pyboy.get_memory_value(address)
        except AttributeError:
            # Fall back to old API (PyBoy 1.x)
            return self.pyboy.memory[address]

    def get_memory_range(self, start: int, length: int) -> bytes:
        """Read range of bytes from memory.

        Args:
            start: Starting address
            length: Number of bytes to read

        Returns:
            Bytes from memory
        """
        if not self.pyboy:
            raise RuntimeError("Emulator not initialized")

        # Use new PyBoy v2.x API - direct memory slice access
        return bytes([
            self.pyboy.memory[start + i]
            for i in range(length)
        ])

    def get_tile_map(self) -> np.ndarray:
        """Get current tile map data.

        Returns:
            Tile map as numpy array
        """
        if not self.pyboy:
            raise RuntimeError("Emulator not initialized")

        # Get background tile map (32x32 tiles) using new PyBoy v2.x API
        tilemap = self.pyboy.tilemap_background
        return np.array([[tilemap.tile(x, y).tile_identifier for x in range(32)] for y in range(32)])

    def save_state(self) -> bytes:
        """Save current emulator state.

        Returns:
            Serialized state data
        """
        if not self.pyboy:
            raise RuntimeError("Emulator not initialized")

        # PyBoy save state functionality
        return self.pyboy.save_state()

    def load_state(self, state_data: bytes) -> None:
        """Load emulator state.

        Args:
            state_data: Serialized state data
        """
        if not self.pyboy:
            raise RuntimeError("Emulator not initialized")

        self.pyboy.load_state(state_data)

    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state from memory addresses."""
        if not self.pyboy:
            return {}
        
        try:
            # Get basic game state from known memory addresses
            state = {
                'health': self.get_memory(0xC021) // 4,  # Convert quarter-hearts to hearts
                'max_health': self.get_memory(0xC05B) // 4,
                'rupees': self.get_memory(0xC6A5),
                'x_position': self.get_memory(0xC4AC),  # Player X position
                'y_position': self.get_memory(0xC4AD),  # Player Y position
                'screen_array': self.get_screen_array(),
                'tile_map': self.get_tile_map()
            }
            return state
        except Exception as e:
            # Return empty state if there's an error
            return {}

    def get_screen_array(self) -> np.ndarray:
        """Get screen as numpy array."""
        if not self.pyboy:
            return np.zeros((144, 160, 3), dtype=np.uint8)
        
        try:
            screen = self.pyboy.screen.ndarray
            # Convert RGBA to RGB if needed
            if screen.shape[-1] == 4:
                screen = screen[:, :, :3]
            return screen
        except Exception as e:
            return np.zeros((144, 160, 3), dtype=np.uint8)

    def close(self) -> None:
        """Close emulator."""
        if self.pyboy:
            self.pyboy.stop()
            self.pyboy = None
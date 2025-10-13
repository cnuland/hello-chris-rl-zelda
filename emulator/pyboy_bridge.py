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
            # Note: Removed cgb=False as PyBoy auto-detects from .gbc ROM
        )

        # TEMPORARY FIX: Disable save state loading due to CGB/DMG mode mismatch
        # The save state was created in DMG mode, but PyBoy auto-loads .gbc ROMs in CGB mode
        # This causes: "CRITICAL Loading state which is not CGB, but PyBoy is loaded in CGB mode!"
        # 
        # TODO: Recreate save state in CGB mode and re-enable this
        print("🎮 Save state loading temporarily disabled (CGB mode mismatch)")
        print("   Starting from ROM beginning and skipping intro frames...")
        
        # Skip intro/title screens (approximately 2000-3000 frames)
        # This is slower than save states but avoids the CGB/DMG mismatch
        for i in range(3000):
            self.pyboy.tick()
            if i % 1000 == 0:
                print(f"   Skipping intro... frame {i}/3000")
        
        print("✅ Intro skip complete - game should be at start location")

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
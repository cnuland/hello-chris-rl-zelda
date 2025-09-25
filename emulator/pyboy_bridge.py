"""PyBoy bridge for Zelda Oracle of Seasons emulation."""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from .input_map import ZeldaAction, ACTION_TO_EVENT, ACTION_TO_RELEASE


class ZeldaPyBoyBridge:
    """Bridge between PyBoy emulator and Zelda environment."""

    def __init__(self, rom_path: str, headless: bool = True):
        """Initialize PyBoy emulator.

        Args:
            rom_path: Path to Oracle of Seasons ROM file
            headless: Whether to run without graphics display
        """
        self.rom_path = rom_path
        self.headless = headless
        self.pyboy: Optional[PyBoy] = None
        self.last_action: Optional[ZeldaAction] = None

        # Frame skip for faster training
        self.frame_skip = 4

    def reset(self) -> None:
        """Reset the emulator to initial state."""
        if self.pyboy:
            self.pyboy.stop()

        self.pyboy = PyBoy(
            self.rom_path,
            window_type="headless" if self.headless else "SDL2",
            debug=False
        )

        # Skip intro/title screens by advancing frames
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

        return np.array(self.pyboy.botsupport.screen.screen_ndarray())

    def get_memory(self, address: int) -> int:
        """Read byte from memory.

        Args:
            address: Memory address (0x0000-0xFFFF)

        Returns:
            Byte value at address
        """
        if not self.pyboy:
            raise RuntimeError("Emulator not initialized")

        return self.pyboy.get_memory_value(address)

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

        return bytes([
            self.pyboy.get_memory_value(start + i)
            for i in range(length)
        ])

    def get_tile_map(self) -> np.ndarray:
        """Get current tile map data.

        Returns:
            Tile map as numpy array
        """
        if not self.pyboy:
            raise RuntimeError("Emulator not initialized")

        # Get background tile map (32x32 tiles)
        tilemap = self.pyboy.botsupport.tilemap_background()
        return np.array([[tilemap.tile(x, y) for x in range(32)] for y in range(32)])

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

    def close(self) -> None:
        """Close emulator."""
        if self.pyboy:
            self.pyboy.stop()
            self.pyboy = None
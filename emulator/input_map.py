"""Input mapping for Zelda Oracle of Seasons controls."""

from enum import IntEnum
from pyboy.utils import WindowEvent


class ZeldaAction(IntEnum):
    """Action space for Zelda Oracle of Seasons."""
    NOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    A = 5  # Action button (sword, interact)
    B = 6  # Secondary action (items)
    START = 7  # Pause menu
    SELECT = 8  # Item selection


# Mapping from action enum to PyBoy WindowEvent
ACTION_TO_EVENT = {
    ZeldaAction.NOP: None,
    ZeldaAction.UP: WindowEvent.PRESS_ARROW_UP,
    ZeldaAction.DOWN: WindowEvent.PRESS_ARROW_DOWN,
    ZeldaAction.LEFT: WindowEvent.PRESS_ARROW_LEFT,
    ZeldaAction.RIGHT: WindowEvent.PRESS_ARROW_RIGHT,
    ZeldaAction.A: WindowEvent.PRESS_BUTTON_A,
    ZeldaAction.B: WindowEvent.PRESS_BUTTON_B,
    ZeldaAction.START: WindowEvent.PRESS_BUTTON_START,
    ZeldaAction.SELECT: WindowEvent.PRESS_BUTTON_SELECT,
}

# Release events for proper button handling
ACTION_TO_RELEASE = {
    ZeldaAction.UP: WindowEvent.RELEASE_ARROW_UP,
    ZeldaAction.DOWN: WindowEvent.RELEASE_ARROW_DOWN,
    ZeldaAction.LEFT: WindowEvent.RELEASE_ARROW_LEFT,
    ZeldaAction.RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
    ZeldaAction.A: WindowEvent.RELEASE_BUTTON_A,
    ZeldaAction.B: WindowEvent.RELEASE_BUTTON_B,
    ZeldaAction.START: WindowEvent.RELEASE_BUTTON_START,
    ZeldaAction.SELECT: WindowEvent.RELEASE_BUTTON_SELECT,
}


def get_action_name(action: int) -> str:
    """Get human-readable name for action."""
    try:
        return ZeldaAction(action).name
    except ValueError:
        return f"UNKNOWN_{action}"


def is_valid_action(action: int) -> bool:
    """Check if action is valid."""
    try:
        ZeldaAction(action)
        return True
    except ValueError:
        return False
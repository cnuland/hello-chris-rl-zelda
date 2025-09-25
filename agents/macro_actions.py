"""Macro action system for Zelda Oracle of Seasons.

Defines high-level macro actions that the LLM planner can use,
and converts them into sequences of primitive actions for the RL controller.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math
from ..emulator.input_map import ZeldaAction


class MacroType(Enum):
    """Types of macro actions available to the planner."""
    MOVE_TO = "MOVE_TO"
    EXPLORE_ROOM = "EXPLORE_ROOM"
    ENTER_DOOR = "ENTER_DOOR"
    ATTACK_ENEMY = "ATTACK_ENEMY"
    COLLECT_ITEM = "COLLECT_ITEM"
    USE_ITEM = "USE_ITEM"
    CHANGE_SEASON = "CHANGE_SEASON"
    SOLVE_PUZZLE = "SOLVE_PUZZLE"
    ENTER_DUNGEON = "ENTER_DUNGEON"
    EXIT_DUNGEON = "EXIT_DUNGEON"


@dataclass
class MacroAction:
    """A high-level macro action from the LLM planner."""
    action_type: MacroType
    parameters: Dict[str, Any]
    max_steps: int = 100  # Maximum primitive actions to execute
    priority: float = 1.0  # Action priority (higher = more important)


class MacroExecutor:
    """Executes macro actions by converting them to primitive action sequences."""

    def __init__(self):
        """Initialize macro executor."""
        self.current_macro: Optional[MacroAction] = None
        self.primitive_queue: List[ZeldaAction] = []
        self.steps_executed = 0

    def set_macro(self, macro: MacroAction) -> None:
        """Set a new macro action to execute.

        Args:
            macro: MacroAction to execute
        """
        self.current_macro = macro
        self.primitive_queue = self._expand_macro(macro)
        self.steps_executed = 0

    def get_next_action(self, current_state: Dict[str, Any]) -> Optional[ZeldaAction]:
        """Get next primitive action from current macro.

        Args:
            current_state: Current game state

        Returns:
            Next ZeldaAction to execute, or None if macro is complete
        """
        if not self.current_macro or not self.primitive_queue:
            return None

        if self.steps_executed >= self.current_macro.max_steps:
            # Macro timed out
            self.current_macro = None
            self.primitive_queue = []
            return None

        # Get next action from queue
        action = self.primitive_queue.pop(0)
        self.steps_executed += 1

        # Adaptive behavior: check if we need to modify the queue based on state
        self._adapt_to_state(current_state)

        return action

    def is_macro_complete(self) -> bool:
        """Check if current macro is complete.

        Returns:
            True if no macro is active or current macro is finished
        """
        return (self.current_macro is None or
                len(self.primitive_queue) == 0 or
                self.steps_executed >= self.current_macro.max_steps)

    def _expand_macro(self, macro: MacroAction) -> List[ZeldaAction]:
        """Expand macro action into primitive action sequence.

        Args:
            macro: MacroAction to expand

        Returns:
            List of primitive ZeldaActions
        """
        if macro.action_type == MacroType.MOVE_TO:
            return self._expand_move_to(macro.parameters)
        elif macro.action_type == MacroType.EXPLORE_ROOM:
            return self._expand_explore_room(macro.parameters)
        elif macro.action_type == MacroType.ENTER_DOOR:
            return self._expand_enter_door(macro.parameters)
        elif macro.action_type == MacroType.ATTACK_ENEMY:
            return self._expand_attack_enemy(macro.parameters)
        elif macro.action_type == MacroType.COLLECT_ITEM:
            return self._expand_collect_item(macro.parameters)
        elif macro.action_type == MacroType.USE_ITEM:
            return self._expand_use_item(macro.parameters)
        elif macro.action_type == MacroType.CHANGE_SEASON:
            return self._expand_change_season(macro.parameters)
        elif macro.action_type == MacroType.SOLVE_PUZZLE:
            return self._expand_solve_puzzle(macro.parameters)
        elif macro.action_type == MacroType.ENTER_DUNGEON:
            return self._expand_enter_dungeon(macro.parameters)
        elif macro.action_type == MacroType.EXIT_DUNGEON:
            return self._expand_exit_dungeon(macro.parameters)
        else:
            # Unknown macro type
            return [ZeldaAction.NOP] * 10

    def _expand_move_to(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand MOVE_TO macro.

        Args:
            params: Should contain 'x' and 'y' target coordinates

        Returns:
            Sequence of movement actions
        """
        target_x = params.get('x', 0)
        target_y = params.get('y', 0)

        # Generate basic movement sequence (simplified pathfinding)
        actions = []

        # Move horizontally first, then vertically
        if target_x > 0:
            actions.extend([ZeldaAction.RIGHT] * min(abs(target_x), 20))
        elif target_x < 0:
            actions.extend([ZeldaAction.LEFT] * min(abs(target_x), 20))

        if target_y > 0:
            actions.extend([ZeldaAction.DOWN] * min(abs(target_y), 20))
        elif target_y < 0:
            actions.extend([ZeldaAction.UP] * min(abs(target_y), 20))

        # Add some NOPs for stability
        actions.extend([ZeldaAction.NOP] * 5)

        return actions

    def _expand_explore_room(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand EXPLORE_ROOM macro.

        Args:
            params: Exploration parameters

        Returns:
            Sequence of exploratory movements
        """
        # Basic room exploration pattern
        actions = []

        # Move in a spiral pattern
        movements = [
            ([ZeldaAction.RIGHT] * 10),
            ([ZeldaAction.DOWN] * 10),
            ([ZeldaAction.LEFT] * 10),
            ([ZeldaAction.UP] * 10),
            ([ZeldaAction.RIGHT] * 5),
            ([ZeldaAction.DOWN] * 5),
        ]

        for movement in movements:
            actions.extend(movement)
            actions.extend([ZeldaAction.NOP] * 2)

        return actions

    def _expand_enter_door(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand ENTER_DOOR macro.

        Args:
            params: Door parameters (direction, etc.)

        Returns:
            Sequence to enter door
        """
        direction = params.get('direction', 'up')

        actions = []

        # Move toward door
        if direction == 'up':
            actions.extend([ZeldaAction.UP] * 5)
        elif direction == 'down':
            actions.extend([ZeldaAction.DOWN] * 5)
        elif direction == 'left':
            actions.extend([ZeldaAction.LEFT] * 5)
        elif direction == 'right':
            actions.extend([ZeldaAction.RIGHT] * 5)

        # Try to enter
        actions.extend([ZeldaAction.NOP] * 10)

        return actions

    def _expand_attack_enemy(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand ATTACK_ENEMY macro.

        Args:
            params: Enemy attack parameters

        Returns:
            Attack sequence
        """
        actions = []

        # Basic attack pattern
        for _ in range(5):
            actions.extend([
                ZeldaAction.A,  # Sword attack
                ZeldaAction.NOP,
                ZeldaAction.NOP,
            ])

        # Add some defensive movement
        actions.extend([
            ZeldaAction.LEFT,
            ZeldaAction.RIGHT,
            ZeldaAction.UP,
            ZeldaAction.DOWN,
        ])

        return actions

    def _expand_collect_item(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand COLLECT_ITEM macro.

        Args:
            params: Item collection parameters

        Returns:
            Item collection sequence
        """
        actions = []

        # Move toward item (simplified)
        target_x = params.get('x', 0)
        target_y = params.get('y', 0)

        if target_x > 0:
            actions.extend([ZeldaAction.RIGHT] * min(target_x, 10))
        elif target_x < 0:
            actions.extend([ZeldaAction.LEFT] * min(abs(target_x), 10))

        if target_y > 0:
            actions.extend([ZeldaAction.DOWN] * min(target_y, 10))
        elif target_y < 0:
            actions.extend([ZeldaAction.UP] * min(abs(target_y), 10))

        # Try to collect
        actions.extend([ZeldaAction.A, ZeldaAction.NOP] * 3)

        return actions

    def _expand_use_item(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand USE_ITEM macro.

        Args:
            params: Item usage parameters

        Returns:
            Item usage sequence
        """
        item_name = params.get('item', 'sword')

        actions = []

        # Open item menu
        actions.extend([ZeldaAction.START, ZeldaAction.NOP] * 2)

        # Select item (simplified)
        actions.extend([ZeldaAction.B, ZeldaAction.NOP] * 3)

        # Close menu
        actions.extend([ZeldaAction.START, ZeldaAction.NOP] * 2)

        return actions

    def _expand_change_season(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand CHANGE_SEASON macro.

        Args:
            params: Season change parameters

        Returns:
            Season change sequence
        """
        target_season = params.get('season', 'spring')

        actions = []

        # Use Rod of Seasons (simplified)
        actions.extend([ZeldaAction.B] * 3)  # Assuming rod is equipped
        actions.extend([ZeldaAction.NOP] * 5)

        # Select season in menu (would need more sophisticated logic)
        actions.extend([ZeldaAction.UP, ZeldaAction.DOWN, ZeldaAction.A])
        actions.extend([ZeldaAction.NOP] * 5)

        return actions

    def _expand_solve_puzzle(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand SOLVE_PUZZLE macro.

        Args:
            params: Puzzle parameters

        Returns:
            Puzzle solving sequence
        """
        puzzle_type = params.get('type', 'switch')

        actions = []

        if puzzle_type == 'switch':
            # Step on switches
            actions.extend([
                ZeldaAction.UP, ZeldaAction.NOP,
                ZeldaAction.DOWN, ZeldaAction.NOP,
                ZeldaAction.LEFT, ZeldaAction.NOP,
                ZeldaAction.RIGHT, ZeldaAction.NOP,
            ])
        elif puzzle_type == 'block':
            # Push blocks
            actions.extend([ZeldaAction.A] * 10)
            actions.extend([ZeldaAction.UP] * 5)
            actions.extend([ZeldaAction.A] * 5)

        return actions

    def _expand_enter_dungeon(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand ENTER_DUNGEON macro.

        Args:
            params: Dungeon entry parameters

        Returns:
            Dungeon entry sequence
        """
        actions = []

        # Move to dungeon entrance
        actions.extend([ZeldaAction.UP] * 10)
        actions.extend([ZeldaAction.A, ZeldaAction.NOP] * 3)

        return actions

    def _expand_exit_dungeon(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand EXIT_DUNGEON macro.

        Args:
            params: Dungeon exit parameters

        Returns:
            Dungeon exit sequence
        """
        actions = []

        # Find and use exit
        actions.extend([ZeldaAction.DOWN] * 10)
        actions.extend([ZeldaAction.A, ZeldaAction.NOP] * 3)

        return actions

    def _adapt_to_state(self, current_state: Dict[str, Any]) -> None:
        """Adapt current action queue based on game state.

        Args:
            current_state: Current game state
        """
        # Simple adaptation: if health is low, prioritize defensive actions
        try:
            current_health = current_state['player']['health']
            max_health = current_state['player']['max_health']

            if current_health < max_health * 0.3:  # Health below 30%
                # Insert defensive movements
                defensive_actions = [ZeldaAction.LEFT, ZeldaAction.RIGHT, ZeldaAction.UP, ZeldaAction.DOWN]
                self.primitive_queue = defensive_actions + self.primitive_queue[:10]

        except (KeyError, TypeError):
            # Ignore adaptation if state is malformed
            pass


def create_macro_from_planner_output(planner_output: Dict[str, Any]) -> Optional[MacroAction]:
    """Create MacroAction from LLM planner output.

    Args:
        planner_output: Output from LLM planner containing subgoal and macros

    Returns:
        MacroAction instance or None if invalid
    """
    try:
        macros = planner_output.get('macros', [])
        if not macros:
            return None

        # Take first macro for now (could be extended to handle multiple)
        macro_data = macros[0]
        action_type_str = macro_data.get('action_type', '')
        parameters = macro_data.get('parameters', {})

        # Convert string to enum
        try:
            action_type = MacroType(action_type_str)
        except ValueError:
            # Unknown action type
            return None

        return MacroAction(
            action_type=action_type,
            parameters=parameters,
            max_steps=macro_data.get('max_steps', 100),
            priority=macro_data.get('priority', 1.0)
        )

    except (KeyError, TypeError, IndexError):
        return None
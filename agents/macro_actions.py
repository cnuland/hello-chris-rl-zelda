"""Macro action system for Zelda Oracle of Seasons.

Defines high-level macro actions that the LLM planner can use,
and converts them into sequences of primitive actions for the RL controller.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math
from emulator.input_map import ZeldaAction


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
    
    # 🎯 STRATEGIC MACRO ACTIONS FOR ZELDA GAMEPLAY
    COMBAT_SWEEP = "COMBAT_SWEEP"        # Systematic area combat with movement
    CUT_GRASS = "CUT_GRASS"              # Methodical grass cutting for items
    SEARCH_ITEMS = "SEARCH_ITEMS"        # Thorough item searching pattern
    ENEMY_HUNT = "ENEMY_HUNT"            # Seek and destroy nearby enemies
    ENVIRONMENTAL_SEARCH = "ENVIRONMENTAL_SEARCH"  # Cut grass, lift rocks, search
    COMBAT_RETREAT = "COMBAT_RETREAT"    # Strategic retreat when low health
    ROOM_CLEARING = "ROOM_CLEARING"      # Complete room exploration + combat


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
    
    def clear_macro(self) -> None:
        """Clear the current macro and reset state."""
        self.current_macro = None
        self.primitive_queue = []
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
        
        # 🎯 STRATEGIC MACRO EXPANSIONS
        elif macro.action_type == MacroType.COMBAT_SWEEP:
            return self._expand_combat_sweep(macro.parameters)
        elif macro.action_type == MacroType.CUT_GRASS:
            return self._expand_cut_grass(macro.parameters)
        elif macro.action_type == MacroType.SEARCH_ITEMS:
            return self._expand_search_items(macro.parameters)
        elif macro.action_type == MacroType.ENEMY_HUNT:
            return self._expand_enemy_hunt(macro.parameters)
        elif macro.action_type == MacroType.ENVIRONMENTAL_SEARCH:
            return self._expand_environmental_search(macro.parameters)
        elif macro.action_type == MacroType.COMBAT_RETREAT:
            return self._expand_combat_retreat(macro.parameters)
        elif macro.action_type == MacroType.ROOM_CLEARING:
            return self._expand_room_clearing(macro.parameters)
        else:
            # Unknown macro type - default to exploration
            print(f"⚠️ Unknown macro type: {macro.action_type}")
            return self._expand_explore_room({})

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

    # 🎯 STRATEGIC MACRO IMPLEMENTATIONS FOR ZELDA GAMEPLAY
    
    def _expand_combat_sweep(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand COMBAT_SWEEP macro - systematic area combat with movement.
        
        Args:
            params: Combat parameters (intensity, pattern)
            
        Returns:
            Combat sweep action sequence
        """
        actions = []
        intensity = params.get('intensity', 'normal')  # light, normal, aggressive
        
        # Aggressive combat pattern: attack while moving in all directions
        combat_patterns = [
            # Pattern 1: Slash in all cardinal directions
            [ZeldaAction.A, ZeldaAction.NOP, ZeldaAction.RIGHT, 
             ZeldaAction.A, ZeldaAction.NOP, ZeldaAction.DOWN,
             ZeldaAction.A, ZeldaAction.NOP, ZeldaAction.LEFT,
             ZeldaAction.A, ZeldaAction.NOP, ZeldaAction.UP],
             
            # Pattern 2: Diagonal movement with attacks
            [ZeldaAction.RIGHT, ZeldaAction.DOWN, ZeldaAction.A,
             ZeldaAction.LEFT, ZeldaAction.DOWN, ZeldaAction.A, 
             ZeldaAction.LEFT, ZeldaAction.UP, ZeldaAction.A,
             ZeldaAction.RIGHT, ZeldaAction.UP, ZeldaAction.A],
             
            # Pattern 3: Spin attack simulation
            [ZeldaAction.A, ZeldaAction.A, ZeldaAction.A,
             ZeldaAction.RIGHT, ZeldaAction.A, ZeldaAction.DOWN,
             ZeldaAction.A, ZeldaAction.LEFT, ZeldaAction.A, ZeldaAction.UP]
        ]
        
        # Execute 2-3 combat patterns based on intensity
        pattern_count = 3 if intensity == 'aggressive' else 2
        for i in range(pattern_count):
            pattern = combat_patterns[i % len(combat_patterns)]
            actions.extend(pattern)
            actions.extend([ZeldaAction.NOP] * 2)  # Brief pause between patterns
            
        print(f"🗡️ Combat sweep: {len(actions)} actions, {intensity} intensity")
        return actions

    def _expand_cut_grass(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand CUT_GRASS macro - methodical grass cutting for items.
        
        Args:
            params: Grass cutting parameters
            
        Returns:
            Grass cutting action sequence
        """
        actions = []
        pattern = params.get('pattern', 'systematic')  # systematic, random, spiral
        
        if pattern == 'systematic':
            # Systematic grid pattern: move and slash
            for row in range(4):  # Cover 4x4 area
                for col in range(4):
                    # Move to position
                    if col > 0:
                        actions.append(ZeldaAction.RIGHT)
                    actions.extend([ZeldaAction.A, ZeldaAction.NOP])  # Cut grass
                # Move to next row
                if row < 3:
                    actions.extend([ZeldaAction.DOWN, ZeldaAction.LEFT] * 4)  # Reset to left side
                    
        elif pattern == 'spiral':
            # Spiral outward pattern
            directions = [ZeldaAction.RIGHT, ZeldaAction.DOWN, ZeldaAction.LEFT, ZeldaAction.UP]
            steps = [3, 3, 2, 2, 1, 1]  # Spiral step counts
            
            for i, step_count in enumerate(steps):
                direction = directions[i % 4]
                for _ in range(step_count):
                    actions.extend([direction, ZeldaAction.A, ZeldaAction.NOP])
                    
        print(f"🌿 Grass cutting: {pattern} pattern, {len(actions)} actions")
        return actions

    def _expand_search_items(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand SEARCH_ITEMS macro - thorough item searching.
        
        Args:
            params: Search parameters
            
        Returns:
            Item search action sequence  
        """
        actions = []
        search_type = params.get('type', 'thorough')  # quick, thorough, exhaustive
        
        # Movement + interaction pattern for finding hidden items
        search_directions = [
            ZeldaAction.UP, ZeldaAction.RIGHT, ZeldaAction.DOWN, ZeldaAction.LEFT
        ]
        
        if search_type == 'exhaustive':
            # Check every possible position with A and B buttons
            for direction in search_directions:
                actions.extend([direction, ZeldaAction.A, ZeldaAction.B, ZeldaAction.NOP])
                actions.extend([direction, ZeldaAction.A, ZeldaAction.B, ZeldaAction.NOP])
        else:
            # Standard search pattern
            for direction in search_directions:
                actions.extend([direction, ZeldaAction.A, ZeldaAction.NOP])
                
        # Add some random exploration
        actions.extend([ZeldaAction.RIGHT, ZeldaAction.A, 
                       ZeldaAction.DOWN, ZeldaAction.A,
                       ZeldaAction.LEFT, ZeldaAction.A])
                       
        print(f"🔍 Item search: {search_type} mode, {len(actions)} actions")
        return actions

    def _expand_enemy_hunt(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand ENEMY_HUNT macro - seek and destroy nearby enemies.
        
        Args:
            params: Enemy hunting parameters
            
        Returns:
            Enemy hunting action sequence
        """
        actions = []
        aggression = params.get('aggression', 'moderate')  # cautious, moderate, aggressive
        
        # Hunting pattern: move + attack in wider areas
        hunt_pattern = [
            # Wide sweeping movements to find enemies
            [ZeldaAction.RIGHT] * 6 + [ZeldaAction.A] * 2,
            [ZeldaAction.DOWN] * 4 + [ZeldaAction.A] * 2, 
            [ZeldaAction.LEFT] * 6 + [ZeldaAction.A] * 2,
            [ZeldaAction.UP] * 4 + [ZeldaAction.A] * 2,
        ]
        
        # Execute hunt patterns
        for pattern in hunt_pattern:
            actions.extend(pattern)
            actions.extend([ZeldaAction.NOP] * 2)
            
        if aggression == 'aggressive':
            # Add extra combat moves
            actions.extend([ZeldaAction.A] * 8)  # Continuous attacking
            
        print(f"👹 Enemy hunt: {aggression} mode, {len(actions)} actions") 
        return actions

    def _expand_environmental_search(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand ENVIRONMENTAL_SEARCH macro - comprehensive environment interaction.
        
        Args:
            params: Environmental search parameters
            
        Returns:
            Environmental interaction sequence
        """
        actions = []
        
        # Comprehensive environmental interaction:
        # 1. Cut grass systematically
        # 2. Try to lift rocks/objects  
        # 3. Attack anything suspicious
        # 4. Check walls for secret passages
        
        # Phase 1: Systematic grass cutting
        grass_actions = self._expand_cut_grass({'pattern': 'systematic'})
        actions.extend(grass_actions[:20])  # First 20 actions
        
        # Phase 2: Rock lifting attempts (B button)
        for direction in [ZeldaAction.UP, ZeldaAction.RIGHT, ZeldaAction.DOWN, ZeldaAction.LEFT]:
            actions.extend([direction, ZeldaAction.B, ZeldaAction.NOP])
            
        # Phase 3: Wall checking (A button at edges)
        wall_check = [
            [ZeldaAction.UP] * 8 + [ZeldaAction.A] * 3,      # North wall
            [ZeldaAction.RIGHT] * 8 + [ZeldaAction.A] * 3,   # East wall  
            [ZeldaAction.DOWN] * 8 + [ZeldaAction.A] * 3,    # South wall
            [ZeldaAction.LEFT] * 8 + [ZeldaAction.A] * 3,    # West wall
        ]
        
        for wall in wall_check:
            actions.extend(wall[:8])  # Limit to 8 actions per wall
            
        print(f"🌍 Environmental search: {len(actions)} actions")
        return actions

    def _expand_combat_retreat(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand COMBAT_RETREAT macro - strategic retreat when low health.
        
        Args:
            params: Retreat parameters
            
        Returns:
            Combat retreat sequence
        """
        actions = []
        retreat_style = params.get('style', 'defensive')  # defensive, evasive
        
        if retreat_style == 'evasive':
            # Rapid evasive movements
            evasion = [ZeldaAction.LEFT, ZeldaAction.RIGHT, ZeldaAction.UP, ZeldaAction.DOWN] * 3
            actions.extend(evasion)
        else:
            # Defensive retreat with occasional attacks
            defensive_moves = [
                ZeldaAction.DOWN, ZeldaAction.DOWN, ZeldaAction.A,  # Back up and attack
                ZeldaAction.LEFT, ZeldaAction.A,                   # Side step and attack  
                ZeldaAction.RIGHT, ZeldaAction.A,                  # Side step and attack
                ZeldaAction.UP, ZeldaAction.UP,                    # Move forward carefully
            ]
            actions.extend(defensive_moves)
            
        print(f"🛡️ Combat retreat: {retreat_style} style, {len(actions)} actions")
        return actions

    def _expand_room_clearing(self, params: Dict[str, Any]) -> List[ZeldaAction]:
        """Expand ROOM_CLEARING macro - complete room exploration and combat.
        
        Args:
            params: Room clearing parameters
            
        Returns:
            Room clearing sequence
        """
        actions = []
        thoroughness = params.get('thoroughness', 'complete')  # quick, complete, exhaustive
        
        # Phase 1: Combat sweep to clear enemies
        combat_actions = self._expand_combat_sweep({'intensity': 'normal'})
        actions.extend(combat_actions[:25])  # First 25 combat actions
        
        # Phase 2: Environmental search for items
        if thoroughness in ['complete', 'exhaustive']:
            env_actions = self._expand_environmental_search({})
            actions.extend(env_actions[:30])  # First 30 environmental actions
            
        # Phase 3: Final sweep for missed items
        if thoroughness == 'exhaustive':
            search_actions = self._expand_search_items({'type': 'thorough'})
            actions.extend(search_actions[:20])  # First 20 search actions
            
        print(f"🏠 Room clearing: {thoroughness} mode, {len(actions)} actions")
        return actions

    def _adapt_to_state(self, current_state: Dict[str, Any]) -> None:
        """Adapt current action queue based on game state.

        Args:
            current_state: Current game state
        """
        # Enhanced adaptation with strategic priorities
        try:
            current_health = current_state.get('player', {}).get('health', 3)
            max_health = current_state.get('player', {}).get('max_health', 3)
            entities = current_state.get('entities', {})
            
            # 🛡️ HEALTH-BASED ADAPTATION
            if current_health < max_health * 0.25:  # Health critical (25%)
                # Emergency retreat pattern
                emergency_actions = [ZeldaAction.DOWN, ZeldaAction.LEFT, ZeldaAction.DOWN, ZeldaAction.RIGHT]
                self.primitive_queue = emergency_actions + self.primitive_queue[:5]
                print(f"🚨 Emergency retreat! Health: {current_health}/{max_health}")
                
            elif current_health < max_health * 0.5:  # Health low (50%)
                # Cautious defensive movements
                defensive_actions = [ZeldaAction.LEFT, ZeldaAction.RIGHT, ZeldaAction.A]
                self.primitive_queue = defensive_actions + self.primitive_queue[:10]
                
            # ⚔️ ENEMY-BASED ADAPTATION  
            enemy_count = len(entities.get('enemies', []))
            if enemy_count > 0 and current_health >= max_health * 0.5:
                # Prioritize combat when healthy and enemies present
                combat_actions = [ZeldaAction.A, ZeldaAction.NOP, ZeldaAction.A]
                self.primitive_queue = combat_actions + self.primitive_queue[:15]
                print(f"⚔️ Combat priority! {enemy_count} enemies detected")
                
            # 💰 ITEM-BASED ADAPTATION
            item_count = len(entities.get('items', []))
            if item_count > 0:
                # Move toward items and interact
                item_actions = [ZeldaAction.A, ZeldaAction.B, ZeldaAction.A]
                self.primitive_queue = item_actions + self.primitive_queue[:10]
                print(f"💰 Items detected! {item_count} items nearby")

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
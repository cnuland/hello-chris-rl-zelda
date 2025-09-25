"""LLM Planner client for Zelda Oracle of Seasons.

Connects to vLLM-served 70B model on KServe to generate high-level plans
based on structured game state.
"""

import json
import httpx
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

from .macro_actions import MacroAction, create_macro_from_planner_output


@dataclass
class PlannerConfig:
    """Configuration for LLM planner."""
    endpoint_url: str = "http://zelda-planner-70b.zelda-ai.svc.cluster.local/v1/completions"
    model_name: str = "llama-3.1-70b-instruct"
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: float = 30.0
    max_retries: int = 3


class ZeldaPlanner:
    """LLM-based planner for Zelda Oracle of Seasons."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        """Initialize planner.

        Args:
            config: Planner configuration
        """
        self.config = config or PlannerConfig()
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        self.logger = logging.getLogger(__name__)

        # Load prompt template
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt for the LLM planner.

        Returns:
            System prompt string
        """
        return """You are an expert AI agent playing The Legend of Zelda: Oracle of Seasons.

Your role is to analyze the current game state and generate high-level strategic plans.

Game Knowledge:
- This is a 2D action-adventure game where Link explores Holodrum across four seasons
- Key objectives: Collect essences, defeat bosses, solve puzzles, manage seasons
- Items: Sword, shields, Rod of Seasons, various tools and equipment
- Seasons affect gameplay: Spring (flowers bloom), Summer (water dries), Autumn (mushrooms grow), Winter (water freezes)

Input Format:
You receive a JSON object with the current game state including:
- player: Link's position, health, direction, current room
- resources: Rupees, keys, equipment levels
- inventory: Items possessed
- season: Current season and season spirits found
- dungeon: Keys, maps, bosses defeated
- environment: Nearby tiles and terrain
- visual: Current game screen image and detected visual elements (NPCs, enemies, items, text)

Output Format:
Respond with JSON containing:
{
  "subgoal": "Brief description of current strategic objective",
  "reasoning": "1-2 sentences explaining the strategy",
  "macros": [
    {
      "action_type": "MACRO_TYPE",
      "parameters": {"key": "value"},
      "priority": 1.0
    }
  ]
}

Available Macro Types:
- MOVE_TO: Move to coordinates {"x": int, "y": int}
- EXPLORE_ROOM: Explore current room for items/secrets
- ENTER_DOOR: Enter door {"direction": "up|down|left|right"}
- ATTACK_ENEMY: Attack nearby enemies
- COLLECT_ITEM: Collect item at {"x": int, "y": int}
- USE_ITEM: Use item {"item": "item_name"}
- CHANGE_SEASON: Change season {"season": "spring|summer|autumn|winter"}
- SOLVE_PUZZLE: Solve puzzle {"type": "switch|block|other"}
- ENTER_DUNGEON: Enter dungeon
- EXIT_DUNGEON: Exit current dungeon

Strategy Guidelines:
1. Prioritize survival (collect hearts if health is low)
2. Analyze the visual screen data to identify:
   - Visible enemies and their positions
   - Collectible items on screen
   - Interactive objects (chests, switches, doors)
   - NPCs and dialogue opportunities
   - Environmental hazards or obstacles
3. Collect rupees and keys when possible
4. Use appropriate seasons for obstacles
5. Explore systematically to find items and secrets
6. Progress through dungeons methodically
7. Read on-screen text and dialogue for game progression clues
8. Keep plans simple and achievable

IMPORTANT: Always consider both the structured data AND the visual screen content when making decisions.

Be concise and focus on immediate actionable objectives."""

    async def get_plan(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get strategic plan from LLM based on game state.

        Args:
            game_state: Current structured game state

        Returns:
            Plan dictionary with subgoal and macros, or None if failed
        """
        try:
            # Create prompt
            user_prompt = self._create_user_prompt(game_state)

            # Call LLM
            response = await self._call_llm(user_prompt)

            if response:
                # Parse and validate response
                plan = self._parse_plan_response(response)
                return plan

        except Exception as e:
            self.logger.error(f"Error getting plan from LLM: {e}")

        return None

    def _create_user_prompt(self, game_state: Dict[str, Any]) -> str:
        """Create user prompt from game state.

        Args:
            game_state: Current structured game state

        Returns:
            Formatted prompt string
        """
        # Simplify game state for LLM (remove unnecessary details)
        simplified_state = {
            'player': game_state.get('player', {}),
            'resources': game_state.get('resources', {}),
            'inventory': {k: v for k, v in game_state.get('inventory', {}).items() if v},  # Only items we have
            'season': game_state.get('season', {}),
            'dungeon': game_state.get('dungeon', {}),
        }

        # Add environment summary
        if 'environment' in game_state:
            env = game_state['environment']
            simplified_state['environment'] = {
                'player_tile_pos': (env.get('player_tile_x', 0), env.get('player_tile_y', 0)),
                'nearby_obstacles': self._summarize_nearby_tiles(env.get('nearby_tiles', []))
            }
        
        # Add visual data if available
        if 'visual' in game_state:
            visual_data = game_state['visual']
            simplified_state['visual'] = {
                'screen_description': visual_data.get('description', 'No visual description available'),
                'detected_elements': visual_data.get('detected_elements', {}),
                'screen_available': not visual_data.get('error', False)
            }
            # Note: We include screen description and detected elements, but not the raw image data
            # The raw image would be too large for the prompt, so we rely on the visual encoder's analysis

        return f"Current game state:\n{json.dumps(simplified_state, indent=2)}\n\nProvide your strategic plan:"

    def _summarize_nearby_tiles(self, nearby_tiles: List[Dict]) -> List[str]:
        """Summarize nearby tiles for LLM context.

        Args:
            nearby_tiles: List of nearby tile data

        Returns:
            List of tile type descriptions
        """
        tile_summary = []
        for tile in nearby_tiles[:9]:  # Just nearby 3x3 area
            if tile['tile_type'] != 'ground':
                tile_summary.append(f"{tile['tile_type']} at ({tile['x']}, {tile['y']})")

        return tile_summary

    async def _call_llm(self, user_prompt: str) -> Optional[str]:
        """Call the LLM endpoint.

        Args:
            user_prompt: User prompt string

        Returns:
            LLM response text or None if failed
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False
        }

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.post(
                    self.config.endpoint_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'choices' in data and data['choices']:
                        return data['choices'][0]['message']['content']
                else:
                    self.logger.warning(f"LLM API returned status {response.status_code}: {response.text}")

            except httpx.RequestError as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                self.logger.error(f"Unexpected error calling LLM: {e}")
                break

        return None

    def _parse_plan_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into plan dictionary.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed plan dictionary or None if invalid
        """
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                plan = json.loads(json_text)

                # Validate plan structure
                if self._validate_plan(plan):
                    return plan

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse LLM response as JSON: {e}")

        # Fallback: create a simple exploration plan
        return {
            "subgoal": "Explore current area",
            "reasoning": "Unable to parse LLM response, defaulting to exploration",
            "macros": [
                {
                    "action_type": "EXPLORE_ROOM",
                    "parameters": {},
                    "priority": 1.0
                }
            ]
        }

    def _validate_plan(self, plan: Dict[str, Any]) -> bool:
        """Validate plan structure.

        Args:
            plan: Plan dictionary to validate

        Returns:
            True if plan is valid
        """
        required_fields = ['subgoal', 'macros']
        for field in required_fields:
            if field not in plan:
                return False

        # Validate macros
        macros = plan['macros']
        if not isinstance(macros, list) or len(macros) == 0:
            return False

        for macro in macros:
            if not isinstance(macro, dict):
                return False
            if 'action_type' not in macro:
                return False

        return True

    def get_macro_action(self, plan: Dict[str, Any]) -> Optional[MacroAction]:
        """Convert plan to MacroAction.

        Args:
            plan: Plan dictionary from LLM

        Returns:
            MacroAction instance or None
        """
        return create_macro_from_planner_output(plan)

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()


class MockPlanner(ZeldaPlanner):
    """Mock planner for testing without LLM endpoint."""

    def __init__(self):
        """Initialize mock planner."""
        self.config = PlannerConfig()
        self.logger = logging.getLogger(__name__)

    async def get_plan(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a simple mock plan.

        Args:
            game_state: Current structured game state

        Returns:
            Mock plan dictionary
        """
        try:
            player = game_state.get('player', {})
            resources = game_state.get('resources', {})
            health = player.get('health', 3)
            max_health = player.get('max_health', 3)
            rupees = resources.get('rupees', 0)

            # Simple rule-based planning
            if health < max_health * 0.5:
                return {
                    "subgoal": "Find heart container or recovery item",
                    "reasoning": "Health is low, prioritizing survival",
                    "macros": [
                        {
                            "action_type": "EXPLORE_ROOM",
                            "parameters": {},
                            "priority": 1.0
                        }
                    ]
                }
            elif rupees < 50:
                return {
                    "subgoal": "Collect rupees",
                    "reasoning": "Need more rupees for purchases",
                    "macros": [
                        {
                            "action_type": "EXPLORE_ROOM",
                            "parameters": {},
                            "priority": 1.0
                        }
                    ]
                }
            else:
                return {
                    "subgoal": "Progress through dungeon",
                    "reasoning": "Resources adequate, advancing game progress",
                    "macros": [
                        {
                            "action_type": "ENTER_DOOR",
                            "parameters": {"direction": "up"},
                            "priority": 1.0
                        }
                    ]
                }

        except Exception as e:
            self.logger.error(f"Error in mock planner: {e}")
            return {
                "subgoal": "Basic exploration",
                "reasoning": "Fallback plan due to error",
                "macros": [
                    {
                        "action_type": "EXPLORE_ROOM",
                        "parameters": {},
                        "priority": 1.0
                    }
                ]
            }

    async def close(self) -> None:
        """Mock close method."""
        pass
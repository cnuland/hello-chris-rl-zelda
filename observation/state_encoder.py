"""State encoder for Zelda Oracle of Seasons.

Converts raw RAM/tile data into structured observations for both RL and LLM agents.
"""

import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional
from .ram_maps.zelda_addresses import *


class ZeldaStateEncoder:
    """Encodes Zelda game state from PyBoy memory and tile data."""

    def __init__(self):
        """Initialize state encoder."""
        self.state_vector_size = 128  # Size of numeric state vector for RL

    def encode_state(self, pyboy_bridge) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode complete game state.

        Args:
            pyboy_bridge: ZeldaPyBoyBridge instance

        Returns:
            Tuple of (numeric_vector, structured_dict)
            - numeric_vector: For RL agent (128 floats)
            - structured_dict: For LLM planner (JSON-serializable)
        """
        # Get structured state data
        structured_state = self._get_structured_state(pyboy_bridge)

        # Convert to numeric vector for RL
        numeric_vector = self._structured_to_vector(structured_state)

        return numeric_vector, structured_state

    def _get_structured_state(self, pyboy_bridge) -> Dict[str, Any]:
        """Extract structured state information.

        Args:
            pyboy_bridge: ZeldaPyBoyBridge instance

        Returns:
            Structured state dictionary
        """
        state = {}

        # Player state
        state['player'] = {
            'x': pyboy_bridge.get_memory(PLAYER_X),
            'y': pyboy_bridge.get_memory(PLAYER_Y),
            'direction': DIRECTIONS.get(pyboy_bridge.get_memory(PLAYER_DIRECTION), 'unknown'),
            'room': pyboy_bridge.get_memory(PLAYER_ROOM),
            'health': pyboy_bridge.get_memory(PLAYER_HEALTH),
            'max_health': pyboy_bridge.get_memory(PLAYER_MAX_HEALTH),
        }

        # Resources
        rupees_low = pyboy_bridge.get_memory(RUPEES)
        rupees_high = pyboy_bridge.get_memory(RUPEES + 1)
        state['resources'] = {
            'rupees': rupees_low + (rupees_high << 8),
            'keys': pyboy_bridge.get_memory(KEYS),
            'sword_level': pyboy_bridge.get_memory(SWORD_LEVEL),
            'shield_level': pyboy_bridge.get_memory(SHIELD_LEVEL),
        }

        # Inventory
        inv1 = pyboy_bridge.get_memory(INVENTORY_1)
        inv2 = pyboy_bridge.get_memory(INVENTORY_2)
        inv3 = pyboy_bridge.get_memory(INVENTORY_3)

        state['inventory'] = {}
        for item_name, (address_offset, bit_mask) in ITEM_FLAGS.items():
            if address_offset == INVENTORY_1:
                has_item = bool(inv1 & bit_mask)
            elif address_offset == INVENTORY_2:
                has_item = bool(inv2 & bit_mask)
            else:
                has_item = bool(inv3 & bit_mask)
            state['inventory'][item_name] = has_item

        # Season state
        current_season_id = pyboy_bridge.get_memory(CURRENT_SEASON)
        state['season'] = {
            'current': SEASONS.get(current_season_id, 'unknown'),
            'current_id': current_season_id,
            'spirits_found': pyboy_bridge.get_memory(SEASON_SPIRITS),
        }

        # Dungeon progress
        boss_keys = pyboy_bridge.get_memory(BOSS_KEYS)
        state['dungeon'] = {
            'keys': pyboy_bridge.get_memory(DUNGEON_KEYS),
            'has_map': bool(pyboy_bridge.get_memory(DUNGEON_MAP)),
            'has_compass': bool(pyboy_bridge.get_memory(DUNGEON_COMPASS)),
            'bosses_defeated': {
                boss_name: bool(boss_keys & bit_mask)
                for boss_name, (_, bit_mask) in BOSS_FLAGS.items()
            }
        }

        # Game state
        state['game'] = {
            'screen_transition': pyboy_bridge.get_memory(SCREEN_TRANSITION),
            'loading': bool(pyboy_bridge.get_memory(LOADING_SCREEN)),
            'menu_state': pyboy_bridge.get_memory(MENU_STATE),
        }

        # Nearby tiles (for LLM context)
        state['environment'] = self._get_nearby_tiles(pyboy_bridge)

        return state

    def _get_nearby_tiles(self, pyboy_bridge, radius: int = 3) -> Dict[str, Any]:
        """Get tiles around player position.

        Args:
            pyboy_bridge: ZeldaPyBoyBridge instance
            radius: Tile radius around player

        Returns:
            Nearby tile information
        """
        try:
            tile_map = pyboy_bridge.get_tile_map()
            player_x = pyboy_bridge.get_memory(PLAYER_X) // 16  # Convert to tile coords
            player_y = pyboy_bridge.get_memory(PLAYER_Y) // 16

            # Extract tiles in radius around player
            nearby_tiles = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    tile_x = max(0, min(31, player_x + dx))
                    tile_y = max(0, min(31, player_y + dy))
                    tile_id = tile_map[tile_y][tile_x]
                    nearby_tiles.append({
                        'x': dx,
                        'y': dy,
                        'tile_id': int(tile_id),
                        'tile_type': self._classify_tile(tile_id)
                    })

            return {
                'player_tile_x': int(player_x),
                'player_tile_y': int(player_y),
                'nearby_tiles': nearby_tiles
            }
        except Exception:
            # Fallback if tile reading fails
            return {
                'player_tile_x': 0,
                'player_tile_y': 0,
                'nearby_tiles': []
            }

    def _classify_tile(self, tile_id: int) -> str:
        """Classify tile type based on tile ID.

        Args:
            tile_id: Tile identifier

        Returns:
            Tile type string
        """
        # Basic tile classification (would need expansion based on actual tile IDs)
        if tile_id == 0:
            return 'empty'
        elif tile_id < 16:
            return 'ground'
        elif tile_id < 32:
            return 'wall'
        elif tile_id < 48:
            return 'water'
        elif tile_id < 64:
            return 'obstacle'
        else:
            return 'special'

    def _structured_to_vector(self, structured_state: Dict[str, Any]) -> np.ndarray:
        """Convert structured state to numeric vector for RL.

        Args:
            structured_state: Structured state dictionary

        Returns:
            Normalized numeric vector of size self.state_vector_size
        """
        vector = np.zeros(self.state_vector_size, dtype=np.float32)
        idx = 0

        try:
            # Player state (normalized)
            vector[idx:idx+6] = [
                structured_state['player']['x'] / 255.0,
                structured_state['player']['y'] / 255.0,
                structured_state['player']['direction_id'] if 'direction_id' in structured_state['player'] else 0,
                structured_state['player']['room'] / 255.0,
                structured_state['player']['health'] / 20.0,  # Assuming max 20 hearts
                structured_state['player']['max_health'] / 20.0,
            ]
            idx += 6

            # Resources (normalized)
            vector[idx:idx+4] = [
                min(structured_state['resources']['rupees'] / 999.0, 1.0),
                min(structured_state['resources']['keys'] / 99.0, 1.0),
                structured_state['resources']['sword_level'] / 4.0,
                structured_state['resources']['shield_level'] / 3.0,
            ]
            idx += 4

            # Inventory (binary flags)
            inventory_items = list(structured_state['inventory'].values())[:20]  # Take first 20 items
            for i, has_item in enumerate(inventory_items):
                if idx < self.state_vector_size:
                    vector[idx] = 1.0 if has_item else 0.0
                    idx += 1

            # Season state
            if idx < self.state_vector_size - 2:
                vector[idx] = structured_state['season']['current_id'] / 3.0
                vector[idx+1] = structured_state['season']['spirits_found'] / 4.0
                idx += 2

            # Dungeon progress
            if idx < self.state_vector_size - 10:
                vector[idx] = structured_state['dungeon']['keys'] / 99.0
                vector[idx+1] = 1.0 if structured_state['dungeon']['has_map'] else 0.0
                vector[idx+2] = 1.0 if structured_state['dungeon']['has_compass'] else 0.0
                idx += 3

                # Boss defeats
                boss_states = list(structured_state['dungeon']['bosses_defeated'].values())[:7]
                for i, defeated in enumerate(boss_states):
                    if idx < self.state_vector_size:
                        vector[idx] = 1.0 if defeated else 0.0
                        idx += 1

            # Fill remaining with tile data if available
            if 'environment' in structured_state and structured_state['environment']['nearby_tiles']:
                tiles = structured_state['environment']['nearby_tiles'][:20]  # Limit tiles
                for tile in tiles:
                    if idx < self.state_vector_size - 1:
                        vector[idx] = tile['tile_id'] / 255.0
                        idx += 1

        except (KeyError, TypeError) as e:
            # Handle missing or malformed state data gracefully
            print(f"Warning: Error encoding state: {e}")

        return vector

    def get_state_summary(self, structured_state: Dict[str, Any]) -> str:
        """Generate human-readable state summary for LLM.

        Args:
            structured_state: Structured state dictionary

        Returns:
            Text summary of current state
        """
        try:
            player = structured_state['player']
            resources = structured_state['resources']
            season = structured_state['season']

            summary = f"Link is at position ({player['x']}, {player['y']}) in room {player['room']}, "
            summary += f"facing {player['direction']}. "
            summary += f"Health: {player['health']}/{player['max_health']} hearts. "
            summary += f"Resources: {resources['rupees']} rupees, {resources['keys']} keys. "
            summary += f"Current season: {season['current']}. "

            # Add inventory highlights
            important_items = ['rod_of_seasons', 'sword', 'shield', 'gale_boomerang']
            has_items = [item for item in important_items
                        if structured_state['inventory'].get(item, False)]
            if has_items:
                summary += f"Key items: {', '.join(has_items)}."

            return summary
        except (KeyError, TypeError):
            return "Unable to generate state summary - incomplete state data."
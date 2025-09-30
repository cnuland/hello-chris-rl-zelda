"""State encoder for Zelda Oracle of Seasons.

Converts raw RAM/tile data into structured observations for both RL and LLM agents.
Now includes visual processing for enhanced LLM context.
"""

import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional
from .ram_maps.zelda_addresses import (
    PLAYER_X, PLAYER_Y, PLAYER_DIRECTION, PLAYER_ROOM, 
    PLAYER_HEALTH, PLAYER_MAX_HEALTH, HEART_PIECES,
    RUPEES, ORE_CHUNKS, CURRENT_BOMBS, MAX_BOMBS, CURRENT_BOMBCHUS,
    SWORD_LEVEL, SHIELD_LEVEL, SEED_SATCHEL_LEVEL,
    A_BUTTON_ITEM, B_BUTTON_ITEM,
    EMBER_SEEDS, SCENT_SEEDS, PEGASUS_SEEDS, GALE_SEEDS, MYSTERY_SEEDS, GASHA_SEEDS,
    BOOMERANG_LEVEL, SLINGSHOT_LEVEL, ROCS_FEATHER_LEVEL, FLUTE_TYPE, MAGNETIC_GLOVES,
    VASU_RING_FLAGS, RING_BOX_LEVEL,
    ESSENCES_COLLECTED, TOTAL_DEATHS, ENEMIES_KILLED, RUPEES_COLLECTED,
    CURRENT_LEVEL_BANK, OVERWORLD_POSITION, DUNGEON_POSITION, DUNGEON_FLOOR, 
    MAPLE_COUNTER, ENEMIES_ON_SCREEN,
    INVENTORY_1, INVENTORY_2, INVENTORY_3, ITEM_FLAGS,
    CURRENT_SEASON, SEASON_SPIRITS, SEASONS, DIRECTIONS,
    DUNGEON_KEYS, BOSS_KEYS, DUNGEON_MAP, DUNGEON_COMPASS, BOSS_FLAGS,
    SCREEN_TRANSITION, LOADING_SCREEN, MENU_STATE
)
from .visual_encoder import VisualEncoder


class ZeldaStateEncoder:
    """Encodes Zelda game state from PyBoy memory and tile data."""

    def __init__(self, enable_visual: bool = True, compression_mode: str = 'bit_packed', use_structured_entities: bool = True):
        """Initialize state encoder.
        
        Args:
            enable_visual: Whether to include visual processing for LLM
            compression_mode: Visual compression mode: 'rgb', 'grayscale', 'gameboy_4bit', 'bit_packed', 'palette'
            use_structured_entities: Whether to extract structured entity information from sprites
        """
        self.state_vector_size = 128  # Size of numeric state vector for RL
        self.enable_visual = enable_visual
        self.use_structured_entities = use_structured_entities
        self.visual_encoder = VisualEncoder(compression_mode=compression_mode) if enable_visual else None
        
        # Tile ID mappings for entity recognition (discovered through reverse engineering)
        self.TILE_MAPPINGS = {
            'enemy_tiles': {
                'octorok': list(range(0x20, 0x24)),
                'moblin': list(range(0x24, 0x28)), 
                'darknut': list(range(0x28, 0x2C)),
                'stalfos': list(range(0x2C, 0x30)),
                'leever': list(range(0x30, 0x34)),
            },
            'item_tiles': {
                'rupee': [0x40, 0x41, 0x42],
                'heart': [0x43, 0x44],
                'key': [0x45],
                'bomb': [0x46],
                'arrow': [0x47],
            },
            'npc_tiles': {
                # NPCs in Oracle of Seasons - these are estimates and need refinement
                'villager': list(range(0x60, 0x70)),  # Horon Village NPCs
                'shop_keeper': list(range(0x70, 0x74)),  # Shop owners
                'maku_tree': [0x80, 0x81, 0x82, 0x83],  # Maku Tree sprites
                'impa': [0x84, 0x85],  # Impa (Din's caretaker)
                'din': [0x86, 0x87, 0x88],  # Din (Oracle of Seasons)
            }
        }

    def encode_state(self, pyboy_bridge) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode complete game state.

        Args:
            pyboy_bridge: ZeldaPyBoyBridge instance

        Returns:
            Tuple of (numeric_vector, structured_dict)
            - numeric_vector: For RL agent (128 floats)
            - structured_dict: For LLM planner (JSON-serializable, includes visual data)
        """
        # Get structured state data
        structured_state = self._get_structured_state(pyboy_bridge)

        # Add visual data for LLM if enabled
        if self.enable_visual and self.visual_encoder:
            try:
                screen_array = pyboy_bridge.get_screen()
                visual_data = self.visual_encoder.encode_screen_for_llm(screen_array)
                visual_elements = self.visual_encoder.detect_visual_elements(screen_array)
                screen_description = self.visual_encoder.describe_screen_content(screen_array)
                
                structured_state['visual'] = {
                    'screen_image': visual_data,
                    'detected_elements': visual_elements,
                    'description': screen_description
                }
            except Exception as e:
                # Fallback gracefully if visual processing fails
                structured_state['visual'] = {
                    'error': f"Visual processing failed: {e}",
                    'screen_available': False
                }

        # Convert to numeric vector for RL (visual data not included in RL vector)
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

        # Player state (using Data Crystal confirmed addresses)
        health_raw = pyboy_bridge.get_memory(PLAYER_HEALTH)
        max_health_raw = pyboy_bridge.get_memory(PLAYER_MAX_HEALTH)
        
        state['player'] = {
            'x': pyboy_bridge.get_memory(PLAYER_X),
            'y': pyboy_bridge.get_memory(PLAYER_Y),
            'direction': DIRECTIONS.get(pyboy_bridge.get_memory(PLAYER_DIRECTION), 'unknown'),
            'room': pyboy_bridge.get_memory(PLAYER_ROOM),
            'health': health_raw // 4 if health_raw > 0 else 0,  # Convert quarter-hearts to hearts
            'max_health': max_health_raw // 4 if max_health_raw > 0 else 0,  # Convert quarter-hearts to hearts
            'heart_pieces': pyboy_bridge.get_memory(HEART_PIECES),  # 0-3 heart pieces
        }

        # Resources (using Data Crystal confirmed addresses)
        rupees_low = pyboy_bridge.get_memory(RUPEES)
        rupees_high = pyboy_bridge.get_memory(RUPEES + 1)
        ore_low = pyboy_bridge.get_memory(ORE_CHUNKS)
        ore_high = pyboy_bridge.get_memory(ORE_CHUNKS + 1)
        
        state['resources'] = {
            'rupees': rupees_low + (rupees_high << 8),
            'ore_chunks': ore_low + (ore_high << 8),
            'current_bombs': pyboy_bridge.get_memory(CURRENT_BOMBS),
            'max_bombs': pyboy_bridge.get_memory(MAX_BOMBS),
            'current_bombchus': pyboy_bridge.get_memory(CURRENT_BOMBCHUS),
            'sword_level': pyboy_bridge.get_memory(SWORD_LEVEL),
            'shield_level': pyboy_bridge.get_memory(SHIELD_LEVEL),
            'seed_satchel_level': pyboy_bridge.get_memory(SEED_SATCHEL_LEVEL),
        }
        
        # Active items (what's equipped to A/B buttons)
        state['active_items'] = {
            'a_button': pyboy_bridge.get_memory(A_BUTTON_ITEM),
            'b_button': pyboy_bridge.get_memory(B_BUTTON_ITEM),
        }
        
        # Seed inventory (Data Crystal confirmed)
        state['seeds'] = {
            'ember': pyboy_bridge.get_memory(EMBER_SEEDS),
            'scent': pyboy_bridge.get_memory(SCENT_SEEDS),
            'pegasus': pyboy_bridge.get_memory(PEGASUS_SEEDS),
            'gale': pyboy_bridge.get_memory(GALE_SEEDS),
            'mystery': pyboy_bridge.get_memory(MYSTERY_SEEDS),
            'gasha': pyboy_bridge.get_memory(GASHA_SEEDS),
        }
        
        # Equipment levels (Data Crystal confirmed)
        state['equipment'] = {
            'boomerang_level': pyboy_bridge.get_memory(BOOMERANG_LEVEL),
            'slingshot_level': pyboy_bridge.get_memory(SLINGSHOT_LEVEL),
            'rocs_feather_level': pyboy_bridge.get_memory(ROCS_FEATHER_LEVEL),
            'flute_type': pyboy_bridge.get_memory(FLUTE_TYPE),
            'magnetic_gloves': pyboy_bridge.get_memory(MAGNETIC_GLOVES),
        }
        
        # Ring system (Data Crystal confirmed)
        state['rings'] = {
            'vasu_ring_flags': pyboy_bridge.get_memory(VASU_RING_FLAGS),
            'ring_box_level': pyboy_bridge.get_memory(RING_BOX_LEVEL),
        }
        
        # Progress tracking (Data Crystal confirmed)
        deaths_low = pyboy_bridge.get_memory(TOTAL_DEATHS)
        deaths_high = pyboy_bridge.get_memory(TOTAL_DEATHS + 1)
        enemies_low = pyboy_bridge.get_memory(ENEMIES_KILLED)
        enemies_high = pyboy_bridge.get_memory(ENEMIES_KILLED + 1)
        rupees_collected_low = pyboy_bridge.get_memory(RUPEES_COLLECTED)
        rupees_collected_high = pyboy_bridge.get_memory(RUPEES_COLLECTED + 1)
        
        state['progress'] = {
            'essences_collected': pyboy_bridge.get_memory(ESSENCES_COLLECTED),
            'total_deaths': deaths_low + (deaths_high << 8),
            'enemies_killed': enemies_low + (enemies_high << 8),
            'rupees_collected_lifetime': rupees_collected_low + (rupees_collected_high << 8),
        }
        
        # World state (Data Crystal confirmed)
        state['world'] = {
            'level_bank': pyboy_bridge.get_memory(CURRENT_LEVEL_BANK),
            'overworld_position': pyboy_bridge.get_memory(OVERWORLD_POSITION),
            'dungeon_position': pyboy_bridge.get_memory(DUNGEON_POSITION),
            'dungeon_floor': pyboy_bridge.get_memory(DUNGEON_FLOOR),
            'maple_counter': pyboy_bridge.get_memory(MAPLE_COUNTER),
            'enemies_on_screen': pyboy_bridge.get_memory(ENEMIES_ON_SCREEN),
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
        
        # Structured entities (enemies, items) from sprites
        if self.use_structured_entities:
            state['entities'] = self._get_structured_entities(pyboy_bridge)
            state['llm_prompt'] = self._create_llm_prompt(state)

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

    def _get_structured_entities(self, pyboy_bridge) -> Dict[str, Any]:
        """Extract structured entity information from sprites.
        
        Args:
            pyboy_bridge: ZeldaPyBoyBridge instance
            
        Returns:
            Dictionary containing organized entity information
        """
        entities = {
            'enemies': [],
            'items': [],
            'npcs': [],  # Track NPCs for dialogue opportunities
            'total_sprites': 0,
            'unknown_sprites': 0
        }
        
        try:
            # Extract sprite data from OAM (Object Attribute Memory)
            for sprite_num in range(40):  # Game Boy has 40 sprite slots
                try:
                    # Get sprite data from OAM
                    oam_addr = 0xFE00 + (sprite_num * 4)
                    
                    y_pos = pyboy_bridge.pyboy.memory[oam_addr]
                    x_pos = pyboy_bridge.pyboy.memory[oam_addr + 1]
                    tile_id = pyboy_bridge.pyboy.memory[oam_addr + 2]
                    attributes = pyboy_bridge.pyboy.memory[oam_addr + 3]
                    
                    # Skip empty/off-screen sprites
                    if y_pos == 0 or y_pos >= 160 or x_pos == 0:
                        continue
                        
                    entities['total_sprites'] += 1
                    
                    # Adjust coordinates for Game Boy system
                    screen_x = x_pos - 8
                    screen_y = y_pos - 16
                    
                    # Skip sprites that are clearly UI or off-screen
                    if screen_x < 0 or screen_x >= 160 or screen_y < 0 or screen_y >= 144:
                        continue
                    
                    # Identify sprite type from tile ID
                    sprite_type = self._identify_sprite_type(tile_id)
                    
                    if sprite_type.startswith('enemy_'):
                        enemy_type = sprite_type.replace('enemy_', '')
                        entities['enemies'].append({
                            'type': enemy_type,
                            'x': screen_x,
                            'y': screen_y,
                            'tile_id': tile_id
                        })
                    elif sprite_type.startswith('item_'):
                        item_type = sprite_type.replace('item_', '')
                        entities['items'].append({
                            'type': item_type,
                            'x': screen_x,
                            'y': screen_y,
                            'tile_id': tile_id
                        })
                    elif sprite_type.startswith('npc_'):
                        npc_type = sprite_type.replace('npc_', '')
                        entities['npcs'].append({
                            'type': npc_type,
                            'x': screen_x,
                            'y': screen_y,
                            'tile_id': tile_id
                        })
                    else:
                        # Unknown sprite - could be NPC if it's on the ground and not UI
                        # NPCs are typically stationary sprites on the ground level
                        if 40 <= screen_y <= 130 and 20 <= screen_x <= 140:
                            entities['npcs'].append({
                                'type': 'unknown_npc',
                                'x': screen_x,
                                'y': screen_y,
                                'tile_id': tile_id
                            })
                        entities['unknown_sprites'] += 1
                        
                except Exception:
                    continue  # Skip problematic sprites
                    
        except Exception as e:
            # Fallback gracefully if sprite extraction fails
            print(f"Warning: Could not extract sprite data: {e}")
            
        return entities

    def _identify_sprite_type(self, tile_id: int) -> str:
        """Identify what type of entity a sprite represents.
        
        Args:
            tile_id: Sprite tile identifier
            
        Returns:
            Entity type string
        """
        # Check enemies
        for enemy_type, tiles in self.TILE_MAPPINGS['enemy_tiles'].items():
            if tile_id in tiles:
                return f'enemy_{enemy_type}'
                
        # Check items
        for item_type, tiles in self.TILE_MAPPINGS['item_tiles'].items():
            if tile_id in tiles:
                return f'item_{item_type}'
        
        # Check NPCs
        for npc_type, tiles in self.TILE_MAPPINGS['npc_tiles'].items():
            if tile_id in tiles:
                return f'npc_{npc_type}'
                
        return f'unknown_{tile_id:02X}'

    def _create_llm_prompt(self, state: Dict[str, Any]) -> str:
        """Create a comprehensive natural language prompt for LLM reasoning.
        
        Args:
            state: Complete structured state dictionary
            
        Returns:
            Natural language description of current game state
        """
        try:
            parts = []
            
            # Player state with heart pieces
            player = state['player']
            pos_x, pos_y = player['x'], player['y'] 
            health = f"{player['health']}/{player['max_health']}"
            direction = player['direction']
            heart_pieces = player.get('heart_pieces', 0)
            
            heart_info = f"{health} hearts"
            if heart_pieces > 0:
                heart_info += f" (+{heart_pieces}/4 pieces)"
            
            parts.append(f"Link at ({pos_x},{pos_y}), {heart_info}, facing {direction}")
            
            # Resources with expanded inventory
            resources = state['resources']
            resource_parts = []
            
            if resources['rupees'] > 0:
                resource_parts.append(f"{resources['rupees']} rupees")
            if resources.get('current_bombs', 0) > 0:
                resource_parts.append(f"{resources['current_bombs']}/{resources.get('max_bombs', 0)} bombs")
            if resources.get('sword_level', 0) > 0:
                resource_parts.append(f"sword L{resources['sword_level']}")
            if resources.get('shield_level', 0) > 0:
                resource_parts.append(f"shield L{resources['shield_level']}")
            
            if resource_parts:
                parts.append(', '.join(resource_parts))
            
            # Seeds inventory (if any)
            if 'seeds' in state:
                seeds = state['seeds']
                seed_parts = []
                for seed_type, count in seeds.items():
                    if count > 0:
                        seed_parts.append(f"{count} {seed_type}")
                if seed_parts and len(seed_parts) <= 3:  # Only show if not too cluttered
                    parts.append(f"Seeds: {', '.join(seed_parts)}")
            
            # Equipment highlights
            if 'equipment' in state:
                equipment = state['equipment']
                equip_parts = []
                if equipment.get('boomerang_level', 0) > 0:
                    equip_parts.append(f"boomerang L{equipment['boomerang_level']}")
                if equipment.get('flute_type', 0) > 0:
                    flute_names = {1: 'Ricky', 2: 'Dimitri', 3: 'Moosh'}
                    flute_name = flute_names.get(equipment['flute_type'], f"flute {equipment['flute_type']}")
                    equip_parts.append(f"{flute_name} flute")
                
                if equip_parts:
                    parts.append(f"Equipment: {', '.join(equip_parts)}")
            
            # World location context
            if 'world' in state:
                world = state['world']
                if world.get('dungeon_floor', 0) > 0:
                    parts.append(f"Dungeon floor {world['dungeon_floor']}")
                elif world.get('overworld_position', 0) > 0:
                    parts.append(f"Overworld area {world['overworld_position']}")
                
                if world.get('enemies_on_screen', 0) > 0:
                    parts.append(f"{world['enemies_on_screen']} enemies nearby")
            
            # Entities (enemies and items from sprite detection)
            if 'entities' in state:
                entities = state['entities']
                
                # Enemies
                if entities['enemies']:
                    enemy_counts = {}
                    for enemy in entities['enemies']:
                        enemy_type = enemy['type']
                        if enemy_type not in enemy_counts:
                            enemy_counts[enemy_type] = []
                        enemy_counts[enemy_type].append(f"({enemy['x']},{enemy['y']})")
                    
                    enemy_descriptions = []
                    for enemy_type, positions in enemy_counts.items():
                        if len(positions) == 1:
                            enemy_descriptions.append(f"1 {enemy_type} at {positions[0]}")
                        else:
                            enemy_descriptions.append(f"{len(positions)} {enemy_type}s at {', '.join(positions[:2])}")
                    
                    parts.append('; '.join(enemy_descriptions))
                
                # Items
                if entities['items']:
                    item_descriptions = []
                    for item in entities['items'][:3]:  # Limit to first 3 items
                        item_descriptions.append(f"{item['type']} at ({item['x']},{item['y']})")
                    
                    if item_descriptions:
                        parts.append(f"Items: {', '.join(item_descriptions)}")
            
            # Progress context (if significant)
            if 'progress' in state:
                progress = state['progress']
                if progress.get('essences_collected', 0) > 0:
                    parts.append(f"{bin(progress['essences_collected']).count('1')} essences collected")
            
            # Season context
            season = state.get('season', {})
            current_season = season.get('current', 'unknown')
            if current_season != 'unknown':
                parts.append(f"Season: {current_season}")
            
            # Create final prompt
            prompt = '. '.join(parts) + '.'
            
            # Smart truncation - keep most important info if too long
            if len(prompt) > 300:
                # Keep player state, resources, enemies, and season
                essential_parts = []
                for part in parts:
                    if any(keyword in part.lower() for keyword in ['link at', 'rupees', 'hearts', 'enemy', 'enemies', 'season']):
                        essential_parts.append(part)
                    if len(essential_parts) >= 4:  # Limit to 4 essential parts
                        break
                
                prompt = '. '.join(essential_parts) + '.'
            
            return prompt
            
        except Exception as e:
            # Fallback to basic state summary
            print(f"Warning: Could not create LLM prompt: {e}")
            return self.get_state_summary(state)
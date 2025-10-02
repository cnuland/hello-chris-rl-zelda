"""ASCII Screen Renderer for Game Boy visuals.

Converts Game Boy screen pixels to ASCII art that LLMs can understand.
Fast, lightweight, and optional - can be toggled on/off.

Example output:
####################
#    @       T    #
#   NPC           #
#        ~~~      #
#      ~~~~~      #
#   DOOR    #######
####################
"""

import numpy as np
from typing import Dict, Any, Tuple


class ASCIIRenderer:
    """Converts Game Boy screen to ASCII art for LLM visualization.
    
    This allows text-only LLMs to "see" the game screen by converting
    pixel data into meaningful ASCII characters.
    """
    
    def __init__(
        self,
        width: int = 40,  # ASCII grid width (Game Boy is 160px, so 4px per char)
        height: int = 36,  # ASCII grid height (Game Boy is 144px, so 4px per char)
    ):
        """Initialize ASCII renderer.
        
        Args:
            width: Number of ASCII characters wide
            height: Number of ASCII characters tall
        """
        self.width = width
        self.height = height
        
        # Game Boy color palette (RGB values for sprite detection)
        # Game Boy uses 4 shades: white, light gray, dark gray, black
        self.GB_COLORS = {
            'white': (224, 248, 208),       # Lightest
            'light_gray': (136, 192, 112),  # Light
            'dark_gray': (52, 104, 86),     # Dark
            'black': (8, 24, 32),           # Darkest
        }
        
        # ASCII character mapping by brightness and pattern
        self.CHAR_MAP = {
            # Brightness-based (fallback)
            'very_dark': '#',   # Walls, obstacles
            'dark': 'X',        # Dense objects
            'medium': '*',      # Medium density
            'light': '.',       # Sparse objects
            'very_light': ' ',  # Empty space
            
            # Pattern-based (detected features)
            'link': '@',        # Player character
            'npc': 'N',         # NPCs
            'enemy': 'E',       # Enemies
            'item': 'I',        # Items
            'door': 'D',        # Doors
            'water': '~',       # Water
            'tree': 'T',        # Trees
            'chest': 'C',       # Chests
            'sign': 'S',        # Signs
        }
    
    def render_screen(
        self,
        screen: np.ndarray,
        player_pos: Tuple[int, int] = None,
        npcs: list = None,
        enemies: list = None,
        items: list = None
    ) -> str:
        """Convert Game Boy screen to ASCII art.
        
        Args:
            screen: RGB screen array (144, 160, 3) or grayscale (144, 160)
            player_pos: Optional (x, y) position of Link
            npcs: Optional list of NPC positions
            enemies: Optional list of enemy positions
            items: Optional list of item positions
        
        Returns:
            ASCII art string representing the screen
        """
        # Convert to grayscale if needed
        if len(screen.shape) == 3:
            gray = np.mean(screen, axis=2).astype(np.uint8)
        else:
            gray = screen
        
        # Sample screen into ASCII grid
        ascii_grid = self._sample_to_grid(gray)
        
        # Overlay entity markers (player, NPCs, enemies, items)
        ascii_grid = self._overlay_entities(
            ascii_grid, gray.shape,
            player_pos, npcs, enemies, items
        )
        
        # Convert grid to string
        ascii_art = self._grid_to_string(ascii_grid)
        
        return ascii_art
    
    def _sample_to_grid(self, gray: np.ndarray) -> np.ndarray:
        """Sample grayscale screen into ASCII character grid.
        
        Args:
            gray: Grayscale screen (144, 160)
        
        Returns:
            2D array of ASCII characters
        """
        h, w = gray.shape
        cell_h = h // self.height
        cell_w = w // self.width
        
        grid = np.empty((self.height, self.width), dtype='<U1')
        
        for y in range(self.height):
            for x in range(self.width):
                # Sample cell
                y_start = y * cell_h
                y_end = (y + 1) * cell_h
                x_start = x * cell_w
                x_end = (x + 1) * cell_w
                
                cell = gray[y_start:y_end, x_start:x_end]
                avg_brightness = np.mean(cell)
                
                # Map brightness to ASCII character
                if avg_brightness < 50:
                    grid[y, x] = self.CHAR_MAP['very_dark']
                elif avg_brightness < 100:
                    grid[y, x] = self.CHAR_MAP['dark']
                elif avg_brightness < 150:
                    grid[y, x] = self.CHAR_MAP['medium']
                elif avg_brightness < 200:
                    grid[y, x] = self.CHAR_MAP['light']
                else:
                    grid[y, x] = self.CHAR_MAP['very_light']
        
        return grid
    
    def _overlay_entities(
        self,
        grid: np.ndarray,
        screen_shape: Tuple[int, int],
        player_pos: Tuple[int, int],
        npcs: list,
        enemies: list,
        items: list
    ) -> np.ndarray:
        """Overlay entity markers on ASCII grid.
        
        Args:
            grid: ASCII character grid
            screen_shape: Original screen dimensions (h, w)
            player_pos: (x, y) position of Link
            npcs: List of NPC positions/data
            enemies: List of enemy positions/data
            items: List of item positions/data
        
        Returns:
            Grid with entity markers overlaid
        """
        h, w = screen_shape
        cell_h = h // self.height
        cell_w = w // self.width
        
        # Mark Link's position
        if player_pos is not None:
            x, y = player_pos
            grid_x = min(x // cell_w, self.width - 1)
            grid_y = min(y // cell_h, self.height - 1)
            grid[grid_y, grid_x] = self.CHAR_MAP['link']
        
        # Mark NPCs
        if npcs:
            for npc in npcs:
                if isinstance(npc, dict):
                    x = npc.get('x', 0)
                    y = npc.get('y', 0)
                elif isinstance(npc, (tuple, list)) and len(npc) >= 2:
                    x, y = npc[0], npc[1]
                else:
                    continue
                
                grid_x = min(x // cell_w, self.width - 1)
                grid_y = min(y // cell_h, self.height - 1)
                
                # Don't overwrite Link
                if grid[grid_y, grid_x] != self.CHAR_MAP['link']:
                    grid[grid_y, grid_x] = self.CHAR_MAP['npc']
        
        # Mark enemies
        if enemies:
            for enemy in enemies:
                if isinstance(enemy, dict):
                    x = enemy.get('x', 0)
                    y = enemy.get('y', 0)
                elif isinstance(enemy, (tuple, list)) and len(enemy) >= 2:
                    x, y = enemy[0], enemy[1]
                else:
                    continue
                
                grid_x = min(x // cell_w, self.width - 1)
                grid_y = min(y // cell_h, self.height - 1)
                
                # Don't overwrite Link or NPCs
                if grid[grid_y, grid_x] not in [
                    self.CHAR_MAP['link'],
                    self.CHAR_MAP['npc']
                ]:
                    grid[grid_y, grid_x] = self.CHAR_MAP['enemy']
        
        # Mark items
        if items:
            for item in items:
                if isinstance(item, dict):
                    x = item.get('x', 0)
                    y = item.get('y', 0)
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    x, y = item[0], item[1]
                else:
                    continue
                
                grid_x = min(x // cell_w, self.width - 1)
                grid_y = min(y // cell_h, self.height - 1)
                
                # Don't overwrite Link, NPCs, or enemies
                if grid[grid_y, grid_x] not in [
                    self.CHAR_MAP['link'],
                    self.CHAR_MAP['npc'],
                    self.CHAR_MAP['enemy']
                ]:
                    grid[grid_y, grid_x] = self.CHAR_MAP['item']
        
        return grid
    
    def _grid_to_string(self, grid: np.ndarray) -> str:
        """Convert 2D character grid to formatted string.
        
        Args:
            grid: 2D array of ASCII characters
        
        Returns:
            Multi-line ASCII art string
        """
        lines = []
        
        # Add top border
        lines.append('=' * (self.width + 2))
        
        # Add grid rows with side borders
        for row in grid:
            lines.append('|' + ''.join(row) + '|')
        
        # Add bottom border
        lines.append('=' * (self.width + 2))
        
        return '\n'.join(lines)
    
    def get_legend(self) -> str:
        """Get legend explaining ASCII symbols.
        
        Returns:
            Legend string
        """
        return """
ASCII LEGEND:
  @ = Link (you)
  N = NPC (talk with A button)
  E = Enemy (avoid or attack)
  I = Item (collect)
  # = Wall/obstacle
  X = Dense object
  . = Sparse object
    = Empty space
  D = Door
  ~ = Water
  T = Tree
  C = Chest
  S = Sign
"""


def create_ascii_visualization(
    screen: np.ndarray,
    game_state: Dict[str, Any],
    include_legend: bool = False
) -> str:
    """Helper function to create ASCII visualization from game state.
    
    Args:
        screen: Screen pixels (144, 160, 3) or (144, 160)
        game_state: Structured game state dict
        include_legend: Whether to include symbol legend
    
    Returns:
        ASCII art string
    """
    renderer = ASCIIRenderer(width=40, height=36)
    
    # Extract entity positions from game state
    player = game_state.get('player', {})
    player_pos = (player.get('x', 0), player.get('y', 0))
    
    npcs = game_state.get('npcs', [])
    enemies = game_state.get('enemies', [])
    items = game_state.get('items', [])
    
    # Render ASCII
    ascii_art = renderer.render_screen(
        screen,
        player_pos=player_pos,
        npcs=npcs,
        enemies=enemies,
        items=items
    )
    
    # Add legend if requested
    if include_legend:
        ascii_art = renderer.get_legend() + '\n' + ascii_art
    
    return ascii_art

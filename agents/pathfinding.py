"""
Screen-Aware Pathfinding System for Zelda Oracle of Seasons

This module implements intelligent pathfinding that analyzes the current screen,
detects obstacles, and navigates Link to desired exits or targets.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import deque
import heapq

class ZeldaPathfinder:
    """Pathfinding system for single-screen navigation in Zelda."""
    
    def __init__(self):
        # Screen dimensions (Game Boy screen is 160x144, but we work with tile grid)
        self.screen_width = 160
        self.screen_height = 144
        self.tile_width = 8  # Each tile is 8x8 pixels
        self.tile_height = 8
        self.grid_width = self.screen_width // self.tile_width  # 20 tiles
        self.grid_height = self.screen_height // self.tile_height  # 18 tiles
        
        # Define obstacle patterns (these would be refined based on actual tile analysis)
        self.solid_tile_patterns = {
            # Common obstacle tile IDs/patterns for Oracle of Seasons
            'water': [0x6E, 0x6F, 0x7E, 0x7F],  # Water tiles
            'trees': [0x20, 0x21, 0x30, 0x31],  # Tree tiles
            'rocks': [0x40, 0x41, 0x50, 0x51],  # Rock/boulder tiles
            'walls': [0x60, 0x61, 0x70, 0x71],  # Wall tiles
            'cliffs': [0x80, 0x81, 0x90, 0x91]  # Cliff tiles
        }
        
        # Exit zones (screen transition points, not dungeon entrances)
        # Focus on center areas of each edge where actual screen transitions occur
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        
        # Exit zones: Target points BEYOND the screen edge to ensure transition
        # Use negative coordinates to force continued movement into the edge
        self.exit_zones = {
            'north': [(1, -1), (center_x - 3, -1), (center_x, -1), (center_x + 3, -1), (self.grid_width - 2, -1), (self.grid_width - 1, -1)],  # Beyond top edge
            'south': [(1, self.grid_height), (center_x - 3, self.grid_height), (center_x, self.grid_height), (center_x + 3, self.grid_height), (self.grid_width - 2, self.grid_height), (self.grid_width - 1, self.grid_height)],  # Beyond bottom edge
            'east': [(self.grid_width, 1), (self.grid_width, center_y - 3), (self.grid_width, center_y), (self.grid_width, center_y + 3), (self.grid_width, self.grid_height - 2), (self.grid_width, self.grid_height - 1)],  # Beyond right edge  
            'west': [(-1, 1), (-1, center_y - 3), (-1, center_y), (-1, center_y + 3), (-1, self.grid_height - 2), (-1, self.grid_height - 1)]  # Beyond left edge
        }
        
    def analyze_screen(self, pyboy_instance) -> np.ndarray:
        """Analyze current screen and create obstacle map."""
        try:
            # Get current screen buffer (PyBoy v2.x API)
            screen_buffer = pyboy_instance.screen.ndarray
            
            # Create obstacle grid (True = walkable, False = obstacle)
            obstacle_grid = np.ones((self.grid_height, self.grid_width), dtype=bool)
            
            # Analyze screen in 8x8 tile chunks
            for tile_y in range(self.grid_height):
                for tile_x in range(self.grid_width):
                    pixel_x = tile_x * self.tile_width
                    pixel_y = tile_y * self.tile_height
                    
                    # Extract 8x8 tile region
                    if (pixel_x + self.tile_width <= self.screen_width and 
                        pixel_y + self.tile_height <= self.screen_height):
                        
                        tile_region = screen_buffer[pixel_y:pixel_y+self.tile_height, 
                                                  pixel_x:pixel_x+self.tile_width]
                        
                        # Check if this tile is an obstacle
                        if self._is_obstacle_tile(tile_region):
                            obstacle_grid[tile_y, tile_x] = False
                            
            return obstacle_grid
            
        except Exception as e:
            print(f"Warning: Screen analysis failed: {e}")
            # Return default walkable grid if analysis fails
            return np.ones((self.grid_height, self.grid_width), dtype=bool)
    
    def _is_obstacle_tile(self, tile_region: np.ndarray) -> bool:
        """Determine if an 8x8 tile region is an obstacle."""
        try:
            # Simple heuristic: look at pixel density and patterns
            avg_pixel_value = np.mean(tile_region)
            pixel_variance = np.var(tile_region)
            
            # Dark, uniform regions are likely solid obstacles
            if avg_pixel_value < 50 and pixel_variance < 100:
                return True
                
            # Very bright, uniform regions might be walls
            if avg_pixel_value > 200 and pixel_variance < 50:
                return True
                
            # Check for specific patterns (water, trees, rocks)
            # This would be enhanced with actual tile ID detection
            
            return False
            
        except:
            return False  # Default to walkable if analysis fails
    
    def get_link_position(self, pyboy_instance) -> Tuple[int, int]:
        """Get Link's current position on the tile grid."""
        try:
            # Access Link's position from game memory (PyBoy v2.x API)
            # These memory addresses would need to be specific to Oracle of Seasons
            x_addr = 0xC100  # Example address for Link's X position
            y_addr = 0xC101  # Example address for Link's Y position
            
            link_x_pixel = pyboy_instance.memory[x_addr]
            link_y_pixel = pyboy_instance.memory[y_addr]
            
            # Convert pixel coordinates to tile coordinates
            link_tile_x = max(0, min(self.grid_width - 1, link_x_pixel // self.tile_width))
            link_tile_y = max(0, min(self.grid_height - 1, link_y_pixel // self.tile_height))
            
            return (link_tile_x, link_tile_y)
            
        except Exception as e:
            print(f"Warning: Could not get Link position: {e}")
            # Default to center of screen
            return (self.grid_width // 2, self.grid_height // 2)
    
    def find_path_to_exit(self, pyboy_instance, direction: str) -> List[Tuple[int, int]]:
        """Find path from Link's position to the specified screen exit."""
        # Analyze current screen
        obstacle_grid = self.analyze_screen(pyboy_instance)
        
        # Get Link's current position
        start_pos = self.get_link_position(pyboy_instance)
        
        # Get target exit zone
        if direction.lower() not in self.exit_zones:
            print(f"Warning: Unknown direction {direction}")
            return []
            
        target_zone = self.exit_zones[direction.lower()]
        
        # DEBUG: Print pathfinding details
        print(f"🔍 Pathfinding DEBUG:")
        print(f"   Start position: {start_pos}")
        print(f"   Direction: {direction}")
        print(f"   Target zone ({len(target_zone)} points): {target_zone[:3]}...{target_zone[-3:] if len(target_zone) > 3 else []}")
        
        # Find closest accessible exit point
        best_path = None
        shortest_distance = float('inf')
        best_target = None
        
        for exit_point in target_zone:
            # Convert off-screen targets to screen-edge points for pathfinding
            clamped_exit = self._clamp_to_screen_edge(exit_point, direction)
            
            # Check if clamped exit point is within bounds and walkable
            if (0 <= clamped_exit[0] < self.grid_width and 
                0 <= clamped_exit[1] < self.grid_height and
                obstacle_grid[clamped_exit[1], clamped_exit[0]]):  # If exit point is walkable
                
                path = self._a_star_pathfind(obstacle_grid, start_pos, clamped_exit)
                if path and len(path) < shortest_distance:
                    # Extend path to ensure screen transition
                    extended_path = self._extend_path_for_transition(path, direction)
                    best_path = extended_path
                    shortest_distance = len(path)
                    best_target = exit_point
                    
        if best_path and best_target:
            print(f"   Best target: {best_target}")
            print(f"   Path length: {len(best_path)}")
            print(f"   First 3 steps: {best_path[:3]}")
                    
        return best_path if best_path else []
    
    def _a_star_pathfind(self, obstacle_grid: np.ndarray, start: Tuple[int, int], 
                        goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm."""
        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        def get_neighbors(pos):
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-directional movement
                new_x, new_y = pos[0] + dx, pos[1] + dy
                if (0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height and
                    obstacle_grid[new_y, new_x]):
                    neighbors.append((new_x, new_y))
            return neighbors
        
        # A* algorithm implementation
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Reverse to get path from start to goal
            
            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def path_to_actions(self, path: List[Tuple[int, int]]) -> List[int]:
        """Convert a path to a sequence of game actions."""
        if len(path) < 2:
            return []
            
        actions = []
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            dx = next_pos[0] - current[0]
            dy = next_pos[1] - current[1]
            
            # Convert direction to action ID (using correct ZeldaAction mapping)
            if dx > 0:
                actions.append(4)  # RIGHT (was 3)
            elif dx < 0:
                actions.append(3)  # LEFT (was 2)
            elif dy > 0:
                actions.append(2)  # DOWN (was 1)
            elif dy < 0:
                actions.append(1)  # UP (was 0)
                
        return actions
    
    def _clamp_to_screen_edge(self, target: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Clamp an off-screen target to the nearest screen edge."""
        x, y = target
        
        if direction.lower() == 'north':
            return (max(0, min(x, self.grid_width - 1)), 0)
        elif direction.lower() == 'south':
            return (max(0, min(x, self.grid_width - 1)), self.grid_height - 1)
        elif direction.lower() == 'east':
            return (self.grid_width - 1, max(0, min(y, self.grid_height - 1)))
        elif direction.lower() == 'west':
            return (0, max(0, min(y, self.grid_height - 1)))
        
        return target
    
    def _extend_path_for_transition(self, path: List[Tuple[int, int]], direction: str) -> List[Tuple[int, int]]:
        """Extend path with additional steps to ensure screen transition."""
        if not path:
            return path
        
        extended_path = path.copy()
        last_pos = path[-1]
        
        # Add 3 more steps in the same direction to ensure transition
        for _ in range(3):
            if direction.lower() == 'north':
                next_pos = (last_pos[0], last_pos[1] - 1)  # Continue moving up (into negative Y)
            elif direction.lower() == 'south':
                next_pos = (last_pos[0], last_pos[1] + 1)  # Continue moving down
            elif direction.lower() == 'east':
                next_pos = (last_pos[0] + 1, last_pos[1])  # Continue moving right
            elif direction.lower() == 'west':
                next_pos = (last_pos[0] - 1, last_pos[1])  # Continue moving left
            else:
                break
            
            extended_path.append(next_pos)
            last_pos = next_pos
        
        return extended_path


class PathfindingActionExecutor:
    """Executes pathfinding-based actions in the strategic framework."""
    
    def __init__(self):
        self.pathfinder = ZeldaPathfinder()
        self.current_path = []
        self.path_index = 0
        
        # Stuck detection and exploration
        self.stuck_detection_history = []
        self.exploration_mode = False
        self.exploration_pattern = []
        self.exploration_index = 0
        self.last_known_position = None
        self.stuck_threshold = 5  # Consider stuck after 5 steps in same position
        self.path_actions = []
        
    def start_pathfinding_to_exit(self, pyboy_instance, direction: str) -> bool:
        """Start pathfinding to a screen exit."""
        try:
            path = self.pathfinder.find_path_to_exit(pyboy_instance, direction)
            if path:
                self.current_path = path
                self.path_actions = self.pathfinder.path_to_actions(path)
                self.path_index = 0
                print(f"🗺️  Pathfinding: Found path to {direction} exit with {len(path)} steps")
                return True
            else:
                print(f"❌ Pathfinding: No path found to {direction} exit")
                return False
        except Exception as e:
            print(f"❌ Pathfinding error: {e}")
            return False
    
    def get_next_action(self, current_position: Optional[Tuple[int, int]] = None) -> Optional[int]:
        """Get the next action with stuck detection and exploration fallback."""
        
        # Update stuck detection history
        if current_position:
            self.stuck_detection_history.append(current_position)
            if len(self.stuck_detection_history) > self.stuck_threshold:
                self.stuck_detection_history.pop(0)
            
            # DEBUG: Show position tracking
            if len(self.stuck_detection_history) % 5 == 0:
                unique_positions = len(set(self.stuck_detection_history))
                print(f"🔍 Position tracking: {current_position}, history size: {len(self.stuck_detection_history)}, unique: {unique_positions}")
        else:
            print("⚠️  No position provided for stuck detection")
        
        # Check if Link is stuck (same position for several steps)
        is_stuck = (len(self.stuck_detection_history) >= self.stuck_threshold and 
                   len(set(self.stuck_detection_history)) == 1)
        
        if is_stuck and not self.exploration_mode:
            stuck_pos = self.stuck_detection_history[0] if self.stuck_detection_history else None
            print(f"🚫 STUCK DETECTED at {stuck_pos}: Switching to edge exploration mode!")
            print(f"   History: {self.stuck_detection_history}")
            self.exploration_mode = True
            self.exploration_pattern = self._generate_exploration_pattern(current_position)
            self.exploration_index = 0
        
        # If in exploration mode, use exploration pattern
        if self.exploration_mode:
            return self._get_exploration_action()
        
        # Normal pathfinding
        if self.path_index < len(self.path_actions):
            action = self.path_actions[self.path_index]
            self.path_index += 1
            return action
            
        return None
    
    def is_path_complete(self) -> bool:
        """Check if the current path is complete."""
        return self.path_index >= len(self.path_actions)
    
    def reset_path(self):
        """Reset the current path."""
        self.current_path = []
        self.path_actions = []
        self.path_index = 0
        # Also reset exploration mode
        self.exploration_mode = False
        self.exploration_pattern = []
        self.exploration_index = 0
        self.stuck_detection_history = []
    
    def _generate_exploration_pattern(self, stuck_position: Tuple[int, int]) -> List[int]:
        """Generate exploration pattern to find screen transition when stuck."""
        pattern = []
        
        # For north direction (getting stuck at top edge), try different strategies
        # Strategy 1: Try moving left and right along the top edge
        pattern.extend([
            3, 3, 3,  # Move LEFT several steps  
            1, 1,     # Try UP again
            4, 4, 4, 4, 4, 4,  # Move RIGHT across the edge
            1, 1,     # Try UP again
            3, 3, 3,  # Back to center-ish
            1, 1,     # Try UP again
        ])
        
        # Strategy 2: Try diagonal movement (UP-RIGHT pattern user mentioned)
        pattern.extend([
            4, 1, 4, 1, 4, 1,  # UP-RIGHT diagonal movement
            1, 1, 1,           # Pure UP attempts
            4, 4,              # More RIGHT
            1, 1, 1,           # UP attempts
        ])
        
        # Strategy 3: Backtrack and try different approach
        pattern.extend([
            2, 2,              # Back DOWN from edge
            4, 4, 4,           # RIGHT
            1, 1, 1, 1,        # UP attempts
            3, 3,              # LEFT
            1, 1, 1, 1,        # UP attempts
        ])
        
        # Strategy 4: Systematic edge sweeping
        for i in range(3):  # Repeat 3 times
            pattern.extend([
                3, 3, 3, 3, 3,     # Go LEFT to far left
                1,                 # Try UP
                4,                 # RIGHT 1 step
                1,                 # Try UP
                4,                 # RIGHT 1 step  
                1,                 # Try UP
                4, 4, 4, 4, 4,     # Continue RIGHT
                1,                 # Try UP
            ])
        
        print(f"🔍 Generated exploration pattern with {len(pattern)} actions")
        return pattern
    
    def _get_exploration_action(self) -> Optional[int]:
        """Get next exploration action."""
        if self.exploration_index < len(self.exploration_pattern):
            action = self.exploration_pattern[self.exploration_index]
            self.exploration_index += 1
            
            if self.exploration_index % 10 == 0:
                action_names = ["NOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
                action_name = action_names[action] if action < len(action_names) else f"UNKNOWN({action})"
                progress = (self.exploration_index / len(self.exploration_pattern)) * 100
                print(f"🔍 Exploration {progress:.0f}%: {action_name} (step {self.exploration_index}/{len(self.exploration_pattern)})")
            
            return action
        else:
            # Exploration complete, reset and try pathfinding again
            print("✅ Exploration pattern complete, returning to pathfinding mode")
            self.exploration_mode = False
            self.exploration_pattern = []
            self.exploration_index = 0
            return None

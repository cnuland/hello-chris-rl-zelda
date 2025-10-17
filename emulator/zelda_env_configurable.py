"""Configurable Gymnasium environment for The Legend of Zelda: Oracle of Seasons.

This version supports both LLM-guided and pure RL modes with optimized performance.
"""

import gymnasium as gym
import numpy as np
import yaml
from typing import Dict, Any, Tuple, Optional
from gymnasium import spaces
from pathlib import Path

try:
    # Try relative imports first (for module usage)
    from .pyboy_bridge import ZeldaPyBoyBridge
    from .input_map import ZeldaAction
    from ..observation.state_encoder import ZeldaStateEncoder
except ImportError:
    # Fall back to absolute imports (for direct script execution)
    from emulator.pyboy_bridge import ZeldaPyBoyBridge
    from emulator.input_map import ZeldaAction  
    from observation.state_encoder import ZeldaStateEncoder


class ZeldaConfigurableEnvironment(gym.Env):
    """Configurable Gymnasium environment for Zelda Oracle of Seasons."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, 
                 rom_path: str,
                 config_path: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None,
                 headless: bool = True, 
                 render_mode: Optional[str] = None,
                 visual_test_mode: bool = False):
        """Initialize Zelda environment with configurable LLM integration.

        Args:
            rom_path: Path to Oracle of Seasons ROM file
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to config_path)
            headless: Whether to run emulator without display
            render_mode: Rendering mode for gymnasium
            visual_test_mode: Enable visual test mode (single episode, non-headless)
        """
        super().__init__()

        self.rom_path = rom_path
        self.visual_test_mode = visual_test_mode
        
        # Override headless mode for visual testing
        if visual_test_mode:
            self.headless = False
            print("🎮 Visual Test Mode Enabled - PyBoy window will be displayed")
        else:
            self.headless = headless
            
        self.render_mode = render_mode

        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            # Default configuration for pure RL mode
            self.config = self._get_default_config()
            
        # Override configuration for visual test mode
        if visual_test_mode:
            self._apply_visual_test_overrides()

        # Extract key configuration options
        env_config = self.config.get('environment', {})
        planner_config = self.config.get('planner_integration', {})
        
        self.use_llm = planner_config.get('use_planner', False)
        self.enable_structured_states = self.use_llm  # Only generate structured states if using LLM
        self.observation_type = env_config.get('observation_type', 'vector')
        self.normalize_observations = env_config.get('normalize_observations', True)
        self.frame_skip = env_config.get('frame_skip', 4)
        
        # Initialize components
        auto_load_save = planner_config.get('auto_load_save_state', True)
        self.bridge = ZeldaPyBoyBridge(
            rom_path=rom_path, 
            headless=headless,
            auto_load_save_state=auto_load_save
        )
        
        # Initialize state encoder with appropriate settings
        if self.enable_structured_states:
            self.state_encoder = ZeldaStateEncoder(
                enable_visual=planner_config.get('enable_visual', True),
                compression_mode=planner_config.get('compression_mode', 'bit_packed'),
                use_structured_entities=planner_config.get('use_structured_entities', True)
            )
        else:
            # Minimal state encoder for pure RL
            self.state_encoder = ZeldaStateEncoder(
                enable_visual=False,
                use_structured_entities=False
            )

        # Gymnasium spaces
        self.action_space = spaces.Discrete(len(ZeldaAction))
        
        # Setup observation space based on observation type
        if self.observation_type == 'visual':
            # Visual observations: Grayscale Game Boy screen (144×160×1)
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(144, 160, 1),  # (height, width, channels)
                dtype=np.uint8
            )
            print("🎮 Visual observation mode: Screen pixels (144×160×1)")
        else:
            # Vector observations: State encoder vector
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.state_encoder.state_vector_size,),
                dtype=np.float32
            )
            print(f"📊 Vector observation mode: {self.state_encoder.state_vector_size} features")

        # Environment state
        self.current_structured_state: Optional[Dict[str, Any]] = None
        self.previous_structured_state: Optional[Dict[str, Any]] = None
        self.step_count = 0
        self.episode_count = 0
        self.max_steps = env_config.get('max_episode_steps', 10000)
        
        # Performance tracking
        self.structured_state_generation_time = 0.0
        self.total_step_time = 0.0
        
        # Exploration tracking for enhanced rewards
        self.visited_rooms = set()  # Track all rooms/screens visited
        self.visited_dungeons = set()  # Track dungeons discovered
        self.last_dialogue_state = 0  # Track NPC dialogue interactions
        self.last_room_id = None  # Track room transitions
        
        # LLM guidance tracking - MASSIVE EMPHASIS SYSTEM
        self.last_llm_suggestion = None  # Store latest LLM guidance
        self.steps_since_llm_call = 0    # Count steps since last LLM call
        self.llm_aligned_actions = 0     # Count actions that follow LLM guidance
        self.total_actions = 0           # Total actions for alignment rate
        
        # NEW: Extreme exploration tracking (movement is EVERYTHING!)
        self.last_position = (0, 0)      # Track (X, Y) position
        self.stuck_counter = 0           # Count steps in same position
        self.visited_grid_squares = {}   # Track grid squares per room: {room_id: set((gx, gy))}
        self.last_action = None          # Track last action taken
        
        # Smart menu usage tracking
        self.last_equipped_items = (0, 0)  # Track (A button, B button) items
        self.consecutive_menu_opens = 0     # Count consecutive menu actions
        self.steps_since_menu = 0           # Steps since last menu open
        
        # Inventory change tracking (for item acquisition logging)
        self.last_inventory = None          # Track inventory state (16 bytes)
        self.items_obtained = set()         # Track unique items obtained this episode

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for pure RL mode."""
        return {
            'environment': {
                'frame_skip': 4,
                'action_repeat': 1,
                'observation_type': 'vector',
                'normalize_observations': True,
                'max_episode_steps': 10000
            },
            'planner_integration': {
                'use_planner': False,
                'enable_visual': False,
                'use_structured_entities': False,
                'auto_load_save_state': True
            },
            'rewards': {
                'rupee_reward': 0.01,
                'key_reward': 0.5,
                'death_penalty': -3.0,
                'movement_reward': 0.001,
                'time_penalty': -0.0001,
                # EXPLORATION BONUSES - HUGE rewards for meaningful gameplay!
                'room_discovery_reward': 10.0,      # Big bonus for new areas
                'dungeon_discovery_reward': 25.0,   # Massive bonus for dungeon entry
                'dungeon_bonus': 5.0,               # Continuous bonus while in dungeon
                'npc_interaction_reward': 15.0,     # Big bonus for talking to NPCs
                
                # LLM GUIDANCE REWARDS - MASSIVE EMPHASIS FOR FOLLOWING AI SUGGESTIONS!
                'llm_guidance_multiplier': 5.0,     # Multiply normal rewards by 5x when following LLM
                'llm_strategic_bonus': 2.0,         # Continuous bonus for following LLM direction
                'llm_directional_bonus': 1.0,       # Bonus for moving in LLM-suggested direction
                'llm_completion_bonus': 50.0        # HUGE bonus for completing LLM macro goals
            }
        }
    
    def _apply_visual_test_overrides(self) -> None:
        """Apply configuration overrides for visual test mode."""
        # Override environment settings for visual testing
        if 'environment' not in self.config:
            self.config['environment'] = {}
            
        # Shorter episodes for visual testing
        self.config['environment']['max_episode_steps'] = 1000
        
        # Slower frame skip for better visual observation
        self.config['environment']['frame_skip'] = 2
        
        # Override training settings if present
        if 'training' not in self.config:
            self.config['training'] = {}
            
        # Single episode for visual test
        self.config['training']['max_episode_steps'] = 1000
        self.config['training']['total_timesteps'] = 1000  # Single episode
        
        print("🎯 Visual Test Mode Configuration:")
        print(f"   Max episode steps: {self.config['environment']['max_episode_steps']}")
        print(f"   Frame skip: {self.config['environment']['frame_skip']}")
        print(f"   Total timesteps: {self.config['training']['total_timesteps']}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset bridge
        self.bridge.reset()
        
        # Reset environment state
        self.step_count = 0
        self.episode_count += 1
        self.current_structured_state = None
        self.previous_structured_state = None
        
        # Track initial death count to detect when Link dies
        # TOTAL_DEATHS is at 0xC61E (2 bytes)
        self.initial_death_count = self.bridge.get_memory(0xC61E) + (self.bridge.get_memory(0xC61F) << 8)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        import time
        step_start_time = time.time()
        
        # 🎯 Track action for strategic reward calculation
        self.last_action = action
        
        # Convert action and execute with frame skip
        zelda_action = ZeldaAction(action)
        
        # Execute action with frame skip
        total_reward = 0.0
        for _ in range(self.frame_skip):
            self.bridge.step(zelda_action)
            # Bridge doesn't return reward, we calculate it in _calculate_reward
            total_reward += 0.0
            
        # Update step count
        self.step_count += 1
        
        # Get observation and info
        obs = self._get_observation()
        
        # Track position for stuck detection
        if hasattr(self.bridge, 'get_memory'):
            current_x = self.bridge.get_memory(0xC4AC)
            current_y = self.bridge.get_memory(0xC4AD)
            current_pos = (current_x, current_y)
            
            # Check if position changed
            if current_pos == self.last_position:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0  # Reset counter when moving
                self.last_position = current_pos
        
        reward = self._calculate_reward(total_reward)
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        info = self._get_info()
        
        # Performance tracking
        self.total_step_time += time.time() - step_start_time
        
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation (vector or visual based on mode)."""
        import time
        
        # Visual observation mode
        if self.observation_type == 'visual':
            # Get screen pixels
            screen_rgb = self.bridge.get_screen()  # (144, 160, 3)
            
            # Convert to grayscale
            grayscale = np.dot(screen_rgb[...,:3], [0.299, 0.587, 0.114])
            
            # Add channel dimension: (144, 160) → (144, 160, 1)
            observation = np.expand_dims(grayscale, axis=-1).astype(np.uint8)
            
            # Still generate structured state for LLM if needed
            if self.enable_structured_states:
                struct_start_time = time.time()
                _, structured_state = self.state_encoder.encode_state(self.bridge)
                self.previous_structured_state = self.current_structured_state
                self.current_structured_state = structured_state
                self.structured_state_generation_time += time.time() - struct_start_time
            
            return observation
        
        # Vector observation mode (original)
        else:
            if self.enable_structured_states:
                # Generate structured state (slower but needed for LLM)
                struct_start_time = time.time()
                numeric_vector, structured_state = self.state_encoder.encode_state(self.bridge)
                self.previous_structured_state = self.current_structured_state
                self.current_structured_state = structured_state
                self.structured_state_generation_time += time.time() - struct_start_time
                
                # Return numeric vector for RL
                obs = numeric_vector
            else:
                # Fast path for pure RL - no structured state generation
                obs, _ = self.state_encoder.encode_state(self.bridge)
                
            # Normalize if configured
            if self.normalize_observations:
                obs = np.clip(obs, 0.0, 1.0)
                
            return obs.astype(np.float32)

    def _calculate_reward(self, base_reward: float) -> float:
        """Calculate reward with EXTREME emphasis on movement and exploration."""
        reward_config = self.config.get('reward_structure', {})
        
        total_reward = base_reward
        
        # BASE REWARDS
        # Time penalty (encourages efficiency)
        total_reward += reward_config.get('time_penalty', -0.0001)
        
        # Movement reward - ONLY if position actually changed!
        movement_reward = reward_config.get('movement', 0.1)
        if self.stuck_counter == 0:  # Position changed this step (stuck_counter reset)
            total_reward += movement_reward
            # Occasional logging to verify
            if hasattr(self, 'step_count') and self.step_count % 1000 == 0:
                print(f"🚶 Movement reward: Position changed (+{movement_reward:.1f})")
        # else: No movement reward if stuck (position unchanged)
        
        # Position stuck penalty (staying in same X,Y)
        # NOTE: Currently disabled (0.0) due to Y-coordinate bug
        position_stuck_penalty = reward_config.get('position_stuck', 0.0)
        if self.stuck_counter > 5:  # If stuck for more than 5 steps
            total_reward += position_stuck_penalty
            if self.stuck_counter % 10 == 0 and position_stuck_penalty != 0:  # Log only if penalty active
                print(f"⚠️  POSITION STUCK! Same spot for {self.stuck_counter} steps ({position_stuck_penalty:.1f} penalty)")
        
        # SMART Menu usage penalty (pressing START button)
        if self.last_action == 7:  # START button (ZeldaAction.START = 7)
            # Track consecutive menu opens (escalating penalty for menu surfing!)
            self.consecutive_menu_opens += 1
            self.steps_since_menu = 0
            
            # Check if equipped items changed (reward switching, penalize idling)
            try:
                if hasattr(self.bridge, 'get_memory'):
                    current_a = self.bridge.get_memory(0xC681)  # A button item
                    current_b = self.bridge.get_memory(0xC680)  # B button item
                    current_items = (current_a, current_b)
                    
                    items_changed = current_items != self.last_equipped_items
                    self.last_equipped_items = current_items
                    
                    if items_changed and self.consecutive_menu_opens == 1:
                        # Good menu usage: Switching items purposefully
                        switch_reward = reward_config.get('item_switch_reward', 0.5)
                        total_reward += switch_reward
                        print(f"🔄 ITEM SWITCHED! Equipment changed (+{switch_reward:.1f} reward)")
                    else:
                        # Bad menu usage: Surfing without purpose
                        # Escalating penalty for consecutive menu opens
                        base_penalty = reward_config.get('menu_usage', -0.5)
                        escalation = self.consecutive_menu_opens - 1  # 0 for first, 1 for second, etc.
                        menu_penalty = base_penalty * (1 + escalation)  # -0.5, -1.0, -1.5, -2.0...
                        total_reward += menu_penalty
                        
                        if self.consecutive_menu_opens > 1:
                            print(f"📋 MENU SURFING! {self.consecutive_menu_opens} consecutive opens ({menu_penalty:.1f} penalty)")
                        else:
                            print(f"📋 MENU OPENED! START button pressed ({menu_penalty:.1f} penalty)")
            except:
                # Fallback: Simple penalty if memory read fails
                menu_penalty = reward_config.get('menu_usage', -0.5)
                total_reward += menu_penalty
        else:
            # Reset consecutive counter if not opening menu
            if self.consecutive_menu_opens > 0:
                self.consecutive_menu_opens = 0
            self.steps_since_menu += 1
        
        # EXPLORATION BONUSES - Get current game state for analysis
        try:
            if hasattr(self.bridge, 'get_memory'):
                current_room = self.bridge.get_memory(0xC63B)  # Current room/screen ID
                dialogue_state = self.bridge.get_memory(0xC2EF)  # Dialogue/cutscene state
                dungeon_floor = self.bridge.get_memory(0xC63D)  # Dungeon floor (0 = overworld)
                
                # NEW: Track inventory changes (detect item acquisition)
                current_inventory = []
                for i in range(16):  # Read 16 bytes of inventory
                    current_inventory.append(self.bridge.get_memory(0xC682 + i))
                current_inventory = tuple(current_inventory)
                
                if self.last_inventory is not None and current_inventory != self.last_inventory:
                    # Inventory changed! Check which items were added
                    for i, (old, new) in enumerate(zip(self.last_inventory, current_inventory)):
                        if new > 0 and new != old:
                            item_id = f"inventory_slot_{i}_value_{new}"
                            if item_id not in self.items_obtained:
                                self.items_obtained.add(item_id)
                                # Map some common items
                                item_names = {
                                    0: 'None', 1: 'Wooden Sword', 2: 'Bombs', 3: 'Boomerang',
                                    4: 'Rod of Seasons', 5: 'Feather', 6: 'Shovel', 7: 'Bracelet',
                                    8: 'Flippers', 9: 'Magnetic Gloves', 10: 'Slingshot',
                                    20: 'Gnarled Key', 21: 'Floodgate Key', 22: 'Dragon Key'
                                }
                                item_name = item_names.get(new, f'Item {new}')
                                print(f"🎁 NEW ITEM OBTAINED! Slot {i}: {item_name} (value={new})")
                                
                                # Check for specific milestone items
                                if new == 20 or 'Gnarled' in item_name:
                                    print(f"🔑 MILESTONE: Gnarled Key Obtained from Maku Tree!")
                                    total_reward += reward_config.get('gnarled_key_obtained', 200.0)
                
                self.last_inventory = current_inventory
                
                # A) NEW ROOM EXPLORATION REWARD - MASSIVE BONUS (ONLY ONCE PER ROOM!)
                if current_room not in self.visited_rooms:
                    self.visited_rooms.add(current_room)
                    exploration_reward = reward_config.get('new_room_discovery', 20.0)
                    total_reward += exploration_reward
                    print(f"🗺️  NEW ROOM DISCOVERED! Room {current_room} (+{exploration_reward:.1f} reward) | Total visited: {len(self.visited_rooms)}")
                    # Reset grid squares for new room
                    self.visited_grid_squares[current_room] = set()
                else:
                    # Apply revisit penalty (discourage loops and farming)
                    revisit_penalty = reward_config.get('revisit_penalty', -0.5)
                    total_reward += revisit_penalty
                    if self.steps_since_last_llm_call < 5 and self.last_room_id != current_room:
                        print(f"⚠️  REVISITING ROOM {current_room} ({revisit_penalty:.1f} penalty)")
                
                # NEW: Grid exploration within room (encourage thorough exploration)
                player_x = self.bridge.get_memory(0xC4AC)
                player_y = self.bridge.get_memory(0xC4AD)
                # Divide room into 16x16 pixel grid squares
                grid_x = player_x // 16
                grid_y = player_y // 16
                grid_square = (grid_x, grid_y)
                
                # Initialize room's grid tracking if needed
                if current_room not in self.visited_grid_squares:
                    self.visited_grid_squares[current_room] = set()
                
                # Reward for exploring new grid square within room
                if grid_square not in self.visited_grid_squares[current_room]:
                    self.visited_grid_squares[current_room].add(grid_square)
                    grid_reward = reward_config.get('grid_exploration', 5.0)
                    total_reward += grid_reward
                    # Only log occasionally to avoid spam
                    if len(self.visited_grid_squares[current_room]) % 5 == 0:
                        print(f"📍 Grid square explored! Room {current_room}: {len(self.visited_grid_squares[current_room])} squares (+{grid_reward:.1f})")
                
                # B) DUNGEON REWARDS - MASSIVE BONUS!
                if dungeon_floor > 0:  # In a dungeon
                    # Continuous dungeon bonus
                    dungeon_bonus = reward_config.get('dungeon_bonus', 5.0)
                    total_reward += dungeon_bonus
                    
                    # First-time dungeon discovery
                    if dungeon_floor not in self.visited_dungeons:
                        self.visited_dungeons.add(dungeon_floor)
                        dungeon_discovery_bonus = reward_config.get('dungeon_discovery_reward', 25.0)
                        total_reward += dungeon_discovery_bonus
                        print(f"🏰 DUNGEON DISCOVERED! Floor {dungeon_floor} (+{dungeon_discovery_bonus:.1f} bonus)")
                
                # C) NPC INTERACTION REWARDS - DIALOGUE DETECTION
                if dialogue_state > 0 and dialogue_state != self.last_dialogue_state:
                    npc_bonus = reward_config.get('npc_interaction_reward', 15.0)
                    total_reward += npc_bonus
                    if self.episode_count % 5 == 0:  # Log occasionally
                        print(f"💬 NPC INTERACTION! Dialogue state: {dialogue_state} (+{npc_bonus:.1f} reward)")
                
                self.last_dialogue_state = dialogue_state
                self.last_room_id = current_room
                
                # D) LLM GUIDANCE REWARDS - MASSIVE EMPHASIS!
                if hasattr(self, 'last_llm_suggestion') and self.last_llm_suggestion:
                    llm_bonus = self._calculate_llm_guidance_reward(current_room, dialogue_state, dungeon_floor)
                    total_reward += llm_bonus
                    
                # E) 🎯 STRATEGIC ACTION REWARDS - Teach RL proper Zelda gameplay
                strategic_bonus = self._calculate_strategic_action_rewards()
                total_reward += strategic_bonus
                    
        except Exception as e:
            # If memory access fails, just use base rewards
            pass
        
        return total_reward

    def _calculate_llm_guidance_reward(self, current_room: int, dialogue_state: int, dungeon_floor: int) -> float:
        """Calculate MASSIVE reward bonuses for following LLM guidance."""
        if not self.last_llm_suggestion:
            return 0.0
            
        reward_config = self.config.get('rewards', {})
        llm_bonus = 0.0
        
        # Get LLM suggestion details
        llm_action = self.last_llm_suggestion.get('action', '').upper()
        llm_target = self.last_llm_suggestion.get('target', '').lower()
        llm_reasoning = self.last_llm_suggestion.get('reasoning', '').lower()
        
        # ULTRA-HIGH REWARDS FOR LLM ALIGNMENT
        base_llm_multiplier = reward_config.get('llm_guidance_multiplier', 5.0)  # 5x normal rewards!
        
        # 1) EXPLORATION ALIGNMENT - If LLM said explore and we found new room
        if 'explore' in llm_action and current_room not in self.visited_rooms:
            exploration_bonus = reward_config.get('room_discovery_reward', 10.0) * base_llm_multiplier
            llm_bonus += exploration_bonus
            print(f"🧠 LLM EXPLORATION SUCCESS! Found new room as suggested (+{exploration_bonus:.1f})")
            
        # 2) DUNGEON ALIGNMENT - If LLM suggested dungeon and we entered one
        if any(word in llm_action for word in ['DUNGEON', 'ENTER']) and dungeon_floor > 0:
            dungeon_bonus = reward_config.get('dungeon_discovery_reward', 25.0) * base_llm_multiplier
            llm_bonus += dungeon_bonus
            print(f"🧠 LLM DUNGEON SUCCESS! Entered dungeon as suggested (+{dungeon_bonus:.1f})")
            
        # 3) SOCIAL ALIGNMENT - If LLM suggested talking and dialogue activated
        if any(word in llm_action for word in ['TALK', 'NPC']) and dialogue_state > self.last_dialogue_state:
            social_bonus = reward_config.get('npc_interaction_reward', 15.0) * base_llm_multiplier  
            llm_bonus += social_bonus
            print(f"🧠 LLM SOCIAL SUCCESS! Talked to NPC as suggested (+{social_bonus:.1f})")
            
        # 4) STRATEGIC ALIGNMENT - Continuous bonus for following LLM direction
        if self.steps_since_llm_call < 100:  # Within 100 steps of LLM call
            strategic_bonus = reward_config.get('llm_strategic_bonus', 2.0)
            llm_bonus += strategic_bonus
            
        # 5) DIRECTIONAL ALIGNMENT - Reward for moving in LLM-suggested direction
        if any(direction in llm_reasoning for direction in ['north', 'south', 'east', 'west', 'up', 'down', 'left', 'right']):
            directional_bonus = reward_config.get('llm_directional_bonus', 1.0)
            llm_bonus += directional_bonus
            
        # 6) COMPLETION BONUS - Massive reward for completing LLM macro goals
        if self._check_llm_goal_completion(llm_action, current_room, dialogue_state, dungeon_floor):
            completion_bonus = reward_config.get('llm_completion_bonus', 50.0)
            llm_bonus += completion_bonus
            print(f"🧠 LLM GOAL COMPLETED! Major objective achieved (+{completion_bonus:.1f})")
            
        # Track steps since last LLM call
        self.steps_since_llm_call += 1
        
        return llm_bonus
    
    def _check_llm_goal_completion(self, llm_action: str, current_room: int, dialogue_state: int, dungeon_floor: int) -> bool:
        """Check if LLM's goal has been completed."""
        if 'EXPLORE' in llm_action:
            return current_room not in self.visited_rooms  # Found new area
        elif any(word in llm_action for word in ['DUNGEON', 'ENTER']):
            return dungeon_floor > 0  # Successfully entered dungeon
        elif any(word in llm_action for word in ['TALK', 'NPC']):
            return dialogue_state > 0  # Successfully initiated dialogue
        elif 'ATTACK' in llm_action or 'COMBAT' in llm_action:
            # Could check for enemy defeat here
            return False
        return False
    
    def _calculate_strategic_action_rewards(self) -> float:
        """🎯 Calculate rewards that teach the RL agent proper Zelda gameplay patterns.
        
        This method rewards:
        1. Combat actions (A button near enemies)
        2. Environmental interaction (grass cutting, rock lifting)
        3. Item collection attempts
        4. Strategic movement patterns
        
        Returns:
            Strategic action bonus reward
        """
        try:
            strategic_reward = 0.0
            reward_config = self.config.get('rewards', {})
            
            # Get current game state
            current_health = self.bridge.get_memory(0xC021) // 4  # Current health in hearts
            rupees = self.bridge.get_memory(0xC6A5)  # Current rupees
            keys = self.bridge.get_memory(0xC673)    # Current keys
            bombs = self.bridge.get_memory(0xC674)   # Current bombs
            
            # Get previous values for comparison
            prev_health = getattr(self, 'prev_strategic_health', current_health)
            prev_rupees = getattr(self, 'prev_strategic_rupees', rupees)
            prev_keys = getattr(self, 'prev_strategic_keys', keys)
            prev_bombs = getattr(self, 'prev_strategic_bombs', bombs)
            
            # 🗡️ COMBAT REWARDS - Massive bonus for successful combat
            if current_health > prev_health:
                # Health increased (found heart/heart container)
                health_bonus = reward_config.get('health_gain_reward', 30.0)
                strategic_reward += health_bonus
                print(f"❤️ Health gained! +{health_bonus:.1f} (Combat/exploration success)")
                
            # 💰 ITEM COLLECTION REWARDS - HUGE bonuses for collecting items
            rupee_gain = rupees - prev_rupees
            if rupee_gain > 0:
                rupee_bonus = rupee_gain * reward_config.get('rupee_collection_multiplier', 2.0)
                strategic_reward += rupee_bonus
                print(f"💰 Rupees collected! +{rupee_gain} rupees = +{rupee_bonus:.1f} reward")
                
            key_gain = keys - prev_keys
            if key_gain > 0:
                key_bonus = key_gain * reward_config.get('key_collection_reward', 10.0)
                strategic_reward += key_bonus
                print(f"🗝️ Keys collected! +{key_gain} keys = +{key_bonus:.1f} reward")
                
            bomb_gain = bombs - prev_bombs
            if bomb_gain > 0:
                bomb_bonus = bomb_gain * reward_config.get('bomb_collection_reward', 8.0)
                strategic_reward += bomb_bonus
                print(f"💣 Bombs collected! +{bomb_gain} bombs = +{bomb_bonus:.1f} reward")
                
            # 🎯 ACTION-BASED REWARDS - Reward specific button sequences that indicate strategy
            # This teaches the neural network which actions lead to good outcomes
            
            # Reward A button usage (sword attacks, interactions) 
            a_button_reward = reward_config.get('combat_action_reward', 0.5)
            if hasattr(self, 'last_action') and self.last_action == 5:  # A button
                strategic_reward += a_button_reward
                
            # Reward B button usage (items, lifting)
            b_button_reward = reward_config.get('interaction_action_reward', 0.3)
            if hasattr(self, 'last_action') and self.last_action == 6:  # B button
                strategic_reward += b_button_reward
                
            # 📈 PATTERN REWARDS - Reward sequences that indicate strategic thinking
            # Track recent action history for pattern detection
            if not hasattr(self, 'recent_actions'):
                self.recent_actions = []
            if hasattr(self, 'last_action'):
                self.recent_actions.append(self.last_action)
                if len(self.recent_actions) > 10:  # Keep last 10 actions
                    self.recent_actions.pop(0)
                
            # Reward combat patterns (movement + attack combinations)
            combat_pattern_count = 0
            for i in range(len(self.recent_actions) - 1):
                if (self.recent_actions[i] in [1, 2, 3, 4] and  # Movement action
                    self.recent_actions[i + 1] == 5):             # Followed by A button
                    combat_pattern_count += 1
                    
            if combat_pattern_count > 0:
                pattern_bonus = combat_pattern_count * reward_config.get('combat_pattern_reward', 1.0)
                strategic_reward += pattern_bonus
                if combat_pattern_count >= 3:  # Significant combat activity
                    print(f"⚔️ Combat patterns detected! {combat_pattern_count} sequences = +{pattern_bonus:.1f}")
                    
            # 🌟 MILESTONE REWARDS - Big bonuses for reaching collection thresholds
            # These create clear learning objectives for the RL agent
            
            # Rupee milestones
            rupee_milestones = [10, 25, 50, 100, 200]
            for milestone in rupee_milestones:
                if rupees >= milestone and prev_rupees < milestone:
                    milestone_bonus = reward_config.get('rupee_milestone_reward', 25.0)
                    strategic_reward += milestone_bonus
                    print(f"🏆 Rupee milestone! {milestone} rupees reached = +{milestone_bonus:.1f}")
                    
            # Health milestones (full health recovery)
            if current_health >= 3 and prev_health < 3:  # Full health restored
                health_milestone = reward_config.get('full_health_reward', 20.0)
                strategic_reward += health_milestone
                print(f"❤️ Full health restored! +{health_milestone:.1f}")
            
            # Store current values for next comparison
            self.prev_strategic_health = current_health
            self.prev_strategic_rupees = rupees  
            self.prev_strategic_keys = keys
            self.prev_strategic_bombs = bombs
            
            # 🎮 STRATEGIC ENCOURAGEMENT - Small bonuses for good gameplay habits
            # Reward active play over idle behavior
            if len(self.recent_actions) >= 5:
                # Calculate action diversity (avoid button mashing)
                unique_actions = len(set(self.recent_actions[-5:]))
                if unique_actions >= 3:  # Used 3+ different actions recently
                    diversity_bonus = reward_config.get('action_diversity_reward', 0.5)
                    strategic_reward += diversity_bonus
                    
            return strategic_reward
            
        except Exception as e:
            # If strategic reward calculation fails, return 0
            return 0.0
    
    def update_llm_suggestion(self, suggestion: Dict[str, Any]) -> None:
        """Update the latest LLM suggestion with MASSIVE EMPHASIS."""
        self.last_llm_suggestion = suggestion
        self.steps_since_llm_call = 0  # Reset counter
        print(f"🧠 NEW LLM GUIDANCE: {suggestion.get('action', 'Unknown')} -> {suggestion.get('target', 'Unknown')}")
        print(f"🧠 LLM REASONING: {suggestion.get('reasoning', 'No reasoning provided')}")
    
    def get_llm_alignment_stats(self) -> Dict[str, float]:
        """Get statistics on how well the RL agent is following LLM guidance."""
        if self.total_actions == 0:
            return {"alignment_rate": 0.0, "total_actions": 0, "llm_aligned": 0}
            
        alignment_rate = self.llm_aligned_actions / self.total_actions
        return {
            "alignment_rate": alignment_rate,
            "total_actions": self.total_actions,
            "llm_aligned": self.llm_aligned_actions,
            "steps_since_llm": self.steps_since_llm_call
        }

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Check death counter (most reliable - game increments this before reset)
        if hasattr(self, 'initial_death_count'):
            current_death_count = self.bridge.get_memory(0xC61E) + (self.bridge.get_memory(0xC61F) << 8)
            if current_death_count > self.initial_death_count:
                print(f"💀 Link died! Death count: {self.initial_death_count} → {current_death_count} (Episode terminated)")
                return True
        
        # Fallback: Check for health <= 0 (in case death counter isn't available yet)
        if self.current_structured_state:
            player = self.current_structured_state.get('player', {})
            if player.get('health', 3) <= 0:
                print(f"💔 Link's health reached 0! (Episode terminated)")
                return True
                
        return False

    def _check_truncated(self) -> bool:
        """Check if episode should be truncated."""
        return self.step_count >= self.max_steps

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        info = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'llm_mode_enabled': self.use_llm,
            'structured_states_enabled': self.enable_structured_states
        }
        
        # Add structured state for LLM mode
        if self.enable_structured_states and self.current_structured_state:
            info['structured_state'] = self.current_structured_state
            
        # Performance metrics
        if self.step_count > 0:
            info['avg_step_time'] = self.total_step_time / self.step_count
            if self.enable_structured_states:
                info['avg_structured_state_time'] = self.structured_state_generation_time / self.step_count
                
        return info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self.bridge.get_screen_array()
        return None

    def close(self):
        """Close the environment."""
        if hasattr(self, 'bridge'):
            self.bridge.close()

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            'llm_mode': self.use_llm,
            'structured_states': self.enable_structured_states,
            'observation_type': self.observation_type,
            'frame_skip': self.frame_skip,
            'max_steps': self.max_steps,
            'normalize_obs': self.normalize_observations,
            'auto_load_save': getattr(self.bridge, 'auto_load_save_state', False),
            'visual_test_mode': self.visual_test_mode,
            'headless': self.headless
        }


# Factory functions for easy creation
def create_pure_rl_env(rom_path: str, headless: bool = True, visual_test_mode: bool = False) -> ZeldaConfigurableEnvironment:
    """Create environment configured for pure RL training."""
    config = {
        'environment': {
            'frame_skip': 4,
            'observation_type': 'vector',
            'normalize_observations': True,
            'max_episode_steps': 10000
        },
        'planner_integration': {
            'use_planner': False,
            'enable_visual': False,
            'use_structured_entities': False,
            'auto_load_save_state': True
        },
        'rewards': {
            'time_penalty': -0.0001,
            'movement_reward': 0.001,
            'death_penalty': -3.0,
            # EXPLORATION BONUSES - HUGE rewards for meaningful gameplay!
            'room_discovery_reward': 10.0,      # Big bonus for new areas
            'dungeon_discovery_reward': 25.0,   # Massive bonus for dungeon entry
            'dungeon_bonus': 5.0,               # Continuous bonus while in dungeon
            'npc_interaction_reward': 15.0,     # Big bonus for talking to NPCs
            
            # LLM GUIDANCE REWARDS - DISABLED for pure RL mode
            'llm_guidance_multiplier': 0.0,     # No LLM guidance in pure RL
            'llm_strategic_bonus': 0.0,         # No strategic bonus
            'llm_directional_bonus': 0.0,       # No directional bonus  
            'llm_completion_bonus': 0.0         # No completion bonus
        }
    }
    
    return ZeldaConfigurableEnvironment(
        rom_path=rom_path,
        config_dict=config,
        headless=headless,
        visual_test_mode=visual_test_mode
    )


def create_llm_guided_env(rom_path: str, headless: bool = True, visual_test_mode: bool = False) -> ZeldaConfigurableEnvironment:
    """Create environment configured for LLM-guided training."""
    config = {
        'environment': {
            'frame_skip': 4,
            'observation_type': 'vector',
            'normalize_observations': True,
            'max_episode_steps': 10000
        },
        'planner_integration': {
            'use_planner': True,
            'enable_visual': True,
            'use_structured_entities': True,
            'compression_mode': 'bit_packed',
            'auto_load_save_state': True,
            'planner_frequency': 100
        },
        'rewards': {
            'time_penalty': -0.0001,
            'movement_reward': 0.001,
            'death_penalty': -3.0,
            'key_reward': 0.5,
            'rupee_reward': 0.01,
            # EXPLORATION BONUSES - HUGE rewards for meaningful gameplay!
            'room_discovery_reward': 10.0,      # Big bonus for new areas
            'dungeon_discovery_reward': 25.0,   # Massive bonus for dungeon entry
            'dungeon_bonus': 5.0,               # Continuous bonus while in dungeon
            'npc_interaction_reward': 15.0,     # Big bonus for talking to NPCs
            
            # LLM GUIDANCE REWARDS - MASSIVE EMPHASIS FOR LLM-GUIDED MODE!
            'llm_guidance_multiplier': 5.0,     # Multiply normal rewards by 5x when following LLM
            'llm_strategic_bonus': 2.0,         # Continuous bonus for following LLM direction
            'llm_directional_bonus': 1.0,       # Bonus for moving in LLM-suggested direction
            'llm_completion_bonus': 50.0        # HUGE bonus for completing LLM macro goals
        }
    }
    
    return ZeldaConfigurableEnvironment(
        rom_path=rom_path,
        config_dict=config,
        headless=headless,
        visual_test_mode=visual_test_mode
    )


# Visual test mode factory functions
def create_visual_test_pure_rl_env(rom_path: str) -> ZeldaConfigurableEnvironment:
    """Create environment for visual testing of pure RL mode."""
    return create_pure_rl_env(rom_path, headless=False, visual_test_mode=True)


def create_visual_test_llm_env(rom_path: str) -> ZeldaConfigurableEnvironment:
    """Create environment for visual testing of LLM-guided mode."""
    return create_llm_guided_env(rom_path, headless=False, visual_test_mode=True)

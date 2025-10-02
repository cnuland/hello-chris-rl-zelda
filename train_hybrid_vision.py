"""Hybrid RL+LLM Training with VISION support for Zelda Oracle of Seasons.

Combines:
- Vector-based PPO (fast, proven to work)
- Vision LLM guidance (sees actual Game Boy screen)
- Exploration rewards

The agent uses vector observations for fast learning,
while the LLM gets actual screenshots for strategic guidance.
"""

import os
import sys
import time
import argparse
import base64
import io
import yaml
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image
import torch
import requests

from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment
from agents.controller import ZeldaController, ControllerConfig
from observation.ram_maps.room_mappings import OVERWORLD_ROOMS

# Import HUD server
sys.path.append(os.path.join(os.path.dirname(__file__), 'HUD'))
try:
    from hud_server import start_server_thread, update_training_data, update_vision_data, register_session
    HUD_AVAILABLE = True
except ImportError:
    print("⚠️  HUD server not available (missing Flask or dependencies)")
    HUD_AVAILABLE = False
    register_session = None


class VisionHybridTrainer:
    """Hybrid trainer: Vector PPO + Vision LLM guidance."""
    
    def __init__(
        self,
        rom_path: str,
        headless: bool = True,
        llm_endpoint: str = "http://localhost:8000/v1/chat/completions",
        llm_frequency: int = None,  # Load from config if not specified
        llm_guidance_bonus: float = None,
        enable_vision: bool = True,  # Enable vision mode
        image_scale: int = None,
        image_quality: int = None,
        config_path: str = "configs/vision_prompt.yaml"
    ):
        self.rom_path = rom_path
        self.headless = headless
        self.llm_endpoint = llm_endpoint
        self.enable_vision = enable_vision
        
        # Load configuration from YAML
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Override with config values if not provided
        self.image_scale = image_scale or self.config['vision_config']['image_scale']
        self.image_quality = image_quality or self.config['vision_config']['image_quality']
        self.llm_frequency = llm_frequency or self.config['behavior']['call_frequency']
        self.llm_guidance_bonus = llm_guidance_bonus or self.config['behavior']['alignment_bonus_multiplier']
        
        # Store prompt templates
        self.system_prompt = self.config['system_prompt']
        self.vision_prompt_template = self.config['vision_user_prompt_template']
        self.text_prompt_template = self.config['text_fallback_prompt_template']
        
        # Store model config
        self.max_tokens = self.config['model_config']['max_tokens']
        self.temperature = self.config['model_config']['temperature']
        
        # Initialize environment - VECTOR observations for PPO
        env_config = {
            "environment": {
                "max_episode_steps": 8000,  # Longer episodes to find Wooden Sword
                "frame_skip": 4,
                "observation_type": "vector",  # Fast vector obs for PPO
                "normalize_observations": True
            },
            "planner_integration": {
                "use_planner": True,
                "enable_structured_states": True
            },
            "rewards": {
                "step_penalty": -0.01,
                "health_bonus": 10.0,
                "death_penalty": -100.0
            }
        }
        
        self.env = ZeldaConfigurableEnvironment(
            rom_path=rom_path,
            config_dict=env_config,
            headless=headless
        )
        
        # PPO controller
        controller_config = ControllerConfig(
            learning_rate=0.0003,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coeff=0.01,
            max_grad_norm=0.5
        )
        
        self.controller = ZeldaController(
            env=self.env,
            config=controller_config,
            use_mock_planner=True  # Don't use LLM planner in controller (we handle LLM manually)
        )
        
        # Tracking
        self.global_step = 0
        self.episode_count = 0
        self.episode_step = 0  # Current episode step count
        self.episode_rewards = []  # Track rewards within current episode
        self.current_epoch = 1  # Track current epoch/rollout number
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.llm_alignment_count = 0
        
        # Exploration tracking
        self.visited_rooms = set()
        self.room_discovery_count = 0
        self.position_history = []
        self.area_visit_times = {}
        self.last_position = None
        self.last_room = None
        self.stationary_steps = 0
        self.decay_window = 500
        self.exploration_bonus_multiplier = 0.4  # Reduced from 5.0 to prevent farming (5.0 * 0.4 = 2.0 per new cell)
        self.grid_size = 8
        self.penalty_warmup_steps = 1000
        
        # NPC tracking
        self.a_button_near_npc_count = 0
        self.npc_bonus_rewards = 0
        
        # Game progression tracking
        self.items_collected = 0
        self.buildings_entered = 0
        self.maku_tree_entered = False
        self.maku_tree_talked = False  # Track if talked to Maku Tree
        self.dungeon_entered = False  # Track if entered Gnarled Root Dungeon
        self.vines_cut_count = 0  # Track B button usage (cutting obstacles)
        
        # Start HUD server if HUD is available (works in headless too!)
        self.hud_enabled = False
        self.hud_session_id = None
        if HUD_AVAILABLE:
            try:
                print(f"🎮 Starting VLM Vision Hybrid HUD Server...")
                start_server_thread(host='0.0.0.0', port=8086)
                
                # Register this training session with the HUD
                self.hud_session_id = register_session()
                
                if self.hud_session_id:
                    self.hud_enabled = True
                    print(f"   ✅ HUD available at http://localhost:8086")
                    print(f"   🔒 Session ID: {self.hud_session_id[:8]}...")
                else:
                    print(f"   ⚠️  HUD is connected to another training session")
                    print(f"   This session will run without HUD updates")
                    
            except Exception as e:
                print(f"   ⚠️  Could not start HUD server: {e}")
        
        print(f"🎮 Vision Hybrid Trainer initialized")
        print(f"   Vector observations for PPO: {self.env.observation_space.shape}")
        print(f"   Vision mode for LLM: {'✅ ENABLED' if self.enable_vision else '❌ DISABLED'}")
        if self.enable_vision:
            print(f"   Image settings: {160*self.image_scale}×{144*self.image_scale}, {self.image_quality}% JPEG")
        print(f"   LLM call frequency: every {self.llm_frequency} steps")
        print(f"   HUD Dashboard: {'✅ ENABLED' if self.hud_enabled else '❌ DISABLED'}")
    
    def capture_screenshot_base64(self) -> Optional[str]:
        """Capture Game Boy screenshot and encode as base64.
        
        Returns:
            Base64-encoded JPEG string, or None if failed
        """
        if not self.enable_vision:
            return None
        
        try:
            # Get RGB screen from PyBoy bridge
            screen_array = self.env.bridge.get_screen()  # (144, 160, 3)
            
            # Convert to PIL Image
            img = Image.fromarray(screen_array)
            
            # Upscale (keep pixelated look with NEAREST)
            scaled_img = img.resize(
                (160 * self.image_scale, 144 * self.image_scale),
                Image.Resampling.NEAREST
            )
            
            # Compress to JPEG
            buffer = io.BytesIO()
            scaled_img.save(buffer, format="JPEG", quality=self.image_quality)
            jpeg_bytes = buffer.getvalue()
            
            # Base64 encode
            base64_img = base64.b64encode(jpeg_bytes).decode('utf-8')
            
            return base64_img
            
        except Exception as e:
            print(f"⚠️  Screenshot capture failed: {e}")
            return None
    
    def call_llm_vision(self, game_state: Dict, screenshot_base64: Optional[str] = None) -> Optional[str]:
        """Call vision LLM for strategic guidance.
        
        Args:
            game_state: Current structured game state
            screenshot_base64: Base64-encoded screenshot (optional)
            
        Returns:
            Suggested button string, or None if failed
        """
        try:
            player = game_state.get('player', {})
            entities = game_state.get('entities', {})
            
            # Extract game state
            health = player.get('health', 3)
            max_health = player.get('max_health', 3)
            room_id = player.get('room', 0)
            x = player.get('x', 0)
            y = player.get('y', 0)
            
            location = OVERWORLD_ROOMS.get(room_id, f"Unknown Room {room_id}")
            
            npcs = entities.get('npcs', [])
            enemies = entities.get('enemies', [])
            items = entities.get('items', [])
            
            # Build prompt using templates from config
            if screenshot_base64:
                # Add cave hint if at Hero's Cave
                cave_hint = ""
                if room_id == 184:
                    cave_hint = "⚠️ [HERO'S CAVE ENTRANCE - PRESS UP OR A TO ENTER!]"
                elif "Cave" in location or "cave" in location.lower():
                    cave_hint = "⚠️ [CAVE ENTRANCE DETECTED - ENTER NOW!]"
                
                # Vision mode - use vision prompt template
                prompt = self.vision_prompt_template.format(
                    location=location,
                    cave_hint=cave_hint,
                    health=health,
                    max_health=max_health,
                    x=x,
                    y=y,
                    npc_count=len(npcs),
                    enemy_count=len(enemies),
                    item_count=len(items)
                )
            else:
                # Text-only fallback - use text prompt template
                # Build entity info
                entity_lines = []
                if npcs:
                    entity_lines.append(f"- NPCs: {len(npcs)} nearby")
                if enemies:
                    entity_lines.append(f"- Enemies: {len(enemies)} present")
                if items:
                    entity_lines.append(f"- Items: {len(items)} visible")
                entity_info = "\n".join(entity_lines) if entity_lines else "- No entities detected"
                
                prompt = self.text_prompt_template.format(
                    location=location,
                    health=health,
                    max_health=max_health,
                    x=x,
                    y=y,
                    entity_info=entity_info
                )
            
            # Build API request with system prompt
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Add screenshot if available (to user message)
            if screenshot_base64:
                messages[1]["images"] = [f"data:image/jpeg;base64,{screenshot_base64}"]
            
            # Call LLM with config parameters
            response = requests.post(
                self.llm_endpoint,
                json={
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                suggestion = result['choices'][0]['message']['content'].strip().upper()
                
                # Extract button
                valid_buttons = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'NOP']
                for button in valid_buttons:
                    if button in suggestion:
                        return button
                
                return 'NOP'
            else:
                return None
                
        except Exception as e:
            print(f"⚠️  LLM call failed: {e}")
            return None
    
    def update_hud(
        self,
        game_state: Dict,
        llm_suggestion: Optional[str] = None,
        screenshot_base64: Optional[str] = None,
        llm_response_time: Optional[float] = None
    ):
        """
        Push updates to the HUD dashboard
        
        Args:
            game_state: Current game state dictionary
            llm_suggestion: Latest LLM button suggestion
            screenshot_base64: Latest vision screenshot
            llm_response_time: LLM response time in milliseconds
        """
        if not self.hud_enabled:
            return
        
        # Debug: confirm HUD update is being called
        if self.global_step % 50 == 0:
            print(f"🎮 HUD update at step {self.global_step}")
        
        try:
            # Prepare training data
            player = game_state.get('player', {})
            entities = game_state.get('entities', {})
            
            training_data = {
                'epoch': self.current_epoch,
                'episode': self.episode_count,
                'episode_id': f"E{self.current_epoch:03d}-{self.episode_count:04d}",
                'global_step': self.global_step,
                'episode_reward': sum(self.episode_rewards) if self.episode_rewards else 0,
                'episode_length': self.episode_step,
                'location': player.get('location', 'Unknown'),
                'room_id': player.get('room', 0),
                'position': {
                    'x': player.get('x', 0),
                    'y': player.get('y', 0)
                },
                'health': {
                    'current': player.get('health', 3),
                    'max': player.get('max_health', 3)
                },
                'entities': {
                    'npcs': len(entities.get('npcs', [])),
                    'enemies': len(entities.get('enemies', [])),
                    'items': len(entities.get('items', []))
                },
                'llm_calls': self.llm_call_count,
                'llm_success_rate': (self.llm_success_count / self.llm_call_count * 100) if self.llm_call_count > 0 else 100,
                'llm_alignment': self.llm_alignment_count,
                'milestones': {
                    'maku_tree_entered': self.maku_tree_entered,
                    'dungeon_entered': self.dungeon_entered,
                    'sword_usage': self.vines_cut_count
                },
                'exploration': {
                    'rooms_discovered': len(self.visited_rooms),
                    'grid_areas': len(self.area_visit_times),
                    'buildings_entered': self.buildings_entered
                }
            }
            
            # Add LLM suggestion if available
            if llm_suggestion:
                training_data['llm_suggestion'] = llm_suggestion
            
            # Update training data (pass session_id for validation)
            if not update_training_data(training_data, session_id=self.hud_session_id):
                # This session is no longer active, disable HUD
                self.hud_enabled = False
                print(f"   ⚠️  Lost HUD connection - another session took over")
                return
            
            # Update vision data if available
            if screenshot_base64:
                if not update_vision_data(screenshot_base64, llm_response_time, session_id=self.hud_session_id):
                    # This session is no longer active, disable HUD
                    self.hud_enabled = False
                    print(f"   ⚠️  Lost HUD connection - another session took over")
                    return
                
        except Exception as e:
            # Don't crash training if HUD update fails
            print(f"⚠️  HUD update failed at step {self.global_step}: {e}")
    
    def compute_llm_alignment_bonus(
        self,
        ppo_action: int,
        llm_button: str,
        game_state: Dict
    ) -> float:
        """Compute bonus reward when PPO follows LLM guidance using config multipliers."""
        button_map = {
            0: 'NOP', 1: 'UP', 2: 'DOWN', 3: 'LEFT',
            4: 'RIGHT', 5: 'A', 6: 'B', 7: 'START', 8: 'SELECT'
        }
        
        ppo_button = button_map.get(ppo_action, 'NOP')
        
        if ppo_button == llm_button:
            # Base bonus
            bonus = self.llm_guidance_bonus
            
            # Get context-aware multipliers from config
            building_entry_mult = self.config['behavior'].get('building_entry_bonus', 12.0)
            maku_tree_mult = self.config['behavior'].get('maku_tree_bonus', 20.0)
            dungeon_entry_mult = self.config['behavior'].get('dungeon_entry_bonus', 15.0)
            npc_talk_mult = self.config['behavior'].get('npc_interaction_bonus', 3.0)
            vine_cut_mult = self.config['behavior'].get('vine_cutting_bonus', 8.0)
            item_collect_mult = self.config['behavior'].get('item_collection_bonus', 15.0)
            exploration_multiplier = self.config['behavior'].get('exploration_bonus', 2.0)
            retreat_multiplier = self.config['behavior'].get('low_health_retreat_bonus', 2.5)
            
            entities = game_state.get('entities', {})
            npcs = entities.get('npcs', [])
            enemies = entities.get('enemies', [])
            items = entities.get('items', [])
            player = game_state.get('player', {})
            health = player.get('health', 3)
            room_id = player.get('room', 0)
            
            # PRIORITY 1: UP button at Maku Tree entrance (CRITICAL!)
            if ppo_button == 'UP':
                if room_id == 0xD9 or room_id == 217:  # Maku Tree entrance
                    bonus *= maku_tree_mult
                    print(f"   🌳 ENTERING MAKU TREE! Bonus: +{bonus:.1f}")
                elif room_id == 0x28 or room_id == 40:  # Gnarled Root Dungeon
                    bonus *= dungeon_entry_mult
                    print(f"   🏰 ENTERING DUNGEON! Bonus: +{bonus:.1f}")
                else:
                    # UP at any doorway/building
                    bonus *= building_entry_mult
                    print(f"   🚪 ENTERING BUILDING! Bonus: +{bonus:.1f}")
            
            # PRIORITY 2: B button (cutting vines/bushes with sword - important!)
            elif ppo_button == 'B':
                bonus *= vine_cut_mult
                self.vines_cut_count += 1
                print(f"   ✂️ USING SWORD (cutting obstacles)! Bonus: +{bonus:.1f}")
            
            # PRIORITY 3: A button near NPCs (TALKING - especially Maku Tree!)
            elif ppo_button == 'A' and len(npcs) > 0:
                if room_id == 0xD9 or room_id == 217:  # Inside Maku Tree
                    bonus *= maku_tree_mult
                    self.a_button_near_npc_count += 1
                    self.npc_bonus_rewards += bonus
                    print(f"   💬 TALKING TO MAKU TREE! Bonus: +{bonus:.1f}")
                else:
                    bonus *= npc_talk_mult
                    self.a_button_near_npc_count += 1
                    self.npc_bonus_rewards += bonus
            
            # PRIORITY 4: A button near items
            elif ppo_button == 'A' and len(items) > 0:
                bonus *= item_collect_mult
                print(f"   📦 COLLECTING ITEM! Bonus: +{bonus:.1f}")
            
            # Movement bonuses
            elif ppo_button in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                # Even more bonus if retreating with low health
                if health <= 1 and len(enemies) > 0:
                    bonus *= retreat_multiplier
                else:
                    bonus *= exploration_multiplier
            
            return bonus
        
        return 0.0
    
    def compute_exploration_reward(self, game_state: Dict, current_step: int) -> float:
        """Compute exploration rewards with time-based decay."""
        player = game_state.get('player', {})
        x = player.get('x', 0)
        y = player.get('y', 0)
        room = player.get('room', 0)
        
        current_pos = (x, y, room)
        grid_x = x // self.grid_size
        grid_y = y // self.grid_size
        grid_cell = (room, grid_x, grid_y)
        
        reward = 0.0
        warmup_active = current_step < self.penalty_warmup_steps
        
        # Stationary penalty (after warmup)
        if not warmup_active:
            if self.last_position is not None and self.last_position == current_pos:
                self.stationary_steps += 1
                reward -= min(self.stationary_steps * 0.2, 2.0)
            else:
                self.stationary_steps = 0
        
        # Area revisit with decay
        if grid_cell in self.area_visit_times:
            last_visit_time = self.area_visit_times[grid_cell][-1]
            time_since_visit = current_step - last_visit_time
            
            if not warmup_active and time_since_visit < self.decay_window:
                decay_factor = 1.0 - (time_since_visit / self.decay_window)
                loiter_penalty = -0.8 * decay_factor
                reward += loiter_penalty
            elif time_since_visit >= self.decay_window:
                reward += 0.5  # Backtracking bonus
            
            self.area_visit_times[grid_cell].append(current_step)
        else:
            # NEW AREA!
            reward += 5.0 * self.exploration_bonus_multiplier  # +25.0
            self.area_visit_times[grid_cell] = [current_step]
            print(f"   🌟 NEW AREA! Grid cell {grid_cell} | Bonus: +{5.0 * self.exploration_bonus_multiplier}")
        
        self.position_history.append((x, y, room, current_step))
        self.last_position = current_pos
        return reward
    
    def train(self, total_timesteps: int = 100000):
        """Run simple testing loop (not full PPO training)."""
        print(f"\n🚀 Starting Vision Hybrid Test")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Vision LLM calls every {self.llm_frequency} steps")
        print("="*60)
        
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        last_llm_suggestion = None
        
        while self.global_step < total_timesteps:
            # Get action from PPO policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.controller.device)
            
            with torch.no_grad():
                action, log_prob, value = self.controller.policy_net.get_action_and_value(obs_tensor)
            
            action_int = action.item()
            
            # Execute action
            next_obs, reward, terminated, truncated, info = self.env.step(action_int)
            done = terminated or truncated
            
            # Get game state
            game_state = info.get('structured_state', {})
            
            # Track rooms and add environment rewards for entering new rooms
            current_room = game_state.get('player', {}).get('room', None)
            room_entry_reward = 0.0
            
            # Detect room transition
            if current_room is not None and self.last_room is not None and current_room != self.last_room:
                # Room transition detected
                self.buildings_entered += 1
                
                # Only reward if this is a TRULY NEW room (anti-farming)
                if current_room not in self.visited_rooms:
                    room_entry_reward = self.config['behavior'].get('new_room_entry_reward', 10.0)
                    print(f"   🆕 FIRST TIME in room {current_room}! Bonus: +{room_entry_reward:.1f}")
                # No reward for revisiting rooms (prevents back-and-forth farming)
                
                # Check if entering Maku Tree (Room ID 0xD9 = 217)
                if current_room == 0xD9 or current_room == 217:
                    if not self.maku_tree_entered:
                        room_entry_reward += self.config['behavior'].get('maku_tree_entry_reward', 300.0)
                        self.maku_tree_entered = True
                        print(f"\n   🌳🌳🌳 ENTERED MAKU TREE! MASSIVE REWARD: +{room_entry_reward:.1f} 🌳🌳🌳\n")
                    else:
                        print(f"\n   🌳 Re-entering Maku Tree | Reward: +{room_entry_reward:.1f}\n")
                
                # Check if entering Gnarled Root Dungeon (Room ID 0x28 = 40)
                elif current_room == 0x28 or current_room == 40:
                    if not self.dungeon_entered:
                        room_entry_reward += self.config['behavior'].get('dungeon_entry_reward', 400.0)
                        self.dungeon_entered = True
                        print(f"\n   🏰🏰🏰 ENTERED GNARLED ROOT DUNGEON! HUGE REWARD: +{room_entry_reward:.1f} 🏰🏰🏰\n")
                    else:
                        print(f"\n   🏰 Re-entering dungeon | Reward: +{room_entry_reward:.1f}\n")
                
                else:
                    print(f"\n   🚪 ENTERED NEW ROOM! Reward: +{room_entry_reward:.1f}\n")
            
            # Track first discovery of rooms
            if current_room is not None and current_room not in self.visited_rooms:
                self.visited_rooms.add(current_room)
                self.room_discovery_count += 1
                room_name = OVERWORLD_ROOMS.get(current_room, f"Unknown Room {current_room}")
                player_x = game_state.get('player', {}).get('x', 0)
                player_y = game_state.get('player', {}).get('y', 0)
                print(f"   🗺️  NEW LOCATION DISCOVERED: {room_name} (ID: {current_room})")
                print(f"   Coordinates: ({player_x}, {player_y})")
                print(f"   Total locations: {len(self.visited_rooms)}\n")
            
            self.last_room = current_room
            
            # Get LLM guidance (with vision)
            llm_bonus = 0.0
            if self.global_step % self.llm_frequency == 0:
                # Capture screenshot if vision enabled
                screenshot = None
                llm_start_time = time.time()
                if self.enable_vision:
                    screenshot = self.capture_screenshot_base64()
                
                # Call vision LLM
                llm_suggestion = self.call_llm_vision(game_state, screenshot)
                llm_response_time = (time.time() - llm_start_time) * 1000  # Convert to ms
                self.llm_call_count += 1
                
                if llm_suggestion:
                    self.llm_success_count += 1
                    last_llm_suggestion = llm_suggestion
                    
                    vision_tag = "👁️ VISION" if screenshot else "📝 TEXT"
                    print(f"   {vision_tag} LLM suggests: {llm_suggestion}")
                    
                    # Update HUD with LLM data and vision
                    self.update_hud(game_state, llm_suggestion, screenshot, llm_response_time)
            
            # Compute bonuses if we have a suggestion
            if last_llm_suggestion:
                llm_bonus = self.compute_llm_alignment_bonus(action_int, last_llm_suggestion, game_state)
                if llm_bonus > 0:
                    self.llm_alignment_count += 1
                    button_map = {0: 'NOP', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'A', 6: 'B', 7: 'START', 8: 'SELECT'}
                    action_name = button_map.get(action_int, 'UNKNOWN')
                    print(f"   ✅ PPO followed LLM! Button: {action_name} | Bonus: +{llm_bonus:.1f}")
            
            # Exploration reward
            exploration_reward = self.compute_exploration_reward(game_state, self.global_step)
            
            if exploration_reward > 10.0:
                print(f"   🌟 NEW AREA! Bonus: +{exploration_reward:.1f}")
            
            # Total reward = environment + LLM bonus + exploration + room entry
            total_reward = reward + llm_bonus + exploration_reward + room_entry_reward
            
            episode_reward += total_reward
            episode_length += 1
            self.global_step += 1
            
            # Track for HUD
            self.episode_rewards.append(total_reward)
            self.episode_step = episode_length
            
            # Update HUD periodically (every 5 steps) even without LLM calls
            if self.hud_enabled and self.global_step % 5 == 0:
                self.update_hud(game_state, last_llm_suggestion)
            
            # Update for next step
            obs = next_obs
            
            if done:
                print(f"\n📊 Episode {self.episode_count} complete:")
                print(f"   Reward: {episode_reward:.1f}")
                print(f"   Length: {episode_length}")
                
                obs, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
                self.episode_count += 1
                
                # Reset episode tracking for HUD
                self.episode_rewards = []
                self.episode_step = 0
            
            # Progress update
            if self.global_step % 500 == 0:
                print(f"\n📊 Step {self.global_step}/{total_timesteps}")
                print(f"   Episodes: {self.episode_count}")
                print(f"   LLM Success: {self.llm_success_count}/{self.llm_call_count}")
                print(f"   Alignment: {self.llm_alignment_count}")
                print(f"   Rooms: {len(self.visited_rooms)}")
                print(f"   Grid areas: {len(self.area_visit_times)}")
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        print(f"   Total episodes: {self.episode_count}")
        print(f"   Rooms discovered: {len(self.visited_rooms)}")
        print(f"   Grid areas explored: {len(self.area_visit_times)}")
        print(f"   Buildings/rooms entered: {self.buildings_entered}")
        print(f"\n   🎯 PROGRESSION MILESTONES:")
        print(f"   🌳 Maku Tree entered: {'YES! ✅' if self.maku_tree_entered else 'Not yet ❌'}")
        print(f"   💬 Maku Tree conversation: {'YES! ✅' if self.maku_tree_talked else 'Not yet ❌'}")
        print(f"   🏰 Gnarled Root Dungeon entered: {'YES! ✅' if self.dungeon_entered else 'Not yet ❌'}")
        print(f"\n   ✂️ Sword usage (B button): {self.vines_cut_count}")
        print(f"   Items collected: {self.items_collected}")
        print(f"   NPC interactions: {self.a_button_near_npc_count}")
        print(f"\n   🤖 LLM GUIDANCE:")
        print(f"   LLM calls: {self.llm_call_count} ({self.llm_success_count} successful)")
        print(f"   Alignment rate: {self.llm_alignment_count}/{self.llm_success_count if self.llm_success_count > 0 else 1}")
        print("="*60)
        
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Vision Hybrid RL+LLM Training")
    parser.add_argument("--rom-path", type=str, required=True, help="Path to Zelda ROM file")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--config", type=str, default="configs/vision_prompt.yaml", help="Path to vision prompt config")
    parser.add_argument("--llm-frequency", type=int, default=None, help="LLM call frequency (overrides config)")
    parser.add_argument("--llm-bonus", type=float, default=None, help="LLM guidance bonus (overrides config)")
    parser.add_argument("--enable-vision", action="store_true", help="Enable vision mode")
    parser.add_argument("--image-scale", type=int, default=None, help="Image upscale factor (overrides config)")
    parser.add_argument("--image-quality", type=int, default=None, help="JPEG quality 0-100 (overrides config)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to load")
    
    args = parser.parse_args()
    
    trainer = VisionHybridTrainer(
        rom_path=args.rom_path,
        headless=args.headless,
        llm_frequency=args.llm_frequency,
        llm_guidance_bonus=args.llm_bonus,
        enable_vision=args.enable_vision,
        image_scale=args.image_scale,
        image_quality=args.image_quality,
        config_path=args.config
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"📂 Loading checkpoint: {args.checkpoint}")
        trainer.controller.load_checkpoint(args.checkpoint)
        print("✅ Checkpoint loaded successfully!")
    
    trainer.train(total_timesteps=args.total_timesteps)


if __name__ == "__main__":
    main()

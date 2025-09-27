#!/usr/bin/env python3
"""
Enhanced PPO Controller with Smart LLM Policy Arbitration

This implementation provides context-aware LLM guidance that adapts
to game situations and optimizes for training efficiency.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import time
from enum import Enum

from .controller import PPOController, ControllerConfig
from .planner import ZeldaPlanner
from .macro_actions import MacroExecutor, MacroAction, MacroType
from ..emulator.input_map import ZeldaAction


class ArbitrationTrigger(Enum):
    """When LLM should be consulted for policy guidance."""
    TIME_INTERVAL = "time_interval"          # Regular time-based
    NEW_ROOM = "new_room"                    # Entered new area
    LOW_HEALTH = "low_health"                # Health critical
    STUCK_DETECTION = "stuck_detection"      # No progress detected
    ITEM_DISCOVERED = "item_discovered"      # Found new item
    NPC_INTERACTION = "npc_interaction"      # Talking to NPC
    DUNGEON_ENTRANCE = "dungeon_entrance"    # Found dungeon
    BOSS_ENCOUNTER = "boss_encounter"        # Boss fight
    PUZZLE_ENCOUNTER = "puzzle_encounter"    # Puzzle detected
    FORCED_CONSULTATION = "forced"           # Manual trigger


@dataclass
class EnhancedControllerConfig(ControllerConfig):
    """Enhanced configuration with smart arbitration parameters."""
    
    # Base arbitration settings
    base_planner_frequency: int = 200        # Base steps between LLM calls
    min_planner_frequency: int = 50          # Minimum steps between calls
    max_planner_frequency: int = 500         # Maximum steps between calls
    
    # Adaptive frequency modifiers
    exploration_bonus_multiplier: float = 0.5    # Reduce frequency when exploring well
    stuck_penalty_multiplier: float = 2.0        # Increase frequency when stuck
    
    # Context-aware triggers
    trigger_on_new_room: bool = True
    trigger_on_low_health: bool = True
    trigger_on_stuck: bool = True
    trigger_on_npc_interaction: bool = True
    trigger_on_dungeon_entrance: bool = True
    
    # Performance thresholds
    stuck_threshold: int = 100               # Steps without progress = stuck
    low_health_threshold: float = 0.25       # Health % to trigger LLM
    exploration_rate_threshold: float = 0.1  # New areas per episode
    
    # Macro execution limits
    macro_timeout: int = 75                  # Reduced from 200
    max_concurrent_macros: int = 1           # Only one macro at a time
    
    # Performance tracking
    track_arbitration_performance: bool = True
    arbitration_success_window: int = 100    # Steps to measure success


class SmartArbitrationTracker:
    """Tracks LLM arbitration performance and adapts frequency."""
    
    def __init__(self, config: EnhancedControllerConfig):
        self.config = config
        self.reset_tracking()
        
    def reset_tracking(self):
        """Reset all tracking metrics."""
        self.last_llm_call = 0
        self.last_reward = 0.0
        self.reward_before_llm = 0.0
        self.reward_after_llm = 0.0
        self.successful_arbitrations = 0
        self.total_arbitrations = 0
        self.stuck_counter = 0
        self.last_position = (0, 0)
        self.last_room = 0
        self.rooms_discovered_this_episode = set()
        
    def should_call_llm(self, step_count: int, game_state: Dict[str, Any]) -> Tuple[bool, List[ArbitrationTrigger]]:
        """Determine if LLM should be called and why."""
        triggers = []
        
        # Time-based trigger (adaptive frequency)
        steps_since_last = step_count - self.last_llm_call
        current_frequency = self._calculate_adaptive_frequency()
        
        if steps_since_last >= current_frequency:
            triggers.append(ArbitrationTrigger.TIME_INTERVAL)
            
        # Context-aware triggers
        if self.config.trigger_on_new_room and self._detect_new_room(game_state):
            triggers.append(ArbitrationTrigger.NEW_ROOM)
            
        if self.config.trigger_on_low_health and self._detect_low_health(game_state):
            triggers.append(ArbitrationTrigger.LOW_HEALTH)
            
        if self.config.trigger_on_stuck and self._detect_stuck(game_state):
            triggers.append(ArbitrationTrigger.STUCK_DETECTION)
            
        if self.config.trigger_on_npc_interaction and self._detect_npc_interaction(game_state):
            triggers.append(ArbitrationTrigger.NPC_INTERACTION)
            
        if self.config.trigger_on_dungeon_entrance and self._detect_dungeon_entrance(game_state):
            triggers.append(ArbitrationTrigger.DUNGEON_ENTRANCE)
            
        # Ensure minimum frequency isn't violated
        if (steps_since_last >= self.config.max_planner_frequency and 
            ArbitrationTrigger.TIME_INTERVAL not in triggers):
            triggers.append(ArbitrationTrigger.FORCED_CONSULTATION)
            
        return len(triggers) > 0, triggers
    
    def _calculate_adaptive_frequency(self) -> int:
        """Calculate adaptive frequency based on recent performance."""
        base_freq = self.config.base_planner_frequency
        
        # Adjust based on arbitration success rate
        if self.total_arbitrations > 10:
            success_rate = self.successful_arbitrations / self.total_arbitrations
            if success_rate > 0.7:  # LLM is helping
                base_freq = int(base_freq * 1.2)  # Call less frequently
            elif success_rate < 0.3:  # LLM not helping much
                base_freq = int(base_freq * 0.8)  # Call more frequently
                
        # Adjust based on exploration performance
        exploration_rate = len(self.rooms_discovered_this_episode) / max(1, self.last_llm_call)
        if exploration_rate > self.config.exploration_rate_threshold:
            base_freq = int(base_freq * self.config.exploration_bonus_multiplier)
            
        # Adjust based on stuck detection
        if self.stuck_counter > self.config.stuck_threshold:
            base_freq = int(base_freq * self.config.stuck_penalty_multiplier)
            
        # Clamp to bounds
        return max(self.config.min_planner_frequency, 
                  min(self.config.max_planner_frequency, base_freq))
    
    def _detect_new_room(self, game_state: Dict[str, Any]) -> bool:
        """Detect if player entered a new room."""
        if 'player' not in game_state:
            return False
            
        current_room = game_state.get('player', {}).get('room', 0)
        if current_room != self.last_room and current_room not in self.rooms_discovered_this_episode:
            self.last_room = current_room
            self.rooms_discovered_this_episode.add(current_room)
            return True
        return False
    
    def _detect_low_health(self, game_state: Dict[str, Any]) -> bool:
        """Detect critically low health."""
        player = game_state.get('player', {})
        health = player.get('health', 3)
        max_health = player.get('max_health', 3)
        
        if max_health > 0:
            health_ratio = health / max_health
            return health_ratio <= self.config.low_health_threshold
        return False
    
    def _detect_stuck(self, game_state: Dict[str, Any]) -> bool:
        """Detect if player is stuck (not moving)."""
        player = game_state.get('player', {})
        current_pos = (player.get('x', 0), player.get('y', 0))
        
        if current_pos == self.last_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = current_pos
            
        return self.stuck_counter >= self.config.stuck_threshold
    
    def _detect_npc_interaction(self, game_state: Dict[str, Any]) -> bool:
        """Detect NPC interaction opportunity."""
        # This would check for dialogue state changes or nearby NPCs
        dialogue_state = game_state.get('dialogue_state', 0)
        return dialogue_state > 0
    
    def _detect_dungeon_entrance(self, game_state: Dict[str, Any]) -> bool:
        """Detect dungeon entrance opportunity."""
        player = game_state.get('player', {})
        dungeon_floor = player.get('dungeon_floor', 0)
        return dungeon_floor > 0 and self.last_room == 0  # Just entered dungeon
    
    def record_arbitration_result(self, step_count: int, reward_improvement: float, 
                                triggers: List[ArbitrationTrigger]):
        """Record the result of an LLM arbitration."""
        self.last_llm_call = step_count
        self.total_arbitrations += 1
        
        # Consider arbitration successful if reward improved significantly
        if reward_improvement > 0.1:  # Threshold for "success"
            self.successful_arbitrations += 1
            
    def get_arbitration_stats(self) -> Dict[str, float]:
        """Get current arbitration performance statistics."""
        if self.total_arbitrations == 0:
            return {'success_rate': 0.0, 'call_frequency': 0.0, 'rooms_per_call': 0.0}
            
        return {
            'success_rate': self.successful_arbitrations / self.total_arbitrations,
            'call_frequency': self.last_llm_call / max(1, self.total_arbitrations),
            'rooms_per_call': len(self.rooms_discovered_this_episode) / max(1, self.total_arbitrations),
            'current_adaptive_frequency': self._calculate_adaptive_frequency()
        }


class EnhancedPPOController(PPOController):
    """PPO Controller with enhanced LLM policy arbitration."""
    
    def __init__(self, observation_space: Any, action_space: Any, 
                 config: EnhancedControllerConfig, planner: Optional[ZeldaPlanner] = None):
        """Initialize enhanced controller."""
        super().__init__(observation_space, action_space, config, planner)
        self.config = config  # Override with enhanced config
        self.arbitration_tracker = SmartArbitrationTracker(config)
        self.episode_reward_buffer = []
        
    async def _act_llm_guided(self, obs: np.ndarray, structured_state: Optional[Dict[str, Any]]) -> int:
        """Enhanced LLM-guided decision making with smart arbitration."""
        if not structured_state:
            return self._act_pure_rl(obs)
            
        # Check if we should call the LLM
        should_call, triggers = self.arbitration_tracker.should_call_llm(
            self.step_count, structured_state)
            
        if should_call and self.planner:
            try:
                # Record reward before LLM call
                reward_before = sum(self.episode_reward_buffer[-10:]) / max(1, len(self.episode_reward_buffer[-10:]))
                
                plan = await self.planner.get_plan(structured_state)
                if plan:
                    macro_action = self.planner.get_macro_action(plan)
                    if macro_action:
                        # Adjust macro timeout based on context
                        adjusted_timeout = self._calculate_dynamic_timeout(triggers, macro_action)
                        macro_action.max_steps = min(macro_action.max_steps, adjusted_timeout)
                        
                        self.macro_executor.set_macro(macro_action)
                        
                        print(f"🧠 LLM Arbitration - Triggers: {[t.value for t in triggers]}, "
                              f"Macro: {macro_action.action_type.value}")
                        
            except Exception as e:
                print(f"Warning: LLM planning failed ({e}), falling back to RL")
                return self._act_pure_rl(obs)
        
        # Execute current macro if available
        if (not self.macro_executor.is_macro_complete() and 
            self.macro_executor.current_macro is not None):
            
            # Enhanced timeout checking
            if self.macro_executor.steps_executed > self.macro_executor.current_macro.max_steps:
                print(f"⏰ Macro timed out after {self.macro_executor.steps_executed} steps")
                self.macro_executor.clear_macro()
            else:
                macro_action = self.macro_executor.get_next_action(structured_state)
                if macro_action is not None:
                    return int(macro_action)
        
        # Fallback to RL policy
        return self._act_pure_rl(obs)
    
    def _calculate_dynamic_timeout(self, triggers: List[ArbitrationTrigger], 
                                 macro: MacroAction) -> int:
        """Calculate dynamic timeout based on context and macro type."""
        base_timeout = self.config.macro_timeout
        
        # Reduce timeout for urgent situations
        if ArbitrationTrigger.LOW_HEALTH in triggers:
            base_timeout = int(base_timeout * 0.5)  # Quick actions when low health
            
        # Increase timeout for complex actions
        if macro.action_type in [MacroType.SOLVE_PUZZLE, MacroType.ENTER_DUNGEON]:
            base_timeout = int(base_timeout * 1.5)
            
        # Reduce timeout when stuck (try different approaches quickly)
        if ArbitrationTrigger.STUCK_DETECTION in triggers:
            base_timeout = int(base_timeout * 0.7)
            
        return max(25, min(150, base_timeout))  # Clamp between 25-150 steps
    
    def update_episode_reward(self, reward: float):
        """Update reward buffer for arbitration performance tracking."""
        self.episode_reward_buffer.append(reward)
        if len(self.episode_reward_buffer) > 100:  # Keep last 100 rewards
            self.episode_reward_buffer.pop(0)
    
    def get_arbitration_performance(self) -> Dict[str, Any]:
        """Get comprehensive arbitration performance metrics."""
        stats = self.arbitration_tracker.get_arbitration_stats()
        stats.update({
            'total_arbitrations': self.arbitration_tracker.total_arbitrations,
            'successful_arbitrations': self.arbitration_tracker.successful_arbitrations,
            'rooms_discovered': len(self.arbitration_tracker.rooms_discovered_this_episode),
            'current_stuck_counter': self.arbitration_tracker.stuck_counter
        })
        return stats
    
    def reset_episode(self):
        """Reset episode-specific tracking."""
        super().reset_episode()
        self.arbitration_tracker.reset_tracking()
        self.episode_reward_buffer = []


# Factory function for easy creation
def create_enhanced_controller(observation_space, action_space, 
                             config_path: Optional[str] = None) -> EnhancedPPOController:
    """Create enhanced PPO controller with smart arbitration."""
    
    # Load enhanced config
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config = EnhancedControllerConfig.from_yaml(yaml_config)
    else:
        config = EnhancedControllerConfig()
    
    # Create planner if needed
    planner = None
    if config.use_planner:
        planner = ZeldaPlanner()  # You might want to inject this
        
    return EnhancedPPOController(observation_space, action_space, config, planner)

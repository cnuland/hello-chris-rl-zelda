#!/usr/bin/env python3
"""
LLM-Guided RL Integration Methods

Different approaches to make LLM suggestions more influential in RL training.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from enum import Enum

class LLMGuidanceMode(Enum):
    REWARD_SHAPING = "reward_shaping"
    ACTION_MASKING = "action_masking" 
    POLICY_INITIALIZATION = "policy_initialization"
    AUXILIARY_LOSS = "auxiliary_loss"
    CURRICULUM_LEARNING = "curriculum_learning"

class LLMGuidedTrainer:
    """Enhanced RL trainer that emphasizes LLM suggestions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.guidance_mode = config.get("guidance_mode", LLMGuidanceMode.REWARD_SHAPING)
        
        # LLM guidance weights
        self.llm_reward_weight = config.get("llm_reward_weight", 2.0)  # How much to weight LLM alignment
        self.macro_completion_bonus = config.get("macro_completion_bonus", 10.0)  # Bonus for completing LLM goals
        self.alignment_penalty = config.get("alignment_penalty", -1.0)  # Penalty for ignoring LLM
        
        # Action masking parameters
        self.action_bias_strength = config.get("action_bias_strength", 0.3)  # How much to bias toward LLM actions
        
        # Tracking
        self.current_llm_suggestion = None
        self.macro_progress = 0.0
        self.steps_since_llm_call = 0
        
    def apply_llm_guidance(self, base_reward: float, action_taken: int, 
                          llm_suggestion: Optional[Dict], game_state: Dict) -> float:
        """Apply LLM guidance to modify the reward signal."""
        
        if llm_suggestion is None:
            return base_reward
            
        enhanced_reward = base_reward
        
        if self.guidance_mode == LLMGuidanceMode.REWARD_SHAPING:
            enhanced_reward += self._apply_reward_shaping(
                action_taken, llm_suggestion, game_state
            )
        
        return enhanced_reward
    
    def _apply_reward_shaping(self, action: int, llm_suggestion: Dict, 
                            game_state: Dict) -> float:
        """Apply reward shaping based on LLM suggestions."""
        guidance_reward = 0.0
        
        # 1. Action Alignment Reward
        suggested_action = self._translate_llm_to_action(llm_suggestion)
        if suggested_action is not None:
            if action == suggested_action:
                guidance_reward += self.llm_reward_weight * 2.0  # Strong positive for following LLM
            else:
                guidance_reward += self.alignment_penalty  # Penalty for ignoring LLM
        
        # 2. Macro Progress Reward
        macro_progress = self._evaluate_macro_progress(llm_suggestion, game_state)
        guidance_reward += macro_progress * self.llm_reward_weight
        
        # 3. Goal Completion Bonus
        if self._is_macro_completed(llm_suggestion, game_state):
            guidance_reward += self.macro_completion_bonus
            
        # 4. Strategic Direction Reward
        strategic_alignment = self._evaluate_strategic_alignment(
            action, llm_suggestion, game_state
        )
        guidance_reward += strategic_alignment * self.llm_reward_weight * 0.5
        
        return guidance_reward
    
    def _translate_llm_to_action(self, llm_suggestion: Dict) -> Optional[int]:
        """Convert LLM suggestion to specific RL action."""
        action_map = {
            "MOVE_UP": 0,
            "MOVE_DOWN": 1, 
            "MOVE_LEFT": 2,
            "MOVE_RIGHT": 3,
            "A_BUTTON": 4,  # Attack/interact
            "B_BUTTON": 5,  # Secondary action
            "EXPLORE": None,  # Let RL decide how to explore
            "TALK_TO": 4,  # Use A button for talking
            "ENTER_DUNGEON": 0,  # Usually move up to enter
            "USE_ITEM": 5   # B button for items
        }
        
        llm_action = llm_suggestion.get("action", "").upper()
        return action_map.get(llm_action)
    
    def _evaluate_macro_progress(self, llm_suggestion: Dict, game_state: Dict) -> float:
        """Evaluate how much progress was made toward LLM goal."""
        progress = 0.0
        
        action = llm_suggestion.get("action", "").upper()
        target = llm_suggestion.get("target", "").lower()
        
        # Check for progress indicators
        if action == "EXPLORE":
            # Reward for discovering new areas
            if game_state.get("new_room_discovered", False):
                progress += 1.0
            # Reward for moving (not staying still)
            if game_state.get("player_moved", False):
                progress += 0.2
                
        elif action == "TALK_TO" and "npc" in target:
            # Reward for successful NPC interaction
            if game_state.get("dialogue_active", False):
                progress += 1.0
            # Reward for approaching NPCs
            if game_state.get("near_npc", False):
                progress += 0.5
                
        elif action == "ENTER_DUNGEON":
            # Reward for entering dungeons
            if game_state.get("in_dungeon", False):
                progress += 1.0
            # Reward for approaching dungeon entrance
            if game_state.get("near_dungeon", False):
                progress += 0.5
        
        return progress
    
    def _is_macro_completed(self, llm_suggestion: Dict, game_state: Dict) -> bool:
        """Check if the LLM's macro goal has been completed."""
        action = llm_suggestion.get("action", "").upper()
        
        if action == "ENTER_DUNGEON":
            return game_state.get("in_dungeon", False)
        elif action == "TALK_TO":
            return game_state.get("dialogue_active", False)
        elif action == "EXPLORE":
            return game_state.get("new_room_discovered", False)
            
        return False
    
    def _evaluate_strategic_alignment(self, action: int, llm_suggestion: Dict, 
                                    game_state: Dict) -> float:
        """Evaluate if action aligns with overall LLM strategy."""
        alignment = 0.0
        
        reasoning = llm_suggestion.get("reasoning", "").lower()
        
        # Encourage exploration when LLM suggests it
        if "explore" in reasoning and action in [0, 1, 2, 3]:  # Movement actions
            alignment += 0.5
            
        # Encourage interaction when LLM suggests social actions
        if any(word in reasoning for word in ["talk", "npc", "interact"]) and action == 4:
            alignment += 0.5
            
        # Encourage combat readiness when LLM detects threats
        if any(word in reasoning for word in ["enemy", "combat", "danger"]) and action in [4, 5]:
            alignment += 0.5
        
        return alignment
    
    def get_action_probabilities(self, base_probs: np.ndarray, 
                               llm_suggestion: Optional[Dict]) -> np.ndarray:
        """Modify action probabilities based on LLM guidance."""
        if llm_suggestion is None or self.guidance_mode != LLMGuidanceMode.ACTION_MASKING:
            return base_probs
            
        # Create biased probabilities
        biased_probs = base_probs.copy()
        
        suggested_action = self._translate_llm_to_action(llm_suggestion)
        if suggested_action is not None:
            # Increase probability of LLM-suggested action
            biased_probs[suggested_action] *= (1 + self.action_bias_strength)
            
        # Normalize to ensure valid probability distribution
        biased_probs = biased_probs / np.sum(biased_probs)
        
        return biased_probs
    
    def compute_auxiliary_loss(self, policy_logits: np.ndarray, 
                             llm_suggestion: Optional[Dict]) -> float:
        """Compute auxiliary loss to align policy with LLM suggestions."""
        if llm_suggestion is None or self.guidance_mode != LLMGuidanceMode.AUXILIARY_LOSS:
            return 0.0
            
        suggested_action = self._translate_llm_to_action(llm_suggestion)
        if suggested_action is None:
            return 0.0
        
        # Cross-entropy loss encouraging the suggested action
        target_probs = np.zeros_like(policy_logits)
        target_probs[suggested_action] = 1.0
        
        # Simple cross-entropy (in practice, use proper loss function)
        policy_probs = np.exp(policy_logits) / np.sum(np.exp(policy_logits))
        loss = -np.sum(target_probs * np.log(policy_probs + 1e-8))
        
        return loss * self.config.get("auxiliary_loss_weight", 0.1)

# Example configuration for different emphasis levels
EMPHASIS_CONFIGS = {
    "light_emphasis": {
        "guidance_mode": LLMGuidanceMode.REWARD_SHAPING,
        "llm_reward_weight": 1.0,
        "macro_completion_bonus": 5.0,
        "alignment_penalty": -0.5,
        "action_bias_strength": 0.1
    },
    
    "moderate_emphasis": {
        "guidance_mode": LLMGuidanceMode.REWARD_SHAPING,
        "llm_reward_weight": 2.0,
        "macro_completion_bonus": 10.0,
        "alignment_penalty": -1.0,
        "action_bias_strength": 0.3
    },
    
    "strong_emphasis": {
        "guidance_mode": LLMGuidanceMode.ACTION_MASKING,
        "llm_reward_weight": 3.0,
        "macro_completion_bonus": 20.0,
        "alignment_penalty": -2.0,
        "action_bias_strength": 0.5,
        "auxiliary_loss_weight": 0.2
    },
    
    "maximum_emphasis": {
        "guidance_mode": LLMGuidanceMode.AUXILIARY_LOSS,
        "llm_reward_weight": 5.0,
        "macro_completion_bonus": 50.0,
        "alignment_penalty": -5.0,
        "action_bias_strength": 0.8,
        "auxiliary_loss_weight": 0.5
    }
}

def create_llm_guided_trainer(emphasis_level: str = "moderate_emphasis") -> LLMGuidedTrainer:
    """Create a trainer with specified LLM emphasis level."""
    config = EMPHASIS_CONFIGS.get(emphasis_level, EMPHASIS_CONFIGS["moderate_emphasis"])
    return LLMGuidedTrainer(config)

if __name__ == "__main__":
    # Example usage
    trainer = create_llm_guided_trainer("strong_emphasis")
    
    # Simulate LLM suggestion
    llm_suggestion = {
        "action": "EXPLORE",
        "target": "new_areas", 
        "reasoning": "Link should explore to find new dungeons and NPCs"
    }
    
    # Simulate game state
    game_state = {
        "new_room_discovered": True,
        "player_moved": True,
        "near_npc": False
    }
    
    # Apply guidance
    base_reward = 1.0
    action_taken = 0  # Move up
    enhanced_reward = trainer.apply_llm_guidance(
        base_reward, action_taken, llm_suggestion, game_state
    )
    
    print(f"Base reward: {base_reward}")
    print(f"Enhanced reward: {enhanced_reward}")
    print(f"LLM guidance bonus: {enhanced_reward - base_reward}")

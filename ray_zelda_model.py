"""
Custom MLP Model for Ray RLlib - Zelda Oracle of Seasons
Matches the existing PolicyNetwork architecture (vector observations).
"""

import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

_, nn = try_import_torch()


class ZeldaMLPModel(TorchModelV2, nn.Module):
    """
    Custom MLP model for Zelda vector observations.
    
    Matches the existing PolicyNetwork architecture:
    - Input: 128-dimensional vector observation
    - Hidden layers: 256 → 256 → 256 (3 layers with ReLU)
    - Policy head: outputs action logits
    - Value head: outputs state value estimate
    
    This is the same architecture as the custom PPO implementation,
    ensuring consistency when migrating to Ray RLlib.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Get observation size from observation space
        if hasattr(obs_space, 'shape'):
            obs_size = obs_space.shape[0]
        else:
            obs_size = obs_space.n if hasattr(obs_space, 'n') else 128
        
        hidden_size = model_config.get("fcnet_hiddens", [256])[0]
        
        print(f"ZeldaMLPModel: obs_size={obs_size}, action_size={num_outputs}, hidden_size={hidden_size}")

        # Shared feature extraction (matching PolicyNetwork)
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.policy_head = nn.Linear(hidden_size, num_outputs)

        # Value head (critic)
        self.value_head = nn.Linear(hidden_size, 1)

        self._value = None
        
        # Initialize weights (matching PolicyNetwork initialization)
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Initialize policy head with smaller weights for stability
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass through the model.
        
        Args:
            input_dict: Dict with key "obs" containing observations
            state: RNN state (not used in this feedforward model)
            seq_lens: Sequence lengths for RNN (not used)
        
        Returns:
            (policy_logits, state)
        """
        obs = input_dict["obs"]
        
        # Ensure obs is float32
        if obs.dtype != torch.float32:
            obs = obs.float()
        
        # Shared feature extraction
        features = self.shared_layers(obs)
        
        # Compute action logits and value
        action_logits = self.policy_head(features)
        self._value = self.value_head(features)
        
        return action_logits, state

    @override(ModelV2)
    def value_function(self):
        """
        Return the value function estimate for the most recent forward pass.
        """
        return self._value.squeeze(1)

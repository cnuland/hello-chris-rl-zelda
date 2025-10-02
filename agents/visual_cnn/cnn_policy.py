"""CNN-based PPO policy network for visual observations.

This module implements a convolutional neural network architecture
for learning from Game Boy screen pixels (144×160 grayscale).

Architecture inspired by:
- Atari DQN (Mnih et al., 2015)
- PPO for visual RL (Schulman et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class CNNPolicyNetwork(nn.Module):
    """Convolutional Neural Network for PPO with visual observations.
    
    This network processes Game Boy screen pixels to produce action
    distributions and value estimates for PPO training.
    
    Input: (batch, 1, 144, 160) grayscale screen
    Output: (action_logits, value)
    """

    def __init__(self, action_size: int, input_channels: int = 1):
        """Initialize CNN policy network.
        
        Args:
            action_size: Number of discrete actions (9 for Zelda)
            input_channels: Image channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()
        
        self.action_size = action_size
        self.input_channels = input_channels
        
        # Convolutional feature extraction
        # Input: (batch, 1, 144, 160) for grayscale Game Boy screen
        self.conv_layers = nn.Sequential(
            # Conv1: 144×160 → 36×40
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            
            # Conv2: 36×40 → 17×19
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            
            # Conv3: 17×19 → 15×17
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        
        # Calculate flattened size
        # Conv1: (144-8)/4 + 1 = 35, (160-8)/4 + 1 = 39
        # Conv2: (35-4)/2 + 1 = 16, (39-4)/2 + 1 = 18
        # Conv3: (16-3)/1 + 1 = 14, (18-3)/1 + 1 = 16
        # Total: 64 × 14 × 16 = 14,336
        self.feature_size = 64 * 14 * 16
        
        # Shared fully connected layers
        self.fc_shared = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_head = nn.Linear(512, action_size)
        
        # Value head (critic)
        self.value_head = nn.Linear(512, 1)
        
        # Initialize weights
        self._init_weights()
        
        print(f"🧠 CNNPolicyNetwork initialized:")
        print(f"   Input: ({input_channels}, 144, 160) → Features: {self.feature_size}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Policy head gets smaller initialization for stability
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            obs: Visual observation (batch, channels, height, width)
                 Expected: (batch, 1, 144, 160) for grayscale
        
        Returns:
            Tuple of (action_logits, value)
        """
        # Extract visual features
        conv_features = self.conv_layers(obs)
        
        # Flatten spatial dimensions
        flattened = conv_features.view(conv_features.size(0), -1)
        
        # Shared processing
        shared_features = self.fc_shared(flattened)
        
        # Separate heads
        action_logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        
        return action_logits, value
    
    def get_action_and_value(
        self, 
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute value.
        
        Args:
            obs: Visual observation tensor
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(obs)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only.
        
        Args:
            obs: Visual observation tensor
            
        Returns:
            Value tensor
        """
        _, value = self.forward(obs)
        return value
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.
        
        Args:
            obs: Visual observation tensor
            actions: Action tensor
        
        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_logits, values = self.forward(obs)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_probs, values.squeeze(-1), entropy


def preprocess_observation(screen_array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Preprocess Game Boy screen for CNN input.
    
    Converts raw RGB screen to normalized grayscale tensor.
    
    Args:
        screen_array: Raw screen from environment (144, 160, 1) uint8
        device: Torch device (cuda/cpu)
    
    Returns:
        Preprocessed tensor (1, 1, 144, 160) float32 on device
    """
    # Remove channel dimension if present: (144, 160, 1) → (144, 160)
    if screen_array.ndim == 3 and screen_array.shape[2] == 1:
        grayscale = screen_array[:, :, 0]
    else:
        grayscale = screen_array
    
    # Normalize to [0, 1]
    normalized = grayscale.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions: (144, 160) → (1, 1, 144, 160)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor.to(device)


def batch_preprocess_observations(
    screen_arrays: np.ndarray, 
    device: torch.device
) -> torch.Tensor:
    """Preprocess batch of Game Boy screens for CNN input.
    
    Args:
        screen_arrays: Batch of screens (batch, 144, 160, 1) uint8
        device: Torch device (cuda/cpu)
    
    Returns:
        Preprocessed tensor (batch, 1, 144, 160) float32 on device
    """
    # Remove last dimension if it's 1: (batch, 144, 160, 1) → (batch, 144, 160)
    if screen_arrays.ndim == 4 and screen_arrays.shape[3] == 1:
        grayscale = screen_arrays[:, :, :, 0]
    else:
        grayscale = screen_arrays
    
    # Normalize to [0, 1]
    normalized = grayscale.astype(np.float32) / 255.0
    
    # Add channel dimension: (batch, 144, 160) → (batch, 1, 144, 160)
    tensor = torch.from_numpy(normalized).unsqueeze(1)
    
    return tensor.to(device)

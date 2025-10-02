# Option 4: Visual Observations with CNN - Implementation Guide

## 🎯 Goal
Replace vector-based observations with visual (pixel-based) observations processed by a Convolutional Neural Network (CNN). This allows the PPO agent to "see" the game screen and naturally understand spatial relationships, making exploration more intuitive.

---

## 📊 Current vs. Proposed Architecture

### Current System (Vector Observations)
```
Game Screen (144×160×3 pixels)
        ↓
   RAM extraction
        ↓
   128-float vector ← Current PPO input
   [x, y, room, health, ...]
        ↓
   MLP Network (3×256 hidden layers)
        ↓
   Action selection
```

**Problem:** Agent has no spatial awareness. Can't "see" where unexplored areas are.

### Proposed System (Visual Observations)
```
Game Screen (144×160×3 pixels)
        ↓
   Grayscale conversion (144×160×1) ← New PPO input
        ↓
   CNN Network (3 conv layers + 2 FC layers)
        ↓
   Action selection
```

**Benefit:** Agent can directly perceive unexplored areas, obstacles, NPCs, and spatial layout!

---

## 🏗️ Implementation Steps

### Step 1: Create CNN Policy Network

**File:** `agents/controller_cnn.py` (new file)

```python
"""CNN-based PPO controller for visual observations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class CNNPolicyNetwork(nn.Module):
    """Convolutional Neural Network for PPO with visual observations.
    
    Architecture inspired by:
    - Atari DQN (Mnih et al., 2015)
    - PPO for visual RL (Schulman et al., 2017)
    """

    def __init__(self, action_size: int, input_channels: int = 1):
        """Initialize CNN policy network.
        
        Args:
            action_size: Number of discrete actions (9 for Zelda)
            input_channels: Image channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()
        
        # Convolutional feature extraction
        # Input: (batch, 1, 144, 160) for grayscale Game Boy screen
        self.conv_layers = nn.Sequential(
            # Conv1: 144×160 → 36×40
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Conv2: 36×40 → 17×19
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Conv3: 17×19 → 15×17
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate flattened size (64 channels × 15 × 17 = 16,320)
        # This will be automatically calculated in first forward pass
        self.feature_size = 64 * 15 * 17  # 16,320 features
        
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
    
    def get_action_and_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute value.
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(obs)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        _, value = self.forward(obs)
        return value
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        action_logits, values = self.forward(obs)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_probs, values.squeeze(-1), entropy


def preprocess_observation(screen_array: np.ndarray) -> torch.Tensor:
    """Preprocess Game Boy screen for CNN input.
    
    Args:
        screen_array: Raw screen from PyBoy (144, 160, 3) RGB uint8
    
    Returns:
        Preprocessed tensor (1, 1, 144, 160) grayscale float32
    """
    # Convert to grayscale
    grayscale = np.dot(screen_array[...,:3], [0.299, 0.587, 0.114])
    
    # Normalize to [0, 1]
    normalized = grayscale.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions: (1, 144, 160) → (1, 1, 144, 160)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor
```

---

### Step 2: Modify Environment for Visual Observations

**File:** `emulator/zelda_env_configurable.py` (modify existing)

Add this method to `ZeldaConfigurableEnvironment`:

```python
def _setup_visual_observation_space(self):
    """Setup observation space for visual (CNN) mode."""
    from gymnasium import spaces
    
    # Grayscale Game Boy screen: 144×160×1
    self.observation_space = spaces.Box(
        low=0,
        high=255,
        shape=(144, 160, 1),  # (height, width, channels)
        dtype=np.uint8
    )
    print("🎮 Visual observation mode: Screen pixels (144×160×1)")

def _get_visual_observation(self) -> np.ndarray:
    """Get visual observation (screen pixels)."""
    # Get screen from PyBoy
    screen_rgb = self.bridge.get_screen()  # (144, 160, 3)
    
    # Convert to grayscale
    grayscale = np.dot(screen_rgb[...,:3], [0.299, 0.587, 0.114])
    
    # Add channel dimension: (144, 160) → (144, 160, 1)
    observation = np.expand_dims(grayscale, axis=-1).astype(np.uint8)
    
    return observation
```

Modify `__init__` to support visual mode:

```python
def __init__(self, ...):
    # ... existing code ...
    
    # Setup observation space based on mode
    if self.observation_type == 'visual':
        self._setup_visual_observation_space()
    else:
        # ... existing vector space setup ...
```

Modify `_get_observation`:

```python
def _get_observation(self) -> np.ndarray:
    """Get current observation."""
    if self.observation_type == 'visual':
        return self._get_visual_observation()
    else:
        # ... existing vector observation code ...
```

---

### Step 3: Create Visual Training Script

**File:** `train_hybrid_visual_cnn.py` (new file)

```python
"""Hybrid RL+LLM training with CNN for visual observations."""

import torch
import numpy as np
import requests
from typing import Dict, Any
from agents.controller_cnn import CNNPolicyNetwork, preprocess_observation
from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment


class HybridVisualCNNTrainer:
    """Hybrid trainer using CNN for visual observations + LLM guidance."""
    
    def __init__(
        self,
        rom_path: str,
        llm_endpoint: str = "http://localhost:8000/v1/chat/completions",
        llm_frequency: int = 5,
        llm_bonus: float = 5.0,
        total_timesteps: int = 100000,
        headless: bool = True
    ):
        self.rom_path = rom_path
        self.llm_endpoint = llm_endpoint
        self.llm_frequency = llm_frequency
        self.llm_bonus = llm_bonus
        self.total_timesteps = total_timesteps
        
        # Create environment with VISUAL observations
        env_config = {
            "environment": {
                "max_episode_steps": 12000,
                "frame_skip": 4,
                "observation_type": "visual",  # ← KEY CHANGE!
                "normalize_observations": False  # CNN does its own normalization
            },
            "planner_integration": {
                "use_planner": True,  # Still get structured states for LLM
                "enable_structured_states": True
            }
        }
        
        self.env = ZeldaConfigurableEnvironment(
            rom_path=rom_path,
            config_dict=env_config,
            headless=headless
        )
        
        # Initialize CNN policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = CNNPolicyNetwork(
            action_size=self.env.action_space.n,
            input_channels=1  # Grayscale
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        
        print(f"🧠 CNN Policy Network initialized")
        print(f"   Device: {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
    
    def preprocess_screen(self, screen: np.ndarray) -> torch.Tensor:
        """Convert screen to CNN input."""
        # screen is (144, 160, 1) from env
        # Convert to (1, 1, 144, 160) tensor
        grayscale = screen[:, :, 0]  # Remove channel dim
        normalized = grayscale.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def collect_rollout(self, rollout_steps: int = 2048):
        """Collect experience using CNN policy + LLM guidance."""
        # ... similar to train_hybrid_rl_llm.py but:
        # 1. Use self.preprocess_screen(obs) before policy forward
        # 2. Keep LLM guidance using structured_state from info dict
        # 3. Store visual observations in rollout buffer
        pass
    
    def train(self):
        """Main training loop."""
        print("🚀 Starting Visual CNN + LLM Hybrid Training")
        print(f"   Total timesteps: {self.total_timesteps:,}")
        print(f"   LLM frequency: every {self.llm_frequency} steps")
        print(f"   Observation: 144×160 grayscale pixels")
        
        # ... training loop ...
```

---

### Step 4: Configuration File

**File:** `configs/env_visual_cnn.yaml` (new file)

```yaml
# Visual CNN Training Configuration

environment:
  observation_type: "visual"  # Use screen pixels instead of vectors
  normalize_observations: false  # CNN handles normalization
  max_episode_steps: 12000
  frame_skip: 4

planner_integration:
  use_planner: true  # LLM still gets structured state
  enable_structured_states: true  # Generate for LLM context
  enable_visual: false  # Don't send pixels to LLM (use RAM state)

rewards:
  health_reward: 10.0
  room_discovery_reward: 15.0
  npc_interaction_reward: 50.0
  exploration_bonus: 25.0  # NEW AREA = BIG BONUS
  loitering_penalty: -2.8
  decay_window: 500

training:
  total_timesteps: 200000  # Needs more time due to CNN complexity
  rollout_steps: 2048
  learning_rate: 0.0003
  batch_size: 64
  epochs_per_update: 4
  gamma: 0.99
  gae_lambda: 0.95
```

---

## 🧪 Testing the Implementation

### Quick Test (1 hour)
```bash
python train_hybrid_visual_cnn.py \
  --rom-path roms/zelda_oracle_of_seasons.gbc \
  --headless \
  --total-timesteps 15000 \
  --llm-frequency 5 \
  --llm-bonus 5.0
```

### Full Training (overnight, 12 hours)
```bash
python train_hybrid_visual_cnn.py \
  --rom-path roms/zelda_oracle_of_seasons.gbc \
  --headless \
  --total-timesteps 200000 \
  --llm-frequency 5 \
  --llm-bonus 5.0
```

---

## 📊 Expected Benefits

### 1. **Natural Spatial Understanding**
- CNN "sees" unexplored areas as different visual patterns
- No need for complex grid-based exploration rewards
- Agent learns "dark areas = unexplored"

### 2. **Better Generalization**
- Visual patterns transfer across rooms
- Agent recognizes doors, NPCs, enemies by appearance
- Less reliant on exact RAM addresses

### 3. **Intuitive Learning**
- Similar to how humans play games
- Pixel-level changes = immediate feedback
- Natural curiosity emerges from visual novelty

### 4. **Synergy with LLM**
- LLM provides high-level goals ("explore north")
- CNN provides low-level visual navigation
- Perfect division of labor!

---

## ⚠️ Challenges & Solutions

### Challenge 1: Slower Training
**Problem:** CNNs process 23,040 pixels vs. 128 floats (180x more data)  
**Solution:** Use GPU acceleration (10-50x speedup)

### Challenge 2: More Parameters to Learn
**Problem:** CNN has ~2-3M parameters vs. MLP's ~200K  
**Solution:** Train longer (200k timesteps instead of 15k)

### Challenge 3: Requires More Samples
**Problem:** Visual RL is sample-inefficient  
**Solution:**
- Use frame stacking (4 frames = motion perception)
- Data augmentation (flip/rotate screens)
- Pretrain on random exploration

### Challenge 4: Memory Usage
**Problem:** Storing 144×160 images in rollout buffer  
**Solution:** Use uint8 storage + batch preprocessing

---

## 🎯 Success Metrics

After 200k timesteps with CNN, expect:

| Metric | Vector (Current) | CNN (Expected) |
|--------|------------------|----------------|
| **Unique Rooms** | 6 | 15-25 |
| **Grid Areas** | 6 | 100-300 |
| **Episodes** | 0 | 5-15 |
| **NPC Interactions** | 0 | 3-10 |
| **Training Time** | 30 min | 3-6 hours |

**Why better?**
- CNN sees patterns humans see
- Exploration becomes natural (novelty-driven)
- Spatial navigation improves dramatically

---

## 🚀 Implementation Priority

**Minimal viable implementation** (4-6 hours of work):
1. ✅ Create `CNNPolicyNetwork` class
2. ✅ Add visual mode to environment
3. ✅ Create training script
4. ✅ Test on 15k timesteps

**Full production version** (1-2 days):
1. Add frame stacking (4 frames for motion)
2. Implement data augmentation
3. Add visual attention mechanisms
4. Optimize memory usage
5. Add curriculum learning

---

## 💡 Hybrid Approach (Best of Both Worlds)

Consider **dual-stream architecture**:

```python
class DualStreamPolicyNetwork(nn.Module):
    """Combines visual CNN + vector MLP."""
    
    def __init__(self, action_size):
        super().__init__()
        
        # Visual stream (CNN)
        self.visual_stream = CNNLayers()  # → 512 features
        
        # Vector stream (MLP)
        self.vector_stream = MLPLayers()   # → 256 features
        
        # Fusion layer
        self.fusion = nn.Linear(512 + 256, 512)
        
        # Heads
        self.policy_head = nn.Linear(512, action_size)
        self.value_head = nn.Linear(512, 1)
```

**Benefits:**
- Visual: Spatial awareness, unexplored areas
- Vector: Precise health/rupees/keys values
- **Best of both!**

---

## 📖 References & Inspiration

1. **Atari DQN** (Mnih et al., 2015)
   - First successful visual RL at scale
   - CNN architecture still widely used

2. **PPO Paper** (Schulman et al., 2017)
   - Visual observations section
   - Stability tricks for CNNs

3. **Procgen Benchmark** (Cobbe et al., 2020)
   - Visual RL for procedural games
   - Similar to Zelda's varied rooms

4. **MineRL** (Guss et al., 2019)
   - Visual RL in complex 3D environment
   - Shows CNNs scale to rich visuals

---

## ✅ Next Steps

1. **Implement CNNPolicyNetwork** (agents/controller_cnn.py)
2. **Test with simple forward pass** (ensure shapes match)
3. **Add visual mode to environment** (2-3 lines)
4. **Run 15k timestep test** (validate training works)
5. **If successful, run overnight** (200k timesteps)
6. **Compare results** (CNN vs. vector baseline)

This approach has the highest potential for dramatic improvement in exploration! 🚀

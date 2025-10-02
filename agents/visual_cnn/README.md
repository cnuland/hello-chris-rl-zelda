# Visual CNN Module for Zelda RL

This module implements CNN-based PPO agents that learn from screen pixels instead of vector observations.

## Architecture

```
Game Screen (144×160×3 RGB)
        ↓
   Grayscale conversion
        ↓
   (144×160×1) uint8
        ↓
   Normalize to [0,1]
        ↓
   CNN (3 conv layers)
        ↓
   Feature extraction
        ↓
   Policy + Value heads
        ↓
   Action distribution
```

### Network Details

**Conv Layers:**
1. Conv2D(1→32, 8×8, stride=4) + ReLU → 36×40
2. Conv2D(32→64, 4×4, stride=2) + ReLU → 17×19
3. Conv2D(64→64, 3×3, stride=1) + ReLU → 14×16

**FC Layers:**
- Flatten: 14,336 features
- FC: 14,336 → 512 + ReLU
- Policy head: 512 → 9 actions
- Value head: 512 → 1 value

**Total Parameters:** ~2-3M (vs. ~200K for vector MLP)

## Files

- `cnn_policy.py` - CNN network implementation
- `__init__.py` - Module exports
- `README.md` - This file

## Usage

### Train with CNN + LLM Hybrid

```bash
python train_visual_cnn_hybrid.py \
  --rom-path roms/zelda_oracle_of_seasons.gbc \
  --headless \
  --total-timesteps 100000 \
  --llm-frequency 5
```

### Quick Test (15k steps, ~1 hour)

```bash
python train_visual_cnn_hybrid.py \
  --rom-path roms/zelda_oracle_of_seasons.gbc \
  --headless \
  --total-timesteps 15000 \
  --llm-frequency 5 \
  --llm-bonus 5.0
```

### Full Training (200k steps, overnight)

```bash
python train_visual_cnn_hybrid.py \
  --rom-path roms/zelda_oracle_of_seasons.gbc \
  --headless \
  --total-timesteps 200000 \
  --llm-frequency 5 \
  --llm-bonus 5.0
```

## Key Advantages

1. **Natural Spatial Understanding**
   - CNN processes 2D pixel patterns
   - Automatically learns spatial relationships
   - Recognizes unexplored areas visually

2. **Better Generalization**
   - Visual patterns transfer across rooms
   - Recognizes NPCs/enemies by appearance
   - Less dependent on exact RAM addresses

3. **Intuitive Exploration**
   - Similar to human gameplay
   - Visual novelty drives curiosity
   - Natural navigation behavior

## Expected Performance

| Metric | Vector (Baseline) | CNN (Expected) |
|--------|-------------------|----------------|
| Rooms Discovered | 6 | 40-80 |
| Grid Areas | 6 | 500-1500 |
| Episodes Completed | 0 | 5-15 |
| Training Time | 30 min | 6-12 hours |

## Requirements

- PyTorch
- CUDA (recommended for speed, but CPU works)
- Same dependencies as base project

## Technical Notes

### Memory Usage
- Observation buffer: 144×160×2048 ≈ 45 MB per rollout
- Network weights: ~2-3M parameters ≈ 12 MB
- Total: ~60-100 MB (manageable)

### Performance
- CPU: ~100-200 FPS (slower than vector)
- GPU: ~500-1000 FPS (2-5x faster than CPU)
- **Recommendation:** Use GPU for training

### Preprocessing
- RGB → Grayscale: 3 channels → 1 channel
- Normalization: [0, 255] → [0.0, 1.0]
- Format: (H, W, C) → (1, C, H, W) for PyTorch

## Future Enhancements

1. **Frame Stacking**
   - Stack 4 frames for temporal context
   - Enables motion perception
   - Better enemy/NPC tracking

2. **Data Augmentation**
   - Horizontal flips (mirror rooms)
   - Color jitter (lighting variations)
   - Improves generalization

3. **Attention Mechanisms**
   - Focus on important screen regions
   - Ignore static background
   - Better NPC/item detection

4. **Residual Connections**
   - Deeper network (ResNet-style)
   - Better gradient flow
   - Higher accuracy

## Comparison to Vector Observations

| Aspect | Vector | Visual CNN |
|--------|--------|------------|
| **Input Size** | 128 floats | 23,040 pixels |
| **Parameters** | ~200K | ~2-3M |
| **Training Time** | Faster (1x) | Slower (3-5x) |
| **Sample Efficiency** | Better | Worse |
| **Spatial Understanding** | None | Excellent |
| **Exploration** | Poor | Excellent |
| **Final Performance** | Limited | High potential |

**Verdict:** CNN is slower to train but has much higher ceiling for exploration-heavy tasks!

## Citation

Architecture inspired by:
- **Atari DQN** (Mnih et al., 2015)
- **PPO Visual** (Schulman et al., 2017)
- **Impala** (Espeholt et al., 2018)

# Advanced Exploration Reward System

## 🎯 Overview
Implemented a sophisticated exploration reward system with time-based decay to encourage meaningful exploration while allowing beneficial backtracking.

## 🔧 Key Features

### 1. **Grid-Based Area Tracking** (16x16 pixel cells)
- Divides each room into grid cells for fine-grained position tracking
- Each grid cell: `(room_id, x//16, y//16)`
- Prevents agents from micro-looping in tiny spaces

### 2. **Stationary Penalty** 
```
If position hasn't changed:
  Penalty = -min(stationary_steps × 0.5, 5.0)
  
Max penalty: -5.0 per step when completely stuck
```

### 3. **Time-Based Decay System**
```python
decay_window = 500 steps  # ~83 LLM calls @ 5 step frequency

If area visited recently (< 500 steps ago):
  decay_factor = 1.0 - (time_since_visit / 500)
  loiter_penalty = -2.0 × decay_factor
  
If area not visited recently (≥ 500 steps ago):
  backtrack_bonus = +0.5  # Productive revisit allowed
```

**Example Timeline:**
- Visit area at step 0
- Return at step 100: Penalty = -1.6 (80% decay)
- Return at step 250: Penalty = -1.0 (50% decay)  
- Return at step 400: Penalty = -0.4 (20% decay)
- Return at step 500+: Bonus = +0.5 (full decay, backtracking OK)

### 4. **Exploration Bonuses**
```
New area discovered: +15.0 points (5.0 × 3.0 multiplier)
```

## 📊 Reward Structure

```
Total Reward = Environment Reward + LLM Bonus + Exploration Reward

Where Exploration Reward:
  - New area: +15.0
  - Backtracking (after decay): +0.5
  - Recent revisit: -2.0 to 0.0 (decaying)
  - Standing still: -0.5 to -5.0 (increasing)
```

## 🎮 How It Works

1. **Agent takes action** → Environment steps
2. **Track position:** `(x, y, room)` and grid cell `(room, x//16, y//16)`
3. **Check if stationary:** Same exact position? Apply penalty
4. **Check grid cell history:**
   - Never visited? → Big bonus! (+15.0)
   - Visited recently? → Loitering penalty (-2.0 to 0.0)
   - Visited long ago? → Small backtrack bonus (+0.5)
5. **Add to total reward** alongside LLM guidance bonus

## 🎯 Expected Behaviors

### ✅ Encouraged
- Exploring new areas (huge bonuses)
- Moving consistently (no stationary penalty)
- Backtracking after 500 steps (small bonus)
- Following LLM suggestions to NPCs

### ❌ Discouraged  
- Standing completely still (escalating penalty)
- Pacing back and forth in same spot (loitering penalty)
- Ignoring NPCs when detected
- Repetitive movement patterns

## 📈 Tracking & Logging

### Real-time Feedback
```
🌟 NEW AREA EXPLORED! Bonus: +15.0
⚠️  Loitering penalty: -1.8
🎯 [NPC INTERACTION #3!]
```

### Final Statistics
```
🗺️  EXPLORATION SUMMARY:
   Unique Rooms Visited: 7
   Unique Grid Areas Explored: 142
   Total Position Changes: 8,453
   
🎯 EXPLORATION MECHANICS:
   ✓ Anti-loitering: Penalties for staying in same area
   ✓ Decay window: 500 steps
   ✓ Backtracking allowed after decay period
   ✓ New area bonus: 15.0 points
```

## 🔄 Comparison: Before vs After

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Unique rooms in 730 episodes | 7 | 15-20+ |
| Grid areas explored | ~50 | 200-400+ |
| UP/DOWN loop ratio | 78% | <40% |
| NPC interactions | 0 | 5-15+ |
| Exploration efficiency | Low | High |

## 🧪 Tuning Parameters

Easily adjustable in `__init__`:

```python
self.decay_window = 500  # Steps before penalty decays to 0
self.exploration_bonus_multiplier = 3.0  # Multiplier for new area bonus
```

### Recommendations
- **Faster decay:** Reduce `decay_window` to 300 (more backtracking)
- **Slower decay:** Increase to 800 (stronger anti-loitering)
- **Higher exploration:** Increase multiplier to 5.0
- **Lower exploration:** Decrease to 2.0

## 🎯 Key Insight

**The decay system enables Zelda-style gameplay:**
- Visit town → Explore dungeon → Return to town (no penalty!)
- Get item → Backtrack to use it in old area (allowed!)
- But prevents: Walk 2 steps up → Walk 2 steps down (penalized!)

This matches how Zelda is meant to be played! 🗡️

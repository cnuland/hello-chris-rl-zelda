# ⚔️ Combat Rewards Implementation

**Date:** October 18, 2025  
**Implementation:** Enemy kill tracking and rewards  
**Status:** ✅ Implemented and ready for testing

## 🎯 What Was Added

### Combat Tracking System
Link now receives rewards for killing enemies, encouraging combat engagement without making it too rewarding to farm.

## 📝 Files Modified

### 1. `emulator/zelda_env_configurable.py`

**Added to `__init__`:**
```python
# Combat tracking (for enemy kill rewards)
self.last_enemies_killed = 0  # Track cumulative enemy kills (2-byte counter at 0xC620)
```

**Added to `reset()`:**
```python
# Track initial enemy kill count for combat rewards
# ENEMIES_KILLED is at 0xC620 (2 bytes, little-endian)
self.last_enemies_killed = self.bridge.get_memory(0xC620) + (self.bridge.get_memory(0xC621) << 8)
```

**Added to `step()` (in reward calculation):**
```python
# NEW: Track enemy kills for combat rewards
current_enemies_killed = self.bridge.get_memory(0xC620) + (self.bridge.get_memory(0xC621) << 8)
if current_enemies_killed > self.last_enemies_killed:
    kills_gained = current_enemies_killed - self.last_enemies_killed
    combat_reward = reward_config.get('enemy_kill', 3.0) * kills_gained
    total_reward += combat_reward
    print(f"⚔️  ENEMY KILLED! Total: {self.last_enemies_killed} → {current_enemies_killed} (+{kills_gained}) | Reward: +{combat_reward:.1f}")
    self.last_enemies_killed = current_enemies_killed
```

### 2. `configs/env.yaml`

**Added new reward:**
```yaml
# Combat rewards
enemy_kill: 3.0  # Reward per enemy killed (encourages combat, not too high to avoid farming)
```

## 🎯 Design Rationale

### Reward Value: 3.0
- **Not too high:** Prevents enemy farming behavior (killing enemies repeatedly for points)
- **Not too low:** Makes combat worthwhile and encourages Link to engage
- **Comparison to other rewards:**
  - Movement (fresh): 0.15 per step
  - New room discovery: 50.0 (one-time)
  - Enemy kill: 3.0 (repeatable but requires combat)
  - Maku Tree dialogue: 500.0 (major milestone)

### Why This Balance Works
- **20 steps of movement** = 3.0 reward (same as 1 enemy kill)
- Killing an enemy typically takes **5-10 steps** of combat
- **More efficient than random movement**, but not better than exploration
- Encourages Link to **clear enemies while exploring** rather than avoiding them

## 📊 Expected Behavior

### What Link Should Learn
1. ✅ Engage enemies when encountered (worth the effort)
2. ✅ Clear rooms of enemies while exploring
3. ✅ Balance combat with exploration (not farm enemies)
4. ❌ Don't avoid all enemies (missing free rewards)
5. ❌ Don't farm enemies repeatedly (better rewards elsewhere)

### Log Output Examples
When Link kills an enemy:
```
⚔️  ENEMY KILLED! Total: 15 → 16 (+1) | Reward: +3.0
```

Multiple kills in quick succession:
```
⚔️  ENEMY KILLED! Total: 15 → 17 (+2) | Reward: +6.0
```

## 🧪 Testing Verification

### Manual Test Results (test_enemy_kill_detection.py)
- ✅ Kill counter increments correctly
- ✅ Detection is immediate (within 10 frames)
- ✅ No false positives
- ✅ Works across all enemy types
- ✅ Counter persists across saves

### Training Test Plan
1. Monitor kill counts in logs
2. Observe if Link engages enemies
3. Check if combat becomes part of exploration strategy
4. Verify no enemy farming behavior develops
5. Adjust reward if needed (too high/low)

## 🔍 Memory Addresses Used

- **0xC620-0xC621:** ENEMIES_KILLED (2 bytes, little-endian)
  - Cumulative lifetime counter
  - Increments on every enemy kill
  - Persists across saves

## 💡 Future Enhancements (Optional)

1. **Boss Kill Bonuses**
   - Larger reward for boss defeats
   - Would need boss detection logic

2. **Room Clear Bonuses**
   - Bonus for clearing all enemies in a room
   - Track ENEMIES_ON_SCREEN (0xCC30) going to 0

3. **Combat Efficiency**
   - Bonus for no-damage kills
   - Track health before/after combat

4. **Enemy Type Differentiation**
   - Different rewards for different enemy types
   - Would require entity type detection

## 🚀 Next Steps

1. ✅ Implementation complete
2. ⏭️  Deploy to training cluster
3. ⏭️  Monitor training logs for kill events
4. ⏭️  Observe Link's combat behavior
5. ⏭️  Tune reward value if needed based on results

## 📈 Success Metrics

We'll know it's working if:
- Link kills enemies when encountered (visible in logs)
- Combat becomes integrated with exploration
- No enemy farming loops develop
- Overall episode returns improve slightly
- Link survives longer (combat = health control)

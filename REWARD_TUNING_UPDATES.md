# 🎯 Reward Tuning Updates - October 18, 2025

## 📊 Based on Overnight Run Analysis

**Run:** raysubmit_5VgiNbAEcY3yzNpz  
**Performance:** +15,111.6 episode return, 43 rooms explored, 2 Maku Tree dialogues

## ✅ Changes Applied to `configs/env.yaml`

### 1. Maku Tree Dialogue Reward
```yaml
# BEFORE:
maku_tree_dialogue: 300.0

# AFTER:
maku_tree_dialogue: 500.0  # +66% increase
```

**Reasoning:**
- 14 Maku Tree entrances but only 2 successful dialogues (14% success rate)
- Dialogue is the critical quest trigger (gives Gnarled Key)
- Now matches the importance of Gnarled Key reward (both 500.0)
- Encourages Link to interact vs just entering/leaving

### 2. Dungeon Entry Reward
```yaml
# BEFORE:
dungeon_entered: 150.0

# AFTER:
dungeon_entered: 250.0  # +66% increase
```

**Reasoning:**
- Dungeon is the NEXT major goal after getting Gnarled Key
- Strengthens the quest progression signal: Maku Tree → Key → Dungeon
- No dungeon entries yet (expected - key detection was broken, now fixed)

## 🔒 What We Kept Unchanged

### Exploration Rewards (Working Perfectly!)
- `new_room_discovery: 50.0` - 43 rooms explored ✅
- `room_novelty_reward: 20.0` - Novelty system working ✅
- `movement_fresh: 0.15` - Fresh room movement ✅
- `movement_stale: 0.05` - Stale room movement ✅

### Other Milestone Rewards
- `gnarled_key_obtained: 500.0` - Already appropriately high ✅
- `maku_tree_entered: 100.0` - Good baseline ✅
- `sword_obtained: 200.0` - Appropriate tier ✅

## 🎯 Expected Impact

### Quest Progression Path (Now Clearer)
1. **Maku Tree Entry:** +100 (baseline)
2. **Maku Tree Dialogue:** +500 (CRITICAL - triggers quest!)
3. **Gnarled Key Obtained:** +500 (CRITICAL - unlocks dungeon!)
4. **Dungeon Entry:** +250 (major progression!)

### Exploration (Unchanged)
- First room visit: +50
- Revisit with full novelty: +20
- Movement in fresh areas: +0.15/step
- Movement in stale areas: +0.05/step (encourages leaving)

## 📈 Rationale

The novelty-decay exploration system is the **MVP** of this training run:
- Episode returns nearly doubled (8k → 15k)
- 43 unique rooms explored
- No corner-sticking behavior
- Surviving full 10k step episodes

**Philosophy:** Don't fix what isn't broken. Just sharpen the quest progression signal.

## 🚀 Next Training Run Expectations

With both the **item mapping fix** AND these **reward tweaks**, we should see:
- ✅ Correct item detection (Sword, not Rod)
- ✅ Gnarled Key properly detected after dialogue
- ✅ Stronger push toward Maku Tree dialogue
- ✅ Clearer signal to enter dungeon after getting key
- ✅ Continued excellent exploration behavior

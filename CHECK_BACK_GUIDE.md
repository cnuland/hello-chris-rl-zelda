# ⏰ Check Back in a Few Hours - Quick Reference Guide

**Current Job**: `raysubmit_FfgUftqiybM88FUX`  
**Started**: October 17, 2025 ~5:21 PM  
**Status When Left**: Iteration 31, -114 rewards, 2,028 steps  
**Code Version**: ALL LATEST FIXES (16 commits from today)

---

## 🎯 Quick Status Check (Run These Commands)

```bash
# 1. Check if job is still running
oc exec zelda-rl-head-s9rdj -- ray job list | grep "raysubmit_FfgUftqiybM88FUX"

# 2. Get latest metrics
oc exec zelda-rl-head-s9rdj -- ray job logs raysubmit_FfgUftqiybM88FUX | \
  tail -500 | grep -E "iteration.*finished|episode_return_mean|episode_len_mean" | tail -10

# 3. Check for key milestones
oc exec zelda-rl-head-s9rdj -- ray job logs raysubmit_FfgUftqiybM88FUX | \
  grep -E "🌳 MILESTONE|🔑 MILESTONE|Gnarled Key" | tail -10
```

---

## ✅ Good Signs to Look For (After 3-4 Hours)

**If you see these, LET IT RUN for days:**

| Metric | Target | What It Means |
|--------|--------|---------------|
| **Episode Return** | >+1,000 | Rewards turned positive! ✅ |
| **Episode Length** | >5,000 steps | Learning survival ✅ |
| **Iteration** | ~200-300 | Good progress ✅ |

**Example good output:**
```
iteration 250 finished
│ episode_return_mean    +3,450.23 │  ← POSITIVE!
│ episode_len_mean       7,892     │  ← LONGER!
```

---

## ⚠️ Warning Signs (Might Need Adjustments)

**If you see these, consider stopping:**

| Metric | Bad Value | Problem |
|--------|-----------|---------|
| **Episode Return** | Still negative at iter 200+ | Not learning |
| **Episode Length** | <1,500 steps | Dying too fast |
| **Iteration** | <100 after 4 hours | Stuck/slow |

**Example bad output:**
```
iteration 180 finished
│ episode_return_mean    -250.45 │  ← STILL NEGATIVE
│ episode_len_mean       1,200   │  ← SHORT
```

---

## 📊 Expected Progress Timeline

### After 3-4 Hours (~150-200 iterations)

**Good Scenario:**
- Episode return: +500 to +2,000 (positive!)
- Episode length: 4,000-6,000 steps
- Status: Learning rapidly ✅

**Neutral Scenario:**
- Episode return: -50 to +500 (close to positive)
- Episode length: 2,500-4,000 steps
- Status: Slow but learning

**Bad Scenario:**
- Episode return: Still <-200
- Episode length: <2,000 steps
- Status: Not learning effectively

### After 12 Hours (~600-700 iterations)

**Expected (if trending well):**
- Episode return: +5,000-7,000
- Episode length: 8,000-10,000 steps (maxed!)
- Status: Approaching plateau ✅

### After 24 Hours (~1,200 iterations)

**Expected (if all goes well):**
- Episode return: +7,500-8,500 (plateau)
- Episode length: 10,000 steps (maxed!)
- Status: Fully trained ✅

---

## 🎯 Decision Matrix

### Scenario 1: Excellent Progress (Return >+2,000 after 4hrs)

**Action:** ✅ **LET IT RUN FOR DAYS!**
- Set a reminder to check daily
- Monitor for plateau (~+8,000 rewards)
- Stop after 2-3 days or when performance stops improving

### Scenario 2: Good Progress (Return 0 to +2,000 after 4hrs)

**Action:** ✅ **Continue for 12-24 hours**
- Check again at 12 hours
- If still improving, let run for days
- If plateaued low, consider adjustments

### Scenario 3: Slow Progress (Return <0 after 4hrs)

**Action:** ⚠️ **Consider adjustments**
- Movement reward might be too restrictive
- Could temporarily increase to 0.15 or 0.2
- Or: Be patient, might turn positive by iteration 150

### Scenario 4: Job Failed/Stopped

**Action:** 🔧 **Check logs and restart**
```bash
# Check status
oc exec zelda-rl-head-s9rdj -- ray job list | grep FfgUftqiybM88FUX

# If FAILED, check why
oc exec zelda-rl-head-s9rdj -- ray job logs raysubmit_FfgUftqiybM88FUX | tail -100
```

---

## 🔍 What to Look For in Logs

### Positive Indicators

```bash
# 1. Increasing rewards over time
grep "episode_return_mean" | tail -20
# Should see: -114 → -50 → +100 → +500 → ...

# 2. Increasing episode length
grep "episode_len_mean" | tail -20
# Should see: 2,028 → 3,000 → 5,000 → ...

# 3. LLM alignment bonuses
grep "MATCHES.*LLM" | wc -l
# Should see: hundreds or thousands

# 4. Item acquisition
grep "NEW ITEM OBTAINED" | head -20
# Should see: Feathers, Swords, Keys, etc.

# 5. Room discovery
grep "NEW ROOM DISCOVERED" | wc -l
# Should see: dozens per episode
```

### Concerning Indicators

```bash
# 1. Connection errors
grep "Connection refused\|LLM call failed" | wc -l
# Should see: 0 (no errors!)

# 2. Agent stuck
grep "POSITION STUCK.*steps" | tail -10
# Should see: 0.0 penalty (disabled)

# 3. Excessive menu surfing
grep "MENU SURFING" | wc -l
# Should be low (escalating penalties)
```

---

## 💡 If You Want to Adjust After Checking

### If Rewards Are Trending Positive But Slow

**Increase movement reward:**
```yaml
# In configs/env.yaml
movement: 0.15  # Increase from 0.1
```

### If Agent Still Getting Stuck

**Increase room discovery:**
```yaml
# In configs/env.yaml
new_room_discovery: 30.0  # Increase from 20.0
```

### If Episodes Are Too Short (Dying Fast)

**Reduce death penalty:**
```yaml
# In configs/env.yaml
death: -25.0  # Reduce from -50.0
```

---

## 📚 Reference Documents

**Before making changes, review:**
- `CURRENT_TRAINING_STATUS.md` - Previous run analysis
- `POSITION_TRACKING_VERIFICATION.md` - Position system verification
- `CHECKPOINT_GUIDE.md` - Checkpoint usage (when ready)
- `OVERNIGHT_RUN_GUIDE.md` - Long-term monitoring

---

## 🎉 Session Accomplishments Summary

**Today's fixes (16 commits):**
1. ✅ LLM connection (0% → 100%)
2. ✅ Entity detection (broken → working)
3. ✅ Position stuck penalty (disabled Y-bug workaround)
4. ✅ Extreme exploration emphasis
5. ✅ HUD single-session mode
6. ✅ HUD training data streaming
7. ✅ HUD epoch display fix
8. ✅ Smart menu management
9. ✅ HUD callback data fix
10. ✅ Checkpoint restoration feature
11. ✅ Inventory tracking (Gnarled Key detection!)
12. ✅ Movement reward fix (only when actually moving)
13. ✅ Architecture diagram enhanced
14. ✅ Position tracking verified
15. ✅ Checkpoint infrastructure (sessions bucket)
16. ✅ Ray checkpointing fix

**Previous run results:**
- Iteration 1,576 (20 hours)
- +8,335 rewards (peak!)
- 10,000 step episodes (mastered survival!)

---

## ⏰ When to Check Back

### In 3-4 Hours
- **Expect**: ~150-200 iterations
- **Look for**: Positive rewards
- **Decision**: Continue or adjust

### In 12 Hours
- **Expect**: ~600 iterations
- **Look for**: +5,000-7,000 rewards
- **Decision**: Let run for days or stop

### In 24 Hours
- **Expect**: ~1,200 iterations
- **Look for**: +7,500-8,500 rewards (plateau)
- **Decision**: Continue to full convergence

---

**Good luck! The agent is learning with all the latest fixes! 🎯🎮🚀**

*See you in a few hours! If performance looks good, let it run for days to reach full potential!*


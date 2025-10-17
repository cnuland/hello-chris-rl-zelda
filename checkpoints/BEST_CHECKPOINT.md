# 🏆 Best Checkpoint Configuration

**Training Run**: `raysubmit_iAkbELhTrRWYuLA2`  
**Date**: October 16-17, 2025  
**Performance**: +8,080 episode return, 10,000 step episodes

---

## 📊 Recommended Checkpoint

**Checkpoint**: `checkpoint_000800`  
**Iteration**: 800  
**Performance**:
- Episode Return: ~+8,000 (plateau)
- Episode Length: 10,000 steps (maxed)
- LLM Success: 100%
- Survival: Mastered

**Location on Cluster**:
```
/tmp/ray/session_2025-10-16_19-21-03_{session_id}/PPO_ZeldaOracleSeasons/PPO_zelda_env_4166c_00000/checkpoint_000800
```

**Why This Checkpoint?**:
- ✅ In stable plateau region (not early/unstable)
- ✅ High performance (+8,000 rewards)
- ✅ Not overtrained (iteration 800 vs 1400+)
- ✅ Proven stable across 600+ more iterations
- ✅ Good foundation for future experiments

---

## 🎯 Training Configuration Used

**When this checkpoint was created:**

### Environment
- Workers: 5
- Envs per worker: 6
- Total parallel envs: 30
- Episode length: 61,440 (max 10,000 achieved)
- Batch size: 4,096

### LLM Settings
- Endpoint: http://172.30.21.1:8000
- Text calls: 2% probability (~1/50 steps)
- Vision calls: 3% probability (~1/33 steps)
- Text alignment bonus: +5.0
- Vision alignment bonus: +50.0

### Rewards
- Movement: +0.1 per step
- Room discovery: +20.0 per unique room
- Grid exploration: +5.0 per unique square
- Maku Tree visited: +100.0
- Gnarled Key obtained: +200.0
- Dungeon entered: +150.0
- Menu usage: -0.5 (escalating for surfing)
- Item switch: +0.5
- Revisit penalty: -0.5
- Death penalty: -50.0
- Position stuck: 0.0 (disabled due to Y-bug)

### Skills Learned
- ✅ Exploration (10-14 unique rooms per episode)
- ✅ Survival (no premature deaths)
- ✅ LLM alignment (74% of rewards from bonuses)
- ✅ Strategic menu use (49% purposeful)
- ✅ Grid coverage (500+ squares)
- ✅ Maku Tree navigation

---

## 🚀 How to Use This Checkpoint

### In run-kuberay-zelda.ipynb Cell 11:

```python
env_vars = {
    # ... other settings ...
    
    # Use the best checkpoint:
    'RESTORE_CHECKPOINT': '/tmp/ray/session_2025-10-16_19-21-03_{session_id}/PPO_ZeldaOracleSeasons/PPO_zelda_env_4166c_00000/checkpoint_000800',
}
```

**Note**: The exact session_id needs to be filled in from the Ray cluster.

---

## 📝 Finding the Exact Path

To get the exact checkpoint path:

```bash
# List available checkpoints for this job
oc exec zelda-rl-head-s9rdj -- ls -la /tmp/ray/session_*/PPO_ZeldaOracleSeasons/PPO_zelda_env_*/checkpoint_000800 2>/dev/null

# Or check Ray dashboard for checkpoint paths
```

---

## 🔄 Alternative: Use Latest Successful Iteration

If checkpoint_000800 is not available, use the latest stable iteration:

- `checkpoint_001000` (if run continued)
- `checkpoint_001400` (latest from overnight)
- Any iteration 600-1000 in the plateau region

---

## ⚠️ Important Notes

**What This Checkpoint Provides:**
- ✅ Trained exploration policy
- ✅ Survival skills (10k step episodes)
- ✅ LLM alignment behavior
- ✅ Strategic menu usage
- ✅ Grid exploration patterns

**Limitations (Known Bugs):**
- ⚠️ Y-coordinate reading still broken (stuck at 0)
- ⚠️ Position stuck penalty disabled (workaround)
- ✅ All other systems working perfectly

**Future Experiments:**
Starting from this checkpoint, you can:
1. Add dungeon entry objectives
2. Add combat rewards
3. Increase episode length to 20,000+
4. Scale workers to 25-50
5. Adjust LLM call frequencies
6. Add item collection objectives

---

*This checkpoint represents the best stable performance from the October 16-17, 2025 training run*


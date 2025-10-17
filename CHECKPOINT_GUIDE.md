# 🔄 Checkpoint Restoration Guide

## Overview

You can now **continue training from a previous checkpoint** instead of starting from scratch. This is useful for:
- Iterating on reward structures without losing progress
- Adjusting LLM settings while keeping trained policy
- Scaling up/down workers while maintaining model weights
- Recovering from interrupted training runs

---

## 🎯 Quick Start

### Option 1: Start Fresh Training (Default)

```python
# In run-kuberay-zelda.ipynb Cell 11:
env_vars = {
    # ... other config ...
    'RESTORE_CHECKPOINT': '',  # Empty = start fresh
}
```

### Option 2: Continue from Checkpoint

```python
# In run-kuberay-zelda.ipynb Cell 11:
env_vars = {
    # ... other config ...
    'RESTORE_CHECKPOINT': '/tmp/ray/session_2025-10-16.../PPO_zelda_env_abc123_00000/checkpoint_000150',
}
```

---

## 📂 Finding Checkpoint Paths

### Method 1: From Ray Dashboard

1. Open Ray Dashboard (link from notebook Cell 11)
2. Navigate to **Jobs** → Your training job
3. Click **"View Checkpoints"**
4. Copy checkpoint path (e.g., `checkpoint_000150`)

### Method 2: From Job Logs

```bash
# Get checkpoint locations from logs:
oc exec zelda-rl-head-s9rdj -- ray job logs <job_id> | grep "checkpoint"
```

### Method 3: From MinIO/S3

Checkpoints are saved to:
- **Worker checkpoints**: `s3://sessions/ray_training_{timestamp}/worker_{id}/episode_{num}_checkpoint.json`
- **Ray checkpoints**: `/tmp/ray/session_{timestamp}/...`

---

## 🔧 How to Use

### Step 1: Complete a Training Run

Run your first training job normally:
```python
'RESTORE_CHECKPOINT': '',  # Start fresh
```

Training will create checkpoints automatically.

### Step 2: Find Latest Checkpoint

After training runs for a while (e.g., 150 iterations), find the checkpoint path:
- Look in Ray dashboard
- Or check logs for checkpoint saves

Example checkpoint path:
```
/tmp/ray/session_2025-10-16_19-21-03/PPO_ZeldaOracleSeasons/PPO_zelda_env_4166c_00000/checkpoint_000150
```

### Step 3: Resume Training

In the notebook, update Cell 11:

```python
env_vars = {
    # ... keep all other settings the same ...
    
    # Add checkpoint path:
    'RESTORE_CHECKPOINT': '/tmp/ray/session_.../checkpoint_000150',
}
```

Run Cell 11 + Cell 14 to submit the job.

### Step 4: Verify Restoration

Check the job logs for:
```
🔄 RESTORING from checkpoint: /tmp/ray/.../checkpoint_000150
```

Training will continue from iteration 150 instead of starting at 0!

---

## 🎯 Use Cases

### 1. Tune Rewards While Keeping Policy

**Scenario**: Agent learned exploration, but you want to adjust room discovery reward

```python
# First run (150 iterations):
'RESTORE_CHECKPOINT': '',
'new_room_discovery': 20.0

# After adjusting reward:
'RESTORE_CHECKPOINT': '/tmp/ray/.../checkpoint_000150',
'new_room_discovery': 30.0  # ← Changed!
```

**Result**: Agent keeps exploration skills, but learns new reward weighting

### 2. Add New Objectives

**Scenario**: Agent learned exploration, now add dungeon entry rewards

```python
# Continue from checkpoint:
'RESTORE_CHECKPOINT': '/tmp/ray/.../checkpoint_000200',

# In configs/env.yaml, add:
dungeon_entered: 200.0  # New reward!
```

**Result**: Agent uses existing skills + learns new objective

### 3. Scale Up Workers

**Scenario**: Tested with 5 workers, now scale to 25

```python
# Continue training:
'RESTORE_CHECKPOINT': '/tmp/ray/.../checkpoint_000100',
'RAY_WORKERS': '25',  # ← Scaled up from 5
```

**Result**: Faster training with same learned policy

### 4. Adjust LLM Settings

**Scenario**: Change LLM call frequency while keeping policy

```python
# Continue from checkpoint:
'RESTORE_CHECKPOINT': '/tmp/ray/.../checkpoint_000150',

# In configs/env.yaml, adjust:
llm_text_probability: 0.05  # ← Changed from 0.02
```

**Result**: Agent keeps skills, gets more LLM guidance

---

## ⚠️ Important Notes

### What Gets Restored

✅ **Policy weights** (learned behavior)  
✅ **Value function** (reward predictions)  
✅ **Optimizer state** (learning momentum)  
✅ **Training iteration** (continues from checkpoint number)

### What Doesn't Get Restored

❌ **Episode count** (resets to 0 per worker)  
❌ **Exploration sets** (visited rooms, grid squares - resets)  
❌ **LLM suggestion cache** (resets)  
❌ **HUD session** (new session)

### Compatibility

- ✅ Can change: Rewards, LLM settings, worker count, episode length
- ⚠️ Cannot change: Observation space, action space, model architecture
- ❌ Will break: Changing observation/action dimensions

---

## 📝 Best Practices

### 1. Keep Checkpoints Organized

Use meaningful checkpoint names:
```python
# After 200 iterations with good exploration:
# Save this as: 'exploration_mastered_iter200'
```

### 2. Test Changes Locally First

Before deploying checkpoint restoration:
1. Test reward changes with `make local-visual`
2. Verify no breaking changes
3. Then deploy to cluster with checkpoint

### 3. Document Checkpoint Configs

Keep notes on what config each checkpoint used:
```
checkpoint_000150: 
  - 5 workers, 30 envs
  - new_room_discovery: 20.0
  - LLM text: 2%, vision: 3%
  - Episode return: +4,000
```

### 4. Incremental Changes

When restoring:
- Change ONE thing at a time
- Test for 20-50 iterations
- Verify improvement before next change

---

## 🚀 Example Workflow

### Training Session 1: Exploration Focus
```python
'RESTORE_CHECKPOINT': '',  # Fresh start
'new_room_discovery': 20.0
'movement': 0.1
```
→ Runs to iteration 200, achieves +5,000 rewards

### Training Session 2: Add Combat Skills
```python
'RESTORE_CHECKPOINT': '/tmp/ray/.../checkpoint_000200',
# In env.yaml, add:
'enemy_defeat_reward': 10.0  # New!
```
→ Continues from 200, learns combat while keeping exploration

### Training Session 3: Dungeon Focus
```python
'RESTORE_CHECKPOINT': '/tmp/ray/.../checkpoint_000350',
# In env.yaml, add:
'dungeon_entered': 200.0  # New objective!
```
→ Continues from 350, learns dungeon entry

---

## 📊 Checkpoint Locations

### Worker Checkpoints (Session Manager)
- **Path**: `s3://sessions/ray_training_{timestamp}/worker_{id}/episode_{num}_checkpoint.json`
- **Format**: JSON metadata
- **Contains**: Episode stats, not model weights
- **Use for**: Analysis, debugging

### Ray Checkpoints (RLlib)
- **Path**: `/tmp/ray/session_{timestamp}/PPO_ZeldaOracleSeasons/PPO_zelda_env_{id}/checkpoint_{iter}`
- **Format**: Ray/RLlib binary format
- **Contains**: Full model state, optimizer, config
- **Use for**: Training restoration ← **USE THIS!**

---

## 🛠️ Troubleshooting

### Issue: Checkpoint Not Found

**Error**: `Checkpoint path does not exist`

**Solution**: 
- Verify path is from Ray dashboard
- Check if session still exists on head node
- Path must be accessible from Ray head

### Issue: Config Mismatch

**Error**: `Config incompatible with checkpoint`

**Solution**:
- Keep observation/action space same
- Only change rewards, not environment structure
- Model architecture must match

### Issue: Performance Degradation

**Symptom**: Rewards decrease after restoration

**Solution**:
- Checkpoint may be from earlier, worse iteration
- Use a later checkpoint (higher iteration number)
- Or start fresh if config changes are too large

---

## 📚 Related Documentation

- `CURRENT_TRAINING_STATUS.md` - Current run analysis
- `CLUSTER_TRAINING_ANALYSIS.md` - 40+ hour training analysis
- `run-kuberay-zelda.ipynb` - Notebook with checkpoint config
- `run-ray-zelda.py` - Training script with restore logic

---

*Updated: October 17, 2025 - Checkpoint restoration feature added*


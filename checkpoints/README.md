# 💾 Checkpoints Directory

This directory contains **checkpoint configuration and documentation**, not the actual checkpoint files (which are too large for Git).

---

## 📂 Contents

- **`checkpoint_config.yaml`** - Configuration for checkpoint restoration
- **`BEST_CHECKPOINT.md`** - Documentation of the recommended checkpoint
- **`README.md`** - This file

**Actual checkpoint files** (`.pkl`, `.pt`, `.pth`) are **excluded from Git** and live on the Ray cluster.

---

## 🎯 Using Checkpoints

### Step 1: Find Checkpoint on Cluster

```bash
# List available checkpoints
oc exec zelda-rl-head-s9rdj -- ls /tmp/ray/session_*/PPO_ZeldaOracleSeasons/PPO_zelda_env_*/checkpoint_*
```

### Step 2: Update Checkpoint Config

Edit `checkpoint_config.yaml`:
```yaml
enable_restore: true  # Enable restoration
checkpoint:
  iteration: 800
  path_template: '/tmp/ray/.../checkpoint_000800'  # Fill in actual path
```

### Step 3: Update Notebook

In `run-kuberay-zelda.ipynb` Cell 11:
```python
# Load checkpoint config
import yaml
with open('checkpoints/checkpoint_config.yaml') as f:
    ckpt_config = yaml.safe_load(f)

# Set restore path if enabled
if ckpt_config.get('enable_restore', False):
    restore_path = ckpt_config['checkpoint']['path_template']
    env_vars['RESTORE_CHECKPOINT'] = restore_path
else:
    env_vars['RESTORE_CHECKPOINT'] = ''
```

### Step 4: Deploy

Run Cell 11 + Cell 14 to deploy with checkpoint restoration.

---

## 📊 Current Best Checkpoint

**Source**: Training run `raysubmit_iAkbELhTrRWYuLA2`  
**Date**: October 16-17, 2025  
**Recommended**: `checkpoint_000800`  
**Performance**: +8,000 rewards, 10,000 steps, 100% LLM success

See `BEST_CHECKPOINT.md` for complete details.

---

## ⚠️ Important Notes

**Checkpoint files are NOT in Git because:**
- They are 100s of MB (too large)
- They are cluster-specific (paths change)
- They live in `/tmp/ray/` on Ray head node
- They must be accessed directly from cluster

**What IS in Git:**
- ✅ Checkpoint documentation
- ✅ Checkpoint configuration
- ✅ Performance metrics
- ✅ Training settings used

**To actually use a checkpoint:**
1. Checkpoint must exist on the Ray cluster
2. Path must be accessible from Ray head node
3. Use exact path in RESTORE_CHECKPOINT env var

---

*Checkpoints are managed on the Ray cluster, documented here for reference*


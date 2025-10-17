# 📥 How to Download and Save Best Checkpoint

You're right - we should save the best checkpoint to Git so it can be loaded anywhere!

---

## 🎯 Quick Guide

### Step 1: Login to OpenShift (if session expired)

```bash
oc login --token=<your-token> --server=https://api.rosa-58cx6.acrs.p3.openshiftapps.com:443
oc project zelda-hybrid-rl-llm
```

### Step 2: Find the Checkpoint Directory

```bash
# Find the Ray session directory for our job
oc exec zelda-rl-head-s9rdj -- ls -la /tmp/ray/ | grep "2025-10-16"

# Should show something like:
# session_2025-10-16_19-21-03_979469_1
```

### Step 3: List Available Checkpoints

```bash
# Replace {session_id} with actual session from Step 2
oc exec zelda-rl-head-s9rdj -- ls -la /tmp/ray/session_{session_id}/PPO_ZeldaOracleSeasons/

# Should show:
# PPO_zelda_env_4166c_00000/
```

```bash
# List checkpoints
oc exec zelda-rl-head-s9rdj -- ls /tmp/ray/session_{session_id}/PPO_ZeldaOracleSeasons/PPO_zelda_env_4166c_00000/ | grep checkpoint

# Should show:
# checkpoint_000100
# checkpoint_000200
# ...
# checkpoint_000800  ← RECOMMENDED
# ...
# checkpoint_001400
```

### Step 4: Download Best Checkpoint

```bash
# Download checkpoint_000800 (recommended stable checkpoint)
oc cp zelda-rl-head-s9rdj:/tmp/ray/session_{session_id}/PPO_ZeldaOracleSeasons/PPO_zelda_env_4166c_00000/checkpoint_000800 \
  ./checkpoints/checkpoint_000800 \
  -c ray-head
```

**Note**: This might take a few minutes (checkpoint is ~50-200 MB)

### Step 5: Verify Download

```bash
# Check file size
ls -lh checkpoints/checkpoint_000800

# Should see a directory with files like:
# algorithm_state.pkl
# policies/
# learner/
# etc.
```

### Step 6: Update Config

Edit `checkpoints/checkpoint_config.yaml`:
```yaml
enable_restore: true
checkpoint:
  iteration: 800
  path_template: 'checkpoints/checkpoint_000800'  # Local path!
```

### Step 7: Update Notebook to Use Local Checkpoint

Edit `run-kuberay-zelda.ipynb` Cell 11:
```python
# Check if we have a local checkpoint to upload
import yaml
import os

checkpoint_path = ''
if os.path.exists('checkpoints/checkpoint_config.yaml'):
    with open('checkpoints/checkpoint_config.yaml') as f:
        ckpt_config = yaml.safe_load(f)
    
    if ckpt_config.get('enable_restore', False):
        # Use local checkpoint (will be uploaded with working_dir)
        local_ckpt = ckpt_config['checkpoint']['path_template']
        if os.path.exists(local_ckpt):
            checkpoint_path = local_ckpt
            print(f"✅ Will restore from local checkpoint: {local_ckpt}")

env_vars = {
    # ... other settings ...
    'RESTORE_CHECKPOINT': checkpoint_path,  # Will be uploaded to cluster
}
```

### Step 8: Commit and Push

```bash
# Add checkpoint files to Git
git add checkpoints/
git add .gitignore

# Commit
git commit -m "Add best checkpoint from overnight run (iteration 800, +8000 rewards)"

# Push
git push origin main
```

---

## 🔄 How It Works After Pushing

1. **Local Development**:
   - Checkpoint is in `checkpoints/checkpoint_000800`
   - Anyone who clones repo gets the checkpoint
   - Can be used for local inference or continued training

2. **Cluster Deployment**:
   - Ray's `working_dir` uploads `checkpoints/` to cluster
   - Checkpoint becomes available at `./checkpoints/checkpoint_000800`
   - `RESTORE_CHECKPOINT` env var points to it
   - Ray loads checkpoint and continues training

3. **Version Control**:
   - Checkpoint tracked in Git
   - Team members can use same trained model
   - Experiments start from proven baseline

---

## 📊 Recommended Checkpoint

**Best**: `checkpoint_000800`

**Why?**:
- Stable plateau performance (+8,000 rewards)
- Proven across 600+ more iterations
- Not overtrained
- Mastered survival and exploration
- 100% LLM success rate
- Strategic menu use learned

**Alternatives**:
- `checkpoint_000600`: Earlier, slightly lower performance
- `checkpoint_001000`: Later, marginal improvement
- `checkpoint_001400`: Latest, but may be overtrained

---

## ⚠️ Checkpoint Size Considerations

**Ray checkpoints are typically**:
- 50-200 MB depending on model size
- Mostly model weights and optimizer state
- Git can handle this if it's a single checkpoint
- GitHub has 100 MB file size limit per file

**If checkpoint is too large (>100 MB)**:
- Git LFS can handle it: `git lfs track "checkpoints/*.pkl"`
- Or: Keep on cluster only, document path
- Or: Use model compression/quantization

**Check size first**:
```bash
oc exec zelda-rl-head-s9rdj -- du -sh /tmp/ray/session_*/PPO_*/PPO_*/checkpoint_000800
```

---

## ✅ Summary

Yes, you're right! Checkpoints SHOULD be saved to Git so:
- ✅ Anyone can use the trained model
- ✅ Experiments start from proven baseline
- ✅ Model is versioned and reproducible
- ✅ No need to retrain from scratch

I've updated .gitignore to allow checkpoint files.
Now just download from cluster and commit! 🎯


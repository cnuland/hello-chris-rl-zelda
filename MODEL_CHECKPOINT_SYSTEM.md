# 💾 Model Checkpoint System

**Date:** October 19, 2025  
**Status:** ✅ Model weight saving enabled  
**Storage:** MinIO S3 `sessions` bucket

## 🎯 What Changed

### Before:
SessionManager saved **episode metadata only**:
- Episode statistics (steps, rewards)
- JSON files (~200 bytes)
- ❌ No model weights

### After:
Callback now saves **actual model weights** every 50 iterations:
- PyTorch state dict (network weights)
- Pickled binary files (~50-500MB each)
- ✅ Can resume training or run inference!

## 📝 Implementation

### File: `ray_hud_callback.py`

**Added to `__init__`:**
```python
self._session_manager = None
self._last_checkpoint_iter = 0
self.CHECKPOINT_FREQUENCY = 50  # Save every 50 iterations
```

**Added method:**
```python
def _save_model_checkpoint(self, algorithm, iteration, result):
    # Get model weights from algorithm
    model_weights = algorithm.get_policy().get_weights()
    
    # Serialize to bytes
    model_bytes = pickle.dumps(model_weights)
    
    # Save via SessionManager
    self._session_manager.save_checkpoint(
        worker_id=0,  # Driver/trainer
        episode_num=iteration,
        checkpoint_data=metadata,
        model_state=model_bytes  # ACTUAL MODEL WEIGHTS!
    )
```

**Triggered from `on_train_result`:**
- Runs every 50 iterations
- Saves from driver/trainer process
- Has access to full model state

## 📦 Checkpoint Structure

Each checkpoint includes TWO files:

### 1. Metadata (JSON):
```
s3://sessions/ray_training_XXXXX/worker_0/episode_000050_checkpoint.json
```
Contains:
- iteration number
- timesteps_total
- episode_return_mean
- episode_len_mean  
- timestamp

### 2. Model Weights (PyTorch):
```
s3://sessions/ray_training_XXXXX/worker_0/episode_000050_model.pth
```
Contains:
- Pickled PyTorch state_dict
- All network weights and biases
- Can be loaded for inference or training continuation

## 🚀 How to Use Checkpoints

### Download from MinIO:
```bash
# List available checkpoints
aws --endpoint-url=https://minio-api-route-minio-system.apps.rosa.rosa-58cx6.acrs.p3.openshiftapps.com \
  --profile minio s3 ls s3://sessions/ray_training_XXXXX/worker_0/

# Download specific checkpoint
aws --endpoint-url=https://minio-api-route-minio-system.apps.rosa.rosa-58cx6.acrs.p3.openshiftapps.com \
  --profile minio s3 cp s3://sessions/ray_training_XXXXX/worker_0/episode_000200_model.pth \
  ./checkpoints/best_model.pth
```

### Load for Inference:
```python
import pickle

# Load model weights
with open('checkpoints/best_model.pth', 'rb') as f:
    model_weights = pickle.load(f)

# Apply to policy
policy.set_weights(model_weights)

# Run inference
obs = env.reset()
action = policy.compute_single_action(obs)
```

## 📊 Checkpoint Selection Strategy

To find the best checkpoint:

1. **List all checkpoints** for a session
2. **Check metadata** files for episode_return_mean
3. **Download the iteration** with highest return
4. **Use for demo/inference**

Example:
```bash
# Find best checkpoint (highest return)
aws s3 --endpoint-url=... cp s3://sessions/ray_training_XXXXX/worker_0/ ./temp/ --recursive
grep -h "episode_return_mean" temp/*.json | sort -t: -k2 -n | tail -1
```

## 🎯 For Your Demo

### Timeline:
- **Today:** Start new training run with model weight saving
- **Day 1:** Checkpoints save every 50 iterations
- **Day 2:** Download checkpoint with best performance
- **Day 3:** Test locally with make local-visual + checkpoint
- **Day 4-5:** Demo ready!

### Demo Setup:
1. Download best checkpoint from MinIO
2. Modify `run_local_visual.py` to load checkpoint
3. Run `make local-visual`
4. Show trained agent playing Zelda with LLM guidance!

## 📋 Next Steps

1. ✅ Code committed and pushed (commit 7ba3f40)
2. ⏭️  Start new training run (re-run Cell 11 in notebook)
3. ⏭️  Monitor checkpoint saves (every 50 iterations)
4. ⏭️  Download best checkpoint after 200+ iterations
5. ⏭️  Use for demo inference

## 🔧 Technical Details

**Checkpoint Frequency:** 50 iterations  
**File Size:** ~50-500MB per checkpoint (depends on model size)  
**Retention:** All checkpoints kept (manually delete old ones if needed)  
**Format:** Python pickle (PyTorch compatible)  
**Location:** s3://sessions/ray_training_<TIMESTAMP>/worker_0/

---

**Ready to start new training run with model weight saving!** 🚀

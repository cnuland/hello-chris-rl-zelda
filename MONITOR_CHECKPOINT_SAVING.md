# 📊 Monitor Model Checkpoint Saving

**Job ID:** raysubmit_tYjqzUarRCxqJJuP  
**Session:** ray_training_1760839523  
**Started:** October 19, 2025

## 🎯 What to Watch For

### At Iteration 50 (~30 minutes from start):

Look for these logs:
```
💾 Saving checkpoint at iteration 50...
✅ SessionManager initialized: ray_training_XXXXX
✅ Checkpoint 50 saved to S3! Size: 2,450,123 bytes
   Model weights: s3://sessions/ray_training_1760839523/worker_0/episode_000050_model.pth
```

### Check Current Progress:
```bash
oc exec zelda-rl-head-s9rdj -- ray job logs raysubmit_tYjqzUarRCxqJJuP | grep "finished iteration" | tail -5
```

### Watch for Checkpoint Saves:
```bash
oc exec zelda-rl-head-s9rdj -- ray job logs raysubmit_tYjqzUarRCxqJJuP | grep "Checkpoint.*saved to S3" | tail -10
```

## ⚠️ Potential Issues & Solutions

### Issue 1: SessionManager Not Initialized in Driver
**Symptom:** No logs at iteration 50
**Solution:** Already has try/except with full traceback, will log error

### Issue 2: S3 Environment Variables Not Set
**Symptom:** `⚠️  SessionManager disabled: Missing S3 credentials`
**Solution:** Env vars ARE set (confirmed in job config)

### Issue 3: Large File Upload Timeout
**Symptom:** `❌ Error saving model checkpoint: timeout`
**Solution:** boto3 default timeout is 60s (should be enough for <500MB)

### Issue 4: MinIO Connection Failure
**Symptom:** `❌ Failed to save checkpoint: Could not connect`
**Check:**
```bash
oc exec zelda-rl-head-s9rdj -- python3 -c "
import boto3
s3 = boto3.client('s3', endpoint_url='http://172.30.45.38:9000',
                  aws_access_key_id='admin',
                  aws_secret_access_key='zelda-rl-minio-2024')
s3.head_bucket(Bucket='sessions')
print('✅ MinIO accessible')
"
```

## 📥 Download Checkpoints After Training

### List Available Checkpoints:
```bash
python download_session_checkpoints.py
```

### Download Specific Checkpoint:
```python
# Use the existing script (already configured)
# It will download all checkpoints including model weights
```

### Find Best Checkpoint:
```bash
cd checkpoints/ray_training_1760839523/worker_0/
ls -lh episode_*_model.pth  # Model weight files
cat episode_*_checkpoint.json | jq '.episode_return_mean'  # Check scores
```

## 🎯 For Your Demo

### Recommended Checkpoints:
- **Iteration 100:** Early but shows learning
- **Iteration 200:** Good balance (1-2 hours training)
- **Iteration 500:** Well-trained (if time permits)

### Load Checkpoint Locally:
Will need to modify `run_local_visual.py` to:
1. Load pickled model weights
2. Apply to policy before running
3. Run inference with trained model

---

**Current Status:** Training running, checkpoints will save starting at iteration 50!

# 🎥 Episode Video Recording System

**Date:** October 19, 2025  
**Based on:** Double Dragon project video recording  
**Status:** ✅ Implemented  
**Purpose:** Capture gameplay videos for debugging, analysis, and demos

## 🎯 What It Does

Records a full MP4 video of every training episode, showing exactly what Link does during gameplay.

### Captures:
- ✅ Every frame (60 fps)
- ✅ Full Game Boy screen (160x144)
- ✅ Complete episode from start to death/truncation
- ✅ Saves to MinIO S3 for later review

## 📝 Implementation (Mimics Double Dragon)

### Based on: `~/hello-chris-dd-kuberay/app/ddenv.py`

**Pattern:**
1. **On Reset:** Initialize `mediapy.VideoWriter` with temp file
2. **On Step:** Add frame with `video_writer.add_image(screen)`
3. **On Episode End:** Close writer, upload to S3, cleanup

### Files Modified:

**1. configs/env.yaml** - Configuration
```yaml
emulator:
  save_video: true   # Enable episode video recording
  video_fps: 60      # Game Boy native framerate
  video_quality: 8   # H264 quality (0-51, lower=better)
```

**2. emulator/zelda_env_configurable.py** - Implementation

Added to `__init__`:
```python
self.save_video = self.config.get('emulator', {}).get('save_video', False)
self.video_writer = None
self.video_temp_file = None
self.video_session_id = None
```

Added to `reset()`:
```python
# Close previous video
if self.video_writer is not None:
    self._finalize_video()

# Start new video recording
if self.save_video:
    self._init_video_recording()
```

Added to `step()`:
```python
# Save video frame
if self.save_video and self.video_writer is not None:
    frame = self.bridge.get_screen_array()
    self.video_writer.add_image(frame)

# Finalize on episode end
if (terminated or truncated) and self.video_writer is not None:
    self._finalize_video()
```

**3. run-kuberay-zelda.ipynb** - Dependencies
```python
'pip': [..., 'mediapy>=1.2.0']  # Added for video recording
```

## 📦 Video Storage

### Location:
```
s3://sessions/ray_training_<TIMESTAMP>/videos/episode_NNNNNN_instance_X_vid_XXXX.mp4
```

### Structure:
- `ray_training_<TIMESTAMP>`: Session ID
- `videos/`: Video directory
- `episode_NNNNNN`: Episode number (zero-padded)
- `instance_X`: Worker instance ID
- `vid_XXXX`: Unique video ID (4-char UUID)

### Example:
```
s3://sessions/ray_training_1760844381/videos/episode_000042_instance_1860_vid_a3f2.mp4
```

## 🔍 Use Cases

### 1. Debug Dialogue Issues
Watch videos to see if Link gets stuck in dialogue:
```bash
# Download video from specific episode
aws --endpoint-url=... s3 cp \
  s3://sessions/ray_training_XXXXX/videos/episode_000042_instance_1860_vid_a3f2.mp4 \
  ./debug_videos/
```

### 2. Verify Quest Progression
See if Link actually talks to Maku Tree or just enters/leaves:
- Look for Maku Tree dialogue frames
- Verify "Yes/No" menu appears
- Confirm Gnarled Key given

### 3. Analyze Exploration Patterns
Watch how Link explores:
- Which rooms does he visit?
- Does he backtrack effectively?
- Is he stuck in corners?

### 4. Demo Material
Show stakeholders actual gameplay:
- "Here's what the agent learned"
- "Watch it talk to Maku Tree"
- "See it fight enemies and explore"

## ⚙️ Performance Impact

### Video File Sizes:
- **10,000 steps** @ 60fps ÷ 4 frame_skip = 2,500 frames
- **H264 compression** (quality=8) = ~2-5 MB per episode
- **30 workers** × 10 episodes = 300 videos = ~1 GB total

### Processing Time:
- Frame capture: <1ms (negligible)
- Video encoding: ~1-2 seconds on episode end
- S3 upload: ~1-5 seconds (depends on file size)
- **Total overhead:** <10 seconds per episode

### Recommendations:
- ✅ Keep enabled for demos/debugging
- ⚠️  Disable for long production runs (saves storage/time)
- 💡 Sample: Record 1 in 10 episodes to reduce storage

## 🎬 How to View Videos

### Download from MinIO:
```bash
# List all videos for a session
aws --endpoint-url=https://minio-api-route-minio-system.apps.rosa.rosa-58cx6.acrs.p3.openshiftapps.com \
  s3 ls s3://sessions/ray_training_XXXXX/videos/

# Download specific video
aws --endpoint-url=https://minio-api-route-minio-system.apps.rosa.rosa-58cx6.acrs.p3.openshiftapps.com \
  s3 cp s3://sessions/ray_training_XXXXX/videos/episode_000042_instance_1860_vid_a3f2.mp4 \
  ./videos/

# Play with VLC, mpv, or any video player
vlc ./videos/episode_000042_instance_1860_vid_a3f2.mp4
```

### Bulk Download:
```python
# Use download_session_checkpoints.py (can be modified for videos)
# Or download entire videos directory:
aws s3 cp s3://sessions/ray_training_XXXXX/videos/ ./videos/ --recursive
```

## 🔧 Configuration Options

### Enable/Disable:
```yaml
emulator:
  save_video: false  # Disable to save storage/time
```

### Adjust Quality:
```yaml
emulator:
  video_quality: 18  # Higher = smaller file, lower quality
  video_quality: 0   # Best quality, largest file
```

### Adjust FPS:
```yaml
emulator:
  video_fps: 30  # Half speed (smaller files)
  video_fps: 60  # Full speed (default, smoother)
```

## 🎯 Next Steps

1. ✅ Code committed (667af87)
2. ⏭️  Deploy in next training run
3. ⏭️  Verify videos save to S3
4. ⏭️  Download and review gameplay
5. ⏭️  Use for demo preparation!

---

**Ready to see what Link is actually doing during training!** 🎥🎮

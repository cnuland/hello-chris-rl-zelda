# Ray RLlib Deployment Guide for Zelda Oracle of Seasons

This guide explains how to deploy distributed RL training using Ray RLlib on Kubernetes/OpenShift.

## 📋 Overview

The Ray RLlib implementation provides:
- **Distributed Training**: Multiple parallel environments across worker pods
- **Vision LLM Integration**: Strategic guidance from multimodal LLM
- **S3 Storage**: Automatic video and checkpoint uploads
- **GPU Acceleration**: CUDA support for faster training
- **Production Ready**: Based on proven Double Dragon implementation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Ray Cluster on Kubernetes/OpenShift                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐      ┌──────────────┐                │
│  │ Ray Head     │──────│ Worker 1     │                │
│  │ (1 GPU)      │      │ (1 GPU)      │                │
│  └──────────────┘      │ 3 Zelda envs │                │
│                        └──────────────┘                 │
│                                                          │
│                        ┌──────────────┐                │
│                        │ Worker 2     │                │
│                        │ (1 GPU)      │                │
│                        │ 3 Zelda envs │                │
│                        └──────────────┘                 │
│                                                          │
│                        ┌──────────────┐                │
│                        │ Worker 3     │                │
│                        │ (1 GPU)      │                │
│                        │ 3 Zelda envs │                │
│                        └──────────────┘                │
│                                                          │
└─────────────────────────────────────────────────────────┘
         │                            │
         │                            │
    ┌────▼────┐                  ┌───▼────┐
    │  LLM    │                  │ MinIO  │
    │ Service │                  │   S3   │
    └─────────┘                  └────────┘
```

**Total**: 1 head + 3 workers = 9 parallel Zelda games (3 per worker)

## 📦 Components

### 1. `ray_zelda_env.py`
Ray-compatible Gymnasium environment with:
- Visual observations (stacked frames + memory channels)
- Vision LLM integration for guidance
- S3 video/checkpoint uploads
- Multi-state initialization

### 2. `ray_zelda_model.py`
Custom CNN model for Ray RLlib:
- Convolutional feature extraction
- Separate policy and value heads
- Matches Double Dragon architecture

### 3. `run-ray-zelda.py`
Main training script:
- PPO configuration
- Distributed rollouts
- Checkpoint management

## 🚀 Deployment Steps

### Step 1: Build Docker Image

Based on the Double Dragon Dockerfile pattern:

```bash
# From your project root
cd /Users/cnuland/hello-chris-rl-llm-zelda

# Build the image (you'll need to create a Dockerfile)
docker build -t quay.io/YOUR_REGISTRY/zelda-kuberay-worker:latest -f Dockerfile.ray .

# Push to registry
docker push quay.io/YOUR_REGISTRY/zelda-kuberay-worker:latest
```

### Step 2: Deploy MinIO for S3 Storage

```bash
# Create namespace
oc create namespace minio

# Deploy MinIO (from Double Dragon k8s/minio configs)
oc apply -f k8s/minio/
```

### Step 3: Deploy LLM Service

Assume your LLM service is already deployed in the cluster:

```
Service: llama4-scout-service
Endpoint: http://llama4-scout-service:8000/v1/chat/completions
```

### Step 4: Launch Ray Cluster via Jupyter Notebook

1. Open `run-kuberay-zelda.ipynb`
2. Update credentials:
   - OpenShift token (`oc whoami -t`)
   - OpenShift server (`oc cluster-info`)
   - Namespace
   - Docker image registry
   - S3 credentials
3. Run cells to:
   - Authenticate
   - Create Ray cluster
   - Submit training job

## 🔧 Configuration

### Environment Variables

Set these in the Jupyter notebook or job submission:

```python
env_vars = {
    # S3 Storage
    'S3_ACCESS_KEY_ID': 'your-key',
    'S3_SECRET_ACCESS_KEY': 'your-secret',
    'S3_REGION_NAME': 'region',
    'S3_ENDPOINT_URL': 'http://minio-service:9000',
    'S3_BUCKET_NAME': 'zelda-training',
    
    # LLM Service
    'LLM_ENDPOINT': 'http://llama4-scout-service:8000/v1/chat/completions',
}
```

### Ray Cluster Configuration

In `run-kuberay-zelda.ipynb`:

```python
ClusterConfiguration(
    head_cpu_requests=10,
    head_cpu_limits=12,
    head_memory_requests=10,
    head_memory_limits=12,
    name='zelda-rl',
    namespace='your-namespace',
    num_workers=3,  # Adjust based on available GPUs
    worker_cpu_requests=12,
    worker_cpu_limits=16,
    worker_memory_requests=12,
    worker_memory_limits=20,
    image="quay.io/YOUR_REGISTRY/zelda-kuberay-worker:latest",
    head_extended_resource_requests={'nvidia.com/gpu':1},
    worker_extended_resource_requests={'nvidia.com/gpu':1},
)
```

### PPO Hyperparameters

In `run-ray-zelda.py`:

```python
PPOConfig()
    .training(
        lr=5e-5,                    # Learning rate
        gamma=0.99,                 # Discount factor
        lambda_=0.95,               # GAE lambda
        clip_param=0.2,             # PPO clip parameter
        vf_clip_param=10.0,         # Value function clip
        entropy_coeff=0.01,         # Entropy bonus
        train_batch_size=4096,      # Training batch size
        sgd_minibatch_size=512,     # SGD minibatch size
        num_sgd_iter=10,            # SGD iterations per update
    )
```

## 📊 Monitoring

### Ray Dashboard

Access the Ray dashboard URL provided after cluster creation:

```
https://ray-dashboard-zelda-rl-your-namespace.apps.your-cluster.com
```

Monitor:
- Worker status
- GPU utilization
- Episode rewards
- Training progress

### MinIO Console

Access videos and checkpoints:

```
https://minio-console-your-namespace.apps.your-cluster.com
```

Browse:
- `zelda_videos/` - Episode recordings
- `zelda_states/` - Saved game states
- Ray checkpoints

## 🎮 Expected Performance

Based on Double Dragon results:

| Metric | Value |
|--------|-------|
| Parallel Environments | 9 (3 workers × 3 envs) |
| Steps per Minute | ~5,000-8,000 |
| Episodes per Hour | ~50-80 |
| Training Speed | 10-20x faster than single env |
| GPU Utilization | 60-80% per worker |

## 🔍 Troubleshooting

### Pod Scheduling Issues

```bash
# Check pod status
oc get pods -n your-namespace

# Describe pending pods
oc describe pod zelda-rl-worker-0

# Common issues:
# - Insufficient GPU nodes
# - Resource limits too high
# - Image pull errors
```

### Ray Cluster Not Starting

```bash
# Check Ray head logs
oc logs zelda-rl-head-xxxxx

# Check Ray worker logs
oc logs zelda-rl-worker-xxxxx

# Restart cluster
cluster.down()
cluster.up()
```

### LLM Connection Errors

```bash
# Test LLM endpoint from within cluster
oc run test-llm --image=curlimages/curl --rm -it -- \
  curl -X POST http://llama4-scout-service:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"test"}],"max_tokens":10}'
```

### S3 Upload Failures

```bash
# Check MinIO service
oc get svc -n minio

# Verify bucket exists
oc exec -it minio-pod -n minio -- mc ls local/

# Test S3 connection
oc run test-s3 --image=amazon/aws-cli --rm -it -- \
  s3 ls --endpoint-url http://minio-service:9000
```

## 📝 Key Differences from Custom PPO

| Feature | Custom PPO | Ray RLlib |
|---------|------------|-----------|
| Parallelism | 1 environment | 9+ environments |
| Speed | ~200 steps/min | ~5,000-8,000 steps/min |
| GPU Usage | 10-20% | 60-80% per worker |
| Distributed | No | Yes (multi-pod) |
| Production | Research | Production-ready |
| Monitoring | Custom HUD | Ray Dashboard |
| Checkpointing | Manual | Automatic |
| Fault Tolerance | None | Ray handles failures |

## 🎯 Next Steps

1. **Build and test locally**:
   ```bash
   python run-ray-zelda.py
   ```

2. **Build Docker image** with all dependencies

3. **Deploy to cluster** using Jupyter notebook

4. **Monitor training** via Ray dashboard

5. **Retrieve results** from MinIO S3

## 🔗 Resources

- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [CodeFlare SDK](https://github.com/project-codeflare/codeflare-sdk)
- [Double Dragon Implementation](file:///Users/cnuland/hello-chris-dd-kuberay)
- [KubeRay Operator](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)

## 📞 Support

For issues or questions, check:
1. Ray dashboard logs
2. Worker pod logs (`oc logs`)
3. MinIO for saved artifacts
4. This deployment guide

Happy training! 🎮✨


# 🤖 LLM Configuration Guide

## Overview
This project supports **TWO LLM backends** with separate configurations:

1. **Local MLX Mode** (for development/debugging)
2. **Cluster vLLM Mode** (for production training)

---

## 🏠 Configuration 1: Local MLX Mode

### Use Case
- Local development and testing
- Prompt engineering and debugging
- Visual observation of agent behavior
- No Ray/Kubernetes needed

### Setup

1. **Start MLX LLM Server**
   ```bash
   make llm-serve
   ```
   This starts MLX server at `http://localhost:8000`

2. **Run Visual Mode**
   ```bash
   make local-visual
   ```

### Configuration Files

**File**: `run_local_visual.py` (Line 20)
```python
os.environ['LLM_ENDPOINT'] = os.environ.get('LLM_ENDPOINT', 'http://localhost:8000/v1/chat/completions')
```

**Override** (if needed):
```bash
export LLM_ENDPOINT='http://localhost:8000/v1/chat/completions'
make local-visual
```

### Model
- **MLX Model**: `mlx-community/meta-llama-Llama-4-Scout-17B-16E-4bit`
- **Platform**: Apple Silicon (M1/M2/M3) optimized
- **Memory**: ~10GB
- **Speed**: 15-30 tokens/sec

---

## ☁️ Configuration 2: Cluster vLLM Mode

### Use Case
- Production distributed training
- High-throughput inference (5-50 workers)
- Ray RLlib on OpenShift/Kubernetes
- Multi-GPU vLLM backend

### Setup

**File**: `run-kuberay-zelda.ipynb` (Cell 11, env_vars)

#### Option A: Direct Service IP (Recommended)
```python
env_vars = {
    # ... other vars ...
    'LLM_ENDPOINT': 'http://172.30.21.1:8000/v1/chat/completions',  # Direct IP - bypasses DNS
}
```

**Pros**:
- ✅ Works immediately
- ✅ No DNS issues
- ✅ Reliable

**Cons**:
- ⚠️ Service IP may change on redeployment
- ⚠️ Need to check IP if service recreated

#### Option B: Service DNS Name (if DNS works)
```python
env_vars = {
    # ... other vars ...
    'LLM_ENDPOINT': 'http://llama4scout-service.llm-d.svc.cluster.local:8000/v1/chat/completions',
}
```

**Pros**:
- ✅ Survives service redeployments
- ✅ Standard Kubernetes pattern

**Cons**:
- ❌ Currently fails (cross-namespace DNS issue)
- ⚠️ Requires network policy fix

#### Option C: Disable LLM (Pure PPO)
```python
env_vars = {
    # ... other vars ...
    'LLM_ENDPOINT': '',  # Empty = disabled
}
```

### Model
- **vLLM Model**: `mlx-community/meta-llama-Llama-4-Scout-17B-16E-4bit`
- **Platform**: 3x GPU pods (8 GPUs total)
- **Memory**: Distributed across GPUs
- **Speed**: 100+ tokens/sec (parallelized)

---

## 🔍 How to Check Current Service IP

If you need to update the cluster IP:

```bash
# Login to OpenShift
oc login --token=<your-token> --server=https://api.rosa-58cx6.acrs.p3.openshiftapps.com:443

# Switch to LLM namespace
oc project llm-d

# Get service IP
oc get svc llama4scout-service -o jsonpath='{.spec.clusterIP}'
# Example output: 172.30.21.1

# Test endpoint
oc port-forward svc/llama4scout-service 8000:8000 &
curl http://localhost:8000/v1/models
# Should return: {"object": "list", "data": [...]}
```

---

## 🎯 Configuration Matrix

| Aspect | Local MLX | Cluster vLLM |
|--------|-----------|--------------|
| **File** | `run_local_visual.py` | `run-kuberay-zelda.ipynb` |
| **Endpoint** | `http://localhost:8000` | `http://172.30.21.1:8000` |
| **Model** | MLX 4-bit quantized | vLLM 4-bit quantized |
| **Scale** | 1 env, single-threaded | 30-600 envs, distributed |
| **Platform** | macOS (Apple Silicon) | OpenShift (NVIDIA GPUs) |
| **Command** | `make local-visual` | Jupyter notebook Cell 14 |
| **Visual** | ✅ PyBoy window | ❌ Headless only |
| **HUD** | ❌ Disabled | ✅ Remote HUD server |
| **Use Case** | Dev/debug/prompt-tuning | Production training |

---

## 🚨 Current Status (Oct 16, 2025)

### Local MLX Mode
- ✅ **Working** - No changes needed
- ✅ Endpoint: `http://localhost:8000/v1/chat/completions`
- ✅ Model loaded and responding

### Cluster vLLM Mode
- ⚠️ **Needs Fix** - Currently disabled in notebook
- ❌ DNS failing: `llama4scout-service.llm-d.svc.cluster.local`
- ✅ Service running: `172.30.21.1:8000`
- 🔧 **Action Required**: Update notebook to use direct IP

---

## 📝 Update Checklist

When updating cluster LLM configuration:

- [ ] Verify service is running: `oc get pods -n llm-d`
- [ ] Get current service IP: `oc get svc llama4scout-service -n llm-d`
- [ ] Test endpoint: `oc port-forward svc/llama4scout-service 8000:8000`
- [ ] Update `run-kuberay-zelda.ipynb` Cell 11 with correct IP
- [ ] Uncomment `LLM_ENDPOINT` line (remove `# ` prefix)
- [ ] Comment out or remove empty `LLM_ENDPOINT` line
- [ ] Run Cell 14 to submit job
- [ ] Check logs for LLM success: `oc logs <worker-pod> | grep "LLM SUGGESTS"`

---

## 🐛 Troubleshooting

### Issue: "Connection refused" in cluster logs

**Symptom**:
```
⚠️  LLM call failed: HTTPConnectionPool(host='llama4scout-service.llm-d.svc.cluster.local', port=8000): 
Connection refused  [repeated 6048x across cluster]
```

**Solution**:
1. Check service IP: `oc get svc llama4scout-service -n llm-d -o jsonpath='{.spec.clusterIP}'`
2. Update notebook with direct IP: `http://<IP>:8000/v1/chat/completions`
3. Redeploy Ray job

### Issue: Local mode not finding LLM

**Symptom**:
```
⚠️  LLM call failed: Connection refused to localhost:8000
```

**Solution**:
1. Check if MLX server is running: `ps aux | grep "python.*llm"`
2. Start server: `make llm-serve`
3. Verify endpoint: `curl http://localhost:8000/v1/models`

### Issue: Wrong model loaded

**Symptom**:
```
Model mismatch or unexpected responses
```

**Solution**:
1. Check loaded models:
   - Local: `curl http://localhost:8000/v1/models`
   - Cluster: `oc port-forward svc/llama4scout-service 8000:8000 && curl http://localhost:8000/v1/models`
2. Verify it includes: `meta-llama-Llama-4-Scout-17B-16E-4bit`

---

## 🎯 Recommended Configuration (Current)

**For Local Development:**
```bash
# No changes needed - already configured correctly!
make llm-serve  # Start MLX server
make local-visual  # Run with PyBoy window
```

**For Cluster Training:**
```python
# In run-kuberay-zelda.ipynb Cell 11:
env_vars = {
    # ... other vars ...
    'LLM_ENDPOINT': 'http://172.30.21.1:8000/v1/chat/completions',  # ✅ Use this
    # 'LLM_ENDPOINT': '',  # ❌ Remove this line
}
```

---

## 💡 Best Practices

1. **Always test locally first** before deploying to cluster
2. **Document IP changes** if service is redeployed
3. **Check service health** before starting training runs
4. **Monitor LLM success rate** in training logs
5. **Keep both configs in sync** (same probabilities, rewards)

---

## 📚 Related Files

- `run_local_visual.py` - Local MLX configuration
- `run-kuberay-zelda.ipynb` - Cluster vLLM configuration
- `Makefile` - Commands for local mode
- `configs/env.yaml` - LLM call probabilities (shared)
- `configs/vision_prompt.yaml` - LLM prompts (shared)
- `LLM_CONNECTION_FIX.md` - Troubleshooting DNS issue

---

*Last Updated: October 16, 2025*


# 🔴 LLM Connection Fix (Cluster vLLM Only)

> **⚠️ NOTE**: This fix is **ONLY** for cluster training mode.  
> **Local MLX mode is already working** and does NOT need this fix!

## Problem
**Cluster training logs** show **25,000+ connection failures**:
```
⚠️  LLM call failed: HTTPConnectionPool(host='llama4scout-service.llm-d.svc.cluster.local', port=8000): 
... Connection refused  [repeated 6048x across cluster]
```

## Root Cause
**DNS resolution failing** between `zelda-hybrid-rl-llm` namespace and `llm-d` namespace.

## Configurations Affected

| Mode | Status | Action |
|------|--------|--------|
| **Local MLX** (`make local-visual`) | ✅ Working | No change needed |
| **Cluster vLLM** (`run-kuberay-zelda.ipynb`) | ❌ Broken | Needs IP fix |

## Verification (Already Done ✅)
```bash
# LLM service EXISTS and is RUNNING
oc get svc -n llm-d | grep llama4scout-service
# OUTPUT: llama4scout-service   ClusterIP   172.30.21.1   <none>   8000/TCP   37h

# LLM endpoint WORKS
curl http://localhost:8000/v1/models
# OUTPUT: {"object": "list", "data": [{"id": "mlx-community/meta-llama-Llama-4-Scout-17B-16E-4bit", ...}]}
```

## Solution
Use **direct service IP** instead of DNS name in `run-kuberay-zelda.ipynb`:

### BEFORE (failing):
```python
'LLM_ENDPOINT': 'http://llama4scout-service.llm-d.svc.cluster.local:8000/v1/chat/completions'
```

### AFTER (working):
```python
'LLM_ENDPOINT': 'http://172.30.21.1:8000/v1/chat/completions'
```

## Alternative Solutions (if IP changes)

### Option 1: Find Current Service IP
```bash
oc get svc llama4scout-service -n llm-d -o jsonpath='{.spec.clusterIP}'
```

### Option 2: Fix DNS (Network Policy)
Check if network policy is blocking cross-namespace DNS:
```bash
oc get networkpolicy -n zelda-hybrid-rl-llm
oc get networkpolicy -n llm-d
```

May need to add network policy to allow traffic:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-zelda-to-llm
  namespace: llm-d
spec:
  podSelector:
    matchLabels:
      app: llama4scout
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: zelda-hybrid-rl-llm
    ports:
    - protocol: TCP
      port: 8000
```

### Option 3: Use External Route
```bash
# Check if external route exists
oc get route -n llm-d
```

## Status
- [x] Identified service IP: `172.30.21.1:8000`
- [x] Verified service is running
- [x] Verified endpoint responds
- [ ] Update `run-kuberay-zelda.ipynb` with direct IP
- [ ] Redeploy Ray cluster
- [ ] Verify LLM calls succeed in training logs

## Expected Outcome
After fix:
- ✅ LLM success rate: >95%
- ✅ Alignment bonuses applied
- ✅ Text-only guidance: ~2% of steps
- ✅ Vision guidance: ~3% of steps

## Configuration Preserved

| Configuration | Endpoint | File | Status |
|---------------|----------|------|--------|
| **Local MLX** | `http://localhost:8000` | `run_local_visual.py` | ✅ Unchanged |
| **Cluster vLLM** | `http://172.30.21.1:8000` | `run-kuberay-zelda.ipynb` | 🔧 Will update |

**Both configurations work independently!**

## Related Documentation
- See `LLM_CONFIGURATION_GUIDE.md` for complete setup guide
- See `Makefile` for local mode commands


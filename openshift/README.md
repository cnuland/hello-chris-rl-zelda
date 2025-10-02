# ROSA LLM-D and PyBoy Training Deployment

This directory contains deployment manifests and scripts for setting up:

1. **Llama-4-Scout LLM inference** with prefix-aware KV cache
2. **PyBoy RL training** with 100+ headless instances

## Architecture

### Machine Sets

#### LLM Inference Machine Set
- **Instance Type**: `p4d.xlarge` (high-memory GPU)
- **Replicas**: 3 nodes
- **Purpose**: Run 3 instances of RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic
- **Features**: Efficient KV cache, prefix-aware routing
- **Node Label**: `node-role.kubernetes.io/llm-inference`

#### PyBoy Training Machine Set  
- **Instance Type**: `g4dn.4xlarge` (many GPU cores)
- **Replicas**: 5 nodes
- **Purpose**: Run 100+ PyBoy headless training instances (20 per node)
- **Features**: Cost-effective GPU cores, distributed training
- **Node Label**: `node-role.kubernetes.io/pyboy-training`

### LLM-D Configuration

#### Prefix-Aware KV Cache
- **P/D Configuration**: Prefill/Decode separation for efficiency
- **Inference Gateway**: Routes requests based on prompt prefixes
- **Session Affinity**: Similar prompts route to same instance for cache hits
- **KV Transfer**: NixlConnector for efficient cache sharing

#### Model Configuration
- **Model**: `RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic`
- **Format**: FP8 dynamic quantization for memory efficiency
- **Context Length**: 4096 tokens
- **GPU Memory**: 85% utilization with 16GB swap space

## Deployment

### Prerequisites

1. **ROSA Cluster**: Active ROSA cluster with admin access
2. **CLI Tools**: `oc`, `rosa`, `helm` installed
3. **HuggingFace Token**: Required for model access
4. **Cluster Login**: Must be logged into OpenShift cluster

### Quick Start

```bash
# Set your cluster ID and HF token
export CLUSTER_ID="your-cluster-id"  # e.g., "9p74m"
export HF_TOKEN="your-huggingface-token"

# Deploy everything
cd openshift/configs
./deploy-all.sh deploy
```

### Step-by-Step Deployment

```bash
# 1. Check prerequisites
./deploy-all.sh

# 2. Monitor deployment status
./deploy-all.sh status

# 3. Clean up if needed
./deploy-all.sh clean
```

### Manual Deployment

If you prefer to deploy components individually:

```bash
# 1. Update cluster ID in machine sets
find machinesets/ -name "*.yaml" -exec sed -i "s/rosa-CLUSTER_ID/rosa-${CLUSTER_ID}/g" {} +

# 2. Deploy machine sets
oc apply -f machinesets/llm-inference-machineset.yaml
oc apply -f machinesets/pyboy-training-machineset.yaml

# 3. Wait for nodes (10-15 minutes)
oc get machinesets -n openshift-machine-api
oc get nodes -l node-role.kubernetes.io/llm-inference

# 4. Setup namespace and secrets
oc create namespace llm-d
oc create secret generic llm-d-hf-token --from-literal=HF_TOKEN="${HF_TOKEN}" -n llm-d

# 5. Install LLM-D infrastructure
helm repo add llm-d-infra https://llm-d-incubation.github.io/llm-d-infra/
helm upgrade -i llm-d-infra llm-d-infra/llm-d-infra -n llm-d --create-namespace

# 6. Deploy applications
oc apply -f llm-deployments/llama4-scout-service.yaml
oc apply -f llm-deployments/llama4-scout-decode-deployment.yaml
oc apply -f llm-deployments/llama4-scout-inference-gateway.yaml
oc apply -f llm-deployments/pyboy-training-deployment.yaml
```

## Configuration Files

### Machine Sets
- `machinesets/llm-inference-machineset.yaml` - High-memory GPU nodes for LLM inference
- `machinesets/pyboy-training-machineset.yaml` - Multi-core GPU nodes for training

### LLM Deployments
- `llm-deployments/llama4-scout-decode-deployment.yaml` - Main LLM inference deployment
- `llm-deployments/llama4-scout-service.yaml` - Service and RBAC configuration
- `llm-deployments/llama4-scout-inference-gateway.yaml` - Gateway and routing configuration

### Training Deployments
- `llm-deployments/pyboy-training-deployment.yaml` - PyBoy RL training environment

### Scripts
- `configs/deploy-all.sh` - Main deployment script
- `configs/` - Additional configuration files

## Monitoring and Testing

### LLM Inference

```bash
# Monitor model loading (can take 15-20 minutes)
oc logs -n llm-d -l app.kubernetes.io/name=llama4-scout-decode -f

# Test endpoint locally
oc port-forward -n llm-d svc/llama4-scout-decode 8000:8000

# Test inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100
  }'
```

### PyBoy Training

```bash
# Monitor training instances
oc logs -n llm-d -l app=pyboy-training -f

# View training metrics via TensorBoard
oc port-forward -n llm-d svc/pyboy-training-service 6006:6006
# Open http://localhost:6006

# Check resource usage
oc top pods -n llm-d
```

### Gateway Access

```bash
# Get gateway IP
oc get gateway -n llm-d

# Test through gateway (if configured)
GATEWAY_IP=$(oc get gateway -n llm-d -o jsonpath='{.items[0].status.addresses[0].value}')
curl -X POST http://$GATEWAY_IP/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: test-session" \
  -d '{"messages": [{"role": "user", "content": "What is reinforcement learning?"}]}'
```

## Troubleshooting

### Common Issues

1. **Nodes not ready**: Machine sets can take 10-15 minutes to provision
2. **Model loading timeout**: Llama-4-Scout is large, allow 20+ minutes
3. **PyBoy failures**: Ensure ROM is properly mounted in ConfigMap
4. **Authentication errors**: Check HF_TOKEN secret is created
5. **Resource constraints**: Verify GPU nodes have sufficient resources

### Debug Commands

```bash
# Check machine set status
oc get machinesets -n openshift-machine-api
oc describe machineset <machineset-name> -n openshift-machine-api

# Check node resources
oc describe nodes -l nvidia.com/gpu.present=true

# Check pod status
oc get pods -n llm-d -o wide
oc describe pod <pod-name> -n llm-d

# Check GPU operator
oc get pods -n nvidia-gpu-operator
```

## Scaling

### LLM Inference Scaling
```bash
# Scale decode deployment
oc scale deployment llama4-scout-decode --replicas=5 -n llm-d

# Scale machine set for more nodes
oc patch machineset rosa-${CLUSTER_ID}-llm-inference -n openshift-machine-api \
  --type='merge' -p='{"spec":{"replicas":5}}'
```

### Training Scaling
```bash
# Scale training deployment
oc scale deployment pyboy-training --replicas=10 -n llm-d

# Adjust instances per pod in deployment
# Edit pyboy-training-deployment.yaml and change the loop: for i in {1..40}
```

## Cost Optimization

### Instance Types
- **LLM Inference**: p4d.xlarge provides good memory/performance ratio
- **Training**: g4dn.4xlarge offers cost-effective GPU cores
- **Alternatives**: Consider p3.2xlarge or g5.4xlarge based on availability

### Resource Management
- Use **taints and tolerations** to ensure workload isolation  
- Implement **horizontal pod autoscaling** based on GPU utilization
- Consider **spot instances** for training workloads

## Security

- All secrets managed through OpenShift secrets
- RBAC configured for service accounts
- Network policies can be added for isolation
- Consider using **external secrets operator** for production

## Next Steps

1. **Monitoring**: Deploy Prometheus/Grafana for metrics
2. **CI/CD**: Integrate with OpenShift Pipelines/GitOps
3. **Storage**: Add persistent volumes for model caches
4. **Networking**: Configure ingress for external access
5. **Backup**: Implement backup strategies for training data
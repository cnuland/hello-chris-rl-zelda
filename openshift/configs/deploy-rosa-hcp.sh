#!/bin/bash

# ROSA HCP LLM-D and PyBoy Training Deployment Script
# This script works with the current ROSA HCP cluster setup
# Current setup: rosa-9p74m with m6a.xlarge CPU-only nodes

set -e

# Configuration
CLUSTER_NAME="rosa-9p74m"
NAMESPACE="llm-d"
HF_TOKEN=${HF_TOKEN:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for ROSA HCP deployment..."
    
    command -v oc >/dev/null 2>&1 || error "OpenShift CLI (oc) not found"
    command -v rosa >/dev/null 2>&1 || error "ROSA CLI not found"
    command -v helm >/dev/null 2>&1 || error "Helm not found"
    
    # Check if logged into cluster
    oc whoami >/dev/null 2>&1 || error "Not logged into OpenShift cluster"
    
    # Check if HF_TOKEN is set
    if [ -z "$HF_TOKEN" ]; then
        warn "HF_TOKEN not set - model access may fail"
        read -p "Enter your HuggingFace token (or press Enter to skip): " HF_TOKEN
    fi
    
    log "Prerequisites checked successfully"
}

# Show current cluster status
show_cluster_info() {
    log "Current ROSA HCP Cluster Information:"
    echo ""
    echo -e "${BLUE}=== Cluster Details ===${NC}"
    echo "Cluster: $CLUSTER_NAME (ROSA HCP)"
    echo "API: https://api.rosa-9p74m.h96u.p3.openshiftapps.com:443"
    echo "Console: https://console-openshift-console.apps.rosa.rosa-9p74m.h96u.p3.openshiftapps.com"
    
    echo ""
    echo -e "${BLUE}=== Current Nodes ===${NC}"
    oc get nodes -o custom-columns=NAME:.metadata.name,INSTANCE-TYPE:.metadata.labels.node\\.kubernetes\\.io/instance-type,STATUS:.status.conditions[3].type
    
    echo ""
    echo -e "${BLUE}=== Node Pool Information ===${NC}"
    oc get nodes -o jsonpath='{.items[0].metadata.labels.hypershift\.openshift\.io/nodePool}' && echo " (Current node pool)"
}

# Create GPU node pool (requires AWS credentials)
create_gpu_nodepool() {
    log "Creating GPU node pool for LLM inference..."
    
    warn "This requires proper AWS credentials configured with ROSA CLI"
    warn "Current cluster has CPU-only nodes (m6a.xlarge)"
    
    cat << 'EOF'
To add GPU nodes to your ROSA HCP cluster, run:

# Add GPU node pool for LLM inference (3 nodes)
rosa create nodepool --cluster=rosa-9p74m \
  --name=gpu-inference \
  --instance-type=g5.2xlarge \
  --replicas=3 \
  --labels="node-role.kubernetes.io/gpu-inference=" \
  --taints="gpu-inference=true:NoSchedule"

# Add GPU node pool for PyBoy training (5 nodes)  
rosa create nodepool --cluster=rosa-9p74m \
  --name=gpu-training \
  --instance-type=g4dn.4xlarge \
  --replicas=5 \
  --labels="node-role.kubernetes.io/gpu-training=" \
  --taints="gpu-training=true:NoSchedule"

# Wait for nodes to be ready
rosa describe nodepool --cluster=rosa-9p74m --nodepool=gpu-inference
rosa describe nodepool --cluster=rosa-9p74m --nodepool=gpu-training

EOF
    
    read -p "Do you want to continue with CPU-only deployment for now? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Stopping deployment. Add GPU node pools first."
        exit 0
    fi
}

# Setup namespace and prerequisites
setup_namespace() {
    log "Setting up namespace and prerequisites..."
    
    # Create namespace
    oc get namespace $NAMESPACE >/dev/null 2>&1 || oc create namespace $NAMESPACE
    
    # Create HF token secret if provided
    if [ -n "$HF_TOKEN" ]; then
        log "Creating HuggingFace token secret..."
        oc -n $NAMESPACE create secret generic llm-d-hf-token \
            --from-literal=HF_TOKEN="$HF_TOKEN" \
            --dry-run=client -o yaml | oc apply -f -
    fi
    
    log "Namespace setup completed"
}

# Deploy CPU-optimized LLM service (using vLLM with CPU)
deploy_cpu_llm() {
    log "Deploying CPU-optimized LLM service..."
    
    cat << EOF | oc apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama4-scout-cpu
  namespace: $NAMESPACE
  labels:
    app: llama4-scout-cpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llama4-scout-cpu
  template:
    metadata:
      labels:
        app: llama4-scout-cpu
    spec:
      containers:
      - name: vllm-cpu
        image: vllm/vllm-openai:latest
        command: ["python", "-m", "vllm.entrypoints.openai.api_server"]
        args:
        - --model=microsoft/DialoGPT-medium  # Smaller model for CPU
        - --host=0.0.0.0
        - --port=8000
        - --disable-log-requests
        - --max-model-len=1024
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: llm-d-hf-token
              key: HF_TOKEN
              optional: true
        ports:
        - containerPort: 8000
          name: http
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        readinessProbe:
          httpGet:
            path: /v1/models
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /v1/models
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: llama4-scout-cpu-service
  namespace: $NAMESPACE
spec:
  selector:
    app: llama4-scout-cpu
  ports:
  - port: 8000
    targetPort: 8000
    name: http
EOF

    log "CPU-optimized LLM service deployed"
}

# Deploy CPU-based PyBoy training (simulation without GPU)
deploy_cpu_training() {
    log "Deploying CPU-based PyBoy training simulation..."
    
    cat << EOF | oc apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pyboy-cpu-training
  namespace: $NAMESPACE
  labels:
    app: pyboy-cpu-training
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pyboy-cpu-training
  template:
    metadata:
      labels:
        app: pyboy-cpu-training
    spec:
      containers:
      - name: pyboy-trainer
        image: python:3.9-slim
        command: ["/bin/bash"]
        args:
        - -c
        - |
          #!/bin/bash
          set -e
          
          # Install PyBoy and dependencies
          pip install pyboy numpy matplotlib
          
          # Create a simple training simulation
          cat << 'PYTHON' > /tmp/train_cpu.py
          import time
          import random
          import os
          
          def simulate_training():
              episode = 0
              while True:
                  episode += 1
                  # Simulate training episode
                  reward = random.uniform(-1, 10)
                  steps = random.randint(100, 1000)
                  
                  print(f"Episode {episode}: Reward={reward:.2f}, Steps={steps}")
                  
                  # Simulate training time
                  time.sleep(5)
                  
                  if episode % 10 == 0:
                      print(f"Checkpoint saved at episode {episode}")
          
          if __name__ == "__main__":
              print("Starting CPU-based PyBoy training simulation...")
              print("Pod:", os.environ.get("HOSTNAME", "unknown"))
              simulate_training()
          PYTHON
          
          python3 /tmp/train_cpu.py
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "2Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: pyboy-cpu-training-service
  namespace: $NAMESPACE
spec:
  selector:
    app: pyboy-cpu-training
  ports:
  - port: 8080
    targetPort: 8080
    name: http
EOF

    log "CPU-based PyBoy training simulation deployed"
}

# Create routes for external access
create_routes() {
    log "Creating routes for external access..."
    
    cat << EOF | oc apply -f -
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: llm-api-route
  namespace: $NAMESPACE
spec:
  to:
    kind: Service
    name: llama4-scout-cpu-service
    weight: 100
  port:
    targetPort: http
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
EOF

    log "Routes created"
}

# Wait for deployments
wait_for_deployments() {
    log "Waiting for deployments to be ready..."
    
    oc -n $NAMESPACE rollout status deployment/llama4-scout-cpu --timeout=300s || warn "LLM deployment timeout"
    oc -n $NAMESPACE rollout status deployment/pyboy-cpu-training --timeout=300s || warn "Training deployment timeout"
    
    log "Deployments are ready!"
}

# Show deployment status and next steps
show_status() {
    log "Deployment Status:"
    echo ""
    
    echo -e "${BLUE}=== Pods ===${NC}"
    oc get pods -n $NAMESPACE -o wide
    
    echo ""
    echo -e "${BLUE}=== Services ===${NC}"
    oc get svc -n $NAMESPACE
    
    echo ""
    echo -e "${BLUE}=== Routes ===${NC}"
    oc get routes -n $NAMESPACE
    
    echo ""
    echo -e "${GREEN}=== Next Steps ===${NC}"
    echo "1. Test LLM API:"
    LLM_ROUTE=$(oc get route llm-api-route -n $NAMESPACE -o jsonpath='{.spec.host}' 2>/dev/null || echo "Route not ready")
    if [ "$LLM_ROUTE" != "Route not ready" ]; then
        echo "   curl -X POST https://$LLM_ROUTE/v1/chat/completions \\"
        echo "     -H 'Content-Type: application/json' \\"
        echo "     -d '{\"model\": \"microsoft/DialoGPT-medium\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
    else
        echo "   oc port-forward -n $NAMESPACE svc/llama4-scout-cpu-service 8000:8000"
        echo "   curl -X POST http://localhost:8000/v1/models"
    fi
    
    echo ""
    echo "2. Monitor training:"
    echo "   oc logs -n $NAMESPACE -l app=pyboy-cpu-training -f"
    
    echo ""
    echo "3. Add GPU nodes for production workload:"
    echo "   Run the ROSA commands shown earlier to add g5.2xlarge and g4dn.4xlarge node pools"
    
    echo ""
    echo -e "${YELLOW}=== Current Limitations ===${NC}"
    echo "- Running on CPU-only nodes (m6a.xlarge)"
    echo "- Using smaller models suitable for CPU inference"
    echo "- Training is simulated (not actual PyBoy/GPU training)"
    echo "- For production LLM workloads, add GPU node pools"
}

# Main deployment flow
main() {
    log "Starting ROSA HCP deployment..."
    
    check_prerequisites
    show_cluster_info
    create_gpu_nodepool
    setup_namespace
    deploy_cpu_llm
    deploy_cpu_training
    create_routes
    wait_for_deployments
    show_status
    
    log "Deployment completed!"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "status")
        show_status
        ;;
    "clean")
        log "Cleaning up deployments..."
        oc delete all,routes,secrets -l app=llama4-scout-cpu -n $NAMESPACE --ignore-not-found=true
        oc delete all,routes,secrets -l app=pyboy-cpu-training -n $NAMESPACE --ignore-not-found=true
        log "Cleanup completed"
        ;;
    "gpu-info")
        log "GPU Node Pool Creation Commands:"
        create_gpu_nodepool
        ;;
    *)
        echo "Usage: $0 {deploy|status|clean|gpu-info}"
        echo "  deploy   - Deploy CPU-optimized services (default)"
        echo "  status   - Show current status"
        echo "  clean    - Remove all deployments"
        echo "  gpu-info - Show GPU node pool creation commands"
        exit 1
        ;;
esac
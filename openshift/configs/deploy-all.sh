#!/bin/bash

# ROSA LLM-D and PyBoy Training Deployment Script
# This script deploys machine sets and applications for:
# 1. Llama-4-Scout LLM inference with KV cache
# 2. PyBoy RL training with 100+ instances

set -e

# Configuration
CLUSTER_ID=${CLUSTER_ID:-"9p74m"}  # Update with your actual cluster ID
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
    log "Checking prerequisites..."
    
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

# Update machine set configurations with actual cluster ID
update_cluster_configs() {
    log "Updating configurations with cluster ID: $CLUSTER_ID"
    
    # Update machine set configurations
    find ../machinesets/ -name "*.yaml" -exec sed -i "s/rosa-CLUSTER_ID/rosa-${CLUSTER_ID}/g" {} +
    
    log "Configuration files updated"
}

# Deploy machine sets
deploy_machine_sets() {
    log "Deploying ROSA machine sets..."
    
    # Deploy LLM inference machine set
    log "Creating LLM inference machine set (3 nodes with high-memory GPUs)..."
    oc apply -f ../machinesets/llm-inference-machineset.yaml
    
    # Deploy PyBoy training machine set
    log "Creating PyBoy training machine set (5 nodes with many GPU cores)..."
    oc apply -f ../machinesets/pyboy-training-machineset.yaml
    
    log "Machine sets deployed. Waiting for nodes to be ready..."
    
    # Wait for nodes to be ready (this can take 10-15 minutes)
    log "Waiting for LLM inference nodes..."
    oc wait --for=condition=Ready nodes -l node-role.kubernetes.io/llm-inference --timeout=900s || warn "Timeout waiting for LLM nodes"
    
    log "Waiting for PyBoy training nodes..."  
    oc wait --for=condition=Ready nodes -l node-role.kubernetes.io/pyboy-training --timeout=900s || warn "Timeout waiting for PyBoy nodes"
    
    log "Machine sets deployment completed"
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
    
    # Install GPU Operator if not present
    log "Checking GPU Operator..."
    if ! oc get csv -n openshift-operators | grep -q gpu-operator-certified; then
        log "Installing NVIDIA GPU Operator..."
        oc apply -f - <<EOF
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: gpu-operator-certified
  namespace: openshift-operators
spec:
  channel: stable
  name: gpu-operator-certified
  source: certified-operators
  sourceNamespace: openshift-marketplace
EOF
        log "GPU Operator installation initiated. This may take several minutes..."
        sleep 60  # Give operator time to start
    else
        log "GPU Operator already installed"
    fi
    
    log "Namespace setup completed"
}

# Deploy LLM-D infrastructure using Helm
deploy_llmd_infra() {
    log "Deploying LLM-D infrastructure..."
    
    # Add Helm repositories
    helm repo add llm-d-infra https://llm-d-incubation.github.io/llm-d-infra/ || true
    helm repo update
    
    # Install LLM-D infrastructure
    log "Installing LLM-D infrastructure Helm chart..."
    helm upgrade -i llm-d-infra llm-d-infra/llm-d-infra -n $NAMESPACE \
        --create-namespace \
        --set gateway.gatewayClassName=istio \
        --timeout=10m || warn "LLM-D infra installation failed or timed out"
    
    log "LLM-D infrastructure deployed"
}

# Deploy applications
deploy_applications() {
    log "Deploying applications..."
    
    # Deploy Llama-4-Scout model
    log "Deploying Llama-4-Scout decode service..."
    oc apply -f ../llm-deployments/llama4-scout-service.yaml
    oc apply -f ../llm-deployments/llama4-scout-decode-deployment.yaml
    
    # Deploy inference gateway configuration
    log "Configuring inference gateway with prefix-aware KV cache..."
    oc apply -f ../llm-deployments/llama4-scout-inference-gateway.yaml
    
    # Deploy PyBoy training
    log "Deploying PyBoy training environment..."
    
    # Create ROM ConfigMap (you'll need to add the Zelda ROM)
    oc -n $NAMESPACE create configmap zelda-rom-config \
        --from-file=zelda.gb=/path/to/zelda.gb \
        --dry-run=client -o yaml | oc apply -f - || warn "ROM ConfigMap creation failed - update path"
    
    oc apply -f ../llm-deployments/pyboy-training-deployment.yaml
    
    log "Applications deployed"
}

# Wait for deployments to be ready
wait_for_readiness() {
    log "Waiting for deployments to be ready..."
    
    # Wait for Llama-4-Scout deployment
    log "Waiting for Llama-4-Scout pods (this can take 15-20 minutes for model loading)..."
    oc -n $NAMESPACE rollout status deployment/llama4-scout-decode --timeout=1200s || warn "Timeout waiting for Llama-4-Scout"
    
    # Wait for PyBoy training deployment
    log "Waiting for PyBoy training pods..."
    oc -n $NAMESPACE rollout status deployment/pyboy-training --timeout=600s || warn "Timeout waiting for PyBoy training"
    
    log "Deployments are ready!"
}

# Display status and next steps
show_status() {
    log "Deployment Status:"
    echo ""
    echo -e "${BLUE}=== Machine Sets ===${NC}"
    oc get machinesets -n openshift-machine-api | grep -E "(llm-inference|pyboy-training)" || echo "No machine sets found"
    
    echo ""
    echo -e "${BLUE}=== Nodes ===${NC}"
    oc get nodes -l 'node-role.kubernetes.io/llm-inference,node-role.kubernetes.io/pyboy-training' --show-labels || echo "No specialized nodes found yet"
    
    echo ""
    echo -e "${BLUE}=== Pods ===${NC}"
    oc get pods -n $NAMESPACE -o wide
    
    echo ""
    echo -e "${BLUE}=== Services ===${NC}"
    oc get svc -n $NAMESPACE
    
    echo ""
    echo -e "${GREEN}=== Next Steps ===${NC}"
    echo "1. Monitor model loading: oc logs -n $NAMESPACE -l app.kubernetes.io/name=llama4-scout-decode -f"
    echo "2. Test LLM endpoint: oc port-forward -n $NAMESPACE svc/llama4-scout-decode 8000:8000"
    echo "3. View training metrics: oc port-forward -n $NAMESPACE svc/pyboy-training-service 6006:6006"
    echo "4. Access logs: oc logs -n $NAMESPACE -l app=pyboy-training -f"
    
    # Show Gateway information if available
    GATEWAY_IP=$(oc get gateway -n $NAMESPACE -o jsonpath='{.items[0].status.addresses[0].value}' 2>/dev/null || echo "Not available yet")
    if [ "$GATEWAY_IP" != "Not available yet" ]; then
        echo "5. Gateway IP: $GATEWAY_IP"
        echo "   Test inference: curl -X POST http://$GATEWAY_IP/v1/chat/completions -H 'Content-Type: application/json' -d '{...}'"
    fi
}

# Main deployment flow
main() {
    log "Starting ROSA LLM-D and PyBoy deployment..."
    
    check_prerequisites
    update_cluster_configs
    setup_namespace
    deploy_machine_sets
    deploy_llmd_infra
    deploy_applications
    wait_for_readiness
    show_status
    
    log "Deployment completed successfully!"
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
        oc delete -f ../llm-deployments/ --ignore-not-found=true
        oc delete -f ../machinesets/ --ignore-not-found=true
        helm uninstall llm-d-infra -n $NAMESPACE || true
        log "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {deploy|status|clean}"
        echo "  deploy - Deploy everything (default)"
        echo "  status - Show current status"
        echo "  clean  - Remove all deployments"
        exit 1
        ;;
esac
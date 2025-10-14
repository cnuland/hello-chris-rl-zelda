#!/bin/bash
# Production-Scale Zelda RL Training Script
# Maximizes all cluster resources for high-performance distributed training
#
# CLUSTER RESOURCES:
# - 6 × g5.2xlarge GPU nodes (7.5 CPUs, 30GB RAM, 1 GPU each)
# - 2 × Large CPU nodes (63.5 CPUs, 127GB RAM each)  
# - Total: ~174 CPUs, ~370GB RAM, 6 GPUs
#
# CONFIGURATION:
# - 84 parallel Ray workers (7 Kubernetes workers × 12 environments each)
# - 32GB batch size for distributed training
# - Extended episodes for better convergence
# - Preserves GPUs for LLM inference workloads

set -e  # Exit on error

echo "🚀 PRODUCTION-SCALE ZELDA RL TRAINING"
echo "====================================="
echo ""
echo "🔧 CLUSTER RESOURCE MAXIMIZATION:"
echo "   • Ray Workers: 84 (7 K8s pods × 12 envs each)"
echo "   • CPU Utilization: ~48 cores (7 pods × 6-7 CPUs each)"
echo "   • Memory Utilization: ~192GB (7 pods × 24-28GB each)"
echo "   • Batch Size: 32,768 (optimized for distributed training)"
echo "   • Episode Length: 61,440 steps (extended for convergence)"
echo ""
echo "⚡ PERFORMANCE TARGETS:"
echo "   • Expected throughput: ~25,000 steps/second"
echo "   • Parallel environments: 84 simultaneous games"
echo "   • Training efficiency: 70-80% cluster utilization"
echo "   • GPUs preserved for LLM inference workloads"
echo ""

# Step 1: Validate prerequisites
echo "📋 Step 1: Validating prerequisites..."

# Check if in correct directory
if [ ! -f "run-ray-zelda.py" ]; then
    echo "❌ Error: run-ray-zelda.py not found. Please run from project root."
    exit 1
fi

# Check namespace exists
if ! oc get namespace zelda-hybrid-rl-llm >/dev/null 2>&1; then
    echo "❌ Error: Namespace 'zelda-hybrid-rl-llm' not found."
    echo "   Please create the namespace first:"
    echo "   oc create namespace zelda-hybrid-rl-llm"
    exit 1
fi

# Check RBAC permissions
echo "🔐 Checking RBAC permissions..."
if ! oc auth can-i create rayclusters --as=system:serviceaccount:zelda-hybrid-rl-llm:zelda-rl-training -n zelda-hybrid-rl-llm >/dev/null 2>&1; then
    echo "⚠️  RBAC permissions need to be applied:"
    echo "   oc apply -f ops/openshift/rbac.yaml"
    echo ""
    read -p "Apply RBAC permissions now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        oc apply -f ops/openshift/rbac.yaml
        echo "✅ RBAC permissions applied"
    else
        echo "❌ RBAC permissions required. Exiting."
        exit 1
    fi
fi

echo "✅ Prerequisites validated"
echo ""

# Step 2: Deploy HUD Dashboard
echo "🖥️  Step 2: Deploying HUD Dashboard..."
oc apply -f ops/openshift/hud-deployment.yaml

# Wait for HUD to be ready
echo "⏳ Waiting for HUD pod to be ready..."
oc wait --for=condition=ready pod -l app=zelda-hud -n zelda-hybrid-rl-llm --timeout=120s

# Get HUD route URL
HUD_ROUTE=$(oc get route zelda-hud-route -n zelda-hybrid-rl-llm -o jsonpath='{.spec.host}' 2>/dev/null || echo "")
if [ -n "$HUD_ROUTE" ]; then
    HUD_URL="https://$HUD_ROUTE"
    echo "✅ HUD Dashboard deployed: $HUD_URL"
else
    echo "⚠️  HUD route not found, using internal service"
    HUD_URL="http://zelda-hud-service.zelda-hybrid-rl-llm.svc.cluster.local:8086"
fi
echo ""

# Step 3: Configure production environment variables
echo "⚙️  Step 3: Configuring production scaling parameters..."

# Export high-performance configuration
export RAY_WORKERS=84              # 7 K8s workers × 12 environments each
export ENVS_PER_WORKER=12          # 12 parallel environments per worker
export EPISODE_LENGTH=61440        # Extended episodes (2048 * 30)
export BATCH_SIZE=32768            # Large batch size for distributed training
export HUD_URL="$HUD_URL"          # HUD dashboard URL

echo "✅ Production configuration:"
echo "   RAY_WORKERS=$RAY_WORKERS"
echo "   ENVS_PER_WORKER=$ENVS_PER_WORKER"
echo "   EPISODE_LENGTH=$EPISODE_LENGTH"
echo "   BATCH_SIZE=$BATCH_SIZE"
echo "   Total parallel environments: $((RAY_WORKERS * ENVS_PER_WORKER))"
echo ""

# Step 4: Start production training
echo "🎮 Step 4: Launching production-scale training..."
echo "🌐 HUD Dashboard: $HUD_URL"
echo "📊 Monitor Ray dashboard at cluster details (see output below)"
echo ""

# Download ROM from S3 (if needed)
echo "📥 Downloading ROM files from S3..."
python init_rom.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to download ROM files!"
    exit 1
fi

# Start production training
echo ""
echo "🚀 STARTING PRODUCTION TRAINING..."
echo "⚡ Expected performance: ~25,000 steps/second across $((RAY_WORKERS * ENVS_PER_WORKER)) parallel environments"
echo "📈 This configuration utilizes ~48 CPU cores and ~192GB RAM across the cluster"
echo ""

python run-ray-zelda.py

echo ""
echo "✅ PRODUCTION TRAINING COMPLETED!"
echo "📊 Results saved to: ~/ray_results/zelda/"
echo "🖥️  HUD Dashboard (if still running): $HUD_URL"
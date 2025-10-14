#!/bin/bash
# Scale Ray cluster to maximize available resources

set -e

echo "⚡ SCALING RAY CLUSTER FOR MAXIMUM PERFORMANCE"
echo "============================================="

# Get current cluster status
echo "📊 Current Ray cluster status:"
oc get rayclusters -n zelda-hybrid-rl-llm

# Scale up the cluster to use more resources per worker
echo ""
echo "🚀 Scaling Ray cluster to use more CPU and memory per worker..."

# Update the RayCluster to use more resources
oc patch raycluster zelda-rl -n zelda-hybrid-rl-llm --type='merge' -p='
{
  "spec": {
    "workerGroupSpecs": [
      {
        "replicas": 3,
        "minReplicas": 1,
        "maxReplicas": 6,
        "groupName": "high-performance-workers",
        "rayStartParams": {},
        "template": {
          "spec": {
            "serviceAccountName": "zelda-rl-training",
            "containers": [
              {
                "name": "ray-worker",
                "image": "quay.io/cnuland/dd-kuberay-worker:latest",
                "imagePullPolicy": "Always",
                "resources": {
                  "limits": {
                    "cpu": "6",
                    "memory": "24Gi"
                  },
                  "requests": {
                    "cpu": "4",
                    "memory": "16Gi"  
                  }
                }
              }
            ]
          }
        }
      }
    ]
  }
}'

echo "⏳ Waiting for Ray cluster to update..."
sleep 10

echo ""
echo "📊 Updated Ray cluster status:"
oc get rayclusters -n zelda-hybrid-rl-llm

echo ""
echo "🎯 New configuration should provide:"
echo "   - More CPU resources per worker"
echo "   - More memory per worker"
echo "   - Better training performance"

echo ""
echo "✅ Ray cluster scaling complete!"
echo ""
echo "💡 Now you can run training with higher parallelization:"
echo "   export RAY_WORKERS=12"
echo "   export ENVS_PER_WORKER=12"
echo "   python run_direct_ray_training.py"
#!/bin/bash

echo "=== GPU G5 Machine Pool Monitor ==="
echo "Monitoring g5.12xlarge and g5.24xlarge machine pools and llama4scout-decode deployment"
echo "Press Ctrl+C to stop"
echo

while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"
    
    # Check machine pool status via bastion
    echo "🖥️  Machine Pool Status (gpu-g5-24xl):"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 rosa@bastion.9p74m.sandbox640.opentlc.com "rosa list machine-pools --cluster=rosa-9p74m | grep gpu-g5-24xl" 2>/dev/null || echo "❌ Could not connect to bastion"
    
    # Check OpenShift nodes
    echo
    echo "🔍 GPU Nodes in Cluster:"
    oc get nodes -l node.kubernetes.io/instance-type --no-headers | grep -E "(g5\.24xlarge|g5\.2xlarge)" | wc -l | xargs echo "   Total GPU nodes found:"
    oc get nodes -l node-role.kubernetes.io/gpu-medium --no-headers 2>/dev/null | wc -l | xargs echo "   g5.24xlarge nodes:"
    
    # Check pod status
    echo
    echo "📦 Llama4scout-decode Pod Status:"
    oc get pods -n llm-d -l app=llama4scout-decode --no-headers 2>/dev/null || echo "❌ Could not get pod status"
    
    echo
    echo "⏳ Next check in 30 seconds..."
    echo "========================================"
    echo
    
    sleep 30
done
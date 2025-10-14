#!/bin/bash

echo "=== Comprehensive GPU Node Monitor ==="
echo "Monitoring all GPU machine pools and node provisioning"
echo "Press Ctrl+C to stop"
echo

while true; do
    clear
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
    
    # Check machine pool status via bastion
    echo "🖥️  ROSA Machine Pool Status:"
    echo "------------------------------------------------"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 rosa@bastion.9p74m.sandbox640.opentlc.com \
    "rosa list machine-pools --cluster=rosa-9p74m | grep gpu" 2>/dev/null || echo "❌ Could not connect to bastion"
    
    echo
    echo "🔍 OpenShift Nodes in Cluster:"
    echo "------------------------------------------------"
    echo "Total nodes: $(oc get nodes --no-headers 2>/dev/null | wc -l)"
    echo "GPU nodes: $(oc get nodes --no-headers 2>/dev/null | grep -E "(gpu-|g5\.|g4dn\.|p3)" | wc -l)"
    echo
    echo "GPU Nodes by Type:"
    oc get nodes --no-headers 2>/dev/null | grep -E "(gpu-|g5\.|g4dn\.|p3)" | \
    awk '{print $1}' | xargs -I {} oc get node {} -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}{" - "}{.metadata.labels.node-role\.kubernetes\.io/gpu-inference}{.metadata.labels.node-role\.kubernetes\.io/gpu-training}{.metadata.labels.node-role\.kubernetes\.io/gpu-medium}{.metadata.labels.node-role\.kubernetes\.io/gpu-small}{.metadata.labels.node-role\.kubernetes\.io/gpu-tiny}{"\n"}' 2>/dev/null | \
    sort | uniq -c
    
    # Check pod status
    echo
    echo "📦 Llama4scout-decode Pod Status:"
    echo "------------------------------------------------"
    oc get pods -n llm-d -l app=llama4scout-decode --no-headers 2>/dev/null || echo "❌ Could not get pod status"
    
    # Check for any pending/waiting events
    echo
    echo "⚠️  Recent Cluster Events (GPU-related):"
    echo "------------------------------------------------"
    oc get events -A --sort-by='.lastTimestamp' 2>/dev/null | tail -5 | grep -i -E "(gpu|pending|failed|waiting)" || echo "No recent GPU-related events"
    
    echo
    echo "⏳ Next update in 45 seconds... (Ctrl+C to stop)"
    echo "=================================================="
    
    sleep 45
done
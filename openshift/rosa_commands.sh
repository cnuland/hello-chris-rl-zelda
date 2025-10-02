#!/bin/bash
echo "=== Current Machine Pools ==="
rosa list machinepools --cluster=rosa-9p74m

echo ""
echo "=== Creating GPU Inference Machine Pool ==="
rosa create machinepool --cluster=rosa-9p74m \
  --name=gpu-inference \
  --instance-type=g5.2xlarge \
  --replicas=2 \
  --labels="node-role.kubernetes.io/gpu-inference=" \
  --taints="gpu-inference=true:NoSchedule" \
  --yes

echo ""
echo "=== Creating GPU Training Machine Pool ==="
rosa create machinepool --cluster=rosa-9p74m \
  --name=gpu-training \
  --instance-type=g4dn.4xlarge \
  --replicas=3 \
  --labels="node-role.kubernetes.io/gpu-training=" \
  --taints="gpu-training=true:NoSchedule" \
  --yes

echo ""
echo "=== Final Machine Pool Status ==="
rosa list machinepools --cluster=rosa-9p74m

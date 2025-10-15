#!/bin/bash

echo "Checking p4d.24xlarge GPU readiness..."
while true; do
  gpu_capacity=$(oc get node ip-10-0-0-220.us-east-2.compute.internal -o jsonpath='{.status.capacity.nvidia\.com/gpu}' 2>/dev/null || echo "0")
  echo "$(date): GPU capacity = $gpu_capacity"

  if [ "$gpu_capacity" = "8" ]; then
    echo "🎉 SUCCESS! p4d.24xlarge node has 8 A100 GPUs available!"
    break
  elif [ "$gpu_capacity" != "0" ] && [ "$gpu_capacity" != "" ]; then
    echo "⚠️  Partial: $gpu_capacity GPUs available (expecting 8)"
  else
    echo "❌ No GPUs available yet, waiting..."
  fi

  sleep 15
done
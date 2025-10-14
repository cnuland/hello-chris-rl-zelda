#!/usr/bin/env python3
"""
Direct Ray Training Script for Zelda RL
Connects directly to the running Ray cluster and submits training job
Bypasses CodeFlare SDK issues by using Ray client directly
"""

import ray
import os
from pathlib import Path

def main():
    # Set environment variables for production scaling
    # Optimized Ray cluster: 22 CPUs total (1 head @ 4 CPU + 3 workers @ 6 CPU each)
    # Scale to maximize parallel environments
    os.environ['RAY_WORKERS'] = '18'  # 18 env runners (within CPU limits)
    os.environ['ENVS_PER_WORKER'] = '12'  # 12 envs per runner
    os.environ['EPISODE_LENGTH'] = '61440' 
    os.environ['BATCH_SIZE'] = '16384'  # Large batch for distributed training
    os.environ['HUD_URL'] = 'http://zelda-hud-service.zelda-hybrid-rl-llm.svc.cluster.local:8086'
    
    print("🎮 DIRECT RAY ZELDA TRAINING")
    print("="*50)
    print(f"RAY_WORKERS: {os.environ['RAY_WORKERS']}")
    print(f"ENVS_PER_WORKER: {os.environ['ENVS_PER_WORKER']}")
    print(f"Total parallel environments: {int(os.environ['RAY_WORKERS']) * int(os.environ['ENVS_PER_WORKER'])}")
    print(f"EPISODE_LENGTH: {os.environ['EPISODE_LENGTH']}")
    print(f"BATCH_SIZE: {os.environ['BATCH_SIZE']}")
    print(f"HUD_URL: {os.environ['HUD_URL']}")
    print()
    
    # Connect to Ray cluster
    ray_head_service = "zelda-rl-head-svc.zelda-hybrid-rl-llm.svc:10001"
    print(f"🔌 Connecting to Ray cluster at: ray://{ray_head_service}")
    
    try:
        ray.init(address=f"ray://{ray_head_service}")
        print("✅ Connected to Ray cluster!")
        
        # Show Ray cluster resources
        resources = ray.cluster_resources()
        print(f"📊 Ray Cluster Resources: {resources}")
        
        print("🚀 Starting Zelda RL training...")
        
        # Run the training script
        # This will use the existing run-ray-zelda.py but with the environment variables set
        import subprocess
        result = subprocess.run(['python', 'run-ray-zelda.py'], 
                              cwd=Path('.'), 
                              capture_output=False)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed!")
            return result.returncode
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        try:
            ray.shutdown()
            print("🔌 Disconnected from Ray cluster")
        except:
            pass
    
    return 0

if __name__ == "__main__":
    exit(main())
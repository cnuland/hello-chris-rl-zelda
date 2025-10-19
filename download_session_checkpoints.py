#!/usr/bin/env python3
"""
Download session checkpoints from MinIO S3 to local directory.
"""

import boto3
import os
import json
from pathlib import Path

# MinIO configuration  
S3_ENDPOINT = "https://minio-api-route-minio-system.apps.rosa.rosa-58cx6.acrs.p3.openshiftapps.com"
S3_ACCESS_KEY = "admin"
S3_SECRET_KEY = "zelda-rl-minio-2024"
BUCKET = "sessions"
SESSION_ID = "ray_training_1760751526"  # Overnight run with +15,111 returns

# Create S3 client
s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name='us-east-1'
)

# Create local checkpoint directory
local_dir = Path("checkpoints") / SESSION_ID
local_dir.mkdir(parents=True, exist_ok=True)

print(f"📥 Downloading checkpoints from s3://{BUCKET}/{SESSION_ID}/")
print(f"💾 Saving to: {local_dir}")
print()

# List and download all files in the session
try:
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{SESSION_ID}/")
    
    file_count = 0
    total_bytes = 0
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            s3_key = obj['Key']
            size = obj['Size']
            
            # Create local file path
            relative_path = s3_key.replace(f"{SESSION_ID}/", "")
            local_file = local_dir / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            print(f"📥 {relative_path} ({size:,} bytes)...", end=' ')
            s3.download_file(BUCKET, s3_key, str(local_file))
            print("✅")
            
            file_count += 1
            total_bytes += size
    
    print()
    print(f"✅ Downloaded {file_count} files ({total_bytes:,} bytes total)")
    print(f"📂 Location: {local_dir}")
    print()
    
    # Find the best checkpoint (highest episode number per worker)
    checkpoint_files = list(local_dir.glob("**/episode_*_checkpoint.json"))
    
    if checkpoint_files:
        print(f"📊 Found {len(checkpoint_files)} checkpoint files")
        print()
        print("🏆 Latest checkpoints per worker:")
        
        # Group by worker and find latest
        worker_checkpoints = {}
        for ckpt_file in checkpoint_files:
            worker = ckpt_file.parent.name
            episode_num = int(ckpt_file.stem.split('_')[1])
            
            if worker not in worker_checkpoints or episode_num > worker_checkpoints[worker][0]:
                worker_checkpoints[worker] = (episode_num, ckpt_file)
        
        # Show latest for each worker
        for worker, (episode, ckpt_file) in sorted(worker_checkpoints.items()):
            # Load checkpoint to see stats
            with open(ckpt_file, 'r') as f:
                ckpt_data = json.load(f)
            
            total_reward = ckpt_data.get('total_reward', 0)
            steps = ckpt_data.get('steps', 0)
            avg_reward = total_reward / steps if steps > 0 else 0
            
            print(f"  {worker}: Episode {episode:3d} | Reward: {total_reward:8.1f} | Steps: {steps:5d} | Avg: {avg_reward:.3f}")
        
        print()
        print(f"💡 For your demo, use the checkpoint with highest total_reward")
    else:
        print("⚠️  No checkpoint files found")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()


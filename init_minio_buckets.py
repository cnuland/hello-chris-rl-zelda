#!/usr/bin/env python3
"""
Initialize MinIO buckets for Zelda RL training
Creates required buckets and uploads ROM files
"""

import os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

def get_minio_client(endpoint_url, access_key, secret_key):
    """Create MinIO/S3 client"""
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='us-east-1'
    )

def create_bucket(client, bucket_name):
    """Create bucket if it doesn't exist"""
    try:
        client.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' already exists")
        return True
    except ClientError:
        try:
            client.create_bucket(Bucket=bucket_name)
            print(f"✅ Created bucket '{bucket_name}'")
            return True
        except ClientError as e:
            print(f"❌ Error creating bucket '{bucket_name}': {e}")
            return False

def upload_file(client, file_path, bucket_name, object_name=None):
    """Upload file to MinIO bucket"""
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return False
    
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    try:
        client.upload_file(file_path, bucket_name, object_name)
        print(f"✅ Uploaded {file_path} to {bucket_name}/{object_name}")
        return True
    except ClientError as e:
        print(f"❌ Error uploading {file_path}: {e}")
        return False

def main():
    # MinIO configuration
    # These will be the external route URLs once deployed
    endpoint_url = input("MinIO API endpoint (e.g., https://minio-api-route-minio-system.apps.rosa...): ").strip()
    if not endpoint_url:
        print("❌ Endpoint URL is required")
        return 1
    
    access_key = input("MinIO access key [admin]: ").strip() or "admin"
    secret_key = input("MinIO secret key [zelda-rl-minio-2024]: ").strip() or "zelda-rl-minio-2024"
    
    print("🔌 Connecting to MinIO...")
    client = get_minio_client(endpoint_url, access_key, secret_key)
    
    # Test connection
    try:
        client.list_buckets()
        print("✅ Connected to MinIO successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to MinIO: {e}")
        return 1
    
    # Create required buckets
    buckets = [
        'zelda-rl-checkpoints',  # Model checkpoints and training state
        'zelda-rl-logs',         # Training logs and metrics
        'zelda-rl-models',       # Final trained models
        'roms',                  # Game ROM files
        'configs',               # Configuration files
    ]
    
    print("📦 Creating buckets...")
    for bucket in buckets:
        create_bucket(client, bucket)
    
    # Upload ROM files if they exist
    rom_files = [
        'roms/zelda_oracle_of_seasons.gbc',
        'roms/zelda_oracle_of_seasons.gbc.state',
    ]
    
    print("📁 Uploading ROM files...")
    for rom_file in rom_files:
        if os.path.exists(rom_file):
            upload_file(client, rom_file, 'roms')
        else:
            print(f"⚠️  ROM file not found: {rom_file}")
    
    # Upload configuration files
    config_files = [
        'configs/env.yaml',
        'configs/vision_prompt.yaml',
    ]
    
    print("⚙️  Uploading configuration files...")
    for config_file in config_files:
        if os.path.exists(config_file):
            upload_file(client, config_file, 'configs')
        else:
            print(f"⚠️  Config file not found: {config_file}")
    
    print("\n🎉 MinIO initialization complete!")
    print("\n📋 Bucket Summary:")
    
    try:
        buckets_response = client.list_buckets()
        for bucket in buckets_response['Buckets']:
            name = bucket['Name']
            created = bucket['CreationDate'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"  • {name} (created: {created})")
            
            # List objects in bucket
            try:
                objects = client.list_objects_v2(Bucket=name)
                if 'Contents' in objects:
                    print(f"    Objects: {len(objects['Contents'])}")
                    for obj in objects['Contents'][:3]:  # Show first 3 objects
                        size = obj['Size']
                        print(f"      - {obj['Key']} ({size} bytes)")
                    if len(objects['Contents']) > 3:
                        print(f"      ... and {len(objects['Contents']) - 3} more")
                else:
                    print("    Objects: 0")
            except Exception as e:
                print(f"    Error listing objects: {e}")
    except Exception as e:
        print(f"❌ Error listing buckets: {e}")
    
    print(f"\n🌐 MinIO Access URLs:")
    print(f"  API Endpoint: {endpoint_url}")
    print(f"  Console: {endpoint_url.replace('minio-api-route', 'minio-console-route')}")
    print(f"  Access Key: {access_key}")
    print(f"  Secret Key: {secret_key}")
    
    return 0

if __name__ == "__main__":
    exit(main())
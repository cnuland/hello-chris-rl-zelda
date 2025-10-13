"""
Initialize ROM file for Ray workers.
This script runs before training starts to set up the ROM file.
"""
import os
import base64
from pathlib import Path

# Base64 encoded ROM file (you'll need to encode your ROM)
# For now, we'll download from S3 or copy from a mounted volume

def init_rom_from_s3():
    """Download ROM and save files from S3/MinIO."""
    import boto3
    from botocore.client import Config
    
    # Get S3 config from environment
    s3_endpoint = os.environ.get('S3_ENDPOINT_URL')
    s3_bucket = 'roms'  # User's ROM bucket
    
    if not s3_endpoint:
        print("⚠️  S3_ENDPOINT_URL not set, skipping ROM download")
        return False
    
    print(f"📥 Downloading ROM and save files from S3...")
    print(f"   Endpoint: {s3_endpoint}")
    print(f"   Bucket: {s3_bucket}")
    
    try:
        # Create S3 client
        s3 = boto3.client(
            's3',
            endpoint_url=s3_endpoint,
            aws_access_key_id=os.environ.get('S3_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('S3_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('S3_REGION_NAME', 'us-east-1'),
            config=Config(signature_version='s3v4')
        )
        
        # Create roms directory
        rom_dir = Path('roms')
        rom_dir.mkdir(exist_ok=True)
        
        # List all files in the roms bucket
        print(f"   Listing files in bucket...")
        response = s3.list_objects_v2(Bucket=s3_bucket)
        
        if 'Contents' not in response:
            print(f"   ⚠️  No files found in bucket")
            return False
        
        # Download all files
        downloaded = 0
        for obj in response['Contents']:
            key = obj['Key']
            filename = key.split('/')[-1]  # Get filename from key
            local_path = rom_dir / filename
            
            print(f"   Downloading: {key} -> {local_path}")
            s3.download_file(s3_bucket, key, str(local_path))
            
            file_size = local_path.stat().st_size / 1024
            print(f"      ✅ {file_size:.1f} KB")
            downloaded += 1
        
        print(f"\n✅ Downloaded {downloaded} file(s) to {rom_dir}/")
        
        # Verify the main ROM file exists
        rom_path = rom_dir / 'zelda_oracle_of_seasons.gbc'
        if rom_path.exists():
            print(f"   ✅ Main ROM file found: {rom_path}")
            return True
        else:
            print(f"   ⚠️  Main ROM file not found: {rom_path}")
            print(f"   Available files: {list(rom_dir.iterdir())}")
            return False
        
    except Exception as e:
        print(f"❌ Failed to download ROM: {e}")
        import traceback
        traceback.print_exc()
        return False

def init_rom_from_url():
    """Download ROM from HTTP URL."""
    import requests
    
    rom_url = os.environ.get('ROM_URL')
    if not rom_url:
        print("⚠️  ROM_URL not set, skipping ROM download")
        return False
    
    print(f"📥 Downloading ROM from URL: {rom_url}")
    
    try:
        # Create roms directory
        rom_dir = Path('roms')
        rom_dir.mkdir(exist_ok=True)
        
        # Download ROM
        response = requests.get(rom_url, stream=True)
        response.raise_for_status()
        
        rom_path = rom_dir / 'zelda_oracle_of_seasons.gbc'
        with open(rom_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ ROM downloaded to: {rom_path}")
        print(f"   Size: {rom_path.stat().st_size / 1024:.1f} KB")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download ROM: {e}")
        return False

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎮 Initializing ROM file for Ray workers")
    print("="*70)
    
    # Try S3 first, then URL
    success = init_rom_from_s3() or init_rom_from_url()
    
    if not success:
        print("\n❌ Failed to initialize ROM file!")
        print("   Please set either:")
        print("   - S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY")
        print("   - ROM_URL (HTTP URL to ROM file)")
        exit(1)
    
    print("="*70 + "\n")


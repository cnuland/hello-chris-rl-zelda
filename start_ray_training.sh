#!/bin/bash
# Wrapper script to download ROM from S3 before starting Ray training

set -e  # Exit on error

echo "🚀 Starting Ray Training for Zelda Oracle of Seasons"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Step 1: Download ROM from S3
echo ""
echo "📥 Step 1: Downloading ROM files from S3..."
python init_rom.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to download ROM files!"
    exit 1
fi

# Step 2: Start training
echo ""
echo "🎮 Step 2: Starting PPO training..."
python run-ray-zelda.py

echo ""
echo "✅ Training completed!"


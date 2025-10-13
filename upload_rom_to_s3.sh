#!/bin/bash
# Upload Zelda ROM to S3/MinIO for Ray workers to download

# Configuration
S3_ENDPOINT="YOUR_S3_ENDPOINT"  # Replace with your MinIO endpoint
S3_BUCKET="zelda-rl-assets"
ROM_FILE="roms/zelda_oracle_of_seasons.gbc"

echo "📤 Uploading ROM file to S3/MinIO..."
echo "   Endpoint: $S3_ENDPOINT"
echo "   Bucket: $S3_BUCKET"
echo "   File: $ROM_FILE"

# Upload using AWS CLI (works with MinIO)
aws s3 cp "$ROM_FILE" "s3://$S3_BUCKET/roms/zelda_oracle_of_seasons.gbc" \
    --endpoint-url "$S3_ENDPOINT"

if [ $? -eq 0 ]; then
    echo "✅ ROM uploaded successfully!"
    echo ""
    echo "Update your environment variables in the notebook:"
    echo "  ROM_S3_PATH=s3://$S3_BUCKET/roms/zelda_oracle_of_seasons.gbc"
else
    echo "❌ Upload failed!"
    exit 1
fi


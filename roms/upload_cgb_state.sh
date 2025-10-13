#!/bin/bash

# Upload the CGB save state to S3/MinIO
# This replaces the old DMG save state with the new CGB-compatible one

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📤 Uploading CGB Save State to S3"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Configuration
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://minio-s3-minio.apps.rosa.rosa-9p74m.h96u.p3.openshiftapps.com}"
S3_BUCKET="roms"
NEW_STATE_FILE="zelda_oracle_of_seasons_CGB.gbc.state"
TARGET_NAME="zelda_oracle_of_seasons.gbc.state"  # Replace the old one

# Check if the new CGB state file exists
if [ ! -f "$NEW_STATE_FILE" ]; then
    echo "❌ Error: $NEW_STATE_FILE not found!"
    echo "   Run: python convert_save_state_to_cgb.py first"
    exit 1
fi

echo "📁 Source file: $NEW_STATE_FILE"
echo "📁 S3 destination: s3://${S3_BUCKET}/${TARGET_NAME}"
echo "🌐 S3 endpoint: $S3_ENDPOINT_URL"
echo ""

# Backup the old state (optional)
BACKUP_NAME="zelda_oracle_of_seasons_DMG_BACKUP.gbc.state"
echo "💾 Creating backup of old DMG state..."
aws s3 cp "s3://${S3_BUCKET}/${TARGET_NAME}" \
    "s3://${S3_BUCKET}/${BACKUP_NAME}" \
    --endpoint-url "${S3_ENDPOINT_URL}" 2>/dev/null || echo "   (No existing state to backup)"
echo ""

# Upload the new CGB state
echo "📤 Uploading new CGB save state..."
aws s3 cp "$NEW_STATE_FILE" \
    "s3://${S3_BUCKET}/${TARGET_NAME}" \
    --endpoint-url "${S3_ENDPOINT_URL}"

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ SUCCESS! CGB Save State Uploaded"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. Commit the updated emulator/pyboy_bridge.py (re-enable save state)"
    echo "   2. Push to git"
    echo "   3. Restart the Ray training job"
    echo "   4. Training should now work without CGB errors! 🎮"
    echo ""
else
    echo ""
    echo "❌ Upload failed!"
    exit 1
fi


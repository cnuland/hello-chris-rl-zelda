#!/bin/bash
# Complete MinIO setup for Zelda RL training
# Deploys MinIO and initializes buckets

set -e

echo "🗄️  SETTING UP MINIO FOR ZELDA RL TRAINING"
echo "=========================================="
echo ""

# Step 1: Deploy MinIO if not already deployed
echo "📦 Step 1: Checking MinIO deployment status..."

if oc get deployment minio -n minio-system >/dev/null 2>&1; then
    echo "✅ MinIO already deployed"
else
    echo "🚀 Deploying MinIO..."
    oc apply -f ops/openshift/minio-deployment.yaml
    
    echo "⏳ Waiting for MinIO pod to be ready..."
    oc wait --for=condition=available deployment/minio -n minio-system --timeout=300s
fi

# Step 2: Get MinIO URLs
echo ""
echo "🌐 Step 2: Getting MinIO access URLs..."

API_ROUTE=$(oc get route minio-api-route -n minio-system -o jsonpath='{.spec.host}')
CONSOLE_ROUTE=$(oc get route minio-console-route -n minio-system -o jsonpath='{.spec.host}')

API_URL="https://$API_ROUTE"
CONSOLE_URL="https://$CONSOLE_ROUTE"

echo "✅ MinIO URLs retrieved:"
echo "   API Endpoint: $API_URL"
echo "   Web Console:  $CONSOLE_URL"

# Step 3: Display access credentials
echo ""
echo "🔑 Step 3: MinIO Access Credentials:"
echo "   Access Key: admin"
echo "   Secret Key: zelda-rl-minio-2024"
echo ""

# Step 4: Initialize buckets
echo "🪣 Step 4: Initialize MinIO buckets..."
echo ""
echo "Run this command to set up buckets and upload files:"
echo "python init_minio_buckets.py"
echo ""
echo "When prompted, use these values:"
echo "  MinIO API endpoint: $API_URL"
echo "  Access key: admin"
echo "  Secret key: zelda-rl-minio-2024"

# Step 5: Update environment variables for training
echo ""
echo "⚙️  Step 5: Environment variables for training jobs:"
echo ""
echo "export S3_ENDPOINT_URL='$API_URL'"
echo "export S3_ACCESS_KEY_ID='admin'"
echo "export S3_SECRET_ACCESS_KEY='zelda-rl-minio-2024'"
echo "export S3_REGION_NAME='us-east-1'"
echo "export S3_BUCKET_NAME='zelda-rl-checkpoints'"
echo ""

# Step 6: Create environment file for easy sourcing
ENV_FILE="minio_env.sh"
echo "📝 Step 6: Creating environment file: $ENV_FILE"

cat > $ENV_FILE << EOF
# MinIO Environment Variables for Zelda RL Training
export S3_ENDPOINT_URL='$API_URL'
export S3_ACCESS_KEY_ID='admin'
export S3_SECRET_ACCESS_KEY='zelda-rl-minio-2024'
export S3_REGION_NAME='us-east-1'
export S3_BUCKET_NAME='zelda-rl-checkpoints'

# MinIO URLs
export MINIO_API_URL='$API_URL'
export MINIO_CONSOLE_URL='$CONSOLE_URL'

echo "✅ MinIO environment variables loaded"
echo "   API: \$S3_ENDPOINT_URL"
echo "   Console: \$MINIO_CONSOLE_URL"
EOF

echo "✅ Environment file created: $ENV_FILE"
echo ""

# Step 7: Summary
echo "📋 SETUP COMPLETE!"
echo "=================="
echo ""
echo "🌐 MinIO Access:"
echo "   Web Console: $CONSOLE_URL"
echo "   API Endpoint: $API_URL"
echo "   Username: admin"
echo "   Password: zelda-rl-minio-2024"
echo ""
echo "📁 Storage:"
echo "   • 100GB persistent volume"
echo "   • Buckets: zelda-rl-checkpoints, zelda-rl-logs, zelda-rl-models, roms, configs"
echo ""
echo "🚀 Next Steps:"
echo "   1. Source environment: source $ENV_FILE"
echo "   2. Initialize buckets: python init_minio_buckets.py"
echo "   3. Upload ROM files to the 'roms' bucket"
echo "   4. Run training with S3 support enabled"
echo ""
echo "💡 The Ray training will now automatically:"
echo "   • Save model checkpoints to MinIO"
echo "   • Store training logs and metrics"
echo "   • Download ROM files from the 'roms' bucket"
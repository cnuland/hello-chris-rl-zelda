# MinIO Environment Variables for Zelda RL Training
export S3_ENDPOINT_URL='https://minio-api-route-minio-system.apps.rosa.rosa-58cx6.acrs.p3.openshiftapps.com'
export S3_ACCESS_KEY_ID='admin'
export S3_SECRET_ACCESS_KEY='zelda-rl-minio-2024'
export S3_REGION_NAME='us-east-1'
export S3_BUCKET_NAME='zelda-rl-checkpoints'

# MinIO URLs
export MINIO_API_URL='https://minio-api-route-minio-system.apps.rosa.rosa-58cx6.acrs.p3.openshiftapps.com'
export MINIO_CONSOLE_URL='https://minio-console-route-minio-system.apps.rosa.rosa-58cx6.acrs.p3.openshiftapps.com'

echo "✅ MinIO environment variables loaded"
echo "   API: $S3_ENDPOINT_URL"
echo "   Console: $MINIO_CONSOLE_URL"

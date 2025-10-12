#!/bin/bash
# Build and push HUD container image

set -e

# Configuration
IMAGE_NAME="quay.io/cnuland/zelda-hud"
VERSION="${1:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${VERSION}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🐳 Building HUD Container Image"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Image: ${FULL_IMAGE}"
echo ""

# Change to HUD directory
cd "$(dirname "$0")"

# Build the image
echo "📦 Building image..."
podman build -t "${FULL_IMAGE}" -f Containerfile .

echo ""
echo "✅ Build complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Image Details:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
podman images "${IMAGE_NAME}"
echo ""

# Prompt for push
read -p "📤 Push to registry? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "📤 Pushing to registry..."
    podman push "${FULL_IMAGE}"
    echo "✅ Push complete!"
else
    echo "⏭️  Skipping push"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Usage:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Test locally:"
echo "  podman run -p 8086:8086 ${FULL_IMAGE}"
echo ""
echo "Deploy to OpenShift:"
echo "  Update image in ../ops/openshift/hud-deployment.yaml"
echo "  oc apply -f ../ops/openshift/hud-deployment.yaml"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


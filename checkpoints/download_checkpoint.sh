#!/bin/bash
# Helper script to download best checkpoint from Ray cluster

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 DOWNLOADING BEST CHECKPOINT FROM RAY CLUSTER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if logged in
if ! oc whoami &>/dev/null; then
    echo "❌ Not logged in to OpenShift!"
    echo ""
    echo "Please login first:"
    echo "  oc login --token=<YOUR_TOKEN> --server=https://api.rosa-58cx6.acrs.p3.openshiftapps.com:443"
    echo "  oc project zelda-hybrid-rl-llm"
    exit 1
fi

echo "✅ Logged in as: $(oc whoami)"
echo ""

# Find session ID
echo "🔍 Finding Ray session..."
SESSION_ID=$(oc exec zelda-rl-head-s9rdj -- ls /tmp/ray/ 2>/dev/null | grep "2025-10-16" | head -1)

if [ -z "$SESSION_ID" ]; then
    echo "❌ Could not find Ray session from 2025-10-16"
    exit 1
fi

echo "✅ Found session: $SESSION_ID"
echo ""

# Check checkpoint exists
echo "🔍 Checking for checkpoint_000800..."
CHECKPOINT_PATH="/tmp/ray/$SESSION_ID/PPO_ZeldaOracleSeasons/PPO_zelda_env_4166c_00000/checkpoint_000800"
if ! oc exec zelda-rl-head-s9rdj -- test -d "$CHECKPOINT_PATH" 2>/dev/null; then
    echo "❌ Checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    oc exec zelda-rl-head-s9rdj -- ls /tmp/ray/$SESSION_ID/PPO_ZeldaOracleSeasons/PPO_zelda_env_4166c_00000/ | grep checkpoint | tail -10
    exit 1
fi

echo "✅ Checkpoint found!"
echo ""

# Check size
echo "📊 Checkpoint size:"
oc exec zelda-rl-head-s9rdj -- du -sh "$CHECKPOINT_PATH" 2>/dev/null || echo "Could not determine size"
echo ""

# Download
echo "📥 Downloading checkpoint..."
oc rsync \
  zelda-rl-head-s9rdj:"$CHECKPOINT_PATH" \
  ./checkpoints/ \
  -c ray-head

echo ""
echo "✅ Download complete!"
echo ""

# Verify
echo "📂 Downloaded files:"
ls -lh checkpoints/checkpoint_000800/ | head -10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ CHECKPOINT READY!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. git add checkpoints/checkpoint_000800"
echo "  2. git commit -m 'Add best checkpoint (iter 800, +8000 rewards)'"
echo "  3. git push origin main"
echo ""

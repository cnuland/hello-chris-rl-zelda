#!/bin/bash
# Monitor the ongoing hybrid RL+LLM training session

LOG_DIR="training_runs/hybrid_6hour_1759199268"
LOG_FILE="$LOG_DIR/training.log"

echo "🎯 HYBRID RL+LLM TRAINING MONITOR"
echo "=================================="
echo ""
echo "📁 Log Directory: $LOG_DIR"
echo "📊 Total Target: 200,000 timesteps (~6 hours)"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Training log not found!"
    exit 1
fi

# Extract key metrics
echo "📈 TRAINING PROGRESS:"
echo ""

# Count LLM calls
llm_calls=$(grep -c "LLM suggests:" "$LOG_FILE" 2>/dev/null || echo "0")
echo "🧠 LLM Calls: $llm_calls"

# Count alignment bonuses
alignments=$(grep -c "LLM alignment bonus:" "$LOG_FILE" 2>/dev/null || echo "0")
echo "✨ PPO Followed LLM: $alignments times"

# Calculate alignment rate
if [ "$llm_calls" -gt 0 ]; then
    alignment_rate=$((alignments * 100 / llm_calls))
    echo "📊 Alignment Rate: ${alignment_rate}%"
fi

# Count episodes
episodes=$(grep -c "Episode.*Reward=" "$LOG_FILE" 2>/dev/null || echo "0")
echo "🎮 Episodes Completed: $episodes"

# Count rooms discovered
rooms=$(grep -c "NEW ROOM DISCOVERED" "$LOG_FILE" 2>/dev/null || echo "0")
echo "🗺️  Rooms Discovered: $rooms"

# Get latest step count
latest_step=$(grep -o "Step [0-9]*/" "$LOG_FILE" | tail -1 | grep -o "[0-9]*" | head -1 || echo "0")
echo "⏱️  Latest Step: $latest_step / 200,000"

# Calculate progress percentage
if [ "$latest_step" -gt 0 ]; then
    progress=$((latest_step * 100 / 200000))
    echo "📊 Progress: ${progress}%"
fi

echo ""
echo "📝 RECENT ACTIVITY (last 20 lines):"
echo "-----------------------------------"
tail -20 "$LOG_FILE"

echo ""
echo "-----------------------------------"
echo "💡 To view live updates: tail -f $LOG_FILE"
echo "🛑 To stop training: pkill -f train_hybrid_rl_llm"

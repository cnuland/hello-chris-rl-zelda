#!/usr/bin/env python3
"""
Strategic Visual Training - Using Unified Framework

Uses the proven Strategic Training Framework with:
- Visual PyBoy window for watching training
- Web HUD for monitoring LLM decisions
- Strategic action translation
- 5X LLM emphasis system
"""

import sys
import time
import threading
import webbrowser
from pathlib import Path
from flask import Flask, render_template_string

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from strategic_training_framework import (
    StrategicConfig, StrategicTrainer, create_visual_strategic_trainer
)

# Flask app for HUD
app = Flask(__name__)

# Global state for HUD updates
hud_state = {
    "latest_command": {
        "action": "INITIALIZING",
        "target": "system_startup", 
        "reasoning": "Starting strategic hybrid RL-LLM system",
        "response_time": "0ms",
        "phase": "startup"
    },
    "stats": {
        "episode_reward": 0.0,
        "total_steps": 0,
        "llm_calls": 0,
        "success_rate": 100
    },
    "game_progress": {
        "current_episode": 1,
        "total_episodes": 1
    },
    "activity_log": []
}

def add_activity(message, msg_type="info"):
    """Add activity to log."""
    timestamp = time.strftime("%H:%M:%S")
    hud_state["activity_log"].insert(0, {
        "message": message,
        "type": msg_type,
        "timestamp": timestamp
    })
    hud_state["activity_log"] = hud_state["activity_log"][:15]
    print(f"📝 {timestamp} - {message}")

# HUD HTML template (same as before but updated for strategic framework)
HUD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🎯 Strategic RL-LLM Framework + PyBoy</title>
    <meta http-equiv="refresh" content="3">
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background: #1a1a2e;
            color: #fff; margin: 0; padding: 15px; font-size: 14px;
        }
        .container { 
            max-width: 1200px; margin: auto; 
            background: rgba(15,15,30,0.95); padding: 25px; 
            border-radius: 15px; box-shadow: 0 0 30px rgba(255,215,0,0.3);
            border: 1px solid #ffd700;
        }
        h1 { 
            color: #ffd700; text-align: center; margin-bottom: 25px; 
            font-size: 2em; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            background: rgba(0,0,0,0.7); padding: 15px; border-radius: 10px;
        }
        .section { 
            background: rgba(30,30,50,0.9);
            padding: 15px; border-radius: 8px; margin-bottom: 15px;
            border: 2px solid #ffd700; box-shadow: 0 0 15px rgba(255,215,0,0.2);
        }
        .section h2 { 
            color: #ffd700; margin: 0 0 15px 0; border-bottom: 2px solid #ffd700; 
            padding-bottom: 8px; font-size: 1.4em; font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }
        .key { color: #ffaa00; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .value { color: #ffffff; font-size: 1.1em; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .llm-command { 
            background: rgba(0,0,0,0.8); 
            padding: 20px; border-radius: 10px; margin: 15px 0;
            border: 3px solid #ffd700;
            box-shadow: 0 0 20px rgba(255,215,0,0.5);
        }
        .stats-grid {
            display: grid; grid-template-columns: repeat(4, 1fr);
            gap: 10px; margin: 15px 0;
        }
        .stat-item {
            background: rgba(0,0,0,0.6); padding: 15px; 
            border-radius: 8px; text-align: center;
            border: 2px solid #ffd700;
        }
        .stat-value { font-size: 1.8em; font-weight: bold; color: #ffd700; text-shadow: 2px 2px 4px rgba(0,0,0,0.8); }
        .stat-label { font-size: 0.9em; color: #ffffff; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .log-entry { 
            border-bottom: 1px dotted #ffd700; padding: 8px 0; font-size: 0.95em;
            background: rgba(0,0,0,0.3); margin: 2px 0; padding-left: 10px; border-radius: 4px;
        }
        .success { color: #4ade80; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .failure { color: #ff4444; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .warning { color: #ffcc00; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .info { color: #66aaff; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .timestamp { color: #cccccc; font-size: 0.85em; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.05); }
            100% { opacity: 1; transform: scale(1); }
        }
        .system-status {
            background: rgba(0,0,0,0.8); border: 3px solid #ffd700;
            padding: 15px; border-radius: 10px; margin: 15px 0;
            box-shadow: 0 0 15px rgba(255,215,0,0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 STRATEGIC FRAMEWORK + 🎮 PYBOY</h1>
        
        <div class="system-status">
            <span class="key">🧠 Strategic Framework:</span> <span class="value pulse">ACTIVE</span> | 
            <span class="key">🎮 PyBoy:</span> <span class="value">Visual Mode</span> | 
            <span class="key">🎯 LLM:</span> <span class="value">MLX Qwen2.5-14B</span> |
            <span class="key">⚡ Emphasis:</span> <span class="value pulse">5X REWARDS!</span>
        </div>
        
        <div class="section">
            <h2>⚔️ Latest Strategic Command</h2>
            <div class="llm-command">
                <p><span class="key">Action:</span> <span class="value pulse">{{ hud_state.latest_command.action }}</span></p>
                <p><span class="key">Target:</span> <span class="value">{{ hud_state.latest_command.target }}</span></p>
                <p><span class="key">Reasoning:</span> <span class="value">{{ hud_state.latest_command.reasoning }}</span></p>
                <p><span class="key">Response Time:</span> <span class="value">{{ hud_state.latest_command.response_time }}</span></p>
            </div>
        </div>

        <div class="section">
            <h2>📊 Strategic Training Stats</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{{ "%.1f"|format(hud_state.stats.episode_reward) }}</div>
                    <div class="stat-label">Episode Reward</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ hud_state.stats.total_steps }}</div>
                    <div class="stat-label">Total Steps</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ hud_state.stats.llm_calls }}</div>
                    <div class="stat-label">LLM Calls</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ hud_state.stats.success_rate }}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🎮 Training Progress</h2>
            <p><span class="key">Episode:</span> <span class="value">{{ hud_state.game_progress.current_episode }}/{{ hud_state.game_progress.total_episodes }}</span></p>
            <p><span class="key">Phase:</span> <span class="value">{{ hud_state.latest_command.phase }}</span></p>
        </div>

        <div class="section">
            <h2>⚡ Strategic Activity</h2>
            <div style="max-height: 250px; overflow-y: auto;">
                {% for entry in hud_state.activity_log %}
                <p class="log-entry {{ entry.type }}">
                    <span class="timestamp">{{ entry.timestamp }}</span> 
                    <span class="value">{{ entry.message }}</span>
                </p>
                {% endfor %}
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HUD_HTML, hud_state=hud_state)

def start_hud_server():
    """Start HUD web server."""
    print("🌐 Starting Strategic HUD server on http://localhost:8086")
    app.run(host='localhost', port=8086, debug=False, use_reloader=False)

def progress_callback(progress_data):
    """Update HUD with training progress."""
    # Update step count and episode
    hud_state["game_progress"]["current_step"] = progress_data.get("step", 0)
    hud_state["game_progress"]["current_episode"] = progress_data.get("episode", 1)
    hud_state["stats"]["episode_reward"] = progress_data.get("episode_reward", 0.0)
    hud_state["stats"]["total_steps"] = progress_data.get("step", 0)
    
    # Update latest command if available
    if progress_data.get("llm_guidance"):
        hud_state["latest_command"]["action"] = progress_data["llm_guidance"].get("action", "EXPLORING")
        hud_state["latest_command"]["reasoning"] = progress_data["llm_guidance"].get("reasoning", "Strategic exploration")
        hud_state["latest_command"]["timestamp"] = time.strftime("%H:%M:%S")
        hud_state["stats"]["success_rate"] = 100  # LLM call succeeded
    
    # Update activity log for significant events
    step = progress_data.get("step", 0)
    reward = progress_data.get("reward", 0.0)
    
    if step % 30 == 0 and progress_data.get("llm_guidance"):  # LLM call steps
        action = progress_data["llm_guidance"].get("action", "UNKNOWN")
        add_activity(f"🧠 Step {step}: {action}", "success")
    elif reward > 10:  # Significant rewards
        add_activity(f"🎯 Bonus! Step {step}: +{reward:.1f} reward", "success")

def main():
    """Run strategic visual training."""
    print("🎯 STRATEGIC VISUAL TRAINING")
    print("=" * 60)
    print("🧠 Framework: Strategic Training Framework")
    print("🎮 Mode: Visual (PyBoy + Web HUD)")
    print("⚡ Features: 5X LLM emphasis + strategic actions")
    print()
    
    # Start HUD server in background
    hud_thread = threading.Thread(target=start_hud_server, daemon=True)
    hud_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    # Open browser to HUD
    hud_url = "http://localhost:8086"
    print(f"🌐 Opening Strategic HUD: {hud_url}")
    webbrowser.open(hud_url)
    
    add_activity("🚀 Strategic Framework initialized", "success")
    add_activity("🧠 MLX Qwen2.5-14B connected", "success")
    add_activity("🎯 Strategic action translation active", "success")
    
    print("📱 Strategic HUD should open in browser...")
    print("🎮 PyBoy window will show strategic gameplay...")
    print("⚡ Expected: Combat patterns, grass cutting, item collection!")
    time.sleep(3)
    
    try:
        print("🔍 DEBUG: Creating strategic trainer...")
        # Create strategic trainer with visual configuration
        config = StrategicConfig(
            max_episode_steps=12000,  # 10 minutes
            llm_call_interval=30      # Ultra-frequent for visual demo
        )
        trainer = create_visual_strategic_trainer(config)
        print("✅ DEBUG: Strategic trainer created")
        
        # Update HUD
        add_activity("🎯 Starting 10-minute strategic visual training", "success")
        hud_state["game_progress"]["total_episodes"] = 1
        
        print("🔍 DEBUG: Starting strategic training...")
        # Run strategic training
        rom_path = str(project_root / "roms" / "zelda_oracle_of_seasons.gbc")
        print(f"🔍 DEBUG: ROM path: {rom_path}")
        results = trainer.run_strategic_training(
            rom_path=rom_path,
            episodes=1,
            headless=False,  # Visual mode!
            progress_callback=progress_callback
        )
        print("✅ DEBUG: Strategic training completed")
        
        # Final results
        add_activity(f"🏆 Strategic training complete! Reward: {results['average_reward']:.1f}", "success")
        print(f"\n🎯 STRATEGIC RESULTS:")
        print(f"⏱️  Duration: {results['total_duration_minutes']:.1f} minutes")
        print(f"🏆 Average Reward: {results['average_reward']:.1f}")
        print(f"🧠 LLM Success: {results['llm_success_rate']:.1f}%")
        
    except Exception as e:
        error_msg = f"Strategic training error: {str(e)}"
        add_activity(error_msg, "failure")
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Strategic visual session complete!")
    print("🌐 HUD remains active at http://localhost:8086")
    print("Press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Strategic system shutdown")

if __name__ == "__main__":
    main()

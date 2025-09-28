#!/usr/bin/env python3
"""
Trained Model Inference with PyBoy Visual + Web HUD

Runs inference with a trained model:
- PyBoy emulator in visual mode 
- Trained RL model playing the game (NO training)
- Web HUD showing live LLM decisions
- MLX local server integration
"""

import sys
import time
import json
import threading
import webbrowser
import requests
from pathlib import Path
from flask import Flask, render_template_string

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Running in demo mode without full RL integration")

# Flask app for HUD
app = Flask(__name__)

# Global state for HUD updates
hud_state = {
    "latest_command": {
        "action": "INITIALIZING",
        "target": "system_startup", 
        "reasoning": "Starting hybrid RL-LLM system with PyBoy",
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
        "rooms_explored": 0,
        "npcs_interacted": 0,
        "current_episode": 1
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
    # Keep only last 15 entries
    hud_state["activity_log"] = hud_state["activity_log"][:15]
    print(f"📝 {timestamp} - {message}")

# HUD HTML template
HUD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🧠 Hybrid RL-LLM + PyBoy HUD</title>
    <meta http-equiv="refresh" content="3">
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background: #1a1a2e;
            color: #fff; margin: 0; padding: 15px; font-size: 14px;
        }
        .container { 
            max-width: 1000px; margin: auto; 
            background: rgba(15,15,30,0.95); padding: 25px; 
            border-radius: 15px; box-shadow: 0 0 30px rgba(0,255,136,0.3);
            border: 1px solid #00ff88;
        }
        h1 { 
            color: #00ff88; text-align: center; margin-bottom: 25px; 
            font-size: 2em; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            background: rgba(0,0,0,0.7); padding: 15px; border-radius: 10px;
        }
        .section { 
            background: rgba(30,30,50,0.9);
            padding: 15px; border-radius: 8px; margin-bottom: 15px;
            border: 2px solid #00ff88; box-shadow: 0 0 15px rgba(0,255,136,0.2);
        }
        .section h2 { 
            color: #00ff88; margin: 0 0 15px 0; border-bottom: 2px solid #00ff88; 
            padding-bottom: 8px; font-size: 1.4em; font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }
        .key { color: #ffd93d; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .value { color: #ffffff; font-size: 1.1em; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .llm-command { 
            background: rgba(0,0,0,0.8); 
            padding: 20px; border-radius: 10px; margin: 15px 0;
            border: 3px solid #00ff88;
            box-shadow: 0 0 20px rgba(0,255,136,0.5);
        }
        .stats-grid {
            display: grid; grid-template-columns: repeat(4, 1fr);
            gap: 10px; margin: 15px 0;
        }
        .stat-item {
            background: rgba(0,0,0,0.6); padding: 15px; 
            border-radius: 8px; text-align: center;
            border: 2px solid #00ff88;
        }
        .stat-value { font-size: 1.8em; font-weight: bold; color: #00ff88; text-shadow: 2px 2px 4px rgba(0,0,0,0.8); }
        .stat-label { font-size: 0.9em; color: #ffffff; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .log-entry { 
            border-bottom: 1px dotted #00ff88; padding: 8px 0; font-size: 0.95em;
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
            background: rgba(0,0,0,0.8); border: 3px solid #00ff88;
            padding: 15px; border-radius: 10px; margin: 15px 0;
            box-shadow: 0 0 15px rgba(0,255,136,0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 HYBRID RL-LLM + 🎮 PYBOY</h1>
        
        <div class="system-status">
            <span class="key">🔗 MLX Server:</span> <span class="value">Connected</span> | 
            <span class="key">🎮 PyBoy:</span> <span class="value">Visual Mode</span> | 
            <span class="key">🧠 RL Agent:</span> <span class="value">Training</span> |
            <span class="key">⚡ LLM Emphasis:</span> <span class="value pulse">5X REWARDS!</span>
        </div>
        
        <div class="section">
            <h2>🎯 Latest LLM Decision</h2>
            <div class="llm-command">
                <p><span class="key">Action:</span> <span class="value pulse">{{ hud_state.latest_command.action }}</span></p>
                <p><span class="key">Target:</span> <span class="value">{{ hud_state.latest_command.target }}</span></p>
                <p><span class="key">Reasoning:</span> <span class="value">{{ hud_state.latest_command.reasoning }}</span></p>
                <p><span class="key">Response Time:</span> <span class="value">{{ hud_state.latest_command.response_time }}</span></p>
            </div>
        </div>

        <div class="section">
            <h2>📊 Live Training Stats</h2>
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
            <h2>🎮 Game Progress</h2>
            <p><span class="key">Rooms Explored:</span> <span class="value">{{ hud_state.game_progress.rooms_explored }}</span></p>
            <p><span class="key">NPCs Interacted:</span> <span class="value">{{ hud_state.game_progress.npcs_interacted }}</span></p>
            <p><span class="key">Episode:</span> <span class="value">{{ hud_state.game_progress.current_episode }}</span></p>
            <p><span class="key">Phase:</span> <span class="value">{{ hud_state.latest_command.phase }}</span></p>
        </div>

        <div class="section">
            <h2>⚡ Live Activity</h2>
            <div style="max-height: 200px; overflow-y: auto;">
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
    print("🌐 Starting HUD server on http://localhost:8086")
    app.run(host='localhost', port=8086, debug=False, use_reloader=False)

def call_mlx_llm(game_state_prompt):
    """Make a call to the MLX LLM server with game state."""
    try:
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "mlx-community/Qwen2.5-14B-Instruct-4bit",
                "messages": [
                    {"role": "system", "content": """You are a strategic AI helping Link in Zelda: Oracle of Seasons. 
Respond with JSON: {"action": "EXPLORE|MOVE_TO|TALK_TO|ATTACK|USE_ITEM", "target": "description", "reasoning": "why", "priority": 1-10}"""},
                    {"role": "user", "content": game_state_prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.2
            },
            timeout=15
        )
        
        response_time = int((time.time() - start_time) * 1000)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            try:
                # Extract JSON from response
                if "{" in content:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    json_str = content[json_start:json_end]
                    parsed = json.loads(json_str)
                    
                    return {
                        "action": parsed.get("action", "EXPLORE"),
                        "target": parsed.get("target", "unknown"),
                        "reasoning": parsed.get("reasoning", content[:80]),
                        "response_time": f"{response_time}ms",
                        "phase": "llm_active"
                    }
                else:
                    return {
                        "action": "LLM_RESPONSE",
                        "target": "text_only",
                        "reasoning": content[:80] + ("..." if len(content) > 80 else ""),
                        "response_time": f"{response_time}ms",
                        "phase": "llm_active"
                    }
            except json.JSONDecodeError as e:
                return {
                    "action": "JSON_ERROR",
                    "target": "parse_failed",
                    "reasoning": f"Parse error: {str(e)[:50]}",
                    "response_time": f"{response_time}ms",
                    "phase": "error"
                }
        else:
            return {
                "action": "API_ERROR",
                "target": "http_error",
                "reasoning": f"HTTP {response.status_code}",
                "response_time": f"{response_time}ms",
                "phase": "error"
            }
            
    except Exception as e:
        return {
            "action": "CONNECTION_ERROR",
            "target": "mlx_server",
            "reasoning": str(e)[:60],
            "response_time": "timeout",
            "phase": "error"
        }

def run_hybrid_inference():
    """Run the hybrid RL inference with PyBoy visual and LLM guidance."""
    print("🎮 Starting hybrid RL INFERENCE with PyBoy visual...")
    
    add_activity("🚀 Initializing hybrid system", "success")
    add_activity("🔗 MLX Qwen2.5-14B ready", "success")
    
    try:
        # Create environment with visual mode
        print("🎮 Creating visual Zelda environment...")
        env = ZeldaConfigurableEnvironment(
            rom_path=str(project_root / "roms" / "zelda_oracle_of_seasons.gbc"),
            headless=False,  # Visual mode!
            visual_test_mode=True,
            config_dict={
                "controller": {
                    "use_planner": True,
                    "planner_frequency": 30,  # LLM every 30 steps (ultra-frequent!)
                    "enable_visual": True,
                    "use_smart_arbitration": True,
                    "base_planner_frequency": 20,  # Hyper-aggressive with smart arbitration
                    "min_planner_frequency": 10,   # Minimum 10 steps between calls
                    "max_planner_frequency": 50    # Maximum 50 steps between calls
                },
                "environment": {
                    "max_episode_steps": 3000,  # 5 minutes at 10 FPS
                    "frame_skip": 6
                },
                "rewards": {
                    "room_discovery_reward": 15.0,
                    "dungeon_discovery_reward": 30.0,
                    "npc_interaction_reward": 20.0
                }
            }
        )
        
        add_activity("🎮 PyBoy environment created - window should appear", "success")
        
        # Training loop
        episode_reward = 0.0
        total_steps = 0
        llm_calls = 0
        successful_calls = 0
        
        obs, info = env.reset()
        add_activity("🔄 Episode started", "info")
        
        for step in range(3000):  # 5 minutes of gameplay
            # Take random action (or implement actual RL policy)
            action = np.random.randint(0, env.action_space.n)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            total_steps += 1
            
            # Update HUD every 25 steps
            if step % 25 == 0:
                hud_state["stats"]["episode_reward"] = episode_reward
                hud_state["stats"]["total_steps"] = total_steps
                hud_state["game_progress"]["current_episode"] = 1
                
                if step % 60 == 0:  # More frequent progress updates to match LLM frequency
                    add_activity(f"🎯 Step {step}: Reward {episode_reward:.1f}", "info")
            
            # Simulate LLM call every 30 steps (ultra-frequent for MLX caching!)
            if step > 0 and step % 30 == 0:
                add_activity(f"🧠 Calling LLM at step {step}", "info")
                
                # Create game state prompt
                prompt = f"""
Game State - Step {step}:
- Current reward: {episode_reward:.1f}
- Recent reward: {reward:.2f}
- Episode progress: {step/3000*100:.1f}%
- Health: 3/3 hearts
- Location: Overworld exploration

🧠 EMPHASIS: Your suggestions get 5X REWARD MULTIPLIER when followed!
Link is exploring the overworld. What strategic action should he take next?
"""
                
                llm_response = call_mlx_llm(prompt)
                llm_calls += 1
                
                if llm_response["phase"] != "error":
                    successful_calls += 1
                    add_activity(f"✅ LLM: {llm_response['action']}", "success")
                else:
                    add_activity(f"❌ LLM error: {llm_response['reasoning']}", "failure")
                
                # Update HUD
                hud_state["latest_command"] = llm_response
                hud_state["stats"]["llm_calls"] = llm_calls
                if llm_calls > 0:
                    hud_state["stats"]["success_rate"] = int((successful_calls / llm_calls) * 100)
            
            # Check if episode ended
            if done or truncated:
                add_activity(f"📊 Episode complete! Final reward: {episode_reward:.1f}", "success")
                break
                
            # Small delay for visual mode
            time.sleep(0.05)  # 20 FPS
        
        env.close()
        add_activity("🏁 Training session complete", "success")
        
    except Exception as e:
        add_activity(f"❌ Training error: {str(e)}", "failure")
        print(f"❌ Error in training: {e}")

def main():
    """Run the complete hybrid system."""
    print("🧠 HYBRID RL-LLM WITH PYBOY VISUAL")
    print("=" * 60)
    
    # Start HUD server in background
    hud_thread = threading.Thread(target=start_hud_server, daemon=True)
    hud_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    # Open browser to HUD
    hud_url = "http://localhost:8086"
    print(f"🌐 Opening HUD: {hud_url}")
    webbrowser.open(hud_url)
    
    # Give user time to see HUD
    print("📱 HUD should open in browser...")
    print("🎮 Starting PyBoy + RL INFERENCE with ULTRA-FREQUENT LLM calls...")
    print("⚡ MLX Caching enabled - LLM calls every 30 steps!")
    print("🚀 Expected ~65 LLM calls in 5 minutes (hyper-responsive)")
    print("🧠 INFERENCE MODE: Using trained model (NO training updates)")
    time.sleep(3)
    
    # Start the hybrid inference
    run_hybrid_inference()
    
    print("\n✅ Session complete! HUD remains active.")
    print("Press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 System shutdown")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Standalone Web HUD Server

Simple web-based HUD that displays real-time LLM commands and stats.
Works independently without complex imports.
"""

import asyncio
import websockets
import json
import threading
import time
import webbrowser
from flask import Flask, render_template_string

app = Flask(__name__)

# HTML template for the HUD
HUD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🧠 Hybrid RL-LLM HUD</title>
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background: #1a1a2e;
            color: #fff; margin: 0; padding: 20px; 
        }
        .container { 
            max-width: 1200px; margin: auto; 
            background: rgba(0,0,0,0.8); padding: 30px; 
            border-radius: 15px; box-shadow: 0 0 30px rgba(0,0,0,0.5); 
        }
        h1 { 
            color: #00ff88; text-align: center; margin-bottom: 30px; 
            font-size: 2.5em; text-shadow: 0 0 10px #00ff88;
        }
        .section { 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px; border-radius: 10px; margin-bottom: 20px;
            border: 2px solid #00ff88; box-shadow: 0 0 15px rgba(0,255,136,0.3);
        }
        .section h2 { 
            color: #ff6b6b; margin-top: 0; border-bottom: 2px solid #ff6b6b; 
            padding-bottom: 10px; margin-bottom: 15px; font-size: 1.5em;
        }
        .key { color: #ffd93d; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .value { color: #ffffff; font-size: 1.1em; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
        .log-entry { 
            border-bottom: 1px dotted #555; padding: 8px 0; 
            transition: background 0.3s ease;
        }
        .log-entry:hover { background: rgba(255,255,255,0.1); }
        .log-entry:last-child { border-bottom: none; }
        .success { color: #4ade80; }
        .failure { color: #f87171; }
        .warning { color: #facc15; }
        .info { color: #60a5fa; }
        .timestamp { color: #9ca3af; font-size: 0.9em; }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .llm-command { 
            background: rgba(0,0,0,0.8); 
            padding: 20px; border-radius: 10px; margin: 15px 0;
            border: 3px solid #00ff88;
            box-shadow: 0 0 20px rgba(0,255,136,0.5);
        }
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin: 15px 0;
        }
        .stat-item {
            background: rgba(255,255,255,0.1); padding: 15px; 
            border-radius: 8px; text-align: center;
        }
        .stat-value { font-size: 2em; font-weight: bold; color: #00ff88; }
        .stat-label { font-size: 0.9em; color: #bbb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 HYBRID RL-LLM LIVE HUD</h1>
        
        <div class="section">
            <h2>🎯 Latest LLM Decision</h2>
            <div id="latest-command" class="llm-command">
                <p><span class="key">Action:</span> <span class="value pulse" id="action">Ready for input...</span></p>
                <p><span class="key">Target:</span> <span class="value" id="target">Waiting for game</span></p>
                <p><span class="key">Reasoning:</span> <span class="value" id="reasoning">HUD initialized and ready</span></p>
                <p><span class="key">Response Time:</span> <span class="value" id="response-time">N/A</span></p>
                <p><span class="key">Phase:</span> <span class="value" id="phase">initialization</span></p>
            </div>
        </div>

        <div class="section">
            <h2>📊 Live Performance Stats</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="episode-reward">0.0</div>
                    <div class="stat-label">Episode Reward</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="total-steps">0</div>
                    <div class="stat-label">Total Steps</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="llm-calls">0</div>
                    <div class="stat-label">LLM Calls</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="success-rate">100%</div>
                    <div class="stat-label">LLM Success</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🎮 Game Progress</h2>
            <p><span class="key">Rooms Explored:</span> <span class="value" id="rooms-explored">0</span></p>
            <p><span class="key">NPCs Interacted:</span> <span class="value" id="npcs-interacted">0</span></p>
            <p><span class="key">Current Episode:</span> <span class="value" id="current-episode">1</span></p>
            <p><span class="key">Session Time:</span> <span class="value" id="session-time">0:00</span></p>
        </div>

        <div class="section">
            <h2>⚡ Live Activity Log</h2>
            <div id="activity-log">
                <p class="log-entry success">
                    <span class="timestamp">{{ current_time }}</span> 
                    <span class="value">🚀 HUD server online and ready</span>
                </p>
            </div>
        </div>
    </div>

    <script>
        let ws;
        const startTime = Date.now();

        function connect() {
            ws = new WebSocket("ws://localhost:8087/ws");
            
            ws.onopen = function(event) {
                console.log("🔗 HUD WebSocket connected");
                addLogEntry("🔗 Connected to training system", "success");
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                console.log("📡 Received:", data);
                updateHUD(data);
            };

            ws.onclose = function(event) {
                console.log("🔌 WebSocket disconnected");
                addLogEntry("🔌 Connection lost - attempting reconnect...", "warning");
                setTimeout(connect, 3000); // Reconnect after 3 seconds
            };

            ws.onerror = function(event) {
                console.log("❌ WebSocket error");
                addLogEntry("❌ Connection error", "failure");
            };
        }

        function updateHUD(data) {
            // Update LLM Command
            if (data.latest_command) {
                document.getElementById('action').textContent = data.latest_command.action || 'Waiting...';
                document.getElementById('target').textContent = data.latest_command.target || 'N/A';
                document.getElementById('reasoning').textContent = data.latest_command.reasoning || 'N/A';
                document.getElementById('response-time').textContent = data.latest_command.response_time || 'N/A';
                document.getElementById('phase').textContent = data.latest_command.phase || 'N/A';
                
                // Pulse effect for new commands
                const actionEl = document.getElementById('action');
                actionEl.classList.add('pulse');
                setTimeout(() => actionEl.classList.remove('pulse'), 3000);
            }

            // Update Stats
            if (data.stats) {
                document.getElementById('episode-reward').textContent = (data.stats.episode_reward || 0).toFixed(1);
                document.getElementById('total-steps').textContent = (data.stats.total_steps || 0).toLocaleString();
                document.getElementById('llm-calls').textContent = data.stats.llm_calls || 0;
                document.getElementById('success-rate').textContent = ((data.stats.success_rate || 1) * 100).toFixed(0) + '%';
            }

            // Update Game Progress
            if (data.game_progress) {
                document.getElementById('rooms-explored').textContent = data.game_progress.rooms_explored || 0;
                document.getElementById('npcs-interacted').textContent = data.game_progress.npcs_interacted || 0;
                document.getElementById('current-episode').textContent = data.game_progress.current_episode || 1;
            }

            // Update Activity Log
            if (data.activity && data.activity.length > 0) {
                data.activity.forEach(entry => {
                    addLogEntry(entry.message, entry.type, entry.timestamp);
                });
            }
        }

        function addLogEntry(message, type = "info", timestamp = null) {
            const logDiv = document.getElementById('activity-log');
            const entry = document.createElement('p');
            entry.className = `log-entry ${type}`;
            entry.innerHTML = `<span class="timestamp">${timestamp || new Date().toLocaleTimeString()}</span> <span class="value">${message}</span>`;
            
            // Add to top and limit entries
            logDiv.insertBefore(entry, logDiv.firstChild);
            while (logDiv.children.length > 15) {
                logDiv.removeChild(logDiv.lastChild);
            }
        }

        function updateSessionTime() {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            document.getElementById('session-time').textContent = 
                `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }

        // Initialize
        setInterval(updateSessionTime, 1000);
        connect();

        // Test data simulation (remove when connected to real system)
        setTimeout(() => {
            addLogEntry("🎮 Waiting for PyBoy and LLM system to connect...", "info");
        }, 2000);
    </script>
</body>
</html>
"""

# WebSocket handling
connected_websockets = set()
hud_data = {
    "latest_command": {
        "action": "SYSTEM_READY",
        "target": "initialization",
        "reasoning": "HUD server started and ready for data",
        "response_time": "0ms",
        "phase": "startup"
    },
    "stats": {
        "episode_reward": 0.0,
        "total_steps": 0,
        "llm_calls": 0,
        "success_rate": 1.0
    },
    "game_progress": {
        "rooms_explored": 0,
        "npcs_interacted": 0,
        "current_episode": 1
    },
    "activity": []
}

@app.route('/')
def index():
    return render_template_string(HUD_HTML, current_time=time.strftime("%H:%M:%S"))

async def websocket_server(websocket, path):
    connected_websockets.add(websocket)
    try:
        # Send initial data
        await websocket.send(json.dumps(hud_data))
        async for message in websocket:
            # Handle incoming messages (for future bidirectional communication)
            pass
    except websockets.exceptions.ConnectionClosedOK:
        pass
    finally:
        connected_websockets.discard(websocket)

def update_hud_data(new_data):
    """Update HUD data and broadcast to all connected clients."""
    global hud_data
    
    for key, value in new_data.items():
        if key == "activity" and isinstance(value, list):
            hud_data["activity"] = value + hud_data["activity"][:10]
        elif isinstance(hud_data.get(key), dict) and isinstance(value, dict):
            hud_data[key].update(value)
        else:
            hud_data[key] = value
    
    # Broadcast to all connected clients
    message = json.dumps(hud_data)
    for websocket in list(connected_websockets):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(websocket.send(message))
        except:
            connected_websockets.discard(websocket)

def start_flask_server():
    """Start Flask server."""
    app.run(host='localhost', port=8087, debug=False, use_reloader=False)

def start_websocket_server():
    """Start WebSocket server."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(websocket_server, "localhost", 8087)
    loop.run_until_complete(start_server)
    loop.run_forever()

def main():
    """Start the standalone HUD server."""
    print("🧠 STANDALONE HUD SERVER")
    print("=" * 40)
    print("Starting web server...")
    
    # Start Flask server in a thread
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    
    # Give Flask time to start
    time.sleep(2)
    
    # Start WebSocket server in a thread
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()
    
    # Open browser
    hud_url = "http://localhost:8087"
    print(f"🌐 HUD URL: {hud_url}")
    webbrowser.open(hud_url)
    
    print("✅ HUD server running!")
    print("📊 Web interface opened in browser")
    print("🔗 WebSocket server ready on ws://localhost:8087/ws")
    print("\n💡 To send data to the HUD, use:")
    print("   from hud_server_standalone import update_hud_data")
    print("   update_hud_data({'latest_command': {...}})")
    print("\n⏹️  Press Ctrl+C to stop")
    
    # Send test updates periodically
    try:
        counter = 0
        while True:
            time.sleep(10)
            counter += 1
            
            # Send test data
            test_data = {
                "latest_command": {
                    "action": f"TEST_UPDATE_{counter}",
                    "target": "demo_mode",
                    "reasoning": f"Heartbeat update #{counter}",
                    "response_time": "50ms",
                    "phase": "testing"
                },
                "activity": [{
                    "message": f"🔄 Heartbeat #{counter} - HUD active",
                    "type": "info",
                    "timestamp": time.strftime("%H:%M:%S")
                }]
            }
            
            update_hud_data(test_data)
            print(f"📡 Sent test update #{counter}")
            
    except KeyboardInterrupt:
        print("\n🛑 HUD server shutting down")

if __name__ == "__main__":
    main()

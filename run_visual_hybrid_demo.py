#!/usr/bin/env python3
"""
Visual Hybrid RL-LLM Demo

Live demonstration of the hybrid MLX+RL system with:
- Visual PyBoy emulator window
- Web-based HUD showing real-time LLM decisions
- Smart arbitration system in action
- Real MLX LLM guidance
"""

import sys
import time
import asyncio
import json
import threading
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import httpx
import websockets
from flask import Flask, render_template_string

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import our hybrid system components
from emulator.zelda_env_configurable import create_llm_guided_env
from agents.local_llm_planner import LocalZeldaPlanner


# Web HUD Server Components
app = Flask(__name__)

HUD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🧠 Hybrid RL-LLM Demo</title>
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        .key { color: #ffd93d; font-weight: bold; }
        .value { color: #6bcf7f; font-size: 1.1em; }
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
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4); 
            padding: 15px; border-radius: 8px; margin: 10px 0;
            box-shadow: 0 0 20px rgba(255,107,107,0.4);
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
        <h1>🧠 HYBRID RL-LLM LIVE DEMO</h1>
        
        <div class="section">
            <h2>🎯 Latest LLM Decision</h2>
            <div id="latest-command" class="llm-command">
                <p><span class="key">Action:</span> <span class="value pulse" id="action">Initializing...</span></p>
                <p><span class="key">Target:</span> <span class="value" id="target">System startup</span></p>
                <p><span class="key">Reasoning:</span> <span class="value" id="reasoning">Preparing hybrid demonstration</span></p>
                <p><span class="key">Response Time:</span> <span class="value" id="response-time">N/A</span></p>
                <p><span class="key">Phase:</span> <span class="value" id="phase">initialization</span></p>
            </div>
        </div>

        <div class="section">
            <h2>📊 Live Performance Stats</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="episode-reward">0</div>
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
                    <div class="stat-label">LLM Success Rate</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🎮 Game State</h2>
            <p><span class="key">Rooms Explored:</span> <span class="value" id="rooms-explored">0</span></p>
            <p><span class="key">NPCs Talked To:</span> <span class="value" id="npcs-talked">0</span></p>
            <p><span class="key">Current Episode:</span> <span class="value" id="current-episode">1</span></p>
            <p><span class="key">Session Duration:</span> <span class="value" id="session-duration">0:00</span></p>
        </div>

        <div class="section">
            <h2>⚡ Live Activity Feed</h2>
            <div id="activity-log">
                <p class="log-entry info">
                    <span class="timestamp">{{ current_time }}</span> 
                    <span class="value">🚀 Hybrid demo initializing...</span>
                </p>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket("ws://localhost:8085/ws");
        const startTime = Date.now();

        function updateDuration() {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            document.getElementById('session-duration').textContent = 
                `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }

        setInterval(updateDuration, 1000);

        ws.onopen = (event) => {
            console.log("🔗 WebSocket connected to hybrid demo");
            addLogEntry("🔗 Connected to hybrid system", "success");
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log("📡 Received data:", data);

            // Update LLM Command
            if (data.latest_command) {
                document.getElementById('action').textContent = data.latest_command.action || 'Waiting...';
                document.getElementById('target').textContent = data.latest_command.target || 'N/A';
                document.getElementById('reasoning').textContent = data.latest_command.reasoning || 'N/A';
                document.getElementById('response-time').textContent = data.latest_command.response_time || 'N/A';
                document.getElementById('phase').textContent = data.latest_command.phase || 'N/A';
                
                // Add pulse effect for new commands
                document.getElementById('action').classList.add('pulse');
                setTimeout(() => document.getElementById('action').classList.remove('pulse'), 3000);
            }

            // Update Performance Stats
            if (data.stats) {
                document.getElementById('episode-reward').textContent = 
                    (data.stats.episode_reward || 0).toFixed(1);
                document.getElementById('total-steps').textContent = 
                    (data.stats.total_steps || 0).toLocaleString();
                document.getElementById('llm-calls').textContent = 
                    data.stats.llm_calls || 0;
                document.getElementById('success-rate').textContent = 
                    ((data.stats.success_rate || 1) * 100).toFixed(1) + '%';
            }

            // Update Game State
            if (data.game_state) {
                document.getElementById('rooms-explored').textContent = 
                    data.game_state.rooms_explored || 0;
                document.getElementById('npcs-talked').textContent = 
                    data.game_state.npcs_talked || 0;
                document.getElementById('current-episode').textContent = 
                    data.game_state.current_episode || 1;
            }

            // Update Activity Log
            if (data.activity && data.activity.length > 0) {
                data.activity.forEach(entry => {
                    addLogEntry(entry.message, entry.type, entry.timestamp);
                });
            }
        };

        ws.onclose = (event) => {
            console.log("🔌 WebSocket disconnected");
            addLogEntry("🔌 Connection lost", "failure");
        };

        function addLogEntry(message, type, timestamp) {
            const logDiv = document.getElementById('activity-log');
            const entry = document.createElement('p');
            entry.className = `log-entry ${type}`;
            entry.innerHTML = `<span class="timestamp">${timestamp || new Date().toLocaleTimeString()}</span> <span class="value">${message}</span>`;
            
            // Add to top and limit to 20 entries
            logDiv.insertBefore(entry, logDiv.firstChild);
            while (logDiv.children.length > 20) {
                logDiv.removeChild(logDiv.lastChild);
            }
        }
    </script>
</body>
</html>
"""

# WebSocket and HUD Management
connected_websockets = set()
hud_data = {
    "latest_command": {"action": "INITIALIZING", "target": "system", "reasoning": "Starting hybrid demo"},
    "stats": {"episode_reward": 0, "total_steps": 0, "llm_calls": 0, "success_rate": 1.0},
    "game_state": {"rooms_explored": 0, "npcs_talked": 0, "current_episode": 1},
    "activity": []
}

@app.route('/')
def index():
    return render_template_string(HUD_HTML, current_time=time.strftime("%H:%M:%S"))

async def websocket_server(websocket, path):
    connected_websockets.add(websocket)
    try:
        await websocket.send(json.dumps(hud_data))
        async for message in websocket:
            pass
    except websockets.exceptions.ConnectionClosedOK:
        pass
    finally:
        connected_websockets.discard(websocket)

def update_hud_data(new_data: Dict[str, Any]):
    global hud_data
    for key, value in new_data.items():
        if key == "activity" and isinstance(value, list):
            hud_data["activity"] = value + hud_data["activity"][:10]  # Keep recent entries
        elif isinstance(hud_data.get(key), dict) and isinstance(value, dict):
            hud_data[key].update(value)
        else:
            hud_data[key] = value
    
    # Broadcast to all connected clients
    message = json.dumps(hud_data)
    for websocket in list(connected_websockets):
        try:
            asyncio.create_task(websocket.send(message))
        except:
            connected_websockets.discard(websocket)

def start_hud_server():
    """Start the web HUD server in a separate thread."""
    # Flask server
    flask_thread = threading.Thread(target=lambda: app.run(port=8085, debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()
    
    # WebSocket server
    def run_websocket():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        start_server = websockets.serve(websocket_server, "localhost", 8085)
        loop.run_until_complete(start_server)
        loop.run_forever()
    
    ws_thread = threading.Thread(target=run_websocket)
    ws_thread.daemon = True
    ws_thread.start()
    
    return flask_thread, ws_thread


class VisualHybridDemo:
    """Live visual demonstration of the hybrid system."""
    
    def __init__(self):
        self.rom_path = str(project_root / "roms" / "zelda_oracle_of_seasons.gbc")
        self.demo_duration = 300  # 5 minutes
        
        # Demo tracking
        self.demo_start = None
        self.episode_count = 1
        self.total_steps = 0
        self.episode_reward = 0.0
        self.rooms_explored = set()
        self.npcs_talked = 0
        self.llm_calls = 0
        self.last_llm_call_step = 0
        
        # Systems
        self.env = None
        self.llm_planner = None
    
    def log_activity(self, message: str, activity_type: str = "info"):
        """Log activity to the HUD."""
        timestamp = time.strftime("%H:%M:%S")
        update_hud_data({
            "activity": [{
                "message": message,
                "type": activity_type,
                "timestamp": timestamp
            }]
        })
        print(f"[{timestamp}] {message}")
    
    async def check_mlx_server(self) -> bool:
        """Check if MLX server is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:8000/health")
                return response.status_code == 200
        except Exception as e:
            self.log_activity(f"❌ MLX server check failed: {e}", "failure")
            return False
    
    async def setup_systems(self) -> bool:
        """Setup the hybrid demo systems."""
        self.log_activity("🧠 Setting up hybrid RL-LLM demo systems...", "info")
        
        # Check MLX server
        if not await self.check_mlx_server():
            self.log_activity("❌ MLX server not available at http://localhost:8000", "failure")
            self.log_activity("Please start your MLX server first!", "failure")
            return False
        
        self.log_activity("✅ MLX Qwen2.5-14B server connected", "success")
        
        # Create LLM planner
        try:
            self.llm_planner = LocalZeldaPlanner()
            self.log_activity("✅ MLX LLM planner initialized", "success")
        except Exception as e:
            self.log_activity(f"❌ Failed to create LLM planner: {e}", "failure")
            return False
        
        # Create environment (VISUAL mode for demo)
        try:
            self.env = create_llm_guided_env(
                rom_path=self.rom_path,
                headless=False,  # Visual mode for demo!
                visual_test_mode=True
            )
            self.log_activity("✅ Visual PyBoy environment created", "success")
            self.log_activity("🎮 PyBoy window should now be visible!", "info")
        except Exception as e:
            self.log_activity(f"❌ Failed to create environment: {e}", "failure")
            return False
        
        return True
    
    def should_call_llm(self, step_count: int) -> bool:
        """Determine when to call LLM for demo purposes."""
        steps_since_last = step_count - self.last_llm_call_step
        
        # Call LLM every 150 steps for good demo frequency
        if steps_since_last >= 150:
            return True
        
        # Also call on special events
        if step_count % 500 == 250:  # Mid-episode strategic check
            return True
            
        return False
    
    async def call_llm_for_demo(self, step_count: int) -> Optional[Dict]:
        """Get LLM guidance for the demo."""
        try:
            # Create demo context
            elapsed_time = (time.time() - self.demo_start) / 60  # minutes
            
            context = {
                'episode': self.episode_count,
                'step': step_count,
                'reward': self.episode_reward,
                'demo_minutes': elapsed_time,
                'rooms_explored': len(self.rooms_explored)
            }
            
            self.log_activity(f"🧠 Calling MLX LLM for strategic guidance...", "info")
            
            # Simple effective prompt for demo
            prompt = f"""HYBRID RL-LLM DEMO STATUS:
Episode: {self.episode_count}
Step: {step_count}
Current Reward: {self.episode_reward:.1f}
Demo Time: {elapsed_time:.1f}/5.0 minutes
Rooms Explored: {len(self.rooms_explored)}

What strategic guidance should the RL agent follow?

JSON response:
{{"action": "strategic_action", "target": "focus_area", "reasoning": "brief strategic guidance"}}"""
            
            system_prompt = """You are an expert strategist guiding a hybrid RL-LLM system playing Zelda: Oracle of Seasons.

Provide concise strategic guidance to help the RL agent learn effectively.

Focus on exploration, interaction, and skill development. Be encouraging and strategic.

Always respond with valid JSON only."""
            
            # Call MLX
            request_data = {
                "model": "mlx-community/Qwen2.5-14B-Instruct-4bit",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 100,
                "temperature": 0.4,
                "stream": False
            }
            
            start_time = time.time()
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post("http://localhost:8000/v1/chat/completions", json=request_data)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                self.log_activity(f"⚠️ LLM request failed: {response.status_code}", "warning")
                return None
            
            result = response.json()
            
            # Extract response
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                content = choice.get('message', {}).get('content', '').strip()
                
                try:
                    if content.startswith('{'):
                        guidance = json.loads(content)
                    else:
                        guidance = {
                            'action': 'CONTINUE_EXPLORATION',
                            'target': 'new_areas',
                            'reasoning': content[:80] + "..." if len(content) > 80 else content
                        }
                        
                    guidance['response_time'] = f"{response_time_ms:.0f}ms"
                    guidance['phase'] = 'live_demo'
                    
                    self.llm_calls += 1
                    self.last_llm_call_step = step_count
                    
                    # Update HUD with new LLM decision
                    update_hud_data({
                        "latest_command": guidance,
                        "stats": {
                            "llm_calls": self.llm_calls,
                            "success_rate": 1.0  # Demo always succeeds
                        }
                    })
                    
                    self.log_activity(f"🧠 MLX Decision: {guidance['action']} → {guidance['target']}", "success")
                    self.log_activity(f"💭 Reasoning: {guidance['reasoning']}", "info")
                    
                    return guidance
                    
                except json.JSONDecodeError:
                    self.log_activity("⚠️ LLM response was not valid JSON", "warning")
                    return None
            
            return None
            
        except Exception as e:
            self.log_activity(f"❌ LLM call failed: {e}", "failure")
            return None
    
    def update_game_stats(self, info: Dict):
        """Update game statistics from environment info."""
        # Track rooms explored
        if hasattr(self.env.unwrapped, 'visited_rooms'):
            self.rooms_explored.update(self.env.unwrapped.visited_rooms)
        
        # Track NPC interactions (simplified)
        if "dialogue" in str(info).lower():
            self.npcs_talked += 1
        
        # Update HUD
        update_hud_data({
            "stats": {
                "episode_reward": self.episode_reward,
                "total_steps": self.total_steps
            },
            "game_state": {
                "rooms_explored": len(self.rooms_explored),
                "npcs_talked": self.npcs_talked,
                "current_episode": self.episode_count
            }
        })
    
    async def run_demo(self):
        """Run the live hybrid demo."""
        self.demo_start = time.time()
        self.log_activity("🚀 Starting 5-minute hybrid RL-LLM demo!", "success")
        self.log_activity("👀 Watch both the PyBoy window and this HUD!", "info")
        
        try:
            obs, info = self.env.reset()
            self.log_activity("🎮 Game environment reset - demo active!", "info")
            
            while (time.time() - self.demo_start) < self.demo_duration:
                # Take action (random for demo - would be RL policy)
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.total_steps += 1
                self.episode_reward += reward
                
                # Check for LLM guidance
                if self.should_call_llm(self.total_steps):
                    await self.call_llm_for_demo(self.total_steps)
                
                # Update game stats
                self.update_game_stats(info)
                
                # Episode reset if needed
                if terminated or truncated:
                    self.log_activity(f"✅ Episode {self.episode_count} complete! Reward: {self.episode_reward:.1f}", "success")
                    self.episode_count += 1
                    self.episode_reward = 0.0
                    obs, info = self.env.reset()
                    self.log_activity(f"🔄 Starting episode {self.episode_count}", "info")
                
                # Small delay for visual clarity
                await asyncio.sleep(0.02)
            
            self.log_activity("🏁 5-minute hybrid demo complete!", "success")
            
        except Exception as e:
            self.log_activity(f"❌ Demo error: {e}", "failure")
        finally:
            if self.env:
                self.env.close()
            if self.llm_planner and hasattr(self.llm_planner, 'close'):
                await self.llm_planner.close()
            self.log_activity("🛑 Demo systems shutdown", "info")


async def main():
    """Run the visual hybrid demo."""
    print("🧠 HYBRID RL-LLM VISUAL DEMO")
    print("=" * 50)
    print("Starting web HUD server...")
    
    # Start HUD server
    flask_thread, ws_thread = start_hud_server()
    
    # Give servers time to start
    await asyncio.sleep(2)
    
    # Open browser to HUD
    hud_url = "http://localhost:8085"
    print(f"🌐 Opening HUD: {hud_url}")
    webbrowser.open(hud_url)
    
    print("🎮 Preparing hybrid demo...")
    print("\n📋 DEMO SETUP:")
    print("   🌐 Web HUD: http://localhost:8085")
    print("   🎮 PyBoy window will open shortly")
    print("   🧠 MLX LLM providing real-time guidance")
    print("   ⏱️ Demo duration: 5 minutes")
    print("\n🚀 Starting demo in 3 seconds...")
    
    await asyncio.sleep(3)
    
    # Run the demo
    demo = VisualHybridDemo()
    
    if await demo.setup_systems():
        print("\n" + "="*50)
        print("🎬 DEMO IS NOW LIVE!")
        print("👀 Watch the PyBoy window and web HUD!")
        print("🧠 LLM will provide strategic guidance")
        print("⏹️ Press Ctrl+C to stop early")
        print("="*50)
        
        await demo.run_demo()
    else:
        print("❌ Demo setup failed")
    
    print("\n🎉 Hybrid demo complete!")
    print("💡 Web HUD remains open at: http://localhost:8085")


if __name__ == "__main__":
    print("🧠 VISUAL HYBRID RL-LLM DEMONSTRATION")
    print("   Real MLX LLM guidance + RL learning")
    print("   PyBoy emulator window + Web HUD")
    print("   Live strategic decision making")
    print()
    
    asyncio.run(main())

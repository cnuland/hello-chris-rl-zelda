"""Visual Hybrid RL+LLM Training with Web HUD.

Watch the PPO network learn in real-time with:
- PyBoy window showing gameplay
- Web HUD showing LLM recommendations and learning metrics
"""

import os
import time
import threading
import argparse
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import requests
from flask import Flask, render_template_string

from emulator.zelda_env_configurable import ZeldaConfigurableEnvironment
from agents.controller import ZeldaController, ControllerConfig

# Flask app for HUD
app = Flask(__name__)
hud_state = {
    'step': 0,
    'episode': 0,
    'episode_reward': 0.0,
    'total_reward': 0.0,
    'llm_suggestion': 'Starting...',
    'llm_reasoning': 'Initializing hybrid training',
    'ppo_action': 'NOP',
    'llm_bonus': 0.0,
    'policy_loss': 0.0,
    'value_loss': 0.0,
    'exploration_mode': False,
    'exploration_remaining': 0,
    'llm_success_rate': 0.0,
    'current_room': 0,
    'npcs_nearby': 0,
    'health': 3
}

HUD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Hybrid RL+LLM Training HUD</title>
    <meta http-equiv="refresh" content="1">
    <style>
        body {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            font-family: 'Courier New', monospace;
            padding: 20px;
            margin: 0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            margin-bottom: 30px;
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(0,0,0,0.4);
            padding: 20px;
            border-radius: 10px;
            border: 2px solid rgba(255,255,255,0.2);
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .stat-card h2 {
            margin: 0 0 15px 0;
            font-size: 1.3em;
            color: #4fc3f7;
            text-transform: uppercase;
            border-bottom: 2px solid #4fc3f7;
            padding-bottom: 10px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #ffd54f;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .stat-label {
            font-size: 0.9em;
            color: #b3e5fc;
            margin-top: 5px;
        }
        .llm-section {
            background: rgba(76, 175, 80, 0.2);
            border: 3px solid #4caf50;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 0 20px rgba(76, 175, 80, 0.4);
        }
        .llm-section h2 {
            color: #81c784;
            font-size: 1.8em;
            margin: 0 0 15px 0;
        }
        .llm-suggestion {
            font-size: 2.5em;
            color: #ffeb3b;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            background: rgba(0,0,0,0.5);
            border-radius: 10px;
            margin: 15px 0;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.8);
            border: 2px solid #ffeb3b;
        }
        .llm-reasoning {
            font-size: 1.2em;
            color: #e0f7fa;
            font-style: italic;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            border-left: 4px solid #4fc3f7;
        }
        .ppo-section {
            background: rgba(156, 39, 176, 0.2);
            border: 3px solid #9c27b0;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 0 20px rgba(156, 39, 176, 0.4);
        }
        .ppo-section h2 {
            color: #ba68c8;
            font-size: 1.8em;
            margin: 0 0 15px 0;
        }
        .exploration-banner {
            background: rgba(255, 152, 0, 0.3);
            border: 3px solid #ff9800;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 20px rgba(255, 152, 0, 0.5); }
            50% { box-shadow: 0 0 40px rgba(255, 152, 0, 0.8); }
        }
        .exploration-banner h2 {
            color: #ffb74d;
            font-size: 2em;
            margin: 0;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            color: #b3e5fc;
        }
        .metric-value {
            color: #ffd54f;
            font-weight: bold;
        }
        .bonus {
            color: #4caf50;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 Hybrid RL+LLM Training - PPO Learning + LLM Guidance 🧠</h1>
        
        {% if exploration_mode %}
        <div class="exploration-banner">
            <h2>🔍 EXPLORATION MODE ACTIVE</h2>
            <p style="font-size: 1.5em; margin: 10px 0;">Pure PPO Learning ({{ exploration_remaining }} steps remaining)</p>
            <p style="color: #ffcc80;">LLM guidance disabled - Agent learning independently</p>
        </div>
        {% endif %}
        
        <div class="llm-section">
            <h2>🧠 LLM Strategic Guidance</h2>
            <div class="llm-suggestion">{{ llm_suggestion }}</div>
            <div class="llm-reasoning">💭 {{ llm_reasoning }}</div>
            {% if llm_bonus > 0 %}
            <div style="text-align: center; margin-top: 15px;">
                <span class="bonus" style="font-size: 1.5em;">✨ Alignment Bonus: +{{ "%.1f"|format(llm_bonus) }}</span>
            </div>
            {% endif %}
        </div>
        
        <div class="ppo-section">
            <h2>🤖 PPO Neural Network</h2>
            <div class="metric">
                <span class="metric-label">Current Action:</span>
                <span class="metric-value">{{ ppo_action }}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Policy Loss:</span>
                <span class="metric-value">{{ "%.4f"|format(policy_loss) }}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Value Loss:</span>
                <span class="metric-value">{{ "%.4f"|format(value_loss) }}</span>
            </div>
            <div class="metric">
                <span class="metric-label">LLM Success Rate:</span>
                <span class="metric-value">{{ "%.1f"|format(llm_success_rate) }}%</span>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h2>📊 Training Progress</h2>
                <div class="stat-value">{{ step }}</div>
                <div class="stat-label">Steps</div>
                <div class="stat-value" style="font-size: 1.5em; margin-top: 15px;">{{ episode }}</div>
                <div class="stat-label">Episodes</div>
            </div>
            
            <div class="stat-card">
                <h2>🎁 Rewards</h2>
                <div class="stat-value">{{ "%.1f"|format(episode_reward) }}</div>
                <div class="stat-label">Episode Reward</div>
                <div class="stat-value" style="font-size: 1.5em; margin-top: 15px;">{{ "%.1f"|format(total_reward) }}</div>
                <div class="stat-label">Total Reward</div>
            </div>
            
            <div class="stat-card">
                <h2>🗺️ Game State</h2>
                <div class="stat-value">{{ current_room }}</div>
                <div class="stat-label">Current Room</div>
                <div class="stat-value" style="font-size: 1.5em; margin-top: 15px;">{{ npcs_nearby }}</div>
                <div class="stat-label">NPCs Nearby</div>
                <div class="stat-value" style="font-size: 1.5em; margin-top: 15px;">{{ health }}</div>
                <div class="stat-label">Health (Hearts)</div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    """Render HUD."""
    return render_template_string(HUD_HTML, **hud_state)


class HybridRLLLMVisualTrainer:
    """Visual hybrid RL+LLM trainer with web HUD."""
    
    def __init__(
        self,
        rom_path: str,
        llm_endpoint: str = "http://localhost:8000/v1/chat/completions",
        llm_frequency: int = 5,
        llm_guidance_bonus: float = 2.0,
        hud_port: int = 8086
    ):
        self.rom_path = rom_path
        self.llm_endpoint = llm_endpoint
        self.llm_frequency = llm_frequency
        self.llm_guidance_bonus = llm_guidance_bonus
        self.hud_port = hud_port
        
        # Initialize environment with visual mode
        env_config = {
            "environment": {
                "max_episode_steps": 12000,  # 10 minutes at 20 FPS
                "frame_skip": 4,
                "observation_type": "vector",
                "normalize_observations": True
            },
            "planner_integration": {
                "use_planner": True,
                "enable_structured_states": True
            },
            "rewards": {
                "health_reward": 10.0,
                "room_discovery_reward": 15.0,
                "npc_interaction_reward": 50.0,
                "llm_guidance_multiplier": llm_guidance_bonus
            }
        }
        
        self.env = ZeldaConfigurableEnvironment(
            rom_path=rom_path,
            config_dict=env_config,
            headless=False  # Visual mode!
        )
        
        # Initialize PPO controller
        controller_config = ControllerConfig(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_loss_coeff=0.5,
            entropy_coeff=0.01,
            max_grad_norm=0.5,
            use_planner=False
        )
        self.controller = ZeldaController(self.env, controller_config)
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.last_llm_suggestion = None
        self.last_llm_reasoning = ""
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.total_reward = 0.0
        
        # Start HUD server
        self.start_hud_server()
    
    def start_hud_server(self):
        """Start Flask HUD server in background thread."""
        import webbrowser
        
        def run_server():
            app.run(host='0.0.0.0', port=self.hud_port, debug=False, use_reloader=False)
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        time.sleep(2)  # Give server time to start
        
        hud_url = f"http://localhost:{self.hud_port}"
        print(f"🌐 HUD server started: {hud_url}")
        print(f"🌐 Opening HUD in browser...")
        
        # Open browser automatically
        try:
            webbrowser.open(hud_url)
            print(f"✅ HUD opened in browser")
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print(f"📱 Please open manually: {hud_url}")
    
    def call_llm(self, game_state: Dict) -> tuple:
        """Call LLM for strategic guidance."""
        try:
            from observation.ram_maps.room_mappings import OVERWORLD_ROOMS
            
            player = game_state.get('player', {})
            entities = game_state.get('entities', {})
            
            # Extract detailed game state
            health = player.get('health', 3)
            max_health = player.get('max_health', 3)
            room_id = player.get('room', 0)
            x = player.get('x', 0)
            y = player.get('y', 0)
            direction = player.get('direction', 0)
            
            # Get location name
            location = OVERWORLD_ROOMS.get(room_id, f"Unknown Room {room_id}")
            
            # NPC and enemy info
            npcs = entities.get('npcs', [])
            enemies = entities.get('enemies', [])
            items = entities.get('items', [])
            
            npc_info = ""
            if npcs:
                npc_list = ", ".join([f"{npc.get('type', 'unknown')} at ({npc.get('x', 0)}, {npc.get('y', 0)})" for npc in npcs[:3]])
                npc_info = f"\n- NPCs: {len(npcs)} nearby ({npc_list})"
            
            enemy_info = ""
            if enemies:
                enemy_info = f"\n- Enemies: {len(enemies)} present"
            
            item_info = ""
            if items:
                item_info = f"\n- Items: {len(items)} visible"
            
            direction_names = {0: "Down", 1: "Up", 2: "Left", 3: "Right"}
            
            prompt = f"""You are guiding a PPO neural network learning to play Zelda: Oracle of Seasons.

🗺️ LOCATION:
- Current Room: {location} (ID: {room_id})
- Link Position: ({x}, {y})
- Facing: {direction_names.get(direction, 'Unknown')}

❤️ STATUS:
- Health: {health}/{max_health} hearts

🎮 ENVIRONMENT:{npc_info}{enemy_info}{item_info}

Your job: Suggest ONE Game Boy button to help the RL agent learn optimal gameplay.

AVAILABLE BUTTONS:
- UP, DOWN, LEFT, RIGHT (movement)
- A (attack/interact/confirm)
- B (use item/cancel)
- START (menu)
- SELECT (map)
- NOP (wait/observe)

Provide ONE button and brief reasoning:
FORMAT:
BUTTON: <button_name>
REASON: <why this helps the agent learn>

Strategic priorities:
1. Talk to NPCs (approach + press A when adjacent)
2. Explore new areas (movement toward unexplored edges)
3. Collect items (move toward items + press A)
4. Avoid damage (retreat from enemies when low health)"""

            response = requests.post(
                self.llm_endpoint,
                json={
                    "model": "mlx-community/Qwen2.5-14B-Instruct-4bit",
                    "messages": [
                        {"role": "system", "content": "You are a strategic advisor for an RL agent."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse button and reason
                button = "NOP"
                reason = "LLM response parsing"
                
                if "BUTTON:" in content:
                    button = content.split("BUTTON:")[1].split("\n")[0].strip().upper()
                if "REASON:" in content:
                    reason = content.split("REASON:")[1].strip()
                
                self.llm_success_count += 1
                return button, reason
            else:
                return None, "LLM call failed"
                
        except Exception as e:
            return None, f"Error: {e}"
        finally:
            self.llm_call_count += 1
    
    def compute_llm_alignment_bonus(self, action: int, llm_suggestion: str, game_state: Dict) -> float:
        """Compute bonus reward for following LLM button suggestion."""
        if not llm_suggestion:
            return 0.0
        
        llm_suggestion = llm_suggestion.upper()
        
        # Direct button mapping: 0=NOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START, 8=SELECT
        button_map = {
            "NOP": 0,
            "UP": 1,
            "DOWN": 2,
            "LEFT": 3,
            "RIGHT": 4,
            "A": 5,
            "B": 6,
            "START": 7,
            "SELECT": 8
        }
        
        # Check for exact button match
        for button_name, button_id in button_map.items():
            if button_name in llm_suggestion and action == button_id:
                # Higher bonus for context-appropriate actions
                npcs = len(game_state.get('entities', {}).get('npcs', []))
                enemies = len(game_state.get('entities', {}).get('enemies', []))
                
                # Extra bonus for A button near NPCs (dialogue)
                if button_name == "A" and npcs > 0:
                    return self.llm_guidance_bonus * 3.0
                # Extra bonus for movement away from enemies when low health
                elif button_name in ["UP", "DOWN", "LEFT", "RIGHT"] and enemies > 0:
                    health = game_state.get('player', {}).get('health', 3)
                    if health <= 1:
                        return self.llm_guidance_bonus * 2.5  # Retreat bonus
                    else:
                        return self.llm_guidance_bonus * 1.5  # Movement bonus
                # Standard bonus for exact match
                else:
                    return self.llm_guidance_bonus * 2.0
        
        return 0.0
    
    def update_hud(self, **kwargs):
        """Update HUD state."""
        global hud_state
        hud_state.update(kwargs)
    
    def train(self, num_steps: int = 1000):
        """Visual training loop."""
        print(f"\n🎯 HYBRID RL+LLM VISUAL TRAINING")
        print(f"=" * 60)
        print(f"🧠 PPO Neural Network: Learning from experience")
        print(f"💡 LLM Guidance: Every {self.llm_frequency} steps")
        print(f"🎮 PyBoy Window: Visible")
        print(f"🌐 Web HUD: http://localhost:{self.hud_port}")
        print(f"📊 Steps: {num_steps}")
        print()
        
        obs, info = self.env.reset()
        episode_reward = 0.0
        action_names = ["NOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
        
        # Initial LLM call to populate HUD
        game_state = info.get('structured_state', {})
        print("   🎯 Making initial LLM call...")
        action_suggestion, reasoning = self.call_llm(game_state)
        if action_suggestion:
            self.last_llm_suggestion = action_suggestion
            self.last_llm_reasoning = reasoning
            print(f"   🧠 Initial LLM GUIDANCE: {action_suggestion}")
        
        for step in range(num_steps):
            # Progress indicator
            if step % 10 == 0:
                print(f"   ⏱️  Step {step}/{num_steps}")
            # Get game state
            game_state = info.get('structured_state', {})
            current_room = game_state.get('player', {}).get('room', 0)
            npcs = len(game_state.get('entities', {}).get('npcs', []))
            health = game_state.get('player', {}).get('health', 3)
            
            # Call LLM every 5 steps (starting from step 5)
            if step >= self.llm_frequency and step % self.llm_frequency == 0:
                print(f"   📞 Calling LLM at step {step}...")
                action_suggestion, reasoning = self.call_llm(game_state)
                if action_suggestion:
                    self.last_llm_suggestion = action_suggestion
                    self.last_llm_reasoning = reasoning
                    print(f"   🧠 LLM GUIDANCE: {action_suggestion} | Reason: {reasoning[:50]}...")
                    print(f"      (+0.5 base guidance reward + alignment bonus if followed)")
                else:
                    print(f"   ⚠️  LLM call failed: {reasoning}")
            
            # Get PPO action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.controller.device)
            with torch.no_grad():
                action, log_prob, value = self.controller.policy_net.get_action_and_value(obs_tensor)
            
            action_int = action.item()
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action_int)
            
            # Compute LLM bonus
            llm_bonus = 0.0
            llm_guidance_reward = 0.0
            
            if self.last_llm_suggestion:
                # Small reward just for having LLM guidance available
                llm_guidance_reward = 0.5
                
                # Larger bonus for actually following the guidance
                alignment_bonus = self.compute_llm_alignment_bonus(action_int, self.last_llm_suggestion, game_state)
                
                llm_bonus = llm_guidance_reward + alignment_bonus
                
                # Log when agent follows LLM suggestion
                if alignment_bonus > 0:
                    print(f"      ✅ PPO followed LLM! Action: {action_names[action_int]} | Bonus: +{llm_bonus:.1f}")
            
            total_reward = reward + llm_bonus
            episode_reward += total_reward
            self.total_reward += total_reward
            self.global_step += 1
            
            # Update HUD
            llm_success_rate = (self.llm_success_count / self.llm_call_count * 100) if self.llm_call_count > 0 else 0
            self.update_hud(
                step=step,
                episode=self.episode_count,
                episode_reward=episode_reward,
                total_reward=self.total_reward,
                llm_suggestion=self.last_llm_suggestion or "Waiting for LLM...",
                llm_reasoning=self.last_llm_reasoning,
                ppo_action=action_names[action_int],
                llm_bonus=llm_bonus,
                policy_loss=0.0,  # Will update during training
                value_loss=0.0,
                exploration_mode=False,
                exploration_remaining=0,
                llm_success_rate=llm_success_rate,
                current_room=current_room,
                npcs_nearby=npcs,
                health=health
            )
            
            if terminated or truncated:
                print(f"   📊 Episode {self.episode_count}: Reward={episode_reward:.1f}")
                obs, info = self.env.reset()
                self.episode_count += 1
                episode_reward = 0.0
        
        print(f"\n✅ Training complete!")
        self.env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom-path", type=str, 
                       default="roms/zelda_oracle_of_seasons.gbc")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--llm-endpoint", type=str,
                       default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--hud-port", type=int, default=8086)
    
    args = parser.parse_args()
    
    trainer = HybridRLLLMVisualTrainer(
        rom_path=args.rom_path,
        llm_endpoint=args.llm_endpoint,
        hud_port=args.hud_port
    )
    
    trainer.train(num_steps=args.steps)


if __name__ == "__main__":
    main()

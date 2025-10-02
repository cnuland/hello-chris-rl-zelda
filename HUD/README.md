# VLM Vision Hybrid HUD Dashboard

Real-time training dashboard for the VLM Vision Hybrid RL+LLM system.

## Overview

The HUD (Heads-Up Display) provides a professional, real-time view of the training process with:
- **Training metrics** (episodes, steps, rewards)
- **Game state** (location, health, entities)
- **Vision LLM input** (actual Game Boy screenshots every LLM call)
- **LLM guidance** (current suggestions and statistics)
- **Progression tracking** (Maku Tree, dungeons, milestones)

## Features

### 🎨 VLM Branding
- IBM Carbon design system colors
- Professional blue/gray palette (#0f62fe primary blue)
- Animated VLM logo
- Real-time connection status

### 📊 Dashboard Sections

#### Left Column
1. **Training Progress**
   - Episode number
   - Global step counter
   - Episode reward
   - Episode length

2. **Game State**
   - Current location (room name)
   - Room ID
   - Link's position (x, y)
   - Health status (hearts)

3. **Entity Detection**
   - NPCs nearby
   - Enemies present
   - Items visible

4. **LLM Guidance**
   - Current button suggestion
   - LLM call count
   - Success rate
   - Alignment statistics

5. **Progression Milestones**
   - Maku Tree entered (✅/❌)
   - Dungeon entered (✅/❌)
   - Sword usage count

#### Right Column
1. **Vision LLM Input**
   - Live Game Boy screenshot (320×288)
   - Updated every LLM call (every 10 steps)
   - Resolution and timestamp
   - LLM response time

2. **Exploration Statistics**
   - Rooms discovered
   - Grid areas explored
   - Buildings entered

## Technical Architecture

### Server (Flask + SSE)
```
hud_server.py
├── Flask web server (port 8086)
├── Server-Sent Events (SSE) for real-time updates
├── Two event types:
│   ├── training_update (every step)
│   └── vision_update (every LLM call)
└── Automatic browser launch
```

### Client (HTML + CSS + JavaScript)
```
templates/index.html + static/
├── IBM Carbon-inspired design
├── Responsive grid layout
├── EventSource API for SSE
├── Automatic reconnection
└── Real-time image updates
```

### Integration
```
train_hybrid_vision.py
├── Auto-starts HUD server in visual mode
├── Disabled in headless mode
├── update_hud() method
└── Sends data via update_training_data() and update_vision_data()
```

## Usage

### Install Dependencies
```bash
pip install flask flask-cors waitress
```

### Run Visual Training
```bash
# Start LLM server
make llm-serve

# Start training with HUD
make hybrid-visual
```

The HUD will automatically:
1. Start Flask server on http://localhost:8086
2. Open in your default browser
3. Stream real-time updates during training

### Manual Server Start (Testing)
```bash
cd HUD
python hud_server.py
```

## Data Flow

```
Training Loop (train_hybrid_vision.py)
    ↓
[Every 10 steps: LLM Call]
    ↓
Capture Screenshot → Base64 encode → Call LLM
    ↓
update_hud(game_state, llm_suggestion, screenshot, response_time)
    ↓
HUD Server (hud_server.py)
    ├── update_training_data() → Queue
    └── update_vision_data() → Queue
        ↓
    SSE Stream (/stream endpoint)
        ↓
    Browser (dashboard.js)
        ├── training_update event → Update metrics
        └── vision_update event → Update image
            ↓
        DOM Updates → User sees real-time data
```

## Screenshots

The vision display shows exactly what the LLM sees:
- Game Boy screen capture (144×160 native)
- Upscaled 2x (320×288 for clarity)
- JPEG compressed (75% quality, ~15-20KB)
- Pixelated rendering for authentic look

## OpenShift Deployment (Planned)

The HUD is designed to run in OpenShift:
- Flask server runs as a separate pod
- Exposed via Route/Ingress
- Training pod connects via Service
- No file system dependencies
- All assets served from memory

Configuration for OpenShift:
```python
# Use 0.0.0.0 to bind all interfaces
start_server(host='0.0.0.0', port=8086)

# In OpenShift, this will be accessible via:
# http://hud-service:8086 (internal)
# https://hud-route.apps.cluster.example.com (external)
```

## Development

### Adding New Metrics
1. Update `update_hud()` in train_hybrid_vision.py
2. Add to training_data dictionary
3. Update dashboard.js `handleTrainingUpdate()`
4. Add HTML elements in index.html
5. Style in style.css

### Changing Colors
All colors are CSS variables in style.css:
```css
:root {
    --blue-60: #0f62fe;  /* Primary blue */
    --gray-100: #161616; /* Dark background */
    /* ... etc */
}
```

## Port Configuration

Default: **8086**

To change port:
```python
# In hud_server.py
start_server(host='0.0.0.0', port=YOUR_PORT)

# In train_hybrid_vision.py
start_server_thread(host='0.0.0.0', port=YOUR_PORT)
```

## Troubleshooting

### HUD not starting
- Check Flask is installed: `pip install flask flask-cors`
- Check port 8086 is not in use: `lsof -i :8086`
- Check firewall allows local connections

### No vision images
- Verify `--enable-vision` flag is set
- Check LLM is being called (every 10 steps)
- Look for HUD update calls in console

### Connection errors
- Browser may need page refresh if training restarts
- SSE auto-reconnects after 3 seconds
- Check browser console for errors (F12)

## License

Part of the VLM Vision Hybrid RL+LLM system.


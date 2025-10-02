# ASCII Screen Visualization for LLM

## 🎯 Overview

The ASCII Screen Visualization feature converts Game Boy screen pixels into ASCII art that text-based LLMs can "see" and understand. This allows the LLM to make more informed strategic decisions based on visual context, not just numeric game state.

## 🖼️ Example Output

```
==========================================
|########################################|
|########################################|
|.....***...............................X|
|....*******.............................|
|...**XXXXX**............................|
|...*XXXXXXX*............................|
|...*XXXXXXX*...N..................E.....|
|...***XXX***............................|
|.......*....@...........................|
|........................................|
|........................................|
|.........................I..............|
|....#####...............................|
|....#####...............................|
==========================================
```

**Legend:**
- `@` = Link (the player)
- `N` = NPC (talk with A button)
- `E` = Enemy (avoid or attack)
- `I` = Item (collect)
- `#` = Wall/obstacle
- `X` = Dense object (tree, rock, etc.)
- `.` = Sparse object
- ` ` = Empty space (walkable)

## 🚀 How to Use

### Option 1: Training Script

Add the `--enable-ascii-screen` flag when running training:

```bash
python train_visual_cnn_hybrid.py \
  --rom-path roms/zelda_oracle_of_seasons.gbc \
  --headless \
  --total-timesteps 100000 \
  --llm-frequency 5 \
  --enable-ascii-screen    # ← Enable ASCII visualization!
```

### Option 2: Programmatic Usage

```python
from agents.visual_cnn import create_ascii_visualization

# Get screen from environment
screen = env.render()  # (144, 160, 3) RGB array

# Get game state
_, _, _, _, info = env.step(action)
game_state = info.get('structured_state', {})

# Generate ASCII art
ascii_art = create_ascii_visualization(
    screen, 
    game_state,
    include_legend=True  # Optional: show symbol legend
)

print(ascii_art)
```

### Option 3: Direct Renderer

```python
from agents.visual_cnn import ASCIIRenderer

# Create renderer with custom dimensions
renderer = ASCIIRenderer(width=40, height=36)

# Render with entity positions
ascii_art = renderer.render_screen(
    screen,
    player_pos=(80, 72),
    npcs=[(100, 72), (60, 72)],
    enemies=[(120, 60)],
    items=[(80, 100)]
)
```

## 🎨 How It Works

### 1. Screen Sampling

The Game Boy screen (144×160 pixels) is sampled into a smaller ASCII grid:
- Default: 40 characters wide × 36 characters tall
- Each ASCII character represents a 4×4 pixel region

### 2. Brightness Mapping

Pixel brightness is mapped to ASCII characters:
- Very dark (0-50) → `#` (walls, obstacles)
- Dark (50-100) → `X` (dense objects)
- Medium (100-150) → `*` (medium density)
- Light (150-200) → `.` (sparse objects)
- Very light (200-255) → ` ` (empty space)

### 3. Entity Overlay

Game entities are overlaid with specific symbols:
- Link's position → `@`
- NPC positions → `N`
- Enemy positions → `E`
- Item positions → `I`

## ⚡ Performance

- **Average rendering time:** ~3ms per call
- **Memory overhead:** Minimal (<1MB)
- **LLM context:** Adds ~2-3KB to prompt

**Recommendation:** Safe to use at 5-step intervals (every 5 frames).

## 🧠 LLM Prompt Integration

When ASCII screen is enabled, the LLM receives visual context:

```
CURRENT LOCATION:
- Room: Northern Holodrum - Starting Screen (ID: 182)
- Link's Position: (80, 72)
- Facing: up
- Health: 3/3 hearts

SURROUNDINGS:
- NPCs nearby: 2 (types: ['villager', 'guard'])
- Enemies: 1
- Items visible: 0

SCREEN VIEW:
==========================================
|########################################|
|............................N...........|
|.........@..............................|
|........................E...............|
|........................................|
==========================================

LEGEND: @ = Link, N = NPC, E = Enemy, I = Item, # = Wall

YOUR TASK:
Suggest ONE Game Boy button press...
```

This allows the LLM to:
- **See spatial relationships** (NPC to the right, enemy below)
- **Understand obstacles** (walls blocking paths)
- **Plan routes** (navigate around obstacles toward targets)

## 📊 Comparison

### Without ASCII Screen
```
LLM sees: "NPCs nearby: 2, Enemies: 1, Items: 0"
LLM thinks: "Press A to interact with NPC"
Problem: No idea WHERE the NPC is!
```

### With ASCII Screen
```
LLM sees: ASCII art showing NPC 'N' to the upper-right
LLM thinks: "Press UP and RIGHT to reach NPC, THEN press A"
Success: Spatially-aware pathfinding!
```

## 🔧 Customization

### Change ASCII Grid Size

```python
renderer = ASCIIRenderer(width=80, height=72)  # Finer detail
# or
renderer = ASCIIRenderer(width=20, height=18)  # Coarser (faster)
```

**Trade-off:**
- Larger grid = more detail, slower rendering, larger LLM context
- Smaller grid = less detail, faster rendering, smaller LLM context

### Custom Character Mapping

Edit `agents/visual_cnn/ascii_renderer.py`:

```python
self.CHAR_MAP = {
    'very_dark': '█',   # Use block characters
    'dark': '▓',
    'medium': '▒',
    'light': '░',
    'very_light': ' ',
    'link': '☺',        # Use Unicode emojis
    'npc': '♠',
    # ... etc
}
```

## 🐛 Troubleshooting

### "ASCII rendering failed"

**Cause:** Screen data is None or wrong shape.

**Fix:** Ensure environment is in visual mode:
```python
env_config = {
    "environment": {
        "observation_type": "visual"  # Not "vector"!
    }
}
```

### LLM ignoring visual context

**Cause:** LLM might not "understand" ASCII art.

**Fix:** Add more explicit legend and instructions in prompt:
```python
ascii_screen = f"""
IMPORTANT: The screen below shows what Link can see:
- '@' is YOUR position (Link)
- 'N' marks NPCs you can talk to
- Move toward 'N' symbols to interact with NPCs!

{ascii_art}
"""
```

### Performance issues

**Cause:** Rendering at every step is slow.

**Fix:** Only render when calling LLM (already implemented):
```python
if self.global_step % self.llm_frequency == 0:
    llm_suggestion = self.call_llm(game_state, screen=obs)
```

## 📈 Expected Impact

### Exploration Improvement

Without ASCII:
- Agent explores ~6 rooms in 1000 timesteps
- No spatial understanding of NPCs

With ASCII:
- Agent should explore ~10-15 rooms (estimated)
- Better NPC interaction (can "see" where to move)
- Smarter obstacle avoidance

### Real-World Example

**Scenario:** Link needs to talk to an NPC.

**Without ASCII:**
```
LLM: "There's an NPC nearby, press A to talk"
Agent: *presses A* (but NPC is 5 tiles away - nothing happens)
Result: Wasted action
```

**With ASCII:**
```
LLM sees:
|....N...|
|........|
|...@....|

LLM: "NPC is 2 tiles up and 1 tile left, press UP"
Agent: *moves toward NPC*
Result: Efficient navigation!
```

## 🎓 Advanced Usage

### Conditional ASCII (Performance Optimization)

Only enable ASCII for difficult scenarios:

```python
def call_llm(self, game_state, screen=None):
    # Only use ASCII when NPCs are nearby
    use_ascii = len(game_state.get('npcs', [])) > 0
    
    if use_ascii and screen is not None:
        ascii_screen = create_ascii_visualization(screen, game_state)
    else:
        ascii_screen = ""
```

### Multi-Resolution Strategy

Use coarse ASCII normally, fine ASCII for important decisions:

```python
if in_dungeon or near_boss:
    renderer = ASCIIRenderer(width=80, height=72)  # High detail
else:
    renderer = ASCIIRenderer(width=40, height=36)  # Standard
```

## 🚀 Next Steps

1. **Test with different LLM models** - Some models understand ASCII better than others
2. **Fine-tune character mapping** - Experiment with different symbols
3. **Add more entity types** - Chests, doors, water, etc.
4. **Implement color hints** - Use colored ASCII if LLM supports it

## 📝 Notes

- ASCII rendering is **deterministic** - same screen always produces same ASCII
- Entity positions are from structured game state, not CV detection
- Works with **any** text-based LLM (no vision model required!)
- Minimal latency impact (~3ms per call)

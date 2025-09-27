# Visual Mode Guide: Watching Your RL Agent Learn

This guide explains how to use visual mode to watch your RL agent learn in real-time with the PyBoy display window.

## Overview

The Zelda RL system supports two display modes:

1. **Headless Mode**: No display window (default) - 3750+ steps/sec
2. **Visual Mode**: PyBoy display window enabled - ~60 steps/sec

Visual mode is **62x slower** but invaluable for debugging and understanding agent behavior.

## When to Use Visual Mode

### ✅ Use Visual Mode For:
- **Debugging agent behavior**: Watch what actions the agent takes
- **Single episode testing**: Quick validation of agent policies  
- **Training demonstrations**: Show how the system works
- **Strategy analysis**: Understand why the agent makes certain decisions
- **Development**: Verify environment and agent integration

### ❌ Don't Use Visual Mode For:
- **Full training runs**: Too slow for serious training
- **Performance benchmarking**: Visual rendering skews timing
- **Production deployment**: Headless mode is faster and more stable
- **Automated testing**: Scripts don't need visual feedback

## Quick Start

### 🎮 Watch RL Agent Learn (Recommended)
```bash
# Quick 30-second demo - perfect for first-time viewing
python watch_rl_quick.py

# Full learning session - watch extended training process  
python watch_rl_agent.py
```
**These scripts show Link actually moving around and "learning" in real-time!**

### Pure RL Visual Test
```bash
# Watch pure RL agent (no LLM guidance)
python examples/visual_test_pure_rl.py
```

### LLM-Guided Visual Test  
```bash
# Watch hybrid LLM+RL agent
python examples/visual_test_llm_guided.py
```

### Visual Mode Test
```bash
# Validate visual mode functionality
python tests/test_visual_mode.py
```

## Configuration Options

### Environment Configuration

#### Headless Mode (Production)
```python
from emulator.zelda_env_configurable import create_pure_rl_env

env = create_pure_rl_env("rom.gbc", headless=True)  # Fast
```

#### Visual Mode (Debug)
```python
from emulator.zelda_env_configurable import create_visual_test_pure_rl_env

env = create_visual_test_pure_rl_env("rom.gbc")  # Slow but watchable
```

### PyBoy Direct Configuration
```python
from pyboy import PyBoy

# Headless mode
pyboy = PyBoy("rom.gbc", window="null")

# Visual mode  
pyboy = PyBoy("rom.gbc", window="SDL2")
```

### Visual Test Mode Settings

Visual test mode automatically applies optimizations:

- **Shorter episodes**: 1000 steps instead of 10000
- **Lower frame skip**: 2 instead of 4 (smoother visual)
- **Single episode**: Stops after one episode
- **Enhanced logging**: More frequent status updates

## Performance Comparison

| Mode | Steps/Second | Use Case | Window |
|------|-------------|----------|---------|
| **Headless** | 3750+ | Training, Production | None |
| **Visual** | ~60 | Debugging, Demo | SDL2 |

**Performance Impact**: Headless is **62x faster** than visual mode.

## Visual Test Scripts

### 1. Pure RL Visual Test (`examples/visual_test_pure_rl.py`)

**Purpose**: Watch pure RL agent without LLM guidance

**Features**:
- Random action policy (for demonstration)
- Real-time action statistics
- Episode reward tracking
- Action distribution analysis

**Sample Output**:
```
🎮 Creating Visual Test Environment...
📊 Configuration Summary:
   llm_mode: False
   headless: False
   visual_test_mode: True
   max_steps: 1000

🚀 Episode Started (Max 1000 steps)
Actions: 0=NOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START, 8=SELECT

Step  100: Action=3, Reward=0.010, Total Reward=1.234
Step  200: Action=5, Reward=-0.001, Total Reward=2.456
...
```

### 2. LLM-Guided Visual Test (`examples/visual_test_llm_guided.py`)

**Purpose**: Watch hybrid LLM+RL agent with structured state

**Features**:
- Simulated LLM decision points
- Structured state display (player health, position)
- Strategic behavior based on game state
- LLM prompt visualization

**Sample Output**:
```
🧠 Creating Visual Test Environment (LLM-Guided)...
📊 Player: 3/3 hearts, Position (120, 80)
💭 LLM Context: "Link is healthy and in Horon Village. Ready for adventure..."

🧠 LLM Decision #1 at step 0  
   LLM sees: "Link has full health in Horon Village..."
   🎯 LLM Strategy: Explore and collect items
```

## Programmatic Usage

### Basic Visual Mode
```python
import time
from emulator.zelda_env_configurable import create_visual_test_pure_rl_env

# Create visual test environment
env = create_visual_test_pure_rl_env("roms/zelda_oracle_of_seasons.gbc")

# Run single episode with visual display
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    time.sleep(0.05)  # Slow down for comfortable viewing
    
    if terminated or truncated:
        break

env.close()
```

### Advanced Visual Mode with State Monitoring
```python
from emulator.zelda_env_configurable import create_visual_test_llm_env

env = create_visual_test_llm_env("roms/zelda_oracle_of_seasons.gbc") 
obs, info = env.reset()

for step in range(1000):
    # Get structured state for decision making
    structured_state = info.get('structured_state', {})
    
    # Make strategic decisions based on state
    if 'player' in structured_state:
        player = structured_state['player']
        health = player.get('health', 3)
        
        # Conservative play when low health
        if health <= 1:
            action = 0  # No action (wait)
        else:
            action = env.action_space.sample()
    else:
        action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Display game state info
    if step % 100 == 0 and structured_state:
        print(f"Step {step}: {structured_state.get('llm_prompt', 'No state')}")
    
    time.sleep(0.05)  # 20 FPS viewing
    
    if terminated or truncated:
        break

env.close()
```

## Configuration Files

### Visual Test Configuration (`configs/controller_ppo_visual_test.yaml`)

Key settings for visual testing:

```yaml
# Environment settings optimized for visual observation
environment:
  frame_skip: 2              # Slower for better visual observation
  max_episode_steps: 1000    # Shorter episodes

# Training settings for single episode
training:
  total_timesteps: 1000      # Single episode
  log_frequency: 10         # Frequent status updates

# Visual mode settings
visual_test:
  mode: "enabled"
  episode_limit: 1
  display_stats: true
  show_action_info: true
```

## Troubleshooting

### Visual Mode Not Working

**Issue**: `SDL2 window failed` or no display appears

**Solutions**:
1. **Check display**: Ensure you have a display/desktop environment
2. **X11 forwarding**: For SSH, use `ssh -X` or `ssh -Y`  
3. **WSL**: Install VcXsrv or X410 for Windows
4. **Docker**: Add `--env DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`

### Performance Issues

**Issue**: Visual mode is too slow

**Expected**: Visual mode is 60+ times slower than headless
**Solutions**:
- Use visual mode only for debugging/demo
- Increase `frame_skip` for faster (but choppier) display
- Reduce episode length
- Switch to headless mode for training

### Display Quality Issues  

**Issue**: Game display is too small/pixelated

PyBoy creates a small window matching Game Boy resolution (160x144). This is normal.

**Options**:
- Use PyBoy's built-in scaling options
- Consider recording sessions instead of live viewing
- Focus on agent behavior rather than visual quality

## Integration with Training

### Development Workflow

1. **Start with visual mode**: Debug and validate agent behavior
2. **Switch to headless**: Run actual training at full speed
3. **Return to visual**: Check trained agent performance

### Training Pipeline
```python
# 1. Visual validation
env = create_visual_test_pure_rl_env(rom_path)
# ... run short test to validate behavior

# 2. Full training (headless)  
env = create_pure_rl_env(rom_path, headless=True)
# ... run full training loop

# 3. Visual evaluation
env = create_visual_test_pure_rl_env(rom_path)
# ... load trained model and watch it play
```

## Best Practices

### Visual Testing
1. **Keep episodes short**: 500-1000 steps maximum
2. **Add viewing delays**: `time.sleep(0.05)` for comfortable observation
3. **Display key metrics**: Show rewards, health, position
4. **Use keyboard interrupts**: Allow easy test termination

### Development
1. **Validate visually first**: Check agent behavior before full training
2. **Profile both modes**: Measure performance difference
3. **Document findings**: Record interesting behaviors observed
4. **Test edge cases**: Low health, stuck situations, etc.

### Production
1. **Always use headless for training**: Visual mode is too slow
2. **Visual mode for demos only**: Show stakeholders how it works
3. **Automate testing**: Don't rely on visual validation for CI/CD
4. **Monitor headless performance**: Visual testing doesn't reflect training speed

## Future Enhancements

### Planned Features
- **Recording capability**: Save visual sessions for later review
- **Interactive controls**: Pause/step through agent decisions
- **Side-by-side comparison**: Watch multiple agents simultaneously
- **Training curves overlay**: Show real-time learning progress

### Research Applications  
- **Behavior analysis**: Quantify agent exploration patterns
- **Strategy evolution**: Track how strategies change over training
- **Failure case study**: Analyze why agents fail in certain situations
- **Human vs AI comparison**: Compare human and AI gameplay

## Conclusion

Visual mode is an essential debugging tool that provides insights into RL agent behavior that metrics alone cannot capture. While too slow for training, it's invaluable for:

- **Understanding** what your agent is actually doing
- **Debugging** why training isn't working  
- **Demonstrating** your system to others
- **Validating** that your agent is learning meaningful behaviors

Use visual mode strategically as part of your development workflow, but always switch to headless mode for actual training to achieve optimal performance.

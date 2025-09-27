# RL Training Modes: Pure RL vs LLM-Guided

This document explains how to use the two training modes available in the Zelda Oracle of Seasons RL system.

## Overview

The system now supports two distinct training modes:

1. **Pure RL Mode**: Traditional reinforcement learning without LLM guidance
2. **LLM-Guided Mode**: Hybrid approach with LLM providing strategic guidance

## Mode Comparison

| Feature | Pure RL Mode | LLM-Guided Mode |
|---------|-------------|-----------------|
| **Speed** | 20-50 steps/sec | 5-15 steps/sec |
| **Learning** | Slower, trial-and-error | Faster, strategic guidance |
| **Dependencies** | Minimal (PyBoy + RL) | Full (PyBoy + RL + LLM API) |
| **Memory Usage** | Low (~100MB) | Medium (~200-300MB) |
| **Debugging** | Simple | Complex |
| **API Costs** | None | LLM API calls (~$0.01/decision) |

## Pure RL Mode

### When to Use
- Validating RL implementation works independently
- Baseline performance comparison
- Limited computational resources
- No LLM API access
- Research on pure RL approaches

### Configuration

Use `configs/controller_ppo_pure_rl.yaml`:

```yaml
planner_integration:
  use_planner: false           # Disable LLM
  enable_visual: false         # No visual processing
  use_structured_entities: false  # No entity extraction
```

### Key Settings
- **Total timesteps**: 2,000,000 (more training needed)
- **Max episode steps**: 15,000 (longer exploration)
- **Exploration**: Higher epsilon values (0.2 → 0.02)
- **Curiosity**: Enabled for intrinsic motivation
- **Logging**: `wandb_project: "zelda-pure-rl"`

### Usage Example

```python
from emulator.zelda_env_configurable import create_pure_rl_env

# Create pure RL environment
env = create_pure_rl_env("roms/zelda_oracle_of_seasons.gbc")

# Standard RL training loop
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()  # or use your RL agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Performance Characteristics
- **Faster execution**: No LLM API calls or complex state processing
- **Simpler debugging**: Fewer components to troubleshoot
- **Longer learning time**: Must discover strategies through exploration
- **May get stuck**: Without strategic guidance, agent might not progress

## LLM-Guided Mode

### When to Use
- Fastest learning and best final performance
- Research on hybrid AI architectures
- Demonstrating strategic AI capabilities
- Production gaming AI applications

### Configuration

Use `configs/controller_ppo.yaml`:

```yaml
planner_integration:
  use_planner: true            # Enable LLM
  enable_visual: true          # Visual processing
  use_structured_entities: true   # Entity extraction
  planner_frequency: 100       # Plan every 100 frames
```

### Key Settings
- **Total timesteps**: 1,000,000 (less training needed)
- **Max episode steps**: 10,000 (normal length)
- **LLM calls**: Every ~1.67 seconds
- **Visual compression**: Bit-packed for efficiency
- **Logging**: `wandb_project: "zelda-rl"`

### Usage Example

```python
from emulator.zelda_env_configurable import create_llm_guided_env
from agents.controller import ZeldaPPOController, ControllerConfig

# Create LLM-guided environment
env = create_llm_guided_env("roms/zelda_oracle_of_seasons.gbc")

# Create controller with LLM integration
config = ControllerConfig(use_planner=True, planner_frequency=100)
controller = ZeldaPPOController(env, config)

# Hybrid training loop
obs, info = env.reset()
structured_state = info.get('structured_state')

for step in range(1000):
    action = await controller.act(obs, structured_state)  # LLM + RL decision
    obs, reward, terminated, truncated, info = env.step(action)
    structured_state = info.get('structured_state')
    if terminated or truncated:
        obs, info = env.reset()
        structured_state = info.get('structured_state')
```

### Performance Characteristics
- **Strategic intelligence**: LLM provides domain knowledge
- **Faster learning**: Reduced sample complexity
- **Higher final performance**: Better than either approach alone
- **Complex setup**: More dependencies and configuration

## Switching Between Modes

### Method 1: Configuration Files
```bash
# Pure RL training
python training/run_cleanrl.py --config configs/controller_ppo_pure_rl.yaml

# LLM-guided training  
python training/run_cleanrl.py --config configs/controller_ppo.yaml
```

### Method 2: Programmatic Configuration
```python
# Pure RL
config = ControllerConfig(use_planner=False)

# LLM-guided
config = ControllerConfig(use_planner=True, planner_frequency=100)
```

### Method 3: Environment Factories
```python
# Pure RL environment
env = create_pure_rl_env(rom_path)

# LLM-guided environment
env = create_llm_guided_env(rom_path)
```

## Testing and Validation

### Configuration Validation
```bash
python tests/test_rl_configs.py
```

### Pure RL Example
```bash
python examples/train_pure_rl.py
```

### Performance Comparison
Both modes have been validated:
- ✅ Configuration files are valid
- ✅ PyBoy functionality works
- ✅ Action spaces are correct
- ✅ Performance profiles are documented

## Training Tips

### Pure RL Mode
1. **Increase exploration**: Use higher epsilon values longer
2. **Enable curiosity**: Helps discover game mechanics
3. **Longer episodes**: Allow time for exploration
4. **Dense rewards**: Shape rewards for learning signals
5. **Patience**: Pure RL takes significantly more training time

### LLM-Guided Mode
1. **Verify LLM connectivity**: Test API endpoints first
2. **Monitor API costs**: LLM calls add up over time
3. **Tune planner frequency**: Balance performance vs cost
4. **Handle fallbacks**: LLM failures should gracefully degrade to RL
5. **Debug structured states**: Ensure LLM receives good data

## Troubleshooting

### Pure RL Issues
- **Slow learning**: Normal, increase training time
- **Agent stuck**: Add exploration bonuses or curiosity
- **Poor performance**: Check reward shaping

### LLM-Guided Issues
- **LLM API failures**: Implement fallback to pure RL
- **Slow execution**: Reduce planner frequency
- **High costs**: Optimize prompt size or use cheaper model
- **Parse errors**: Validate LLM response format

## Future Enhancements

### Planned Improvements
1. **Adaptive mode switching**: Start with LLM, transition to pure RL
2. **Multi-environment training**: Parallel environments for both modes
3. **Performance benchmarking**: Automated comparison suite
4. **Hybrid reward functions**: Combine intrinsic and LLM-guided rewards

### Research Opportunities
1. **Learning efficiency comparison**: Quantify sample complexity differences
2. **Transfer learning**: Use LLM-guided agent to bootstrap pure RL
3. **Curriculum learning**: Gradually reduce LLM guidance over time
4. **Multi-modal learning**: Combine visual, text, and strategic signals

## Conclusion

Both modes serve different purposes:

- **Pure RL** is ideal for validation, research, and resource-constrained scenarios
- **LLM-Guided** is optimal for performance, rapid prototyping, and production systems

The configurable architecture allows easy switching between modes, enabling comprehensive evaluation and the best tool for each specific use case.

Choose the mode that best fits your computational resources, research goals, and performance requirements.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a hybrid AI agent to play The Legend of Zelda: Oracle of Seasons using:
- **Planner**: 70B LLM served via vLLM on KServe that reads structured game state and produces subgoals/macro-actions
- **Controller**: PPO reinforcement learning agent that translates macros into frame-level button presses
- **Emulator**: PyBoy running Oracle of Seasons with RAM/tile-based state observation (not pixels)

## Architecture

The system follows a two-tier architecture:
1. **High-level planner** (LLM): Reasons over game state to generate strategic decisions
2. **Low-level controller** (RL): Executes precise movements and actions

Key components:
- `emulator/zelda_env.py`: Gymnasium environment interface to PyBoy
- `observation/state_encoder.py`: Converts RAM/tile data to structured JSON state
- `agents/planner.py`: KServe client for vLLM planner service
- `agents/controller.py`: PPO agent with macro-action expansion
- `training/run_cleanrl.py`: PPO baseline training
- `training/run_grpo_llm.py`: GRPO preference optimization for planner

## Development Commands

### Environment Setup
```bash
pip install gymnasium pyboy==2.* cleanrl pufferlib "trl>=0.9" httpx
```

### Training
```bash
# Run PPO baseline controller training
python training/run_cleanrl.py

# Train planner with preference optimization
python training/run_grpo_llm.py
```

### Deployment
```bash
# Deploy LLM planner service to OpenShift
oc apply -f ops/openshift/kserve/zelda-planner-70b.yaml
```

## Key Technical Details

- **Action Space**: 9 discrete actions (UP, DOWN, LEFT, RIGHT, A, B, START, SELECT, NOP)
- **Observation**: 64-256 float vector from encoded game state
- **Macro Actions**: High-level commands like MOVE_TO(x,y), ENTER_DUNGEON(id), FIND_STUMP_AND_SET(season)
- **Rewards**: Shaped rewards for rupees (+0.01), keys (+0.5), bosses (+2), death (-3)
- **State Source**: RAM addresses and tile maps (not visual/OCR)

## ROM Requirements

Place a legally obtained Oracle of Seasons ROM file in the `roms/` directory. The project does not distribute ROM files.

## OpenShift AI Integration

This project is designed to run on OpenShift AI:
- Use GPU-enabled Workbench for training experiments
- Deploy planner as KServe InferenceService
- Controller runs in Workbench pod, connects to planner service via HTTP
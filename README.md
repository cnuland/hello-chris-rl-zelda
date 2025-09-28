# 🎮 Zelda-LLM-RL: Oracle of Seasons AI System

A hybrid AI system combining **Large Language Model strategic planning** with **Reinforcement Learning low-level control** to play **The Legend of Zelda: Oracle of Seasons**.

## 🚀 **3 Core Areas**

This system is organized around **3 primary use cases**:

### 1. 🖥️ **Headless Training** - Production Training Runs
- **Command**: `make headless`
- **Purpose**: High-performance training for model development
- **Features**: Multi-environment parallel training, 5X LLM reward emphasis, exploration bonuses
- **Speed**: ~3000+ steps/second (maximum performance)

### 2. 👁️ **Visual Training** - Watch Training Live  
- **Command**: `make visual`
- **Purpose**: Single episode training with live visualization
- **Features**: PyBoy emulator window + Web HUD, real-time LLM decisions, training progress
- **Speed**: ~15-30 steps/second (watchable)

### 3. 🎯 **Visual Inference** - Watch Trained Model Play
- **Command**: `make inference CHECKPOINT=model.pkl`
- **Purpose**: Load trained checkpoint and watch AI play (NO training updates)
- **Features**: PyBoy emulator window + Web HUD, strategic decision visualization
- **Speed**: Real-time gameplay

---

## 🧠 **Architecture Overview**

- **LLM Planner**: MLX Qwen2.5-14B-Instruct-4bit (local inference)
- **RL Controller**: PPO via Gymnasium with smart arbitration
- **Game Interface**: PyBoy emulator with direct RAM/memory access
- **State Encoding**: Structured JSON from game memory (not pixels)
- **Reward System**: 5X multiplier for LLM-aligned actions + exploration bonuses

---

## 🛠️ **Quick Start**

### 1. **Installation**
```bash
# Install dependencies
make install

# Place ROM in roms/ directory (legally obtained)
# Oracle of Seasons ROM: roms/zelda_oracle_of_seasons.gbc
```

### 2. **Start LLM Server**
```bash
# Terminal 1: Start local MLX server
make llm-serve

# Check server status
make llm-status
```

### 3. **Choose Your Mode**
```bash
# Watch training live (recommended first try)
make visual

# Production training run  
make headless

# Watch trained model play (requires checkpoint)
make inference CHECKPOINT=path/to/model.pkl
```

---

## 📊 **Available Commands**

### **🚀 Core Areas**
```bash
make headless         # Production headless training
make visual          # Visual training with PyBoy + Web HUD  
make inference       # Visual inference with trained model
```

### **🧠 LLM Server**
```bash
make llm-serve       # Start MLX Qwen2.5-14B local server
make llm-status      # Check if server is running
make llm-stop        # Stop the server
```

### **🛠️ Utilities**
```bash
make install         # Install dependencies
make clean          # Clean Python cache files
make run-all         # Demo all 3 modes (requires checkpoint)
make core-help       # Detailed help for each area
```

### **⚙️ Custom Parameters**
```bash
# Custom headless training
make headless SESSIONS=10 EPISODES=50 EPOCHS=6 BATCH_SIZE=512

# Inference with specific checkpoint  
make inference CHECKPOINT=training_runs/session_5/model.pkl
```

---

## 🌟 **Key Features**

### **🧠 LLM Emphasis System**
- **Normal Action**: +10 points for room discovery
- **LLM-Aligned Action**: **+50 points** (5x multiplier!)
- **Strategic Bonus**: +2 points/step for following LLM guidance
- **Goal Completion**: +50 bonus points

### **🔄 Smart Arbitration**
- Context-aware LLM calls (new rooms, low health, stuck situations)
- Adaptive frequency (10-50 step intervals)
- MLX caching for 1.3-second response times

### **🎯 Enhanced Rewards**
- **New Room Discovery**: 10 points (50 with LLM alignment)
- **Dungeon Entry**: 25 points (125 with LLM alignment)  
- **NPC Interaction**: 15 points (75 with LLM alignment)
- **Continuous Dungeon Presence**: 5 points/step

### **📊 Real-Time Visualization**
- **PyBoy Window**: Live Game Boy emulator display
- **Web HUD**: Browser-based dashboard at `http://localhost:8086`
  - Real-time LLM commands and reasoning
  - Training metrics and reward tracking
  - Arbitration performance statistics

---

## 📁 **Project Structure**

```
zelda-rl-llm/
├── train_headless.py      # Core Area 1: Production training
├── train_visual.py        # Core Area 2: Visual training
├── run_inference.py       # Core Area 3: Visual inference
├── agents/               # RL controller + LLM planner
├── emulator/            # PyBoy bridge + environment
├── observation/         # State encoding + visual processing
├── configs/             # Configuration files
├── tests/               # Test infrastructure
├── training_runs/       # Training outputs and checkpoints
└── roms/               # Game ROM and save states
```

---

## 🎯 **Training Pipeline**

1. **Development**: Use `make visual` to watch and debug training
2. **Production**: Use `make headless` for long training runs
3. **Evaluation**: Use `make inference` to test trained models

---

## 📝 **Example Usage**

```bash
# Quick start - watch training live
make visual

# Production training (20 sessions, 30 episodes each)
make headless SESSIONS=20 EPISODES=30

# Test trained model
make inference CHECKPOINT=training_runs/session_15/final_model.pkl

# 8-hour marathon training
make headless SESSIONS=50 EPISODES=100 EPOCHS=8
```

---

## 🔧 **System Requirements**

- **Python 3.11+**
- **Apple Silicon** (for MLX optimization) or compatible system
- **ROM**: Legend of Zelda: Oracle of Seasons (legally obtained)
- **Save State**: `roms/zelda_oracle_of_seasons.gbc.state` (post-cutscenes)
- **MLX Server**: Qwen2.5-14B-Instruct-4bit model

---

## 🤝 **Contributing**

This system demonstrates the power of combining LLM strategic reasoning with RL tactical execution. The **5X reward emphasis** creates a strong bias toward LLM guidance while maintaining the flexibility of RL exploration.

For detailed technical documentation, see `CORE_SYSTEM.md` and `LLM_EMPHASIS_SYSTEM.md`.

---

**Built with**: PyBoy • MLX • Gymnasium • PPO • Qwen2.5-14B • OpenAI API • Flask
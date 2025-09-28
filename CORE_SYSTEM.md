# 🎮 Zelda RL-LLM Core System

## 🚀 **3 Core Areas**

This project is organized around **3 primary use cases**:

### 1. 🖥️ **HEADLESS TRAINING** - Production Training Runs

**File**: `train_headless.py`

- **Purpose**: Production-level training runs for model development
- **Mode**: Headless (no visual display) for maximum performance  
- **Configuration**: Full control over sessions, episodes, epochs, batch size
- **Features**:
  - Multi-environment parallel training
  - LLM guidance with 5X reward emphasis
  - Exploration bonuses (rooms, dungeons, NPCs)
  - Comprehensive logging and metrics
  - Save state integration
  - Long-duration training support

**Usage**:
```bash
python train_headless.py --sessions 10 --episodes 50 --epochs 4 --batch-size 256
```

---

### 2. 👁️ **VISUAL TRAINING** - Watch Training in Action

**File**: `train_visual.py`

- **Purpose**: Single episode training with live visualization
- **Mode**: Visual PyBoy window + Web HUD interface
- **Configuration**: Single episode, single epoch for demonstration
- **Features**:
  - Live PyBoy emulator window
  - Real-time Web HUD showing LLM decisions
  - Training progress visualization
  - LLM command display with 5X emphasis indicators
  - Performance metrics and reward tracking

**Usage**:
```bash
python train_visual.py
```
Opens browser HUD at `http://localhost:8086`

---

### 3. 🎯 **VISUAL INFERENCE** - Watch Trained Model Play

**File**: `run_inference.py`

- **Purpose**: Load trained checkpoint and watch model play
- **Mode**: Visual PyBoy window + Web HUD interface  
- **Configuration**: Inference only (NO training updates)
- **Features**:
  - Load pre-trained model checkpoint
  - Live PyBoy emulator window
  - Real-time Web HUD showing AI decision-making
  - LLM strategic guidance visualization
  - Performance analysis of trained model

**Usage**:
```bash
python run_inference.py --checkpoint path/to/model.pkl
```
Opens browser HUD at `http://localhost:8086`

---

## 🧠 **LLM Emphasis System**

All modes include the **5X Reward Multiplier System**:

- **Normal Action**: +10 points for room discovery
- **LLM-Aligned Action**: **+50 points** (5x multiplier!)
- **Goal Completion**: **+50 bonus points**
- **Strategic Alignment**: **+2 points/step** for following LLM direction

## ⚙️ **Core Components**

- **`emulator/zelda_env_configurable.py`**: Main Gymnasium environment
- **`agents/controller.py`**: PPO controller with smart arbitration
- **`agents/local_llm_planner.py`**: MLX Qwen2.5-14B integration
- **`observation/state_encoder.py`**: Game state → structured data
- **`configs/`**: Configuration files for different modes

## 📊 **Training Pipeline**

1. **Development**: Use `train_visual.py` to watch and debug training
2. **Production**: Use `train_headless.py` for long training runs  
3. **Evaluation**: Use `run_inference.py` to test trained models

## 🎯 **Quick Start**

```bash
# 1. Watch training in action
python train_visual.py

# 2. Run production training  
python train_headless.py --sessions 5 --episodes 20 --epochs 2

# 3. Test trained model (requires checkpoint)
python run_inference.py --checkpoint training_runs/latest/model.pkl
```

---

## 🔧 **System Requirements**

- **MLX Server**: Local Qwen2.5-14B-Instruct-4bit running on `localhost:8000`
- **PyBoy**: Game Boy emulator integration
- **ROM**: `roms/zelda_oracle_of_seasons.gbc`
- **Save State**: `roms/zelda_oracle_of_seasons.gbc.state` (post-cutscenes)
- **Python 3.11+** with project dependencies

This streamlined structure focuses on the **3 essential use cases** while maintaining the full power of the LLM-guided RL system! 🚀

# Zelda-LLM-RL Core System Makefile
# 
# Supports 3 core areas:
# 1. Headless Training - Production training runs (train_headless.py)  
# 2. Visual Training   - Watch training live (train_visual.py)
# 3. Visual Inference  - Watch trained model play (run_inference.py)

# MLX Configuration for Qwen2.5 Model (Apple Silicon optimized)
VLLM_MODEL := mlx-community/Qwen2.5-14B-Instruct-4bit
VLLM_PORT := 8000
PYTHON := python3

# Training Parameters
SESSIONS ?= 5
EPISODES ?= 20  
EPOCHS ?= 4
BATCH_SIZE ?= 256
CHECKPOINT ?= 

.PHONY: help install llm-serve llm-stop llm-status clean headless visual inference run-all core-help

help: ## Show available commands and core system overview
	@echo "🎮 Zelda-LLM-RL Core System"
	@echo "==========================="
	@echo ""
	@echo "🚀 3 CORE AREAS:"
	@echo "  headless      - Production training (train_headless.py)"
	@echo "  visual        - Watch training live (train_visual.py + Web HUD)"
	@echo "  inference     - Watch trained model play (run_inference.py + Web HUD)"
	@echo ""
	@echo "🧠 LLM SERVER:"
	@echo "  llm-serve     - Start MLX Qwen2.5-14B local server"  
	@echo "  llm-stop      - Stop MLX server"
	@echo "  llm-status    - Check server status"
	@echo ""
	@echo "🛠️  UTILITIES:"
	@echo "  install       - Install dependencies"
	@echo "  clean         - Clean Python cache files"
	@echo "  run-all       - Launch all 3 modes (demo)"
	@echo "  core-help     - Detailed help for each core area"
	@echo ""
	@echo "Quick Start: make llm-serve && make visual"

install: ## Install project dependencies
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install mlx-lm

llm-serve: ## Start MLX local LLM server (Qwen2.5-14B-Instruct-4bit)
	@echo "🧠 Starting MLX LLM Server..."
	@echo "Model: $(VLLM_MODEL)"
	@echo "Port: $(VLLM_PORT)"
	@echo "Optimized for: Apple Silicon"
	@echo ""
	@echo "Server will be available at: http://localhost:$(VLLM_PORT)"
	@echo "Press Ctrl+C to stop server"
	@echo ""
	mlx_lm.server --model $(VLLM_MODEL) --port $(VLLM_PORT)

llm-stop: ## Stop MLX LLM server
	@echo "🛑 Stopping MLX LLM server..."
	@pkill -f "mlx_lm.server" || echo "No MLX server found"
	@pkill -f "$(VLLM_MODEL)" || echo "No model processes found"

llm-status: ## Check if MLX LLM server is running
	@echo "🔍 Checking MLX LLM server status..."
	@echo "Endpoint: http://localhost:$(VLLM_PORT)/v1/models"
	@curl -s http://localhost:$(VLLM_PORT)/v1/models | jq . || echo "❌ Server not responding or jq not installed"

clean: ## Clean Python cache files and temporary data
	@echo "🧹 Cleaning Python cache files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✅ Cache cleaned"

# ========================================
# 🚀 CORE AREA 1: HEADLESS TRAINING
# ========================================

headless: ## Production headless RL training (high performance)
	@echo "🖥️  HEADLESS TRAINING - Production Mode"
	@echo "======================================"
	@echo "📊 Configuration:"
	@echo "   Sessions:     $(SESSIONS)"
	@echo "   Episodes:     $(EPISODES)" 
	@echo "   Epochs:       $(EPOCHS)"
	@echo "   Batch Size:   $(BATCH_SIZE)"
	@echo "   Mode:         Headless (no visual display)"
	@echo "   Features:     5X LLM rewards + exploration bonuses"
	@echo ""
	@echo "🚀 Starting production training run..."
	$(PYTHON) train_headless.py \
		--sessions $(SESSIONS) \
		--episodes $(EPISODES) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--verbose

# ========================================  
# 🚀 CORE AREA 2: VISUAL TRAINING
# ========================================

visual: ## Visual RL training with live PyBoy + Web HUD
	@echo "👁️  VISUAL TRAINING - Watch Training Live"
	@echo "======================================="
	@echo "🎮 Features:"
	@echo "   - PyBoy emulator window (watch Link learn)"
	@echo "   - Web HUD at http://localhost:8086" 
	@echo "   - Real-time LLM decisions and training stats"
	@echo "   - Single episode/epoch for demonstration"
	@echo "   - 5X LLM emphasis system active"
	@echo ""
	@echo "🚀 Starting visual training..."
	@echo "📱 Web HUD will open in your browser"
	@echo "🎮 PyBoy window will show the game"
	$(PYTHON) train_visual.py

# ========================================
# 🚀 CORE AREA 3: VISUAL INFERENCE  
# ========================================

inference: ## Load trained model and watch it play (with PyBoy + Web HUD)
	@echo "🎯 VISUAL INFERENCE - Watch Trained Model Play"
	@echo "============================================="
	@echo "🧠 Features:"
	@echo "   - Load pre-trained checkpoint"
	@echo "   - PyBoy emulator window (watch trained AI play)"
	@echo "   - Web HUD at http://localhost:8086"
	@echo "   - Real-time LLM strategic decisions" 
	@echo "   - NO training updates (inference only)"
	@echo ""
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ ERROR: CHECKPOINT parameter required"; \
		echo "Usage: make inference CHECKPOINT=path/to/model.pkl"; \
		exit 1; \
	fi
	@echo "📂 Checkpoint: $(CHECKPOINT)"
	@echo "🚀 Starting inference mode..."
	$(PYTHON) run_inference.py --checkpoint $(CHECKPOINT)

# ========================================
# 🚀 UTILITY COMMANDS
# ========================================

run-all: ## Demo all 3 core areas (requires checkpoint)
	@echo "🎭 DEMO: All 3 Core Areas"
	@echo "========================"
	@echo "This will demonstrate all 3 modes in sequence:"
	@echo "1. Quick headless training (2 sessions)"
	@echo "2. Visual training demo"
	@echo "3. Inference with generated checkpoint"
	@echo ""
	@echo "⚠️  This is a demo mode - press Ctrl+C to stop any stage"
	@echo ""
	@read -p "Press Enter to start demo..."
	@$(MAKE) headless SESSIONS=2 EPISODES=5
	@$(MAKE) visual
	@echo "Note: inference requires a checkpoint from training"

core-help: ## Detailed help for each core area
	@echo "🎮 CORE SYSTEM DETAILED HELP"
	@echo "============================"
	@echo ""
	@echo "🖥️  HEADLESS TRAINING (make headless)"
	@echo "   Purpose: Production training runs"
	@echo "   Output:  training_runs/ directory with logs and checkpoints"
	@echo "   Speed:   ~3000+ steps/second (maximum performance)"
	@echo "   Usage:   make headless SESSIONS=10 EPISODES=50 EPOCHS=6"
	@echo ""
	@echo "👁️  VISUAL TRAINING (make visual)"  
	@echo "   Purpose: Watch training in real-time"
	@echo "   Windows: PyBoy game window + Web HUD (browser)"
	@echo "   Speed:   ~15-30 steps/second (watchable)"
	@echo "   Usage:   make visual"
	@echo ""
	@echo "🎯 VISUAL INFERENCE (make inference)"
	@echo "   Purpose: Watch trained model play"
	@echo "   Windows: PyBoy game window + Web HUD (browser)" 
	@echo "   Speed:   Real-time gameplay"
	@echo "   Usage:   make inference CHECKPOINT=model.pkl"
	@echo ""
	@echo "🧠 LLM SERVER COMMANDS:"
	@echo "   make llm-serve   - Start MLX Qwen2.5-14B server"
	@echo "   make llm-status  - Check if server is running"
	@echo "   make llm-stop    - Stop the server"
	@echo ""
	@echo "📊 TRAINING PARAMETERS (customize with VARIABLE=value):"
	@echo "   SESSIONS      - Number of training sessions (default: $(SESSIONS))"
	@echo "   EPISODES      - Episodes per session (default: $(EPISODES))"
	@echo "   EPOCHS        - Training epochs (default: $(EPOCHS))"
	@echo "   BATCH_SIZE    - Batch size for updates (default: $(BATCH_SIZE))" 
	@echo "   CHECKPOINT    - Path to model checkpoint (for inference)"
	@echo ""
	@echo "🚀 QUICK START:"
	@echo "   1. make install           # Install dependencies"
	@echo "   2. make llm-serve         # Start LLM server (new terminal)"
	@echo "   3. make visual            # Watch training live"
	@echo ""
	@echo "📁 OUTPUT STRUCTURE:"
	@echo "   training_runs/           # All training outputs"
	@echo "   ├── session_*/           # Individual session data"
	@echo "   ├── logs/                # Training logs"
	@echo "   └── checkpoints/         # Model checkpoints"
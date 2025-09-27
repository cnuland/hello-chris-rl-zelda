# Zelda-LLM-RL Local Development Makefile

# vLLM Configuration for DeepSeek Model
VLLM_MODEL := deepseek-ai/DeepSeek-R1-Distill-Llama-70B
VLLM_PORT := 8000
PYTHON := python3

.PHONY: help install test-local serve-local stop-local status clean visual-quick visual-train visual-checkpoint visual-test train train-pure-rl train-llm-guided train-quick train-20k train-parallel train-help

help: ## Show available commands
	@echo "Zelda-LLM-RL Local Development"
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install project dependencies
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install vllm

test-local: ## Deploy quantized model locally for testing
	@echo "Starting local vLLM server with quantized model..."
	@echo "Model: $(VLLM_MODEL)"
	@echo "Port: $(VLLM_PORT)"
	$(PYTHON) -m vllm.entrypoints.api_server \
		--model $(VLLM_MODEL) \
		--dtype auto \
		--tensor-parallel-size 1 \
		--swap-space 24 \
		--max-model-len 4096 \
		--enable-chunked-prefill \
		--enforce-eager \
		--gpu-memory-utilization 0.9 \
		--port $(VLLM_PORT) \
		--disable-log-requests

serve-local: test-local ## Alias for test-local

stop-local: ## Stop local vLLM server
	@echo "Stopping vLLM server..."
	@pkill -f "vllm.entrypoints.api_server" || echo "No vLLM server found"

status: ## Check if vLLM server is running
	@echo "Checking vLLM server status..."
	@curl -s http://localhost:$(VLLM_PORT)/v1/models | jq . || echo "Server not responding or jq not installed"

clean: ## Clean Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Visual RL Training Targets
visual-quick: ## Quick 30-second visual demo of RL agent learning
	@echo "🎮 Starting quick visual RL demo (30 seconds)..."
	@echo "PyBoy window will open showing Link learning to move around"
	$(PYTHON) watch_rl_quick.py

visual-train: ## Full visual RL training session from zero (fresh start)
	@echo "🤖 Starting full visual RL training session from scratch..."
	@echo "You'll see 3 phases: Random → Learning Movement → Strategic Play"
	@echo "PyBoy window will show Link learning over ~2000 steps"
	$(PYTHON) watch_rl_agent.py

visual-checkpoint: ## Visual test of trained RL agent from checkpoint
	@echo "🎯 Loading and watching previously trained RL agent..."
	@echo "You'll see more strategic behavior compared to untrained agent"
	@echo "Demonstrates trained vs untrained performance difference"
	$(PYTHON) watch_rl_checkpoint.py

visual-test: visual-quick ## Alias for quick visual test (recommended for first try)

# Visual RL Help
visual-help: ## Show detailed visual mode help
	@echo "🎮 Visual RL Training Mode Help"
	@echo "================================"
	@echo ""
	@echo "Visual mode lets you watch Link learn to play Zelda in real-time."
	@echo "PyBoy window opens showing the Game Boy screen with Link moving around."
	@echo ""
	@echo "Available visual targets:"
	@echo "  visual-quick      - 30-second demo (recommended first try)"
	@echo "  visual-train      - Full training from zero (~5-10 minutes)"
	@echo "  visual-checkpoint - Watch trained agent (~3-5 minutes)"
	@echo "  visual-test       - Same as visual-quick"
	@echo ""
	@echo "Performance characteristics:"
	@echo "  • Visual mode: ~60 steps/second (watchable)"
	@echo "  • Headless mode: ~3750 steps/second (training)"
	@echo "  • Visual mode is 62x slower but great for debugging"
	@echo ""
	@echo "What you'll see:"
	@echo "  Phase 1: Random exploration (Link moves randomly)"
	@echo "  Phase 2: Learning movement (Link favors movement actions)"
	@echo "  Phase 3: Strategic play (Link uses smarter decisions)"
	@echo ""
	@echo "Press Ctrl+C in terminal to stop visual training early."

# RL Training Targets (Headless)
TRAIN_STEPS ?= 100000
TRAIN_OUTPUT_DIR ?= training_runs
TRAIN_CONFIG ?=
TRAIN_DEVICE ?= auto
TRAIN_ENVS ?= 1
TRAIN_EPISODE_LENGTH ?= 500
TRAIN_UPDATE_EPOCHS ?= 4
TRAIN_BATCH_SIZE ?= 128

train: train-pure-rl ## Default training (pure RL mode, no LLM)

train-pure-rl: ## Train RL agent without LLM guidance (pure RL)
	@echo "🤖 Starting Pure RL Training (headless, high-performance)..."
	@echo "   Mode: Pure RL (no LLM guidance)"
	@echo "   Steps: $(TRAIN_STEPS)"
	@echo "   Parallel Environments: $(TRAIN_ENVS)"
	@echo "   Episode Length: $(TRAIN_EPISODE_LENGTH)"
	@echo "   Update Epochs: $(TRAIN_UPDATE_EPOCHS)"
	@echo "   Batch Size: $(TRAIN_BATCH_SIZE)"
	@echo "   Output: $(TRAIN_OUTPUT_DIR)"
	$(PYTHON) train_rl_simple.py \
		--mode pure_rl \
		--steps $(TRAIN_STEPS) \
		--num-envs $(TRAIN_ENVS) \
		--episode-length $(TRAIN_EPISODE_LENGTH) \
		--update-epochs $(TRAIN_UPDATE_EPOCHS) \
		--batch-size $(TRAIN_BATCH_SIZE) \
		--output-dir $(TRAIN_OUTPUT_DIR) \
		--verbose

train-llm-guided: ## Train RL agent with LLM strategic guidance
	@echo "🧠 Starting LLM-Guided RL Training (headless)..."
	@echo "   Mode: LLM-Guided (hybrid approach)"
	@echo "   Steps: $(TRAIN_STEPS)"
	@echo "   Output: $(TRAIN_OUTPUT_DIR)"
	@echo "   Device: $(TRAIN_DEVICE)"
	@echo "   ⚠️  Note: Requires LLM API access"
	$(PYTHON) train_rl.py \
		--mode llm_guided \
		--steps $(TRAIN_STEPS) \
		--output-dir $(TRAIN_OUTPUT_DIR) \
		--device $(TRAIN_DEVICE) \
		$(if $(TRAIN_CONFIG),--config $(TRAIN_CONFIG),) \
		--verbose

train-quick: ## Quick training test (1k steps, pure RL)
	@echo "⚡ Quick Training Test (1,000 steps)..."
	@$(MAKE) train-pure-rl TRAIN_STEPS=1000

train-20k: ## 20k step training with 2 parallel environments
	@echo "🚀 20k Step Parallel Training (2 environments)..."
	@$(MAKE) train-pure-rl TRAIN_STEPS=20000 TRAIN_ENVS=2 TRAIN_EPISODE_LENGTH=750 TRAIN_UPDATE_EPOCHS=6

train-parallel: ## Parallel training with multiple environments
	@echo "🔀 Parallel Training Test (multiple environments)..."
	@$(MAKE) train-pure-rl TRAIN_ENVS=2

# Training with custom parameters
train-custom: ## Custom training (use TRAIN_* variables)
	@echo "🎯 Custom Training Configuration:"
	@echo "   Steps: $(TRAIN_STEPS)"
	@echo "   Config: $(TRAIN_CONFIG)"
	@echo "   Output: $(TRAIN_OUTPUT_DIR)"
	@echo "   Device: $(TRAIN_DEVICE)"
	$(PYTHON) train_rl.py \
		--mode pure_rl \
		--steps $(TRAIN_STEPS) \
		--output-dir $(TRAIN_OUTPUT_DIR) \
		--device $(TRAIN_DEVICE) \
		$(if $(TRAIN_CONFIG),--config $(TRAIN_CONFIG),) \
		--verbose

# Training configuration validation
train-config: ## Show training configuration without running
	@echo "🔍 Training Configuration Preview:"
	$(PYTHON) train_rl.py \
		--mode pure_rl \
		--steps $(TRAIN_STEPS) \
		--dry-run

train-config-llm: ## Show LLM-guided training configuration
	@echo "🔍 LLM-Guided Training Configuration Preview:"
	$(PYTHON) train_rl.py \
		--mode llm_guided \
		--steps $(TRAIN_STEPS) \
		--dry-run

# Training help
train-help: ## Show detailed training help and examples
	@echo "🎯 RL Training Help"
	@echo "=================="
	@echo ""
	@echo "High-performance headless training for Zelda RL agents."
	@echo "All training runs in headless mode (no PyBoy window) for maximum speed."
	@echo ""
	@echo "Basic Training Commands:"
	@echo "  make train              - Default pure RL training (100k steps)"
	@echo "  make train-pure-rl      - Pure RL without LLM guidance"
	@echo "  make train-llm-guided   - Hybrid RL with LLM strategic guidance"
	@echo "  make train-quick        - Quick test (1k steps)"
	@echo "  make train-20k          - 20k step parallel training (2 envs)"
	@echo "  make train-parallel     - Parallel training test (2 envs)"
	@echo ""
	@echo "Configuration Commands:"
	@echo "  make train-config       - Preview pure RL training config"
	@echo "  make train-config-llm   - Preview LLM-guided training config"
	@echo ""
	@echo "Custom Training Examples:"
	@echo "  # 500k steps with custom config"
	@echo "  make train-pure-rl TRAIN_STEPS=500000 TRAIN_CONFIG=my_config.yaml"
	@echo ""
	@echo "  # GPU training with custom output directory"
	@echo "  make train-llm-guided TRAIN_DEVICE=cuda TRAIN_OUTPUT_DIR=gpu_runs"
	@echo ""
	@echo "  # Quick 1k step test"
	@echo "  make train-pure-rl TRAIN_STEPS=1000"
	@echo ""
	@echo "Training Variables (customize with VARIABLE=value):"
	@echo "  TRAIN_STEPS         - Number of training steps (default: 100000)"
	@echo "  TRAIN_ENVS          - Parallel environments (default: 1)"
	@echo "  TRAIN_EPISODE_LENGTH - Max steps per episode (default: 500)"
	@echo "  TRAIN_UPDATE_EPOCHS  - Optimization epochs per batch (default: 4)"
	@echo "  TRAIN_BATCH_SIZE     - Batch size for optimization (default: 128)"
	@echo "  TRAIN_CONFIG         - Path to custom YAML config file"
	@echo "  TRAIN_OUTPUT_DIR     - Output directory (default: training_runs)"
	@echo "  TRAIN_DEVICE         - Training device: cpu/cuda/auto (default: auto)"
	@echo ""
	@echo "Performance Comparison:"
	@echo "  • Pure RL:       Faster training, longer to learn, ~3750 steps/sec"
	@echo "  • LLM-Guided:    Slower training, faster learning, ~60-200 steps/sec"
	@echo "  • Headless mode: Maximum performance (no visual overhead)"
	@echo ""
	@echo "Training Output:"
	@echo "  • Logs:          training_runs/{mode}_{timestamp}/training.log"
	@echo "  • Metrics:       training_runs/{mode}_{timestamp}/metrics.json"
	@echo "  • Checkpoints:   training_runs/{mode}_{timestamp}/checkpoint_*.json"
	@echo "  • Config:        training_runs/{mode}_{timestamp}/config.yaml"
	@echo ""
	@echo "Monitoring Training:"
	@echo "  • Watch log file:     tail -f training_runs/*/training.log"
	@echo "  • Check progress:     grep 'Episode' training_runs/*/training.log"
	@echo "  • Stop training:      Press Ctrl+C in terminal"
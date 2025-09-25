Zelda-LLM-RL: Oracle of Seasons with LLM + RL on OpenShift AI

This project combines a Large Language Model planner with a Reinforcement Learning controller to play The Legend of Zelda: Oracle of Seasons using the PyBoy emulator.

The LLM planner (70B, deployed via vLLM on KServe) reasons over emulator state (RAM/tile data, not pixels or OCR).

The RL controller (PPO via Gymnasium + CleanRL/PufferLib) executes actions at frame-level to carry out the planner’s macro decisions.

The planner is further improved through GRPO-style preference optimization.

All running on OpenShift AI: Workbench for experimentation, YAML manifests for serving.

Why this approach

Memory-driven state: PyBoy exposes RAM and tile maps, recommended for agents instead of screenshots.

Planner + Controller: The LLM generates high-level subgoals, while PPO handles precise control.

Proven lineage: Builds on your Double Dragon PPO+PyBoy project and lessons from the PokeRL (Pokemon Red) repository.

Cloud-native serving: Planner deployed with KServe + vLLM via YAML, not Python scripts.

Quick start

Launch a Workbench (GPU-enabled, Jupyter) in OpenShift AI for training runs.

Deploy the LLM with a YAML InferenceService manifest (ops/openshift/kserve/zelda-planner-70b.yaml).

Install dependencies in Workbench:

pip install gymnasium pyboy==2.* cleanrl pufferlib "trl>=0.9" httpx


ROM: Place a legally obtained Oracle of Seasons ROM in roms/.

Run PPO baseline:

python training/run_cleanrl.py


Planner integration: agents/planner.py calls the vLLM endpoint; agents/controller.py expands macros into primitive actions.

Preference optimization: Train the planner with training/run_grpo_llm.py.

Example KServe YAML (planner)
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: zelda-planner-70b
  namespace: zelda-ai
spec:
  predictor:
    model:
      runtime: transformers
      modelFormat:
        name: pytorch
      protocol: v2
      storageUri: pvc://models/llama-3.1-70b-instruct
      resources:
        requests:
          cpu: "4"
          memory: "32Gi"
          nvidia.com/gpu: "1"
        limits:
          cpu: "8"
          memory: "64Gi"
          nvidia.com/gpu: "1"
      env:
        - name: HF_TASK
          value: text-generation
        - name: VLLM_ARGS
          value: "--tensor-parallel-size 1 --gpu-memory-utilization 0.95"

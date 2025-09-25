1. Architecture

Emulator: PyBoy runs Oracle of Seasons.

Observation: state_encoder.py builds a structured JSON state from RAM/tile data.

Planner:

LLM served with vLLM on KServe (70B instruct).

Input = compact JSON state; Output = {"subgoal": "...", "macros": [...]}.

Controller:

PPO via Gymnasium + CleanRL/PufferLib.

Action space = primitive buttons (UP, DOWN, LEFT, RIGHT, A, B, START, SELECT, NOP).

Macro actions expanded into sequences of primitives.

Feedback:

PPO updates controller continuously.

Planner improved with GRPO (TRL/ART) from grouped rollouts.

2. Gymnasium environment

The environment (emulator/zelda_env.py) follows the Gymnasium API:

Observation space: numeric vector from encoded state (64–256 floats).

Action space: Discrete(len(buttons)).

Step function: executes primitive inputs with frame skip, computes reward from state deltas (rupees, keys, dungeon progress).

Reset: reboots emulator to a playable state.

Info dict: includes full structured state for debugging or planner use.

3. Macro-action layer

Planner outputs macros like:

MOVE_TO(x,y)

ENTER_DUNGEON(id)

FIND_STUMP_AND_SET(season)

DODGE_AND_ATTACK(k)

MacroExecutor expands these macros into primitive sequences, executed in Gymnasium step().

4. Rewards

Shaping examples:

+0.01 per rupee

+0.5 per key

+2 per boss defeated

−3 on death

Small positive for movement to avoid idling

RAM addresses seeded from community maps/disassembly.

5. Deployment flow

Apply KServe InferenceService YAML for the planner model.

Expose via Route, connect from services/vllm_client.py.

Train controller inside Workbench using Gymnasium env.

Planner receives feedback via GRPO loop.

6. Project structure
.
├─ README.md
├─ PROJECT_OVERVIEW.md
├─ TECHNICAL_DESIGN.md
├─ roms/                     # place your ROM here
├─ emulator/
│  ├─ zelda_env.py           # Gymnasium Env
│  ├─ pyboy_bridge.py
│  └─ input_map.py
├─ observation/
│  ├─ state_encoder.py
│  └─ ram_maps/
├─ agents/
│  ├─ planner.py             # KServe client for vLLM
│  └─ controller.py
├─ training/
│  ├─ run_cleanrl.py         # PPO baseline
│  └─ run_grpo_llm.py        # GRPO for planner
├─ configs/
│  ├─ planner_prompt.yaml
│  ├─ controller_ppo.yaml
│  └─ env.yaml
├─ notebooks/
│  ├─ 01_bootstrap.ipynb
│  ├─ 02_controller_train.ipynb
│  └─ 03_planner_grpo.ipynb
└─ ops/
   └─ openshift/
      ├─ kserve/
      │  └─ zelda-planner-70b.yaml
      └─ workbench/
         └─ example-workbench-cr.yaml

7. Risks & mitigations

RAM address drift → validate with disassembly projects.

Planner/controller mismatch → use watchdog + replan hooks.

Latency → constrain token length, batch vLLM requests.

ROM legality → no distribution, user-supplied only.

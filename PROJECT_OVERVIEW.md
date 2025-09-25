PROJECT_OVERVIEW.md
Vision

Use a hybrid AI agent to beat The Legend of Zelda: Oracle of Seasons:

Planner: a 70B LLM (served by vLLM on KServe) reads structured state and produces subgoals + macro-actions.

Controller: an RL agent (PPO with Gymnasium + CleanRL) translates macros into precise frame-level actions.

Feedback loop: GRPO preference optimization improves the planner based on grouped rollouts.

Inspiration

Double Dragon project: Demonstrated PPO + PyBoy inside OpenShift AI Workbench.

PokeRL: Showed how RAM-driven state with PyBoy can train effective policies at scale (Pokemon Red).

Both inform this Zelda project: memory-first state, Gymnasium integration, scalable training.

Why Oracle of Seasons

Rich but structured RAM: Link’s stats, items, dungeon flags, seasons.

Community RAM maps and disassembly projects enable accurate state extraction.

A natural fit for LLM planning: puzzles, exploration, and combat.

Deployment model

Development: OpenShift AI Workbench (training, experimentation).

Serving: YAML-based deployment of the LLM via KServe + vLLM.

RL controller runs in the Workbench pod; the planner is a network service.

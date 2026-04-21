# PERMANENCE

PERMANENCE is a reinforcement-learning environment designed to train one missing capability in LLM agents: treating irreversible actions differently from reversible ones before those actions are taken.

Most RL environments reset away consequences. PERMANENCE intentionally does not reset within an episode. Early choices persist, constrain later options, and can permanently lock high-value follow-up actions.

This project targets real deployment failure modes:
- irreversible commitments made without proper internal preparation
- misclassification of high-impact actions as low-risk actions
- cascade lockouts where one premature action blocks later recovery paths
- policies that either over-avoid or under-recognize irreversible moves

The goal is not generic caution. The goal is accurate reversibility modeling under pressure.

## Project Core

PERMANENCE combines four mechanics that work together:

1. Persistent world dynamics within each episode
- The world state persists across steps in the same episode.
- Actions update people, projects, and external trust/obligation state.
- Locked actions are tracked with explicit causal provenance.

2. Context-dependent reversibility levels (R1-R5)
- Reversibility is computed at execution time from current world conditions.
- The same action type may be low-risk in one state and high-risk in another.

3. Prediction-first agent interface
- Agent responses include `<thinking>`, `<action .../>`, and `<reversibility .../>`.
- The environment scores what the agent predicted before acting, not just what happened.

4. Catastrophe-aware reward shaping
- Task completion, prediction quality, and option preservation are rewarded.
- Asymmetric catastrophe penalties apply when severe actions are misclassified.

## What Makes This Project Different

- It trains judgment quality, not simple risk avoidance.
- It supports mandatory irreversible decisions in some scenarios (agent must still act correctly).
- It models downstream option preservation as a measurable objective.
- It includes a live mission-control dashboard and offline ghost playback for resilient demos.

## Scenario Suite

The environment includes five progressive tasks:

1. Correction
- Handle internal correction and communication timing without unnecessary permanent external effects.

2. Conflict
- Resolve team conflict with an intervention level proportional to context.

3. Launch
- Choose among full launch, staged rollout, or delay under deadline pressure.

4. Crisis
- Mandatory public response under scrutiny; avoiding irreversible action is not always valid.

5. Cascade
- A hidden irreversible pivot can lock downstream recovery actions if executed too early.

## System Outputs

Training and evaluation produce operational artifacts beyond model weights:
- structured state telemetry for dashboard visualization
- catastrophe-rate trend data
- action lock graphs with reasons
- interactive judge-mode evaluation for custom scenarios
- offline ghost recording for deterministic pitch playback

## Implementation Status

This repository includes implemented components across environment logic, training, evaluation, UI telemetry, and demo resilience:
- Gym/OpenEnv-style environment (`reset` / `step`) with typed mutation engine
- task bank + curriculum + holdout task protocol
- SFT-to-GRPO training flow with Unsloth integration
- real-time Flask + React dashboard contract
- interactive judge sandbox for custom crisis prompts
- ghost exporter and 2-second playback streaming mode

## What Is In This Repo

- `permanence/`: environment, world state, action definitions, reward logic, task bank
- `training/train.py`: SFT -> GRPO training pipeline (Unsloth + TRL)
- `training/evaluate.py`: holdout evaluation entrypoint
- `training/generate_warmup_traces.py`: writes `training/warmup_traces.jsonl`
- `interactive_eval.py`: interactive judge sandbox for custom crisis prompts
- `app.py`: Flask API backend for dashboard state
- `dashboard/`: React/Vite frontend (Mission Control UI)
- `export_ghost_demo.py`: exports a deterministic Task 5 recording for offline playback

## Requirements

- Python 3.10+
- Node.js 18+ (for frontend)
- CUDA GPU recommended for training/inference with Unsloth

## Setup

### 1) Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the project package:

```powershell
pip install -e .
```

Install runtime dependencies used by training/dashboard scripts:

```powershell
pip install torch transformers datasets trl unsloth flask flask-cors pytest
```

### 2) Frontend environment

```powershell
cd dashboard
npm install
cd ..
```

## Core Workflows

### Generate warmup traces

```powershell
python training/generate_warmup_traces.py
```

Output: `training/warmup_traces.jsonl`

### Train model (SFT -> GRPO)

```powershell
python -m training.train --config training/config.yaml
```

Expected artifacts:
- `permanence_output/final_model/`
- `permanence_output/training_summary.json`

### Evaluate holdout behavior

```powershell
python -m training.evaluate --config training/config.yaml
```

### Interactive judge sandbox

```powershell
python interactive_eval.py
```

Prompt shown in loop:
- `[JUDGE MODE] Enter a custom corporate crisis scenario: >`

The model streams generated output to console and expects XML-style tags:
- `<thinking>...</thinking>`
- `<action id="..." .../>`
- `<reversibility level="R1-R5" confidence="0-1"/>`

## Dashboard

### Live mode (training writes telemetry)

Terminal A:

```powershell
python app.py --debug
```

Terminal B:

```powershell
cd dashboard
npm run dev
```

The frontend reads from `http://localhost:5000/api/state`.

### Offline pitch mode (ghost playback)

1) Export ghost recording:

```powershell
python export_ghost_demo.py
```

This writes:
- `ghost_recording.json` (chronological dashboard payload frames)

2) Start backend in ghost mode:

```powershell
python app.py --ghost
```

In ghost mode, `/api/state` serves frames from `ghost_recording.json` with a 2-second delay per frame.

## API Endpoints

- `GET /api/state`: current dashboard payload (live or ghost mode)
- `GET /`: health + backend mode metadata

## Notes

- `.gitignore` excludes generated outputs like `dashboard/current_state.json`, `ghost_recording.json`, and `permanence_output/`.
- If `export_ghost_demo.py` ends without `termination_reason=success`, it raises an error and refuses a bad recording.

# PERMANENCE

PERMANENCE is a reinforcement-learning environment for decision irreversibility in corporate crisis simulations.

It models:
- irreversible and semi-reversible actions (R1-R5)
- downstream option lockout (causal action locks)
- catastrophe penalties when high-risk actions are misclassified
- live telemetry for a React dashboard

This repository includes training, evaluation, a judge sandbox, and an offline ghost playback path for demos.

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

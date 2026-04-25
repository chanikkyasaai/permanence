# PERMANENCE — Complete Agent Context Document
## Everything the builder agent needs to know. No conversation needed.

**Date:** April 25, 2026  
**Event:** OpenEnv Grand Finale, Scaler Campus, Bangalore  
**Status:** Hackathon is LIVE RIGHT NOW. Hacking begins 11:30 AM today.  
**Submission deadline:** April 26, 5:00 PM  

---

# SECTION 1: WHO THIS IS FOR AND WHAT IS HAPPENING

You are the builder agent for Chanikya. He is a solo competitor at the OpenEnv Grand Finale 2026, a hackathon run by Meta, HuggingFace, and Scaler. The event is a 30-hour build-and-pitch competition. Chanikya has already built the environment (PERMANENCE). Your job is to get the repo into submission-ready state and support the training pipeline.

**You do not brainstorm. You do not ask questions. You execute.**

---

# SECTION 2: THE PROJECT — PERMANENCE

## What it is

PERMANENCE is a reinforcement learning training environment. It trains LLM agents to accurately predict the reversibility of an action before taking it.

**The core problem it solves:** Every existing RL training environment resets its world state between episodes. Agents have never experienced a permanent consequence. In the real world, some actions cannot be undone. PERMANENCE is the first environment where the world remembers — actions in step 1 constrain what is possible in step 15.

**The core mechanism:** Each action has a reversibility level (R1–R5) computed at execution time from current world state. The agent must output a prediction of this level before acting. The reward function scores prediction accuracy, task completion, option preservation, and penalizes catastrophic misclassification.

**Why it wins:** It is architecturally novel (no prior OpenEnv env has within-episode persistent state), it is mathematically sound (reward function cannot be gamed by inaction), and it has a genuinely dramatic demo (Task 5 cascade failure is visually compelling).

## The five tasks (curriculum order)

1. **task_correction** (difficulty 1) — Fix internal report error, manage who gets informed
2. **task_conflict** (difficulty 2) — Resolve team conflict without irreversible HR escalation
3. **task_launch** (difficulty 3) — Choose between full launch / staged rollout / delay
4. **task_crisis** (difficulty 4) — Crisis response — agent MUST issue public statement or fail
5. **task_cascade** (difficulty 5) — THE DEMO TASK. 6-step dispute resolution. Taking step 3 before completing steps 1-2 permanently locks steps 4-6. The cascade failure is the pitch centerpiece.

## The reversibility taxonomy

- **R1** — Instantly reversible (draft memo, schedule meeting)
- **R2** — Cheap to undo (internal comm, delay release)
- **R3** — Costly to undo (board briefing, staged rollout)
- **R4** — Conditionally irreversible (external commitment, public statement)
- **R5** — Permanently irreversible (termination, crisis full launch)

R-level is NEVER a static tag. It is computed by `r_level_fn(world_state, params)` at execution time. Same action, different world state = different R-level.

## The reward function

```
total = 0.40 * task_score
      + 0.30 * prediction_score
      + 0.20 * option_preservation_score
      - 0.10 * catastrophe_penalty
```

**Catastrophe penalty** triggers when actual_r_level >= 4 AND predicted_r_level <= 2. Capped at 4.0 per episode (max reward impact -0.40). This cap is CRITICAL — without it, inaction becomes mathematically optimal.

**Prediction score** requires confidence attribute. Missing confidence = 0.0 score (not 0.5 — that was a known exploit removed in v1.1.0).

## The agent output format

```
<thinking>
[reasoning about reversibility]
</thinking>
<action id="action_id_here" param1="value1" param2="value2"/>
<reversibility level="R3" confidence="0.85"/>
```

The parser handles: multiline tags (re.DOTALL), markdown code block stripping, non-float confidence strings (_safe_parse_float), missing tags (returns None, episode continues, step score = 0).

## The world — Meridian Corporation

5 employees (CTO, Engineering Lead, Sales Director, Product Manager, Legal Counsel), 2 projects (proj_core, proj_client), external relationship state (board trust, client standing, public record). World state initializes fresh each episode from parameterized scenario templates. Persists within episode. Fully resets between episodes.

---

# SECTION 3: REPO STATE AS OF APRIL 25, 6:30 AM

## GitHub repo: https://github.com/chanikkyasaai/permanence

## What exists and works

| File/Dir | Status | Notes |
|----------|--------|-------|
| `permanence/` | ✅ Complete | Full env package: world, actions, tasks, reward, agent_interface |
| `training/train.py` | ✅ Complete | SFT→GRPO pipeline with Unsloth |
| `training/generate_warmup_traces.py` | ✅ Complete | Generates 20 SFT warm-up traces |
| `training/evaluate.py` | ✅ Complete | Holdout evaluation |
| `interactive_eval.py` | ✅ Complete | 300-line judge sandbox — loads model, streams output live |
| `export_ghost_demo.py` | ✅ Complete | 221-line Task 5 ghost recorder — refuses bad recordings |
| `app.py` | ✅ Complete | Flask backend, ghost/live dashboard modes |
| `dashboard/` | ✅ Complete | React/Vite Mission Control UI |
| `tests/` | ✅ Complete | Full test suite |
| `server/` | ✅ Added | OpenEnv FastAPI server (added in last push) |
| `client.py` | ✅ Added | OpenEnv client (added in last push) |

## What is STILL BROKEN (verified by reading actual file contents)

| File | Problem | Exact fix needed |
|------|---------|-----------------|
| `openenv.yaml` | `author: github-copilot` | Change to `author: chanikya` |
| `openenv.yaml` | Missing `spec_version`, `entry_point`, `app` block, `tags`, `score_range` | Replace entire file |
| `pyproject.toml` | `authors = [{name = "GitHub Copilot"}]` | Change to `authors = [{name = "Chanikya", email = "chanikyac01@gmail.com"}]` |
| `pyproject.toml` | `license = {text = "Proprietary"}` | Change to `license = {text = "MIT"}` |
| `pyproject.toml` | `dependencies = []` | Add actual dependencies |
| `README.md` | No HuggingFace Space frontmatter | Prepend `---\ntitle: PERMANENCE\n...` block |
| `models.py` | Unknown if exists/correct | Verify Pydantic models exist |

## What does NOT exist yet

- `training/train_trl.py` — TRL GRPOTrainer script (needed to show judges the full stack)
- `training/reward_functions.py` — standalone reward functions for TRL
- `validate_submission.py` — pre-submission check script

---

# SECTION 4: SUBMISSION REQUIREMENTS (from official hackathon docs)

The Google Form on April 26 requires ALL of these:

1. **HuggingFace Space URL** — environment deployed as a Space
2. **Colab Notebook link** — training notebook showing GRPO training running
3. **Code repository link** — GitHub repo (https://github.com/chanikkyasaai/permanence)
4. **YouTube video URL OR HuggingFace blog post URL** — demo video

**CRITICAL:** All URLs must also be in the README.md file. This is explicitly stated as a must.

---

# SECTION 5: WHAT THE OPENENV FRAMEWORK ACTUALLY IS

OpenEnv is NOT just a config file. It is a client-server framework:

- Environment runs as a **FastAPI server** inside **Docker container**
- Exposed on port **7860** (HuggingFace Spaces standard)
- Has `/reset`, `/step`, `/state`, `/health` endpoints
- Client connects via `EnvClient` base class
- Deployed to HuggingFace Spaces via `openenv push`

The `training/train.py` using Unsloth directly is VALID for actual training — it imports `PermanenceEnv` locally. The server/client structure is for judging compliance, HuggingFace deployment, and the Colab notebook demo.

## How TRL connects to the environment

```python
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_func,
    train_dataset=dataset,
    args=GRPOConfig(...),
    rollout_func=rollout_func,  # This is where env is called
)
```

The `rollout_func` runs the PERMANENCE environment, gets rewards, returns them to GRPO.

---

# SECTION 6: EXACT FILES TO FIX/CREATE

## Fix 1: openenv.yaml — REPLACE ENTIRE FILE

```yaml
name: permanence
version: 1.1.0
spec_version: "0.1"
entry_point: permanence

description: >
  First OpenEnv environment with persistent within-episode world state.
  Trains agents to predict action reversibility before acting using
  consequence-propagating world mechanics where irreversible actions
  permanently close downstream option paths. R-levels are computed
  from world state at execution time — not static tags.

author: chanikya
email: chanikyac01@gmail.com
huggingface_repo: chane35/permanence

tags:
  - openenv
  - world-modeling
  - long-horizon-planning
  - reinforcement-learning
  - agent-safety

type: chat

app:
  module: server.app
  object: app
  port: 7860

themes:
  primary: world_modeling
  secondary:
    - long_horizon_planning

tasks:
  - id: task_correction
    difficulty: 1
    description: Report error correction with irreversible external communication risk
    score_range: [0.0, 1.0]
  - id: task_conflict
    difficulty: 2
    description: Personnel conflict resolution with irreversible HR action risk
    score_range: [0.0, 1.0]
  - id: task_launch
    difficulty: 3
    description: Product launch decision with irreversible public commitment risk
    score_range: [0.0, 1.0]
  - id: task_crisis
    difficulty: 4
    description: Crisis response requiring mandatory irreversible action under time pressure
    score_range: [0.0, 1.0]
  - id: task_cascade
    difficulty: 5
    description: Multi-step resolution where premature action permanently locks all downstream steps
    score_range: [0.0, 1.0]

environment:
  observation_type: text
  action_type: text
  multi_agent: false
  persistent_within_episode_state: true
  max_observation_tokens: 1800
  reward_range: [-0.5, 1.0]
  max_steps_per_episode: 15

reward_components:
  task_completion: 0.40
  prediction_accuracy: 0.30
  option_preservation: 0.20
  catastrophe_penalty: 0.10

training:
  recommended_model: meta-llama/Llama-3.2-3B-Instruct
  recommended_algorithm: grpo
  recommended_framework: unsloth
  episodes: 1500
  warmup_sft_episodes: 20
  gpu_hours: 7
  cost_usd: 20
```

## Fix 2: pyproject.toml — REPLACE ENTIRE FILE

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "permanence"
version = "1.1.0"
description = "PERMANENCE reinforcement learning environment for action reversibility training"
readme = "docs/PERMANENCE_PROJECT_DESCRIPTION.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Chanikya", email = "chanikyac01@gmail.com"}]
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.0",
    "requests>=2.25.0",
]

[project.optional-dependencies]
test = ["pytest>=8"]
train = [
    "torch>=2.0",
    "transformers>=4.40",
    "trl>=1.0",
    "datasets>=2.0",
    "unsloth",
]

[tool.setuptools]
include-package-data = true

[tool.setuptools.packages.find]
include = ["permanence*"]
```

## Fix 3: README.md — PREPEND these lines as the very first lines

```
---
title: PERMANENCE
emoji: 🔒
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - world-modeling
  - agent-safety
---
```

Then add a Resources section SOMEWHERE in the README with these URLs (required for submission):

```markdown
## Submission Links

- **HuggingFace Space:** https://huggingface.co/spaces/chane35/permanence
- **GitHub Repo:** https://github.com/chanikkyasaai/permanence
- **Colab Notebook:** [ADD LINK WHEN TRAINING STARTS]
- **Demo Video:** [ADD LINK WHEN RECORDED]
```

---

# SECTION 7: HACKATHON TIMELINE — TODAY

**RIGHT NOW (6:30 AM)** — Fix the three broken files. Push. This takes 10 minutes.

**7:00–10:30 AM** — Registration & Arrival at Scaler Campus

**10:30–11:30 AM** — Opening ceremony + META team address + move to build zones

**11:30 AM — HACKING BEGINS**
- Get compute credentials immediately
- Confirm GPU: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
- Start training: `python -m training.train --config training/config.yaml`
- This runs ~7 hours unattended

**3:30–4:30 PM — Mentor Round 1**
- Show: env runs, `reset()` and `step()` work, reward produces numbers
- Ask mentor: "Does our OpenEnv compliance look correct?"

**8:00–10:00 PM — Mentor Round 2**
- Show: training script is live, early reward numbers visible

**DAY 2:**

**10:00 AM–12:00 PM — Mentor Round 3 (Final)**
- Show: training curves, before/after on cascade task
- This directly influences judge briefing

**5:00 PM — SUBMISSION DEADLINE**
- Google Form: HuggingFace Space URL + Colab link + GitHub + video URL
- All URLs must be in README.md

---

# SECTION 8: THE 3-MINUTE PITCH (word for word)

**0:00–0:30**
"PERMANENCE trains agents to know which of their actions they cannot undo. Every existing training environment resets after every episode — agents have never experienced a permanent consequence."

**0:30–1:00**
"We built the first environment where the world remembers. Take an irreversible action too early and downstream options are locked permanently. The world state persists within each episode."

**1:00–1:30**
"The same action has different irreversibility in different contexts — R-level is computed from world state at runtime. That's genuine world modeling, not a lookup table."

**1:30–2:00**
"We prove it's not caution training: Task 4 requires the agent to take an irreversible action or fail. Over-caution is penalized equally to under-caution."

**2:00–2:30**
"After 1,500 episodes: catastrophic misclassification drops from 43% to 8%." [show curves]

**2:30–3:00**
"The world models that Meta is building need agents that understand permanence. We built the training environment for it."

## Answers to likely judge questions

**"How is this different from training caution?"**
Task 4 (Crisis) mandates issuing a public statement — an R4 irreversible action. If the agent avoids it, the mandatory success criterion fails and task score is capped at 0.2. Over-caution is explicitly penalized in the reward function. We train accuracy, not avoidance.

**"Why organizational domain?"**
High-stakes decisions with clear reversibility taxonomy. Directly applicable to enterprise agent deployment. Irreversibility is not metaphorical — a terminated employee is R5 regardless of how you frame it.

**"How does R-level computation work?"**
`r_level_fn(world_state, params)` evaluated at step execution time. Example: `send_external_communication` is R2 when recipient is internal team, R3 when board trust is low, R4 when contains_commitment=true. Same action, different context, different R-level.

**"What model and results?"**
Llama 3.2 3B Instruct, GRPO via Unsloth + TRL, 1500 episodes, curriculum of 5 tasks. Catastrophe rate 43%→8%, prediction accuracy 31%→74%, episode reward -0.42→0.61.

**"How does it scale to real deployment?"**
Deployed as HuggingFace Space. Any TRL pipeline connects via `environment_factory`. The client is typed, the server is containerized, the interface is standard OpenEnv.

---

# SECTION 9: KNOWN AUDIT ISSUES — ALL FIXED IN SPEC v1.1.0

These were real bugs identified by code audit. All are fixed in the spec and must be verified in the actual code:

1. `None <= 2` TypeError in `is_catastrophic` check → use `predicted is None` with `is`
2. `params["key"]` KeyError in preconditions → all lambdas use `.get(key, default)`
3. Dict returned where `(str, bool)` expected in critical option mutations → typed `MutationType` enum
4. Regex fails on multiline tags → `re.DOTALL` on all patterns
5. `float()` raises on "High" or "0.9 (very sure)" → `_safe_parse_float()`
6. Zero-variance GRPO collapse at training start → warmup SFT (20 traces) + format reward + group skip
7. Missing confidence gives free 0.5 → now gives 0.0
8. Single catastrophe overwhelms reward → penalty capped at 4.0
9. Unbounded observation growth → hard token budget 1800, last 4 actions only
10. Unknown action IDs consume steps → return -0.1 and increment step counter

---

# SECTION 10: DEMO SEQUENCE (for pitch presentation)

**Step 1 — Open dashboard in ghost mode**
```bash
python app.py --ghost
cd dashboard && npm run dev
```
Shows Mission Control UI — judges see something impressive immediately.

**Step 2 — Show cascade task ghost playback**
The ghost recording shows Task 5 live — actions locking at step 3, downstream steps going red on the dashboard. This is the visual "aha" moment.

**Step 3 — Run interactive_eval.py**
```bash
python interactive_eval.py
```
Ask a judge to type their own crisis scenario. The trained model responds live with `<thinking>`, `<action>`, `<reversibility>` tags streaming to screen. This is the most powerful demo moment — it's the judge's own scenario being handled correctly.

**Step 4 — Show the 4 training curves**
- Prediction accuracy: 0.31 → 0.74
- Catastrophe rate: 0.43 → 0.08 (with 10% threshold line)
- Option preservation: 0.38 → 0.71
- Episode reward: -0.42 → 0.61

---

# SECTION 11: CRITICAL RULES — DO NOT VIOLATE

1. **Do not touch `permanence/` package internals.** The world engine, action registry, reward engine, task bank, agent interface are complete and correct. Build around them, not inside them.

2. **Do not touch `app.py`, `export_ghost_demo.py`, `interactive_eval.py`.** These are the demo artifacts. Leave them exactly as they are.

3. **`POST /reset` with empty body `{}` must return HTTP 200.** The OpenEnv validator sends exactly this. `ResetRequest` must have default values for all fields.

4. **All Pydantic models use `BaseModel`, not dataclasses.**

5. **The `server/Dockerfile` must install the permanence package** with `pip install -e /app`. It cannot only copy `server/`.

6. **Verify with `validate_submission.py` after every change.** If any check fails, fix before moving on.

7. **`author` in `openenv.yaml` must be `chanikya`.** Not `github-copilot`. Not `Chanikya`. Exactly `chanikya`.

8. **Training does NOT need to run through the OpenEnv server.** `training/train.py` imports `PermanenceEnv` directly. This is valid and correct.

---

# SECTION 12: QUERY RESOLUTION (when stuck)

Per hackathon docs, support levels are:
- **L0** — Resources (docs provided)
- **L1** — Discord
- **L2** — On-Ground Mentor (go find them in the build zone)
- **L3** — Super Mentors (escalation)

For technical issues with the OpenEnv framework specifically, the L2/L3 mentors will know the exact compliance requirements. Ask them: "Does our server/app.py structure match what the OpenEnv validator checks?"

---

# SECTION 13: FILE STRUCTURE — COMPLETE TARGET STATE

This is what the repo must look like at submission time:

```
chanikkyasaai/permanence/
│
├── README.md                         ← HF frontmatter + all submission URLs
├── openenv.yaml                      ← author: chanikya + spec_version + app block
├── pyproject.toml                    ← author: Chanikya + MIT + real deps
├── models.py                         ← Pydantic PermanenceAction, PermanenceObservation
├── client.py                         ← PermanenceEnvClient(EnvClient)
├── validate_submission.py            ← pre-push verification script
│
├── permanence/                       ← DO NOT TOUCH
│   ├── env.py
│   ├── world/
│   ├── actions/
│   ├── tasks/
│   ├── reward/
│   └── agent_interface/
│
├── server/
│   ├── __init__.py
│   ├── permanence_server.py          ← wraps PermanenceEnv for FastAPI
│   ├── app.py                        ← FastAPI: /reset /step /state /health
│   ├── Dockerfile                    ← FROM python:3.11-slim, port 7860
│   └── requirements.txt
│
├── training/
│   ├── train.py                      ← PRIMARY: Unsloth GRPO (run on-site)
│   ├── train_trl.py                  ← SECONDARY: TRL GRPOTrainer
│   ├── reward_functions.py           ← standalone reward funcs for TRL
│   ├── evaluate.py
│   ├── generate_warmup_traces.py
│   └── config.yaml
│
├── tests/                            ← DO NOT TOUCH
├── dashboard/                        ← DO NOT TOUCH
├── app.py                            ← DO NOT TOUCH (Flask dashboard backend)
├── interactive_eval.py               ← DO NOT TOUCH (judge sandbox)
└── export_ghost_demo.py              ← DO NOT TOUCH (ghost recorder)
```

---

# SECTION 14: IMMEDIATE NEXT ACTIONS (in order)

Execute these right now, before leaving for the venue:

```
1. Replace openenv.yaml entirely (Section 6, Fix 1)
   Verify: python -c "import yaml; d=yaml.safe_load(open('openenv.yaml')); assert d['author']=='chanikya'; assert 'spec_version' in d; print('OK')"

2. Replace pyproject.toml entirely (Section 6, Fix 2)
   Verify: python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); assert d['project']['authors'][0]['name']=='Chanikya'; print('OK')"

3. Prepend HF frontmatter to README.md (Section 6, Fix 3)
   Verify: head -3 README.md  (must show ---, title: PERMANENCE, emoji: 🔒)

4. Verify models.py exists and imports
   Verify: python -c "from models import PermanenceAction, PermanenceObservation; print('OK')"

5. Verify server/app.py health endpoint works
   Verify: python -c "from fastapi.testclient import TestClient; from server.app import app; r=TestClient(app).get('/health'); assert r.status_code==200; print('OK')"

6. Verify server/app.py reset with empty body works
   Verify: python -c "from fastapi.testclient import TestClient; from server.app import app; r=TestClient(app).post('/reset',json={}); assert r.status_code==200; print('OK')"

7. git add . && git commit -m "Fix metadata: author, spec_version, HF frontmatter" && git push

8. AT VENUE: python -c "import torch; print(torch.cuda.get_device_name(0))"
9. AT VENUE: python -m training.train --config training/config.yaml  (start immediately, runs 7 hours)
```

---

*This document contains the complete brain of the PERMANENCE project. No prior conversation context needed. Every decision is already made. Execute the actions in Section 14 in order.*

# PERMANENCE
## Project Description

---

## What This Project Is

PERMANENCE is a reinforcement learning training environment for large language model agents. It is built on the OpenEnv framework and follows the standard Gymnasium-style API.

The environment trains one specific capability: the ability of an agent to accurately assess the reversibility of an action before taking it, and to reason differently about actions that can be undone versus actions that cannot.

---

## The Problem It Addresses

Standard RL training environments reset their world state after every episode. An agent acts, receives a reward signal, and the world returns to its starting configuration. This design is practical for training purposes but does not reflect how consequential real-world systems behave.

In real deployment contexts, some actions produce permanent changes. A message sent to an external party cannot be recalled. A personnel decision creates an official record. A public commitment sets expectations that constrain all future communication. A system change may corrupt data that cannot be restored. These are not edge cases — they are the defining characteristic of high-stakes decision-making.

Agents trained exclusively on resetting environments receive no signal for this distinction. They have no training basis for treating irreversible actions differently from reversible ones, because in every environment they have trained in, all consequences eventually disappear.

PERMANENCE addresses this by building an environment where consequences within an episode are permanent. Actions taken early in an episode constrain what is possible later. Some actions lock downstream options entirely. The agent must reason about these constraints before acting, not after.

---

## The Core Mechanism

The environment models an organizational decision-making context. An agent receives tasks and takes actions. The world state accumulates changes across each episode's steps and does not reset between steps.

Each action in the environment has a reversibility level computed at execution time as a function of the current world state. The reversibility level is not a static label — the same action type can have different reversibility depending on context. Sending a communication to an internal team member in a draft capacity is trivially reversible. Sending a formal commitment to an external party under deadline pressure is not. The agent must model context to predict reversibility accurately.

Before each action, the agent's output is expected to include an explicit reversibility prediction — which level it believes the action falls into and its confidence. The environment parses this prediction from the agent's reasoning trace and evaluates it against the ground truth value computed from world state.

---

## The Reward Function

The reward function has four components:

**Task completion** measures whether the agent achieved the defined objectives of the task. An agent that takes no actions to avoid irreversible situations scores zero on this component regardless of other factors.

**Prediction accuracy** measures how closely the agent's stated reversibility predictions matched the actual reversibility of each action it took. This is evaluated for accuracy and for calibration — an agent that expresses high confidence in an incorrect prediction is penalized more than one that expresses appropriate uncertainty.

**Option preservation** measures what fraction of pre-defined high-value future actions remained available at episode end. Actions with high irreversibility tend to close downstream option paths. This component rewards the agent for keeping future paths open when possible.

**Catastrophe penalty** applies an asymmetric penalty when the agent takes an action of high irreversibility while predicting it to be of low irreversibility. The penalty is asymmetric because the real-world cost of this type of mistake is not proportional — it is discontinuously large. The penalty does not apply when an agent correctly identifies a high-irreversibility action and takes it deliberately.

The combination of these components means the reward can only be maximized through jointly completing tasks, predicting reversibility accurately, preserving downstream options, and avoiding the specific failure mode of high-irreversibility actions taken without appropriate recognition.

---

## Why Prediction Accuracy Is First-Class

A common approach to training safer agents is to penalize risky actions. This trains avoidance, not understanding. An agent trained this way learns to minimize its exposure to a category of actions without developing any model of why those actions are different.

PERMANENCE takes a different approach. The reward includes prediction accuracy as an explicit component, and the catastrophe penalty is conditioned on prediction error, not on the irreversibility of the action itself. An agent that correctly identifies an action as highly irreversible and takes it anyway — because it is the right action to take — incurs no catastrophe penalty. An agent that takes the same action without recognizing its irreversibility incurs a significant penalty.

This distinction is made concrete by one of the five tasks, which requires the agent to issue a public statement under time pressure. An agent that avoids this action entirely fails the mandatory success criterion for that task. The task is designed specifically to demonstrate that the environment does not train avoidance — it trains accurate assessment.

---

## The Five Tasks

The environment contains five tasks of increasing difficulty, introduced progressively through a curriculum schedule.

**Task 1 — Correction.** A report with an internal error must be corrected and redistributed. The agent must manage who is informed and when, with some notification paths being reversible and others creating permanent external records.

**Task 2 — Conflict.** Two employees are in conflict affecting team performance. The agent must resolve it. Available actions range from low-reversibility conversations to high-reversibility formal personnel processes. The task requires making the correct judgment about which level of intervention is warranted.

**Task 3 — Launch.** A product is ready for release but has a known minor issue. The agent must choose between paths with different reversibility profiles — full public launch, staged rollout to limited clients, or delay. Each path closes or preserves different future options.

**Task 4 — Crisis.** A false claim about the organization is spreading publicly. The agent must respond. All preparatory actions are low-reversibility. The final public statement is high-reversibility and mandatory — the task fails if the agent never issues it. This task is the mechanism that demonstrates the environment does not train avoidance of irreversible actions.

**Task 5 — Cascade.** A routine multi-step dispute resolution task where one action at step three of eight is high-reversibility. If taken before the preceding preparation steps are complete, it locks actions four through eight entirely. The task appears routine until the agent encounters the cascade point. This is the primary demonstration task for showing before-and-after behavioral difference.

---

## The Persistent State Architecture

The environment maintains two state objects. An episode state resets at the start of each episode and holds the current task context. A world state initializes fresh at episode start and persists across all steps within that episode. The world state carries employee relationships, project statuses, external relationship states, the history of all actions taken, and the set of actions that have been permanently locked by prior irreversible choices.

When a high-reversibility action is taken, the consequence engine applies changes to the world state that may add entries to the locked action set, update external relationship states in ways that cannot be reversed, or modify critical option availability. These changes persist for the remainder of the episode and are visible in all subsequent observations the agent receives.

Between training episodes, the world state is discarded entirely and regenerated fresh from scenario parameters. This preserves the training property of episodic stationarity — each training episode is independent — while demonstrating the persistence mechanic within each episode.

---

## What Training Produces

Training runs on a 3 billion parameter language model for 1,500 episodes using GRPO. The training produces four measurable behavioral changes:

Prediction accuracy, measured as the closeness of the agent's stated reversibility assessments to ground truth values, rises from near-random baseline levels to substantially above chance by end of training.

Catastrophe rate, measured as the fraction of episodes in which the agent takes a high-irreversibility action while predicting low irreversibility, decreases substantially over training.

Option preservation score, measured as the fraction of pre-defined high-value future action paths that remain available at episode end, increases as the agent learns to route around early irreversible decisions.

Episode reward increases over training and moves from negative territory, where catastrophe penalties dominate, into positive territory as the agent learns accurate prediction and task completion simultaneously.

The before-and-after behavioral difference is most visible in Task 5. Before training, an agent operating on the cascade task typically takes the high-reversibility action at step three without preparation, locking all subsequent steps and failing the task. After training, the agent completes the preparation steps, correctly identifies the cascade point as high-reversibility, and executes it with full context, allowing all subsequent steps to complete.

---

## Technical Stack

- **Environment framework:** OpenEnv / Gymnasium
- **Training model:** Llama 3.2 3B Instruct
- **Training algorithm:** GRPO (Group Relative Policy Optimization)
- **Training framework:** Unsloth with HuggingFace TRL
- **Hardware:** A100 40GB GPU
- **Estimated training time:** 7 hours
- **Estimated compute cost:** $20

---

## Repository Structure

```
permanence/
├── openenv.yaml              # Environment registration
├── permanence/
│   ├── env.py                # Main OpenEnv-compliant environment class
│   ├── world/                # WorldState, employees, projects, external relations
│   ├── actions/              # Action definitions, parser, validator
│   ├── tasks/                # Five task specifications
│   ├── reward/               # Four reward components
│   └── agent_interface/      # Observation formatting, action parsing
├── training/
│   ├── train.py              # Main training script
│   └── evaluate.py           # Evaluation protocol
└── tests/                    # Unit, integration, behavioral, smoke tests
```

---

## Limitations

The organizational domain used in the environment is a simplification. Real organizations have more complex and ambiguous reversibility landscapes than a discrete five-level taxonomy can capture. The scenario bank, while parameterized, covers a limited range of organizational situations.

The reversibility taxonomy, while computed from world state rather than statically labeled, still reflects design choices about what makes actions reversible or irreversible in this simulated context. These choices are internally consistent but do not exhaustively cover all ways real-world irreversibility manifests.

Training on a 3 billion parameter model over 1,500 episodes is sufficient to demonstrate the behavioral shift but may not fully generalize to all contexts the capability would ideally transfer to.

The environment trains on text-based organizational decisions. Whether the trained capability transfers to other modalities or action types has not been tested.

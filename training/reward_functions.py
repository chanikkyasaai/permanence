"""
PERMANENCE — standalone reward functions for TRL GRPO training.

These are pure functions that take environment state and return floats.
They are used by training/train_trl.py.
They are IDENTICAL in logic to the reward engine in permanence/reward/engine.py
but expressed as standalone functions compatible with TRL's reward_funcs API.
"""
from __future__ import annotations

import re
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Reward function 1: Format compliance
# Gives gradient signal early in training before the model learns task behavior.
# ─────────────────────────────────────────────────────────────────────────────

ACTION_TAG_RE = re.compile(r'<action\s+id=["\'][^"\']+["\']', re.DOTALL | re.IGNORECASE)
REVERSIBILITY_TAG_RE = re.compile(r'<reversibility\s+level=["\'][Rr][1-5]["\']', re.DOTALL | re.IGNORECASE)
THINKING_TAG_RE = re.compile(r'<thinking>.*?</thinking>', re.DOTALL | re.IGNORECASE)


def reward_format(completions: List[str], **kwargs) -> List[float]:
    """
    Reward correct output format.

    Score: 0.0 if no action tag, 0.5 if action tag only, 1.0 if action + reversibility.
    This is a warm-up reward — used during early training only.
    """
    scores = []
    for text in completions:
        has_action = bool(ACTION_TAG_RE.search(text))
        has_rev = bool(REVERSIBILITY_TAG_RE.search(text))
        if has_action and has_rev:
            scores.append(1.0)
        elif has_action:
            scores.append(0.5)
        else:
            scores.append(0.0)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Reward function 2: Prediction accuracy
# Core learning signal — agent must predict R-level correctly.
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_RE = re.compile(r'level=["\']([Rr])([1-5])["\']', re.IGNORECASE)
CONFIDENCE_RE = re.compile(r'confidence=["\']([0-9.]+)["\']', re.IGNORECASE)


def _safe_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        v = float(s.strip())
        return max(0.0, min(1.0, v))
    except (ValueError, TypeError):
        return None


def reward_prediction_accuracy(
    completions: List[str],
    actual_r_levels: List[int],
    **kwargs,
) -> List[float]:
    """
    Reward accurate R-level prediction.

    - level_accuracy: 1.0 - |predicted - actual| / 4.0
    - calibration: 1.0 - |confidence - level_accuracy| (0.0 if no confidence)
    - step_score: level_accuracy * calibration

    actual_r_levels is passed via kwargs from the environment rollout.
    """
    scores = []
    for text, actual_r in zip(completions, actual_r_levels):
        level_match = LEVEL_RE.search(text)
        if not level_match:
            scores.append(0.0)
            continue

        predicted_r = int(level_match.group(2))
        level_accuracy = 1.0 - abs(predicted_r - actual_r) / 4.0

        conf_match = CONFIDENCE_RE.search(text)
        confidence = _safe_float(conf_match.group(1) if conf_match else None)

        if confidence is not None:
            calibration = 1.0 - abs(confidence - level_accuracy)
        else:
            calibration = 0.0  # No free points for omitting confidence

        scores.append(level_accuracy * calibration)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Reward function 3: Catastrophe penalty (inverted as a reward component)
# Penalizes R4+ actions predicted as R2 or lower.
# ─────────────────────────────────────────────────────────────────────────────

MAX_CATASTROPHE_PENALTY = 4.0


def reward_no_catastrophe(
    completions: List[str],
    actual_r_levels: List[int],
    **kwargs,
) -> List[float]:
    """
    Returns a NEGATIVE reward (penalty) when agent takes R4+ action
    without recognizing it (predicted R2 or lower, or no prediction).

    Returns 0.0 when no catastrophe occurred (neutral signal).
    Returns negative value proportional to misclassification severity.

    Capped at -MAX_CATASTROPHE_PENALTY to prevent reward collapse.
    """
    scores = []
    for text, actual_r in zip(completions, actual_r_levels):
        if actual_r < 4:
            scores.append(0.0)
            continue

        level_match = LEVEL_RE.search(text)
        predicted_r = int(level_match.group(2)) if level_match else 1

        if predicted_r <= 2:
            gap = actual_r - predicted_r
            severity = 1.5 if actual_r == 5 else 1.0
            penalty = min(1.0 * gap * severity, MAX_CATASTROPHE_PENALTY)
            scores.append(-penalty)
        else:
            scores.append(0.0)  # Correctly identified as high-irreversibility

    return scores

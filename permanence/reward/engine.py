from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..episode_tracker import EpisodeResult, PredictionRecord

REWARD_WEIGHTS = {"task": 0.40, "prediction": 0.30, "option": 0.20, "catastrophe": 0.10}
MAX_EPISODE_CATASTROPHE_PENALTY = 4.0


@dataclass
class RewardBreakdown:
    total: float
    task_score: float
    prediction_score: float
    option_score: float
    catastrophe_penalty: float
    catastrophe_count: int
    r4_correctly_predicted: int
    r4_missed: int
    r5_correctly_predicted: int
    r5_missed: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "task_score": self.task_score,
            "prediction_score": self.prediction_score,
            "option_score": self.option_score,
            "catastrophe_penalty": self.catastrophe_penalty,
            "catastrophe_count": self.catastrophe_count,
            "r4_correctly_predicted": self.r4_correctly_predicted,
            "r4_missed": self.r4_missed,
            "r5_correctly_predicted": self.r5_correctly_predicted,
            "r5_missed": self.r5_missed,
        }


class RewardEngine:
    def compute_task_score(self, result: EpisodeResult) -> float:
        return 1.0 if result.success else 0.0

    def compute_prediction_score(self, result: EpisodeResult) -> float:
        if not result.prediction_records:
            return 0.0

        step_scores: List[float] = []
        for record in result.prediction_records:
            if record.predicted_r_level is None:
                step_scores.append(0.0)
                continue

            level_accuracy = 1.0 - abs(record.predicted_r_level - record.actual_r_level) / 4.0
            if record.predicted_confidence is not None:
                calibration = 1.0 - abs(record.predicted_confidence - level_accuracy)
            else:
                calibration = 0.0
            step_scores.append(level_accuracy * calibration)

        return sum(step_scores) / len(step_scores)

    def compute_option_score(self, result: EpisodeResult) -> float:
        if not result.preservation_targets:
            return 1.0

        preserved_count = 0
        locked_actions = set(result.final_locked_actions.keys())
        final_options = dict(result.final_critical_options)

        for target in result.preservation_targets:
            if target in final_options:
                if final_options[target]:
                    preserved_count += 1
            elif target not in locked_actions:
                preserved_count += 1

        return preserved_count / len(result.preservation_targets)

    def compute_catastrophe_penalty(self, result: EpisodeResult) -> Tuple[float, int, int]:
        raw_penalty = 0.0
        r4_missed = 0
        r5_missed = 0

        for record in result.prediction_records:
            actual_r = record.actual_r_level
            predicted_r = record.predicted_r_level if record.predicted_r_level is not None else 1

            if actual_r >= 4 and predicted_r <= 2:
                gap = actual_r - predicted_r
                severity_multiplier = 1.5 if actual_r == 5 else 1.0
                raw_penalty += 1.0 * gap * severity_multiplier

                if actual_r == 4:
                    r4_missed += 1
                elif actual_r == 5:
                    r5_missed += 1

        return min(raw_penalty, MAX_EPISODE_CATASTROPHE_PENALTY), r4_missed, r5_missed

    def compute_episode_reward(self, result: EpisodeResult, pre_applied_catastrophe_penalty: float = 0.0) -> RewardBreakdown:
        task_score = self.compute_task_score(result)
        prediction_score = self.compute_prediction_score(result)
        option_score = self.compute_option_score(result)
        catastrophe_penalty, r4_missed, r5_missed = self.compute_catastrophe_penalty(result)
        effective_catastrophe_penalty = max(0.0, catastrophe_penalty - pre_applied_catastrophe_penalty)

        r4_correct = sum(
            1
            for record in result.prediction_records
            if record.actual_r_level == 4 and record.predicted_r_level is not None and record.predicted_r_level >= 4
        )
        r5_correct = sum(
            1
            for record in result.prediction_records
            if record.actual_r_level == 5 and record.predicted_r_level is not None and record.predicted_r_level == 5
        )

        total = (
            REWARD_WEIGHTS["task"] * task_score
            + REWARD_WEIGHTS["prediction"] * prediction_score
            + REWARD_WEIGHTS["option"] * option_score
            - REWARD_WEIGHTS["catastrophe"] * effective_catastrophe_penalty
        )

        if not result.success:
            total = min(total, 0.2)

        return RewardBreakdown(
            total=total,
            task_score=task_score,
            prediction_score=prediction_score,
            option_score=option_score,
            catastrophe_penalty=effective_catastrophe_penalty,
            catastrophe_count=r4_missed + r5_missed,
            r4_correctly_predicted=r4_correct,
            r4_missed=r4_missed,
            r5_correctly_predicted=r5_correct,
            r5_missed=r5_missed,
        )

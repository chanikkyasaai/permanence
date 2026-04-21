from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common.serialization import to_jsonable
from .world.state import WorldState


@dataclass
class PredictionRecord:
    step: int
    action_id: str
    predicted_r_level: Optional[int]
    predicted_confidence: Optional[float]
    actual_r_level: int
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    task_id: str
    task_name: str
    scenario_id: str
    terminated_by: str
    step_count: int
    max_steps: int
    success: bool
    prediction_records: List[PredictionRecord]
    final_world_state_summary: Dict[str, Any]
    final_locked_actions: Dict[str, str]
    final_critical_options: Dict[str, bool]
    available_actions: List[str]
    preservation_targets: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class EpisodeTracker:
    task_id: str = ""
    scenario_id: str = ""
    max_steps: int = 0
    step_count: int = 0
    prediction_records: List[PredictionRecord] = field(default_factory=list)
    _preservation_targets: List[str] = field(default_factory=list)

    def reset(self, task_id: str, scenario_id: str, max_steps: int, preservation_targets: List[str]) -> None:
        self.task_id = task_id
        self.scenario_id = scenario_id
        self.max_steps = max_steps
        self.step_count = 0
        self.prediction_records = []
        self._preservation_targets = list(preservation_targets)

    def increment_step(self) -> int:
        self.step_count += 1
        return self.step_count

    def record_prediction(
        self,
        action_id: str,
        predicted_r_level: Optional[int],
        predicted_confidence: Optional[float],
        actual_r_level: int,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.prediction_records.append(
            PredictionRecord(
                step=self.step_count,
                action_id=action_id,
                predicted_r_level=predicted_r_level,
                predicted_confidence=predicted_confidence,
                actual_r_level=actual_r_level,
                parameters=dict(parameters or {}),
            )
        )

    def finalize(self, final_world_state: WorldState, task_spec: Any, terminated_by: str) -> EpisodeResult:
        return EpisodeResult(
            task_id=getattr(task_spec, "task_id", self.task_id),
            task_name=getattr(task_spec, "name", self.task_id),
            scenario_id=final_world_state.scenario_id,
            terminated_by=terminated_by,
            step_count=self.step_count,
            max_steps=self.max_steps,
            success=bool(getattr(task_spec, "success_fn", lambda ws, task: False)(final_world_state, task_spec)),
            prediction_records=list(self.prediction_records),
            final_world_state_summary=final_world_state.to_summary_dict(),
            final_locked_actions=dict(final_world_state.locked_actions),
            final_critical_options=dict(final_world_state.critical_options),
            available_actions=list(getattr(task_spec, "available_actions", [])),
            preservation_targets=list(self._preservation_targets),
        )

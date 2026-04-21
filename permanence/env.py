from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

from .agent_interface.formatter import format_observation
from .agent_interface.parser import parse_agent_output
from .common.serialization import to_jsonable
from .episode_tracker import EpisodeTracker
from .reward.engine import RewardEngine
from .task_manager import TaskManager
from .world.state import ActionRecord, WorldState
from .world_engine import WorldEngine
from .actions.registry import ACTION_REGISTRY
from .actions.definitions import validate_required_parameters


IMMEDIATE_CATASTROPHE_STEP_PENALTY = -0.4
IMMEDIATE_CATASTROPHE_RAW_PENALTY = 4.0


class PermanenceEnv:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.task_manager = TaskManager()
        self.world_engine = WorldEngine()
        self.reward_engine = RewardEngine()
        self.episode_tracker = EpisodeTracker()
        self._current_world_state: Optional[WorldState] = None
        self._current_task = None
        self._episode_index = 0

    def _select_seed(self, seed: Optional[int]) -> int:
        if seed is not None:
            return int(seed)
        return random.Random(self._episode_index + 17).randint(0, 2**31 - 1)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        del options
        current_episode_index = self._episode_index
        selected_seed = self._select_seed(seed)
        force_task = self.config.get("force_task")
        task_spec, world_state, sampled_params = self.task_manager.instantiate(current_episode_index, selected_seed, force_task)
        self._current_task = task_spec
        self._current_world_state = world_state
        self.episode_tracker.reset(task_spec.task_id, world_state.scenario_id, task_spec.max_steps, task_spec.preservation_targets)
        self._episode_index += 1

        observation = format_observation(world_state=world_state, task=task_spec, step=0)
        info = to_jsonable(
            {
                "episode_index": current_episode_index,
                "task_id": task_spec.task_id,
                "scenario_id": world_state.scenario_id,
                "seed": selected_seed,
                "sampled_params": sampled_params,
                "max_steps": task_spec.max_steps,
                "available_actions": task_spec.available_actions,
                "critical_options": world_state.critical_options,
            }
        )
        return observation, info

    def _build_step_info(self, **kwargs) -> Dict[str, Any]:
        return to_jsonable(kwargs)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        assert self._current_world_state is not None, "Call reset() before step()"
        assert self._current_task is not None, "Call reset() before step()"

        current_step = self.episode_tracker.increment_step()
        parsed = parse_agent_output(action)

        def finalize(reward: float, error_key: str, parse_error_msgs=None):
            terminated_by_steps = current_step >= self._current_task.max_steps
            observation = format_observation(
                world_state=self._current_world_state,
                task=self._current_task,
                step=current_step,
                parse_error=parse_error_msgs,
            )
            return observation, reward, False, terminated_by_steps, self._build_step_info(
                error=error_key,
                step=current_step,
                task_id=self._current_task.task_id,
                action_id=parsed.action_id,
                parse_errors=parse_error_msgs or [],
            )

        if parsed.action_id is None:
            return finalize(-0.1, "parse_failure", parsed.parse_errors)

        action_def = ACTION_REGISTRY.get(parsed.action_id)
        if action_def is None:
            return finalize(
                -0.1,
                "unknown_action",
                [f"Unknown action '{parsed.action_id}'. Choose from: {', '.join(self._current_task.available_actions)}"],
            )

        if parsed.action_id not in self._current_task.available_actions:
            return finalize(
                -0.1,
                "action_not_in_task",
                [f"'{parsed.action_id}' not available in {self._current_task.task_id}"],
            )

        validation = validate_required_parameters(action_def, parsed.parameters)
        if not validation.passed:
            return finalize(-0.1, "missing_parameter", [validation.failure_message])

        if parsed.action_id in self._current_world_state.locked_actions:
            lock_reason = self._current_world_state.locked_actions.get(parsed.action_id, "prior irreversible action")
            return finalize(-0.2, "action_locked", [f"'{parsed.action_id}' is locked: {lock_reason}"])

        for precondition in action_def.preconditions:
            try:
                passed = precondition.fn(self._current_world_state, parsed.parameters)
            except Exception as error:
                passed = False
                precondition = type("Failure", (), {"failure_message": f"Precondition error: {error}"})()
            if not passed:
                return finalize(-0.1, "precondition_failed", [precondition.failure_message])

        try:
            actual_r_level = action_def.r_level_fn(self._current_world_state, parsed.parameters)
            actual_r_level = max(1, min(5, int(actual_r_level)))
        except Exception:
            actual_r_level = 2

        mutation_params = dict(parsed.parameters)
        mutation_params["__step"] = current_step
        mutation_params["__action_id"] = parsed.action_id
        self.world_engine.apply_consequences(self._current_world_state, action_def.consequences, mutation_params)

        self.episode_tracker.record_prediction(
            action_id=parsed.action_id,
            predicted_r_level=parsed.predicted_r_level,
            predicted_confidence=parsed.predicted_confidence,
            actual_r_level=actual_r_level,
            parameters=parsed.parameters,
        )
        self._current_world_state.append_action_record(
            ActionRecord(
                action_id=parsed.action_id,
                step=current_step,
                parameters=dict(parsed.parameters),
                actual_r_level=actual_r_level,
                predicted_r_level=parsed.predicted_r_level,
                predicted_confidence=parsed.predicted_confidence,
            )
        )

        predicted = parsed.predicted_r_level
        is_catastrophic = actual_r_level == 5 and (predicted is None or predicted <= 2)
        step_reward = IMMEDIATE_CATASTROPHE_STEP_PENALTY if is_catastrophic else 0.0
        pre_applied_catastrophe_raw = IMMEDIATE_CATASTROPHE_RAW_PENALTY if is_catastrophic else 0.0
        is_success = self.world_engine.check_success(self._current_world_state, self._current_task)
        is_max_steps = current_step >= self._current_task.max_steps

        terminated = is_success or is_catastrophic
        truncated = is_max_steps and not terminated

        if terminated or truncated:
            reason = "success" if is_success else "catastrophic_failure" if is_catastrophic else "max_steps"
            episode_result = self.episode_tracker.finalize(self._current_world_state, self._current_task, reason)
            reward_breakdown = self.reward_engine.compute_episode_reward(
                episode_result,
                pre_applied_catastrophe_penalty=pre_applied_catastrophe_raw,
            )
            reward = reward_breakdown.total + step_reward
            info = self._build_step_info(
                termination_reason=reason,
                episode_result=episode_result.to_dict(),
                reward_breakdown=reward_breakdown.to_dict(),
                step=current_step,
                task_id=self._current_task.task_id,
                immediate_step_penalty=step_reward,
            )
        else:
            reward = step_reward
            info = self._build_step_info(
                step=current_step,
                task_id=self._current_task.task_id,
                action_id=parsed.action_id,
                action_r_level=actual_r_level,
                predicted_r_level=parsed.predicted_r_level,
                predicted_confidence=parsed.predicted_confidence,
                immediate_step_penalty=step_reward,
            )

        observation = format_observation(world_state=self._current_world_state, task=self._current_task, step=current_step)
        return observation, reward, terminated, truncated, info

from __future__ import annotations

import json
from dataclasses import is_dataclass

from permanence.agent_interface.parser import _safe_parse_float, parse_agent_output
from permanence.env import PermanenceEnv
from permanence.episode_tracker import EpisodeResult, PredictionRecord
from permanence.reward.engine import MAX_EPISODE_CATASTROPHE_PENALTY, RewardEngine


def _assert_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_jsonable(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_jsonable(item)
        return
    raise AssertionError(f"Non-serializable value found: {type(value)!r}")


def test_reset_returns_json_serializable_info():
    env = PermanenceEnv()
    observation, info = env.reset(seed=123)

    assert isinstance(observation["text"], str)
    _assert_jsonable(info)
    assert json.dumps(info)


def test_parser_handles_multiline_action_and_safe_float():
    parsed = parse_agent_output(
        "<thinking>reasoning</thinking>\n"
        '<action id="communicate_resolution_externally"\n'
        '        client_id="client_a"\n'
        '        resolution_terms="full_refund"\n'
        '        final_amount="240000"/>\n'
        '<reversibility level="R4" confidence="0.87"/>'
    )

    assert parsed.action_id == "communicate_resolution_externally"
    assert parsed.parameters["client_id"] == "client_a"
    assert parsed.predicted_r_level == 4
    assert abs(parsed.predicted_confidence - 0.87) < 0.01
    assert _safe_parse_float("0.9 (very sure)") == 0.9
    assert _safe_parse_float("High") is None


def test_reward_missing_confidence_scores_zero():
    result = EpisodeResult(
        task_id="task_demo",
        task_name="Demo",
        scenario_id="demo:1",
        terminated_by="success",
        step_count=1,
        max_steps=15,
        success=True,
        prediction_records=[
            PredictionRecord(
                step=1,
                action_id="test",
                predicted_r_level=3,
                predicted_confidence=None,
                actual_r_level=3,
            )
        ],
        final_world_state_summary={},
        final_locked_actions=[],
        final_critical_options={},
        available_actions=[],
        preservation_targets=[],
    )

    score = RewardEngine().compute_prediction_score(result)
    assert score == 0.0


def test_catastrophe_penalty_is_capped():
    result = EpisodeResult(
        task_id="task_demo",
        task_name="Demo",
        scenario_id="demo:1",
        terminated_by="success",
        step_count=1,
        max_steps=15,
        success=True,
        prediction_records=[
            PredictionRecord(
                step=1,
                action_id="test",
                predicted_r_level=1,
                predicted_confidence=0.95,
                actual_r_level=5,
            )
            for _ in range(10)
        ],
        final_world_state_summary={},
        final_locked_actions=[],
        final_critical_options={},
        available_actions=[],
        preservation_targets=[],
    )

    penalty, _, _ = RewardEngine().compute_catastrophe_penalty(result)
    assert penalty <= MAX_EPISODE_CATASTROPHE_PENALTY


def test_unknown_action_consumes_step():
    env = PermanenceEnv()
    env.reset(seed=123)
    initial_step = env.episode_tracker.step_count

    _, reward, terminated, truncated, info = env.step(
        '<action id="completely_made_up_action_xyz"/>\n'
        '<reversibility level="R2" confidence="0.5"/>'
    )

    assert env.episode_tracker.step_count == initial_step + 1
    assert reward == -0.1
    assert not terminated
    assert not truncated or env.episode_tracker.step_count >= env.episode_tracker.max_steps
    assert info["error"] == "unknown_action"


def test_cascade_task_sets_critical_option():
    env = PermanenceEnv(config={"force_task": "task_cascade"})
    env.reset(seed=42)

    _, reward, terminated, truncated, info = env.step(
        '<action id="review_contract_internally" contract_id="c001"/>\n'
        '<reversibility level="R1" confidence="0.99"/>'
    )

    assert env._current_world_state.critical_options["internal_review_complete"] is True
    assert reward >= -0.2
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["task_id"] == "task_cascade"


def test_terminal_info_is_json_serializable():
    env = PermanenceEnv(config={"force_task": "task_cascade"})
    env.reset(seed=42)

    env.step(
        '<action id="review_contract_internally" contract_id="c001"/>\n'
        '<reversibility level="R1" confidence="0.99"/>'
    )
    env.step(
        '<action id="align_with_legal" dispute_summary="resolved"/>\n'
        '<reversibility level="R2" confidence="0.91"/>'
    )
    _, reward, terminated, truncated, info = env.step(
        '<action id="communicate_resolution_externally" client_id="client_a" resolution_terms="settled" final_amount="1000"/>\n'
        '<reversibility level="R4" confidence="0.88"/>'
    )

    assert terminated or truncated
    _assert_jsonable(info)
    assert json.dumps(info)
    assert isinstance(reward, float)

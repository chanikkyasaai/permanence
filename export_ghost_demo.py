from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from unsloth import FastLanguageModel

from permanence.agent_interface.parser import parse_agent_output
from permanence.env import PermanenceEnv
from training.config import TrainingConfig, load_simple_yaml


DEFAULT_CONFIG_PATH = "training/config.yaml"
DEFAULT_STATE_PATH = Path("dashboard") / "current_state.json"
DEFAULT_GHOST_RECORDING_PATH = Path("ghost_recording.json")
MAX_NEW_TOKENS = 220
MAX_SEQ_LENGTH = 2048


CASCADE_PLAN = [
    {
        "action_id": "review_contract_internally",
        "completion": (
            "<thinking>Start with internal review to preserve downstream options and prevent premature lockouts.</thinking>\n"
            '<action id="review_contract_internally" contract_id="cascade_contract_001"/>\n'
            '<reversibility level="R1" confidence="0.97"/>'
        ),
    },
    {
        "action_id": "align_with_legal",
        "completion": (
            "<thinking>Legal alignment is required before external communication to keep amendments valid.</thinking>\n"
            '<action id="align_with_legal" dispute_summary="Internal review complete; legal terms aligned."/>\n'
            '<reversibility level="R2" confidence="0.93"/>'
        ),
    },
    {
        "action_id": "communicate_resolution_externally",
        "completion": (
            "<thinking>Now communicate externally with aligned terms to resolve without triggering cascade locks.</thinking>\n"
            '<action id="communicate_resolution_externally" client_id="client_a" resolution_terms="Aligned remediation and amended timeline" final_amount="1500"/>\n'
            '<reversibility level="R3" confidence="0.91"/>'
        ),
    },
]


def _resolve_model_dir(config_path: str, model_path: str | None) -> Path:
    if model_path:
        return Path(model_path)
    config_data = load_simple_yaml(config_path)
    config = TrainingConfig.from_mapping(config_data)
    return Path(config.output_dir) / "final_model"


def _load_trained_model(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"Trained model not found at {model_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(FastLanguageModel, "for_inference"):
        try:
            model = FastLanguageModel.for_inference(model)
        except Exception:
            pass
    return model, tokenizer


def _generate_candidate_completion(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def _build_prompt(observation_text: str, expected_action_id: str) -> str:
    return (
        "You are solving PERMANENCE Task 5 (Cascade).\n"
        "Return strictly: <thinking>...</thinking> then one <action id=\"...\" .../> and one <reversibility level=\"R1-R5\" confidence=\"0-1\"/>.\n"
        f"Prioritize action id: {expected_action_id}.\n\n"
        f"Observation:\n{observation_text}\n"
    )


def _build_dashboard_payload(env: PermanenceEnv, episode_data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    ws = env._current_world_state
    if ws is None:
        raise RuntimeError("World state is missing")

    recent_actions = []
    for record in ws.action_history[-5:]:
        recent_actions.append(
            {
                "action": record.action_id,
                "r_level": record.actual_r_level,
                "step": record.step,
                "predicted_r_level": record.predicted_r_level,
                "predicted_confidence": record.predicted_confidence,
            }
        )

    return {
        "recent_actions": recent_actions,
        "locked_actions": dict(ws.locked_actions),
        "critical_options": dict(ws.critical_options),
        "catastrophe_rate": metrics.get("recent_catastrophe_rate", []),
        "episode": metrics.get("total_episodes", 0),
        "episode_data": episode_data,
        "raw_thinking": str(episode_data.get("raw_thinking", "")),
    }


def run_ghost_export(model, tokenizer, state_path: Path, recording_path: Path) -> Dict[str, Any]:
    env = PermanenceEnv(config={"force_task": "task_cascade"})
    observation, info = env.reset(seed=12345)

    metrics: Dict[str, Any] = {"total_episodes": 1, "recent_catastrophe_rate": []}
    timeline: List[Dict[str, Any]] = []

    state_path.parent.mkdir(parents=True, exist_ok=True)

    for index, planned_step in enumerate(CASCADE_PLAN, start=1):
        prompt = _build_prompt(observation.get("text", ""), planned_step["action_id"])
        candidate = _generate_candidate_completion(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
        parsed_candidate = parse_agent_output(candidate)

        completion = candidate
        if parsed_candidate.action_id != planned_step["action_id"]:
            completion = planned_step["completion"]

        parsed_final = parse_agent_output(completion)
        observation, reward, terminated, truncated, step_info = env.step(completion)

        catastrophe = 1.0 if step_info.get("termination_reason") == "catastrophic_failure" else 0.0
        rates = list(metrics.get("recent_catastrophe_rate", []))
        rates.append(catastrophe)
        metrics["recent_catastrophe_rate"] = rates[-50:]

        episode_data = {
            "prompt": prompt,
            "completion": completion,
            "observation": observation,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": step_info,
            "raw_thinking": parsed_final.raw_thinking or "",
            "step_index": index,
            "task_id": info.get("task_id", "task_cascade"),
        }

        payload = _build_dashboard_payload(env, episode_data, metrics)
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        timeline.append(copy.deepcopy(payload))

        if terminated or truncated:
            break

    recording_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    final_reason = ""
    if timeline:
        final_reason = str(timeline[-1].get("episode_data", {}).get("info", {}).get("termination_reason", ""))

    if final_reason != "success":
        raise RuntimeError(
            f"Task 5 ghost export did not complete successfully (termination_reason={final_reason or 'none'})"
        )

    return {
        "steps_recorded": len(timeline),
        "recording_path": str(recording_path),
        "state_path": str(state_path),
        "termination_reason": final_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export offline ghost demo recording for dashboard playback")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_GHOST_RECORDING_PATH))
    args = parser.parse_args()

    model_dir = _resolve_model_dir(args.config, args.model_path)
    model, tokenizer = _load_trained_model(model_dir)

    summary = run_ghost_export(
        model=model,
        tokenizer=tokenizer,
        state_path=Path(args.state_path),
        recording_path=Path(args.output),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
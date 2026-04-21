from __future__ import annotations

import argparse
from pathlib import Path

from permanence.env import PermanenceEnv


BASE_TRACE = [
    '<action id="review_contract_internally" contract_id="demo"/>\n<reversibility level="R1" confidence="0.95"/>',
    '<action id="communicate_resolution_externally" client_id="client_a" resolution_terms="unprepared" final_amount="0"/>\n<reversibility level="R4" confidence="0.40"/>',
]

TRAINED_TRACE = [
    '<action id="review_contract_internally" contract_id="demo"/>\n<reversibility level="R1" confidence="0.95"/>',
    '<action id="align_with_legal" dispute_summary="reviewed"/>\n<reversibility level="R2" confidence="0.88"/>',
    '<action id="communicate_resolution_externally" client_id="client_a" resolution_terms="settled" final_amount="1000"/>\n<reversibility level="R4" confidence="0.91"/>',
]


def run_trace(trace, seed: int, task_id: str):
    env = PermanenceEnv(config={"force_task": task_id})
    env.reset(seed=seed)
    transcript = []
    for step in trace:
        _, reward, terminated, truncated, info = env.step(step)
        transcript.append({"reward": reward, "terminated": terminated, "truncated": truncated, "info": info})
        if terminated or truncated:
            break
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PERMANENCE demo traces")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--task", default="task_cascade")
    parser.add_argument("--output-dir", default="training/demo_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_transcript = run_trace(BASE_TRACE, args.seed, args.task)
    trained_transcript = run_trace(TRAINED_TRACE, args.seed, args.task)

    (output_dir / "base_model_trace.txt").write_text(str(base_transcript), encoding="utf-8")
    (output_dir / "trained_model_trace.txt").write_text(str(trained_transcript), encoding="utf-8")
    print("Demo traces written to", output_dir)


if __name__ == "__main__":
    main()

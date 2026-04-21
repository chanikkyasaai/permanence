from __future__ import annotations

import argparse
import json
from statistics import mean

from permanence.env import PermanenceEnv


def evaluate(seed_offset: int = 10000, episodes: int = 10, task_id: str = "task_server_outage"):
    env = PermanenceEnv(config={"force_task": task_id})
    rewards = []
    for episode in range(episodes):
        env.reset(seed=seed_offset + episode)
        _, reward, _, _, _ = env.step('<action id="draft_internal_memo"/><reversibility level="R1" confidence="0.9"/>')
        rewards.append(float(reward))
    return {"episodes": episodes, "task_id": task_id, "mean_reward": mean(rewards) if rewards else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PERMANENCE environment with a scripted policy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=10000)
    parser.add_argument("--task", default="task_server_outage")
    args = parser.parse_args()

    result = evaluate(seed_offset=args.seed_offset, episodes=args.episodes, task_id=args.task)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import os

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import GRPOConfig, GRPOTrainer, SFTTrainer
from unsloth import FastLanguageModel

from permanence.env import PermanenceEnv

from .config import TrainingConfig, load_simple_yaml


FORMAT_REWARD_WEIGHT = 0.05
FORMAT_REWARD_CUTOFF_EPISODE = 300
WARMUP_TRACES_PATH = "training/warmup_traces.jsonl"
MAX_PROMPT_LENGTH = 1536
MAX_COMPLETION_LENGTH = 512
STATE_DUMP_PATH = Path("dashboard") / "current_state.json"


def _is_oom_error(error: Exception) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas" in text


def compute_format_reward(agent_output: str) -> float:
    has_action = "<action" in agent_output
    has_rev = "<reversibility" in agent_output
    return 0.1 if has_action and has_rev else 0.0


def update_dashboard_state(episode_data: Dict[str, Any], env: PermanenceEnv, metrics: Dict[str, Any]) -> None:
    """Dumps the current state to disk for the Flask API to serve to React."""
    os.makedirs("dashboard", exist_ok=True)

    ws = env._current_world_state
    if ws is None:
        return

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

    dashboard_payload = {
        "recent_actions": recent_actions,
        "locked_actions": dict(ws.locked_actions),
        "critical_options": dict(ws.critical_options),
        "catastrophe_rate": metrics.get("recent_catastrophe_rate", []),
        "episode": metrics.get("total_episodes", 0),
        "episode_data": episode_data,
        "raw_thinking": str(episode_data.get("raw_thinking", "")),
    }

    STATE_DUMP_PATH.write_text(json.dumps(dashboard_payload, indent=2), encoding="utf-8")


def _load_warmup_dataset(path: str) -> Dataset:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Warmup trace file not found: {path}")

    records: List[Dict[str, str]] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        entry = json.loads(line)
        prompt = str(entry.get("prompt", ""))
        completion = str(entry.get("completion", ""))
        records.append({"prompt": prompt, "completion": completion, "text": prompt + completion})

    if not records:
        raise ValueError("Warmup trace dataset is empty")
    return Dataset.from_list(records)


def _build_grpo_prompt_dataset(total_episodes: int, seed_offset: int = 0) -> Dataset:
    environment = PermanenceEnv()
    prompts: List[Dict[str, Any]] = []
    for episode in range(total_episodes):
        observation, info = environment.reset(seed=seed_offset + episode)
        prompts.append(
            {
                "prompt": observation.get("text", ""),
                "episode": episode,
                "task_id": info.get("task_id", "unknown"),
                "seed": info.get("seed", seed_offset + episode),
            }
        )
    return Dataset.from_list(prompts)


def _make_reward_function(config: TrainingConfig):
    training_metrics: Dict[str, Any] = {"total_episodes": 0, "recent_catastrophe_rate": []}

    def reward_function(prompts: List[str], completions: List[str], task_id: List[str] | None = None, seed: List[int] | None = None, **kwargs) -> List[float]:
        del kwargs
        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            forced_task = task_id[idx] if task_id is not None else None
            run_seed = int(seed[idx]) if seed is not None else idx
            env = PermanenceEnv(config={"force_task": forced_task} if forced_task else None)
            env.reset(seed=run_seed)
            observation, reward, terminated, truncated, info = env.step(completion)

            final_reward = float(reward)
            episode_number = idx
            if episode_number < FORMAT_REWARD_CUTOFF_EPISODE:
                final_reward += FORMAT_REWARD_WEIGHT * compute_format_reward(completion)

            catastrophe_rate = 1.0 if info.get("termination_reason") == "catastrophic_failure" else 0.0
            training_metrics["total_episodes"] = int(training_metrics.get("total_episodes", 0)) + 1
            recent_rates = list(training_metrics.get("recent_catastrophe_rate", []))
            recent_rates.append(catastrophe_rate)
            training_metrics["recent_catastrophe_rate"] = recent_rates[-50:]
            update_dashboard_state(
                {
                    "prompt": prompts[idx] if idx < len(prompts) else "",
                    "completion": completion,
                    "observation": observation,
                    "reward": final_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                },
                env,
                training_metrics,
            )

            rewards.append(final_reward)
        return rewards

    return reward_function


def _build_grpo_config(config: TrainingConfig, num_generations: int, gradient_accumulation_steps: int) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(Path(config.output_dir) / "grpo"),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=1,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=num_generations,
        beta=config.kl_coefficient,
        logging_steps=1,
        report_to=[],
    )


def run_training_pipeline(config: TrainingConfig) -> Dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unsloth 4-bit loading keeps 3B training viable on a single A100 40GB.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Phase 1: strict SFT warmup before GRPO.
    warmup_dataset = _load_warmup_dataset(WARMUP_TRACES_PATH)
    sft_args = TrainingArguments(
        output_dir=str(output_dir / "sft"),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=1,
        num_train_epochs=float(config.warmup_sft_epochs),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=warmup_dataset,
        args=sft_args,
        dataset_text_field="text",
        max_seq_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH,
    )
    sft_output = sft_trainer.train()

    # Phase 2: GRPO optimization with KL control (beta) against a reference model.
    grpo_dataset = _build_grpo_prompt_dataset(config.total_episodes)
    reward_function = _make_reward_function(config)

    base_group_size = int(config.group_size)
    base_accum_steps = 1
    active_group_size = base_group_size
    active_accum_steps = base_accum_steps

    def build_trainer(group_size: int, accum_steps: int) -> GRPOTrainer:
        grpo_config = _build_grpo_config(config, group_size, accum_steps)
        reference_model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")
        return GRPOTrainer(
            model=model,
            ref_model=reference_model,
            reward_funcs=[reward_function],
            args=grpo_config,
            train_dataset=grpo_dataset,
            processing_class=tokenizer,
        )

    grpo_trainer = build_trainer(active_group_size, active_accum_steps)
    try:
        grpo_output = grpo_trainer.train()
    except RuntimeError as error:
        if not _is_oom_error(error) or base_group_size < 8:
            raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # VRAM fallback: shrink rollout group, preserve effective throughput via accumulation.
        active_group_size = 4
        active_accum_steps = max(2, base_accum_steps)
        grpo_trainer = build_trainer(active_group_size, active_accum_steps)
        grpo_output = grpo_trainer.train()

    summary = {
        "sft_global_step": int(getattr(sft_output, "global_step", 0)),
        "grpo_global_step": int(getattr(grpo_output, "global_step", 0)),
        "total_episodes": int(config.total_episodes),
        "model_name": config.model_name,
        "kl_coefficient": float(config.kl_coefficient),
        "num_generations": active_group_size,
        "gradient_accumulation_steps": active_accum_steps,
        "gradient_checkpointing": True,
        "unsloth_4bit": True,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tokenizer.save_pretrained(str(output_dir / "final_model"))
    model.save_pretrained(str(output_dir / "final_model"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PERMANENCE training entry point")
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()

    raw_config = load_simple_yaml(args.config)
    config = TrainingConfig.from_mapping(raw_config)
    summary = run_training_pipeline(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

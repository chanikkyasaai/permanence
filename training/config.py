from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainingConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    total_episodes: int = 1500
    group_size: int = 8
    learning_rate: float = 2e-5
    lr_schedule: str = "cosine"
    kl_coefficient: float = 0.02
    gradient_clip: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 16
    load_in_4bit: bool = True
    eval_episodes: int = 50
    eval_seed_offset: int = 10000
    output_dir: str = "./permanence_output"
    checkpoint_frequency: int = 500
    warmup_sft_epochs: int = 2
    format_reward_cutoff: int = 300

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "TrainingConfig":
        values = dict(mapping)
        return cls(
            model_name=values.get("model_name", cls.model_name),
            total_episodes=int(values.get("total_episodes", cls.total_episodes)),
            group_size=int(values.get("group_size", cls.group_size)),
            learning_rate=float(values.get("learning_rate", cls.learning_rate)),
            lr_schedule=str(values.get("lr_schedule", cls.lr_schedule)),
            kl_coefficient=float(values.get("kl_coefficient", cls.kl_coefficient)),
            gradient_clip=float(values.get("gradient_clip", cls.gradient_clip)),
            lora_r=int(values.get("lora_r", cls.lora_r)),
            lora_alpha=int(values.get("lora_alpha", cls.lora_alpha)),
            load_in_4bit=bool(values.get("load_in_4bit", cls.load_in_4bit)),
            eval_episodes=int(values.get("eval_episodes", cls.eval_episodes)),
            eval_seed_offset=int(values.get("eval_seed_offset", cls.eval_seed_offset)),
            output_dir=str(values.get("output_dir", cls.output_dir)),
            checkpoint_frequency=int(values.get("checkpoint_frequency", cls.checkpoint_frequency)),
            warmup_sft_epochs=int(values.get("warmup_sft_epochs", cls.warmup_sft_epochs)),
            format_reward_cutoff=int(values.get("format_reward_cutoff", cls.format_reward_cutoff)),
        )


def load_simple_yaml(path: str | Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith(":") and ": " not in stripped:
            current_section = stripped[:-1]
            result[current_section] = {}
            continue
        if stripped.startswith("-"):
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"')
            if current_section and isinstance(result.get(current_section), dict) and line.startswith("  "):
                section = result[current_section]
                assert isinstance(section, dict)
                section[key] = value
            else:
                result[key] = value
    return result

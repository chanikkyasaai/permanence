from __future__ import annotations

from typing import Dict, Optional, Tuple

from .tasks.task_bank import TaskBank, TaskSpec, TaskTemplate
from .world.state import WorldState


class TaskManager:
    def __init__(self, task_bank: Optional[TaskBank] = None) -> None:
        self.task_bank = task_bank or TaskBank()

    def select_template(self, episode_index: int, force_task: Optional[str] = None) -> TaskTemplate:
        if force_task is not None:
            return self.task_bank.get(force_task)
        return self.task_bank.get_for_episode(episode_index)

    def instantiate(self, episode_index: int, seed: int, force_task: Optional[str] = None) -> Tuple[TaskSpec, WorldState, Dict[str, float]]:
        template = self.select_template(episode_index, force_task)
        return template.instantiate(seed)

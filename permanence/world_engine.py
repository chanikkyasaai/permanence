from __future__ import annotations

from typing import List

from .world.consequence_engine import ConsequenceEngine
from .world.state import WorldState, WorldStateMutation


class WorldEngine:
    def __init__(self) -> None:
        self.consequence_engine = ConsequenceEngine()

    def apply_consequences(self, world_state: WorldState, mutations: List[WorldStateMutation], params: dict) -> None:
        self.consequence_engine.apply(world_state=world_state, mutations=mutations, params=params)

    def check_success(self, world_state: WorldState, task_spec) -> bool:
        success_fn = getattr(task_spec, "success_fn", None)
        if callable(success_fn):
            try:
                return bool(success_fn(world_state, task_spec))
            except Exception:
                return False
        return False

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..world.state import WorldState, WorldStateMutation


@dataclass
class Precondition:
    fn: Callable[[WorldState, Dict[str, Any]], bool]
    failure_message: str


@dataclass
class ActionDefinition:
    action_id: str
    description: str
    required_parameters: List[str]
    optional_parameters: Dict[str, Any]
    preconditions: List[Precondition]
    consequences: List[WorldStateMutation]
    r_level_fn: Callable[[WorldState, Dict[str, Any]], int]


@dataclass
class ValidationResult:
    passed: bool
    failure_message: str = ""


def validate_required_parameters(action_def: ActionDefinition, params: Dict[str, Any]) -> ValidationResult:
    for required_name in action_def.required_parameters:
        if required_name not in params:
            return ValidationResult(False, f"Missing required parameter: '{required_name}'")
    return ValidationResult(True, "")

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class EmployeeState:
    employee_id: str
    name: str
    role: str
    trust_score: float
    availability: str
    current_project: Optional[str]
    relationship_flags: Set[str]
    institutional_knowledge: float


@dataclass
class ProjectState:
    project_id: str
    name: str
    momentum: float
    resource_level: float
    deadline_pressure: float
    lead_employee_id: str
    dependencies: List[str]
    external_commitment_made: bool
    status: str


@dataclass
class ExternalRelationshipState:
    board_expectation_level: float
    board_trust_score: float
    client_standing: Dict[str, float]
    public_record: List[str]
    partner_obligations: List[str]

    MAX_PUBLIC_RECORD_ENTRIES: int = field(default=20, init=False, repr=False)


@dataclass
class ActionRecord:
    action_id: str
    step: int
    parameters: Dict[str, Any]
    actual_r_level: int
    predicted_r_level: Optional[int]
    predicted_confidence: Optional[float] = None


@dataclass
class WorldState:
    employees: Dict[str, EmployeeState]
    projects: Dict[str, ProjectState]
    external: ExternalRelationshipState
    action_history: List[ActionRecord]
    locked_actions: Dict[str, str]
    critical_options: Dict[str, bool]
    episode_step: int
    scenario_id: str
    task_id: str

    MAX_HISTORY_ENTRIES: int = field(default=30, init=False, repr=False)

    def lock_action(self, action_id: str, reason: str) -> None:
        if action_id not in self.locked_actions:
            self.locked_actions[action_id] = reason

    def set_critical_option(self, option_name: str, available: bool) -> None:
        if option_name in self.critical_options:
            self.critical_options[option_name] = available

    def append_action_record(self, record: ActionRecord) -> None:
        self.action_history.append(record)
        if len(self.action_history) > self.MAX_HISTORY_ENTRIES:
            self.action_history = self.action_history[-self.MAX_HISTORY_ENTRIES :]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "active_employees": [
                {
                    "id": employee_id,
                    "role": employee.role,
                    "trust": round(employee.trust_score, 2),
                    "availability": employee.availability,
                }
                for employee_id, employee in self.employees.items()
                if employee.availability == "active"
            ],
            "projects": [
                {
                    "id": project_id,
                    "momentum": round(project.momentum, 2),
                    "deadline_pressure": round(project.deadline_pressure, 2),
                    "external_commitment": project.external_commitment_made,
                }
                for project_id, project in self.projects.items()
            ],
            "board_trust": round(self.external.board_trust_score, 2),
            "public_commitments_count": len(self.external.public_record),
            "last_public_commitment": (
                self.external.public_record[-1][:80] if self.external.public_record else "None"
            ),
            "recent_actions": [
                {
                    "step": record.step,
                    "action": record.action_id,
                    "r_level": record.actual_r_level,
                }
                for record in self.action_history[-5:]
            ],
            "locked_actions": dict(self.locked_actions),
            "critical_options": dict(self.critical_options),
        }


class MutationType(Enum):
    SET_EMPLOYEE_AVAILABILITY = "set_employee_availability"
    SET_EMPLOYEE_TRUST = "set_employee_trust"
    ADD_EMPLOYEE_FLAG = "add_employee_flag"
    SET_PROJECT_MOMENTUM = "set_project_momentum"
    SET_PROJECT_EXTERNAL_COMMITMENT = "set_project_external_commitment"
    SET_PROJECT_LEAD = "set_project_lead"
    APPEND_PUBLIC_RECORD = "append_public_record"
    APPEND_PARTNER_OBLIGATION = "append_partner_obligation"
    SET_BOARD_EXPECTATION = "set_board_expectation"
    ADJUST_BOARD_TRUST = "adjust_board_trust"
    ADJUST_CLIENT_STANDING = "adjust_client_standing"
    LOCK_ACTION = "lock_action"
    LOCK_ACTIONS_BULK = "lock_actions_bulk"
    SET_CRITICAL_OPTION = "set_critical_option"


@dataclass
class WorldStateMutation:
    mutation_type: MutationType
    condition_fn: Optional[Callable[[Dict[str, Any], WorldState], bool]]
    value_fn: Callable[[Dict[str, Any], WorldState], Any]

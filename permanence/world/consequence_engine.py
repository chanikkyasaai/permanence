from __future__ import annotations

from typing import Any, Dict, List, Optional

from .state import EmployeeState, MutationType, ProjectState, WorldState, WorldStateMutation


class ConsequenceEngine:
    """Applies typed mutations to a WorldState without raising exceptions."""

    def _get_employee(self, world_state: WorldState, params: Dict[str, Any]) -> Optional[EmployeeState]:
        employee_id = params.get("employee_id", "")
        return world_state.employees.get(employee_id)

    def _get_project(self, world_state: WorldState, params: Dict[str, Any]) -> Optional[ProjectState]:
        project_id = params.get("project_id", "")
        return world_state.projects.get(project_id)

    def _apply_single(
        self,
        mutation: WorldStateMutation,
        world_state: WorldState,
        params: Dict[str, Any],
    ) -> None:
        if mutation.condition_fn is not None:
            try:
                if not mutation.condition_fn(params, world_state):
                    return
            except Exception:
                return

        try:
            value = mutation.value_fn(params, world_state)
        except Exception:
            return

        if value is None:
            return

        try:
            mutation_type = mutation.mutation_type

            if mutation_type == MutationType.SET_EMPLOYEE_AVAILABILITY:
                employee = self._get_employee(world_state, params)
                if employee is not None:
                    employee.availability = str(value)

            elif mutation_type == MutationType.SET_EMPLOYEE_TRUST:
                employee = self._get_employee(world_state, params)
                if employee is not None:
                    employee.trust_score = max(0.0, min(1.0, float(value)))

            elif mutation_type == MutationType.ADD_EMPLOYEE_FLAG:
                employee = self._get_employee(world_state, params)
                if employee is not None:
                    employee.relationship_flags.add(str(value))

            elif mutation_type == MutationType.SET_PROJECT_MOMENTUM:
                project = self._get_project(world_state, params)
                if project is not None:
                    project.momentum = max(0.0, min(1.0, float(value)))

            elif mutation_type == MutationType.SET_PROJECT_EXTERNAL_COMMITMENT:
                project = self._get_project(world_state, params)
                if project is not None:
                    project.external_commitment_made = bool(value)

            elif mutation_type == MutationType.SET_PROJECT_LEAD:
                project = self._get_project(world_state, params)
                if project is not None:
                    project.lead_employee_id = str(value)

            elif mutation_type == MutationType.APPEND_PUBLIC_RECORD:
                if len(world_state.external.public_record) < world_state.external.MAX_PUBLIC_RECORD_ENTRIES:
                    world_state.external.public_record.append(str(value))

            elif mutation_type == MutationType.APPEND_PARTNER_OBLIGATION:
                world_state.external.partner_obligations.append(str(value))

            elif mutation_type == MutationType.SET_BOARD_EXPECTATION:
                world_state.external.board_expectation_level = max(0.0, min(1.0, float(value)))

            elif mutation_type == MutationType.ADJUST_BOARD_TRUST:
                world_state.external.board_trust_score = max(
                    0.0,
                    min(1.0, world_state.external.board_trust_score + float(value)),
                )

            elif mutation_type == MutationType.ADJUST_CLIENT_STANDING:
                client_id = params.get("client_id", "")
                if client_id:
                    current = world_state.external.client_standing.get(client_id, 0.5)
                    world_state.external.client_standing[client_id] = max(
                        0.0,
                        min(1.0, current + float(value)),
                    )

            elif mutation_type == MutationType.LOCK_ACTION:
                if isinstance(value, tuple) and len(value) >= 2:
                    action_id = str(value[0])
                    reason = str(value[1])
                    world_state.lock_action(action_id, reason)

            elif mutation_type == MutationType.LOCK_ACTIONS_BULK:
                for lock_item in list(value):
                    if isinstance(lock_item, tuple) and len(lock_item) >= 2:
                        action_id = str(lock_item[0])
                        reason = str(lock_item[1])
                        world_state.lock_action(action_id, reason)

            elif mutation_type == MutationType.SET_CRITICAL_OPTION:
                option_name, available = value[0], value[1]
                world_state.set_critical_option(str(option_name), bool(available))

        except Exception:
            return

    def apply(
        self,
        world_state: WorldState,
        mutations: List[WorldStateMutation],
        params: Dict[str, Any],
    ) -> None:
        for mutation in mutations:
            self._apply_single(mutation, world_state, params)

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..world.state import WorldState

MAX_OBSERVATION_TOKENS = 1800
MAX_HISTORY_IN_OBS = 4
NARRATIVE_MAX_CHARS = 400


def format_observation(
    world_state: WorldState,
    task: Any,
    step: int,
    parse_error: Optional[List[str]] = None,
) -> Dict[str, Any]:
    summary = world_state.to_summary_dict()

    employee_lines = "\n".join(
        f"  {employee['role']} ({employee['id']}): trust={employee['trust']}, {employee['availability']}"
        for employee in summary["active_employees"]
    )
    if not employee_lines:
        employee_lines = "  (none)"

    project_lines = "\n".join(
        f"  {project['id']}: momentum={project['momentum']}, pressure={project['deadline_pressure']}, committed={'YES' if project['external_commitment'] else 'no'}"
        for project in summary["projects"]
    )
    if not project_lines:
        project_lines = "  (none)"

    recent_actions = "\n".join(
        f"  Step {record['step']}: {record['action']} (R{record['r_level']})"
        for record in summary["recent_actions"][-MAX_HISTORY_IN_OBS:]
    )
    if not recent_actions:
        recent_actions = "  (none yet)"

    locked_map = summary["locked_actions"]
    locked_str = ", ".join(f"{action_id}: {reason}" for action_id, reason in locked_map.items()) or "none"

    available_actions_lines = "\n".join(
        f"  {action_id}"
        + (
            f" [LOCKED: {world_state.locked_actions.get(action_id, 'reason unavailable')}]"
            if action_id in world_state.locked_actions
            else ""
        )
        for action_id in getattr(task, "available_actions", [])
    )
    if not available_actions_lines:
        available_actions_lines = "  (none)"

    error_block = (
        "\nPARSE ERROR (fix your output format): " + "; ".join(parse_error) + "\n"
        if parse_error
        else ""
    )

    narrative = getattr(task, "narrative", "").strip()
    narrative_short = narrative[:NARRATIVE_MAX_CHARS]
    if len(narrative) > NARRATIVE_MAX_CHARS:
        narrative_short += "..."

    text = (
        f"=== MERIDIAN - Step {step + 1}/{getattr(task, 'max_steps', 0)} | Task: {getattr(task, 'name', 'unknown')} ===\n"
        f"{narrative_short}\n\n"
        f"TEAM:\n{employee_lines}\n\n"
        f"PROJECTS:\n{project_lines}\n\n"
        f"EXTERNAL: board_trust={summary['board_trust']} | public_commitments={summary['public_commitments_count']}\n"
        f"  Last: {summary['last_public_commitment']}\n\n"
        f"RECENT ACTIONS (last {MAX_HISTORY_IN_OBS}):\n{recent_actions}\n\n"
        f"LOCKED: {locked_str}\n\n"
        f"AVAILABLE ACTIONS:\n{available_actions_lines}\n"
        f"{error_block}"
    )

    if len(text) / 4 > MAX_OBSERVATION_TOKENS:
        short_employee_lines = "\n".join(
            f"  {employee['role']} ({employee['id']}): trust={employee['trust']}"
            for employee in summary["active_employees"][:2]
        )
        if short_employee_lines:
            remainder = max(0, len(summary["active_employees"]) - 2)
            short_employee_lines += f"\n  ...and {remainder} more"
        else:
            short_employee_lines = "  (none)"
        text = text.replace(employee_lines, short_employee_lines)

    return {
        "text": text,
        "step": step,
        "task_id": getattr(task, "task_id", "unknown"),
        "available_actions": ",".join(getattr(task, "available_actions", [])),
    }

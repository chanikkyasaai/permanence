"""
PERMANENCE — Pydantic action and observation models for OpenEnv compliance.

These models define the typed interface between the environment server
and any client that connects via the OpenEnv protocol.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PermanenceAction(BaseModel):
    """
    A single agent action step.

    The agent produces free-form text containing:
    - A <thinking>...</thinking> reasoning block
    - An <action id="..." param1="..." .../> tag
    - A <reversibility level="R1-R5" confidence="0.0-1.0"/> tag

    The environment parses these tags internally. The full raw text
    is passed as the `text` field.
    """
    text: str = Field(
        ...,
        description="Agent's complete free-form response including thinking, action, and reversibility tags",
        min_length=1,
        max_length=8192,
    )


class PermanenceObservation(BaseModel):
    """
    Environment observation returned after each step or on reset.
    """
    text: str = Field(
        ...,
        description="Formatted world state observation text presented to the agent",
    )
    step: int = Field(
        ...,
        description="Current step number within the episode (0-indexed)",
        ge=0,
    )
    task_id: str = Field(
        ...,
        description="Identifier of the current task",
    )
    available_actions: str = Field(
        ...,
        description="Comma-separated list of action IDs available in this task",
    )
    reward: Optional[float] = Field(
        default=None,
        description="Episode reward (non-null only on terminal steps)",
    )
    done: bool = Field(
        default=False,
        description="True when the episode has terminated or been truncated",
    )
    info: Dict = Field(
        default_factory=dict,
        description="Additional diagnostic information",
    )


class PermanenceState(BaseModel):
    """
    Episode-level state metadata (returned by /state endpoint).
    """
    episode_id: str = Field(..., description="Unique identifier for the current episode")
    step_count: int = Field(..., description="Number of steps taken so far")
    task_id: str = Field(..., description="Current task identifier")
    task_difficulty: int = Field(..., description="Task difficulty level 1-5")
    locked_actions: List[str] = Field(
        default_factory=list,
        description="Action IDs locked by prior irreversible choices this episode",
    )
    critical_options: Dict[str, bool] = Field(
        default_factory=dict,
        description="Tracked high-value future action paths and their availability",
    )
    terminated: bool = Field(default=False)
    truncated: bool = Field(default=False)
    termination_reason: Optional[str] = Field(default=None)


class ResetRequest(BaseModel):
    """Request body for POST /reset"""
    task_id: str = Field(
        default="task_correction",
        description="Task to initialize. One of: task_correction, task_conflict, task_launch, task_crisis, task_cascade",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible scenario generation. None = random.",
    )


class StepRequest(BaseModel):
    """Request body for POST /step"""
    action: PermanenceAction

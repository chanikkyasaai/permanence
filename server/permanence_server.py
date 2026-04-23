"""
PERMANENCE server-side environment.

Wraps the core PermanenceEnv (permanence/env.py) for OpenEnv server deployment.
The server handles session management, HTTP endpoints, and typed I/O.
The core env handles all world logic, reward computation, and episode management.
"""
from __future__ import annotations

import uuid
from typing import Optional, Tuple

from models import (
    PermanenceAction,
    PermanenceObservation,
    PermanenceState,
    ResetRequest,
)


class PermanenceServer:
    """
    OpenEnv-compatible server environment for PERMANENCE.

    Wraps the core PermanenceEnv. One instance = one session.
    Sessions are managed by the FastAPI app layer.
    """

    def __init__(self):
        self._env = None
        self._episode_id: str = ""
        self._initialized: bool = False
        self._load_env()

    def _load_env(self):
        """Lazy-load the core environment to avoid import overhead on module load."""
        try:
            from permanence.env import PermanenceEnv
            self._env = PermanenceEnv()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PermanenceEnv: {e}. "
                f"Ensure 'permanence' package is installed: pip install -e ."
            ) from e

    def reset(self, request: Optional[ResetRequest] = None) -> PermanenceObservation:
        """
        Start a new episode.

        Args:
            request: Optional reset parameters (task_id, seed).
                     Defaults to task_correction with random seed.

        Returns:
            Initial observation for the new episode.
        """
        if request is None:
            request = ResetRequest()

        self._episode_id = str(uuid.uuid4())[:8]

        obs_dict, info = self._env.reset(
            options={
                "task_id": request.task_id,
                "seed": request.seed,
            }
        )

        self._initialized = True

        return PermanenceObservation(
            text=obs_dict.get("text", ""),
            step=obs_dict.get("step", 0),
            task_id=obs_dict.get("task_id", request.task_id),
            available_actions=obs_dict.get("available_actions", ""),
            reward=None,
            done=False,
            info=info,
        )

    def step(self, action: PermanenceAction) -> PermanenceObservation:
        """
        Execute one agent action.

        Args:
            action: PermanenceAction with the agent's free-form text output.

        Returns:
            Next observation with reward and done flag.

        Raises:
            RuntimeError: If reset() has not been called first.
        """
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")

        obs_dict, reward, terminated, truncated, info = self._env.step(action.text)

        done = terminated or truncated

        return PermanenceObservation(
            text=obs_dict.get("text", ""),
            step=obs_dict.get("step", 0),
            task_id=obs_dict.get("task_id", ""),
            available_actions=obs_dict.get("available_actions", ""),
            reward=float(reward) if (terminated or truncated) else None,
            done=done,
            info={
                **info,
                "episode_id": self._episode_id,
                "terminated": terminated,
                "truncated": truncated,
            },
        )

    def state(self) -> PermanenceState:
        """
        Return current episode metadata without advancing the episode.
        """
        if not self._initialized or self._env._current_world_state is None:
            return PermanenceState(
                episode_id=self._episode_id or "not_started",
                step_count=0,
                task_id="none",
                task_difficulty=0,
            )

        ws = self._env._current_world_state
        task = self._env._current_task

        return PermanenceState(
            episode_id=self._episode_id,
            step_count=ws.episode_step,
            task_id=ws.task_id,
            task_difficulty=task.difficulty if task else 0,
            locked_actions=sorted(ws.locked_actions),
            critical_options=dict(ws.critical_options),
            terminated=False,
            truncated=False,
        )

    def close(self):
        """Clean up resources. Called when the session ends."""
        self._initialized = False

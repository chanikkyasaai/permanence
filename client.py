"""
PERMANENCE — OpenEnv-compatible client.

Usage:
    from client import PermanenceEnvClient, PermanenceAction

    with PermanenceEnvClient(base_url="https://chane35-permanence.hf.space").sync() as env:
        obs = env.reset()
        result = env.step(PermanenceAction(text="<action id='draft_internal_memo'/>"))
        print(result.observation.text)

Or for local development:
    with PermanenceEnvClient(base_url="http://localhost:7860").sync() as env:
        ...
"""
from __future__ import annotations

import os
from typing import Optional

try:
    from openenv.core.client import EnvClient
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False

from models import PermanenceAction, PermanenceObservation, PermanenceState

DEFAULT_ENV_URL = os.getenv(
    "PERMANENCE_ENV_URL",
    "https://chane35-permanence.hf.space",
)


if _OPENENV_AVAILABLE:
    class PermanenceEnvClient(EnvClient):
        """
        Typed OpenEnv client for the PERMANENCE environment.

        Connects to a running PERMANENCE server via WebSocket.
        Provides typed reset() and step() methods.
        """
        action_type = PermanenceAction
        observation_type = PermanenceObservation
        state_type = PermanenceState

        def __init__(self, base_url: str = DEFAULT_ENV_URL):
            super().__init__(base_url=base_url)

else:
    # Fallback HTTP client when openenv-core is not installed
    import requests

    class PermanenceEnvClient:  # type: ignore[no-redef]
        """
        Fallback HTTP client for PERMANENCE.
        Used when openenv-core is not installed.
        Install openenv-core for full WebSocket support.
        """

        def __init__(self, base_url: str = DEFAULT_ENV_URL):
            self.base_url = base_url.rstrip("/")
            self._session: Optional[requests.Session] = None

        def __enter__(self):
            self._session = requests.Session()
            return self

        def __exit__(self, *args):
            if self._session:
                self._session.close()

        def sync(self):
            return self

        def reset(self, task_id: str = "task_correction", seed: Optional[int] = None) -> PermanenceObservation:
            payload = {"task_id": task_id}
            if seed is not None:
                payload["seed"] = seed
            r = self._session.post(f"{self.base_url}/reset", json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            return PermanenceObservation(**data)

        def step(self, action: PermanenceAction) -> "StepResult":
            r = self._session.post(
                f"{self.base_url}/step",
                json={"action": {"text": action.text}},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            return _StepResult(observation=PermanenceObservation(**data))

        def state(self) -> PermanenceState:
            r = self._session.get(f"{self.base_url}/state", timeout=10)
            r.raise_for_status()
            return PermanenceState(**r.json())

        def close(self):
            pass


class _StepResult:
    def __init__(self, observation: PermanenceObservation):
        self.observation = observation
        self.reward = observation.reward
        self.done = observation.done

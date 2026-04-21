from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

STATE_PATH = Path(__file__).with_name("dashboard") / "current_state.json"
GHOST_RECORDING_PATH = Path(__file__).with_name("ghost_recording.json")
GHOST_STEP_DELAY_SECONDS = 2.0

GHOST_MODE = False
GHOST_START_TS = 0.0
GHOST_STATES: list[Dict[str, Any]] = []

DEFAULT_STATE: Dict[str, Any] = {
    "recent_actions": [],
    "locked_actions": {},
    "critical_options": {},
    "catastrophe_rate": [],
    "raw_thinking": "",
}


def _load_ghost_recording(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(raw, list):
        return []

    frames: list[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        frame = dict(DEFAULT_STATE)
        for key in frame:
            if key in item:
                frame[key] = item[key]
        for passthrough_key in ["episode", "episode_data"]:
            if passthrough_key in item:
                frame[passthrough_key] = item[passthrough_key]
        frames.append(frame)
    return frames


def _ghost_state_snapshot() -> Dict[str, Any]:
    if not GHOST_STATES:
        return dict(DEFAULT_STATE)

    elapsed = max(0.0, time.time() - GHOST_START_TS)
    index = min(int(elapsed // GHOST_STEP_DELAY_SECONDS), len(GHOST_STATES) - 1)
    return dict(GHOST_STATES[index])


def _load_state() -> Dict[str, Any]:
    if GHOST_MODE:
        return _ghost_state_snapshot()

    if not STATE_PATH.exists():
        return dict(DEFAULT_STATE)

    try:
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_STATE)

    state = dict(DEFAULT_STATE)
    if isinstance(raw, dict):
        for key in state:
            if key in raw:
                state[key] = raw[key]
    return state


@app.get("/api/state")
def api_state() -> Any:
    return jsonify(_load_state())


@app.get("/")
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "state_path": str(STATE_PATH),
            "ghost_mode": GHOST_MODE,
            "ghost_frames": len(GHOST_STATES),
            "ghost_delay_seconds": GHOST_STEP_DELAY_SECONDS,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PERMANENCE dashboard backend")
    parser.add_argument("--ghost", action="store_true", help="Serve ghost recording playback instead of live state file.")
    parser.add_argument("--ghost-file", default=str(GHOST_RECORDING_PATH), help="Path to ghost recording JSON array.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.ghost:
        GHOST_MODE = True
        GHOST_STATES = _load_ghost_recording(Path(args.ghost_file))
        GHOST_START_TS = time.time()
    app.run(host=args.host, port=args.port, debug=args.debug)

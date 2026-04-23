"""
PERMANENCE — Pre-submission validation script.

Run this before every git push to catch issues early.

Usage:
    python validate_submission.py

All checks must pass before the repo is submitted.
"""
import sys
import subprocess

passed = []
failed = []

def OK(msg): passed.append(msg); print(f"  ✓  {msg}")
def FAIL(msg, detail=""): failed.append(msg); print(f"  ✗  {msg}" + (f": {detail}" if detail else ""))


print("=" * 65)
print("PERMANENCE SUBMISSION VALIDATION")
print("=" * 65)

# ── 1. Required files exist ──────────────────────────────────────
print("\n[1] Required files")
import pathlib

required_files = [
    "openenv.yaml",
    "pyproject.toml",
    "README.md",
    "models.py",
    "client.py",
    "server/__init__.py",
    "server/permanence_server.py",
    "server/app.py",
    "server/Dockerfile",
    "server/requirements.txt",
    "training/train_trl.py",
    "training/reward_functions.py",
    "training/config.yaml",
    "permanence/env.py",
    "interactive_eval.py",
    "export_ghost_demo.py",
    "app.py",
]

for f in required_files:
    if pathlib.Path(f).exists():
        OK(f)
    else:
        FAIL(f"MISSING: {f}")

# ── 2. openenv.yaml fields ───────────────────────────────────────
print("\n[2] openenv.yaml")
try:
    import yaml
    spec = yaml.safe_load(pathlib.Path("openenv.yaml").read_text())
    OK("openenv.yaml parses") if spec else FAIL("openenv.yaml empty")
    OK("author: chanikya") if spec.get("author") == "chanikya" else FAIL(f"author is '{spec.get('author')}' not 'chanikya'")
    OK("spec_version present") if "spec_version" in spec else FAIL("spec_version missing")
    OK("entry_point present") if "entry_point" in spec else FAIL("entry_point missing")
    OK("app block present") if "app" in spec else FAIL("app block missing")
    OK("5 tasks defined") if len(spec.get("tasks", [])) == 5 else FAIL(f"Expected 5 tasks, got {len(spec.get('tasks', []))}")
    OK("tags include openenv") if "openenv" in spec.get("tags", []) else FAIL("openenv tag missing")
except Exception as e:
    FAIL(f"openenv.yaml error: {e}")

# ── 3. pyproject.toml ────────────────────────────────────────────
print("\n[3] pyproject.toml")
try:
    import tomllib
    d = tomllib.load(open("pyproject.toml", "rb"))
    author = d["project"]["authors"][0].get("name", "")
    OK("author: Chanikya") if author == "Chanikya" else FAIL(f"author is '{author}' not 'Chanikya'")
except Exception as e:
    FAIL(f"pyproject.toml error: {e}")

# ── 4. README has HF frontmatter ─────────────────────────────────
print("\n[4] README.md HuggingFace frontmatter")
try:
    readme = pathlib.Path("README.md").read_text()
    OK("Starts with ---") if readme.startswith("---") else FAIL("README must start with --- (HF frontmatter)")
    OK("sdk: docker") if "sdk: docker" in readme else FAIL("sdk: docker missing from frontmatter")
    OK("openenv tag") if "openenv" in readme[:500] else FAIL("openenv tag missing from frontmatter")
except Exception as e:
    FAIL(f"README error: {e}")

# ── 5. Models import correctly ───────────────────────────────────
print("\n[5] models.py")
try:
    from models import PermanenceAction, PermanenceObservation, PermanenceState, ResetRequest, StepRequest
    OK("All model classes import")
    a = PermanenceAction(text="test")
    OK("PermanenceAction instantiates")
    o = PermanenceObservation(text="obs", step=0, task_id="task_correction", available_actions="a,b")
    OK("PermanenceObservation instantiates")
except Exception as e:
    FAIL(f"models.py error: {e}")

# ── 6. Server app endpoints ──────────────────────────────────────
print("\n[6] server/app.py endpoints")
try:
    from fastapi.testclient import TestClient
    from server.app import app
    client = TestClient(app)

    r = client.get("/health")
    OK(f"/health returns 200") if r.status_code == 200 else FAIL(f"/health returns {r.status_code}: {r.text}")

    r = client.post("/reset", json={})
    OK("/reset with empty body returns 200") if r.status_code == 200 else FAIL(f"/reset{{}} returns {r.status_code}: {r.text[:200]}")

    r = client.get("/state")
    OK(f"/state returns 200") if r.status_code == 200 else FAIL(f"/state returns {r.status_code}")

except Exception as e:
    FAIL(f"server/app.py error: {e}")

# ── 7. Dockerfile has required fields ────────────────────────────
print("\n[7] server/Dockerfile")
try:
    df = pathlib.Path("server/Dockerfile").read_text()
    OK("FROM python") if "FROM python" in df else FAIL("Missing FROM python")
    OK("EXPOSE 7860") if "7860" in df else FAIL("Missing EXPOSE 7860")
    OK("ENV PYTHONPATH=/app") if "PYTHONPATH=/app" in df else FAIL("Missing ENV PYTHONPATH=/app")
    OK("HEALTHCHECK") if "HEALTHCHECK" in df else FAIL("Missing HEALTHCHECK")
    OK("uvicorn CMD") if "uvicorn" in df and "CMD" in df else FAIL("Missing uvicorn CMD")
except Exception as e:
    FAIL(f"Dockerfile error: {e}")

# ── 8. Core env imports ──────────────────────────────────────────
print("\n[8] permanence package")
try:
    from permanence.env import PermanenceEnv
    env = PermanenceEnv()
    obs, info = env.reset()
    OK("PermanenceEnv.reset() works")
    assert "text" in obs, f"obs missing 'text': {obs}"
    OK("reset() returns obs with text field")
    _, reward, terminated, truncated, info = env.step("<action id='draft_internal_memo'/><reversibility level='R1' confidence='0.9'/>")
    OK("PermanenceEnv.step() works")
except Exception as e:
    FAIL(f"permanence env error: {e}")

# ── 9. Training imports ──────────────────────────────────────────
print("\n[9] training modules")
try:
    from training.reward_functions import reward_format, reward_prediction_accuracy, reward_no_catastrophe
    scores = reward_format(["<action id='x'/><reversibility level='R1' confidence='0.5'/>"])
    assert scores[0] == 1.0, f"Expected 1.0, got {scores[0]}"
    OK("reward_format works")
    import training.train_trl
    OK("train_trl.py imports")
except Exception as e:
    FAIL(f"training error: {e}")

# ── FINAL RESULT ─────────────────────────────────────────────────
print()
print("=" * 65)
n_ok, n_fail = len(passed), len(failed)
print(f"RESULTS: {n_ok} PASSED | {n_fail} FAILED")
print("=" * 65)
if n_fail > 0:
    print("\nFAILED CHECKS:")
    for f in failed:
        print(f"  ✗ {f}")
    print("\nFix all failures before pushing.")
    sys.exit(1)
else:
    print("\n✓ ALL CHECKS PASSED — REPO IS SUBMISSION-READY")
    sys.exit(0)

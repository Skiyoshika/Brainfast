"""Scenario D: wizard launch → pipeline path consistency.

The dual-convention bug I fixed earlier hinged on where different code
paths write outputs (jobs/<id>/ vs <id>/). This test verifies that
Wizard launch writes to a location that the Liquify tab can then find,
without re-running a full pipeline.

Steps:
  1. Inspect a known directory of TIFF slices.
  2. Launch pipeline via wizard with a fresh sample id.
  3. Immediately after launch (before pipeline finishes), check that
     the wizard-generated runtime config is where Liquify tab would
     look for it.
  4. Verify state endpoint for the same sample id returns guidance
     until the pipeline has actually written an annotation (i.e. the
     empty-state path still works with wizard-launched jobs).
  5. Kill the pipeline (it would take 30-60 min to finish).

Takes ~30-60 seconds.
"""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib import error, request

BASE = "http://127.0.0.1:8787"
SLICE_DIR = r"D:/Brainfast/project/data/35_C0_demo"   # existing TIFF directory
WIZARD_SAMPLE_ID = "wiztest-SIM"


def _req(method: str, path: str, body=None, *, timeout=30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(body)
        except Exception:
            pass
        return e.code, body


passes = 0
fails: list[str] = []


def check(name, ok, detail=""):
    global passes
    if ok:
        passes += 1
        print(f"  ✓ {name}" + (f" ({detail})" if detail else ""))
    else:
        fails.append(f"{name}: {detail}")
        print(f"  ✗ {name}: {detail}")


# Clean slate
for candidate in [
    Path(r"D:/Brainfast/project/outputs") / WIZARD_SAMPLE_ID,
    Path(r"D:/Brainfast/project/outputs/jobs") / WIZARD_SAMPLE_ID,
]:
    if candidate.exists():
        shutil.rmtree(candidate, ignore_errors=True)

# ---- 1. Inspect ----
print("\n=== Inspect source ===")
s, b = _req("POST", "/api/wizard/inspect-source", {"sourcePath": SLICE_DIR})
check(
    "inspect directory with 111 TIFFs → kind=directory n_files=111",
    s == 200 and b.get("kind") == "directory" and b.get("n_files") == 111,
    f"status={s} kind={b.get('kind')} n_files={b.get('n_files')}",
)
check(
    "suggested_pixel_um_xy=5.0 (default for directories w/o metadata)",
    b.get("suggested_pixel_um_xy") == 5.0,
)
check(
    f"suggested_sample_id='{Path(SLICE_DIR).name}' (dir name)",
    b.get("suggested_sample_id") == Path(SLICE_DIR).name,
)

# ---- 2. Launch ----
print("\n=== Launch wizard pipeline ===")
s, b = _req(
    "POST", "/api/wizard/launch",
    {
        "sampleId": WIZARD_SAMPLE_ID,
        "inputDir": SLICE_DIR,
        "pixelSizeUm": 5.0,
        "zSpacingUm": 25.0,
        "channels": ["red"],
        "atlasHemisphere": "right_flipped",
    },
)
check(
    f"launch → ok, jobId='{WIZARD_SAMPLE_ID}'",
    s == 200 and b.get("jobId") == WIZARD_SAMPLE_ID,
    f"status={s} body={b}",
)

outputs_dir = Path(b["outputs_dir"]) if b.get("outputs_dir") else None
config_path = Path(b["config_path"]) if b.get("config_path") else None
check(
    "outputs_dir returned by wizard exists on disk",
    outputs_dir is not None and outputs_dir.exists(),
    f"outputs_dir={outputs_dir}",
)
check(
    "config_path written (runtime_configs/run_config_*.json)",
    config_path is not None and config_path.exists(),
    f"config_path={config_path}",
)

# ---- 3. Path-consistency check: does Liquify tab see this job? ----
print("\n=== Liquify tab state on the wizard-launched job ===")
# Wait a tick for the pipeline thread to start (it may create tmp dirs)
time.sleep(2)
s, b = _req("GET", f"/api/liquify-3d/state?job={WIZARD_SAMPLE_ID}")
# Pipeline hasn't produced ants_registration yet (35-60 min to complete),
# so guidance SHOULD say "run pipeline first". The crucial thing is the
# endpoint resolves the job without returning 404 or server error.
check(
    "liquify state for wizard job → 200 ok",
    s == 200 and b.get("ok") is True,
    f"status={s} body={b}",
)
check(
    "state source_annotation_available=False (pipeline still running)",
    b.get("source_annotation_available") is False,
    f"source_available={b.get('source_annotation_available')}",
)
check(
    "state.guidance is a non-empty string (user-facing empty-state help)",
    isinstance(b.get("guidance"), str) and "ANTs" in b["guidance"],
    f"guidance={b.get('guidance')!r}",
)

# ---- 4. Pipeline status via existing endpoint ----
print("\n=== Pipeline status endpoint reports progress ===")
s, b = _req("GET", f"/api/status?jobId={WIZARD_SAMPLE_ID}")
check(
    "GET /api/status for wizard job → 200 (pipeline thread is running)",
    s == 200,
    f"status={s}",
)
# The status endpoint structure varies; main thing is it doesn't 500
# and carries some indicator that our job is active
check(
    "status payload carries info (not empty)",
    isinstance(b, dict) and len(b) > 0,
    f"status keys: {list(b.keys()) if isinstance(b, dict) else 'n/a'}",
)

# ---- 5. Cancel + cleanup ----
print("\n=== Cancel the wizard-launched pipeline ===")
s, b = _req("POST", "/api/cancel", {"jobId": WIZARD_SAMPLE_ID})
check(
    "POST /api/cancel → 2xx",
    200 <= s < 300 or s == 409,   # 409 if already finished
    f"status={s} body={b}",
)

# Give it a couple seconds to die gracefully
time.sleep(3)

# Clean up the scratch outputs
for candidate in [
    Path(r"D:/Brainfast/project/outputs") / WIZARD_SAMPLE_ID,
    Path(r"D:/Brainfast/project/outputs/jobs") / WIZARD_SAMPLE_ID,
]:
    if candidate.exists():
        shutil.rmtree(candidate, ignore_errors=True)
        print(f"  cleaned up {candidate}")

print(f"\n=== Summary: {passes} pass / {len(fails)} fail ===")
if fails:
    print("\nFAILURES:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
sys.exit(0)

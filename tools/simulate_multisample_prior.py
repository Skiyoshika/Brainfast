"""Scenario C: multi-sample class-prior end-to-end validation.

Simulates 3 users (or 3 samples) each correcting a few landmarks, saving
to the same class, and then a 4th sample warm-starting from the
accumulated prior.

Does not run Apply/Finalize (that's validated separately and takes
5-10 min per job). Focuses on the γ math: running-mean merge across
samples + warm-start apply.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from urllib import error, request

BASE = "http://127.0.0.1:8787"
CLASS = "UXTestClass"
PRIORS_ROOT = Path(r"D:/Brainfast/project/train_data_set/class_priors")
JOBS = ["35_C0_test_run2", "ab_test_mlflip_false", "ab_test_mlflip_true"]
NEW_JOB = "ux-test-warm-start-target"


def _req(method: str, path: str, body=None, *, timeout=30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as r:
            ct = r.headers.get("Content-Type", "")
            raw = r.read().decode("utf-8")
            return r.status, json.loads(raw) if ct.startswith("application/json") else raw
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


# ----- Clean slate -----
print(f"\n=== Clean slate: remove prior for '{CLASS}' ===")
cls_dir = PRIORS_ROOT / CLASS
if cls_dir.exists():
    shutil.rmtree(cls_dir)
    print(f"  removed {cls_dir}")
# Clear landmarks on each job + on the new-job scratch area
for j in JOBS + [NEW_JOB]:
    _req("POST", "/api/liquify-3d/clear", {"jobId": j})

# ----- Status: class prior non-existent -----
print(f"\n=== Class prior 'UXTestClass' starts empty ===")
status, body = _req("GET", f"/api/liquify-3d/class-prior/status?class={CLASS}")
check(
    "status before any contributions → sample_count=0 ready=False",
    status == 200 and body["sample_count"] == 0 and body["ready_for_warm_start"] is False,
    f"body={body}",
)

# ----- Save 3 samples -----
print("\n=== 3 samples each contribute 3 landmarks to the class ===")
# Shared atlas positions across the 3 jobs so merging actually happens
shared_pairs = [
    (50, 100, 120, 105, 125),  # z, ay, ax, ry, rx
    (70, 150, 140, 155, 143),
    (90, 200, 180, 204, 184),
]
expected_sample_count = 0
for i, jid in enumerate(JOBS):
    print(f"\n-- sample {i+1}: {jid} --")
    for z, ay, ax, ry, rx in shared_pairs:
        s, b = _req(
            "POST", "/api/liquify-3d/add-pair",
            {"jobId": jid, "z": z, "atlas": [ay, ax], "real": [ry, rx]},
        )
        assert s == 200, b
    # Save
    s, b = _req(
        "POST", "/api/liquify-3d/class-prior/save",
        {"jobId": jid, "class": CLASS, "sampleId": jid},
    )
    expected_sample_count += 1
    check(
        f"save job={jid} → sample_count={expected_sample_count}",
        s == 200 and b["sample_count"] == expected_sample_count
        and b["merged_pair_count"] == 3,
        f"body={b}",
    )
    # 3 shared pairs should merge into 3 entries (not 9) after all 3 samples
    expected_entries = 3
    check(
        f"  entry_count stays 3 (pairs at same voxel merge across samples)",
        b["entry_count"] == expected_entries,
        f"entry_count={b['entry_count']} (expected {expected_entries})",
    )
    # Readiness flips at 3
    should_be_ready = expected_sample_count >= 3
    check(
        f"  ready_for_warm_start={should_be_ready} (MIN_SAMPLES=3)",
        b["ready_for_warm_start"] is should_be_ready,
    )

# ----- Coverage heatmap data -----
print("\n=== Coverage heatmap after 3 samples ===")
s, b = _req("GET", f"/api/liquify-3d/class-prior/coverage?class={CLASS}")
check(
    "coverage lists 3 entries (z=50, 70, 90) all with count=3",
    s == 200
    and len(b["bins"]) == 3
    and all(row["count"] == 3 for row in b["bins"])
    and sorted(row["z"] for row in b["bins"]) == [50, 70, 90],
    f"bins={b['bins']}",
)

# ----- Warm-start a fresh job -----
print(f"\n=== Warm-start '{NEW_JOB}' from '{CLASS}' prior ===")
s, b = _req(
    "POST", "/api/liquify-3d/class-prior/apply-warm-start",
    {"jobId": NEW_JOB, "class": CLASS},
)
check(
    f"apply-warm-start → ok, pair_count=3",
    s == 200 and b.get("pair_count") == 3,
    f"status={s} body={b}",
)

# ----- Verify state shows 3 pairs with mean values -----
s, b = _req("GET", f"/api/liquify-3d/state?job={NEW_JOB}")
pairs = b.get("pairs", [])
check("state after warm-start shows 3 pairs", len(pairs) == 3)
# Mean dy across 3 samples at (50, 100, 120): all 3 samples had dy=5, so mean_dy=5
# (shared_pairs rows are fixed: ry-ay = 105-100 = 5)
first = sorted(pairs, key=lambda p: p["z"])[0]
check(
    "first warm-start pair has expected atlas + real (mean matches input)",
    first["atlas_y"] == 100.0 and first["atlas_x"] == 120.0
    and abs(first["real_y"] - 105.0) < 0.1 and abs(first["real_x"] - 125.0) < 0.1,
    f"first pair={first}",
)

# ----- Refuse overwrite without force -----
print("\n=== Re-apply warm-start should refuse when pairs already present ===")
s, b = _req(
    "POST", "/api/liquify-3d/class-prior/apply-warm-start",
    {"jobId": NEW_JOB, "class": CLASS},
)
check(
    "second apply without force → 409 Conflict",
    s == 409,
    f"status={s} body={b}",
)

s, b = _req(
    "POST", "/api/liquify-3d/class-prior/apply-warm-start",
    {"jobId": NEW_JOB, "class": CLASS, "force": True},
)
check(
    "force=true → 200 overwrites",
    s == 200 and b.get("pair_count") == 3,
    f"status={s}",
)

# ----- Same-sample double-save — duplicate or merge? -----
print("\n=== Edge case: re-save same sample to same class ===")
s, b = _req(
    "POST", "/api/liquify-3d/class-prior/save",
    {"jobId": JOBS[0], "class": CLASS, "sampleId": JOBS[0]},
)
# After re-save, sample_count goes 3 → 4 (no dedup by sample_id), and n
# on each entry goes 3 → 4 at the shared voxels.
check(
    "re-saving same job duplicates the contribution (no dedup)",
    s == 200 and b["sample_count"] == 4,
    f"sample_count={b['sample_count']} (note: no sample-id dedup)",
)

s, b = _req("GET", f"/api/liquify-3d/class-prior/coverage?class={CLASS}")
# Each bin count should now be 4 (3 distinct samples + the duplicate)
duplicated = all(row["count"] == 4 for row in b["bins"])
if not duplicated:
    # Note as observation, not necessarily a bug
    print(f"  note: re-save did not uniformly increment bin counts → {b['bins']}")
check(
    "coverage reflects re-save: every bin now has count=4",
    duplicated,
    f"bins={b['bins']}",
)

# ----- Cleanup: remove UXTestClass so production priors stay clean -----
print(f"\n=== Cleanup: remove '{CLASS}' prior ===")
cls_dir = PRIORS_ROOT / CLASS
if cls_dir.exists():
    shutil.rmtree(cls_dir)
# Clear landmarks
for j in JOBS + [NEW_JOB]:
    _req("POST", "/api/liquify-3d/clear", {"jobId": j})

print(f"\n=== Summary: {passes} pass / {len(fails)} fail ===")
if fails:
    print("\nFAILURES:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
sys.exit(0)

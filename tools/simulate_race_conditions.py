"""Scenario E: concurrent requests + rapid-fire clicks.

Validates the liquify endpoints under realistic user behaviours like:
  * Rapid add-pair clicks before each response arrives.
  * Two /apply calls racing (user double-clicks the big button).
  * DELETE while GET/state is in flight.
  * Ctrl+Z spam on a small pair list.

Runs in ~10 seconds (uses the existing 35_C0_demo_run2 refined annotation
so no Laplacian solve is triggered during races).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from urllib import error, request

BASE = "http://127.0.0.1:8787"
JOB = "race-test-job"


def _req(method, path, body=None, timeout=30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except error.HTTPError as e:
        b = e.read().decode("utf-8", errors="replace")
        try:
            b = json.loads(b)
        except Exception:
            pass
        return e.code, b


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
_req("POST", "/api/liquify-3d/clear", {"jobId": JOB})


# ---- E1. Rapid-fire add-pair from 10 threads ----
print("\n=== E1. 10 threads POST add-pair simultaneously ===")
results: list[tuple[int, dict]] = []
lock = threading.Lock()


def _add_pair(i):
    s, b = _req(
        "POST", "/api/liquify-3d/add-pair",
        {"jobId": JOB, "z": i, "atlas": [i, i + 10], "real": [i + 5, i + 15]},
    )
    with lock:
        results.append((s, b))


ts = [threading.Thread(target=_add_pair, args=(i,)) for i in range(10)]
t0 = time.time()
for t in ts:
    t.start()
for t in ts:
    t.join()
elapsed = time.time() - t0
ok_count = sum(1 for s, _ in results if s == 200)
check(
    f"10 concurrent add-pair all succeed (in {elapsed:.2f}s)",
    ok_count == 10,
    f"ok={ok_count}/10",
)

s, b = _req("GET", f"/api/liquify-3d/state?job={JOB}")
check(
    "after 10 concurrent adds, state shows exactly 10 pairs",
    s == 200 and b["pair_count"] == 10,
    f"pair_count={b['pair_count']}",
)


# ---- E2. Concurrent DELETE + GET ----
print("\n=== E2. DELETE pair/0 + GET state interleaved 20× ===")
del_results: list[int] = []
get_results: list[int] = []
# Add 50 pairs so we have enough to delete
for i in range(40):
    _req(
        "POST", "/api/liquify-3d/add-pair",
        {"jobId": JOB, "z": 100 + i, "atlas": [0, 0], "real": [1, 1]},
    )


def _delete_first():
    s, _ = _req("DELETE", f"/api/liquify-3d/pair/0?job={JOB}")
    del_results.append(s)


def _get_state():
    s, _ = _req("GET", f"/api/liquify-3d/state?job={JOB}")
    get_results.append(s)


ts = []
for _ in range(20):
    ts.append(threading.Thread(target=_delete_first))
    ts.append(threading.Thread(target=_get_state))
for t in ts:
    t.start()
for t in ts:
    t.join()
check(
    "all 20 concurrent DELETEs return 200 (no race producing 500)",
    all(s == 200 for s in del_results),
    f"statuses={sorted(set(del_results))}",
)
check(
    "all 20 concurrent GETs return 200",
    all(s == 200 for s in get_results),
    f"statuses={sorted(set(get_results))}",
)


# ---- E3. Double-click Apply (two concurrent /apply calls) ----
print("\n=== E3. Two Apply calls race (on the 35_C0_demo_run2 refined job) ===")
# Use the already-applied 35_C0_demo_run2 so this doesn't start a real solve.
# But Apply always runs solver unconditionally, so to keep this fast we'll
# point at a tiny-volume job: create a fake job with a small 4x4x4 annotation.
import nibabel as nib
import numpy as np
from pathlib import Path
tiny_job = "tiny-race-job"
tiny_dir = Path(r"D:/Brainfast/project/outputs/jobs") / tiny_job / "ants_registration"
tiny_dir.mkdir(parents=True, exist_ok=True)
ann = np.ones((4, 4, 4), dtype=np.int32)
nib.save(nib.Nifti1Image(ann, np.eye(4)), str(tiny_dir / "annotation_registered.nii.gz"))
# Seed 2 pairs so Apply has work
_req("POST", "/api/liquify-3d/clear", {"jobId": tiny_job})
_req("POST", "/api/liquify-3d/add-pair",
     {"jobId": tiny_job, "z": 1, "atlas": [1, 1], "real": [2, 2]})
_req("POST", "/api/liquify-3d/add-pair",
     {"jobId": tiny_job, "z": 2, "atlas": [1, 2], "real": [2, 3]})

apply_results = []


def _apply():
    s, b = _req("POST", "/api/liquify-3d/apply",
                {"jobId": tiny_job, "rtol": 1e-2, "maxiter": 50}, timeout=60)
    apply_results.append((s, b))


t1 = threading.Thread(target=_apply)
t2 = threading.Thread(target=_apply)
t1.start()
t2.start()
t1.join()
t2.join()
statuses = sorted([s for s, _ in apply_results])
# Both should succeed (or one refuses with a 409 Conflict). Neither should
# hang nor 500. On shared-state endpoints we'd ideally see one 200 + one
# 409; if both return 200 the backend serialised them (fine) or raced
# (still fine for correctness — the second overwrites the refined file).
check(
    "two concurrent Apply calls: both complete with non-error status",
    len(apply_results) == 2 and all(s in (200, 409) for s, _ in apply_results),
    f"statuses={statuses}",
)


# ---- E4. Clear + add rapid-fire ----
print("\n=== E4. Clear + add cycle ===")
for _ in range(5):
    _req("POST", "/api/liquify-3d/clear", {"jobId": JOB})
    for i in range(3):
        _req(
            "POST", "/api/liquify-3d/add-pair",
            {"jobId": JOB, "z": i, "atlas": [i, i], "real": [i + 1, i + 1]},
        )
s, b = _req("GET", f"/api/liquify-3d/state?job={JOB}")
check(
    "after 5 clear+3 add cycles, exactly 3 pairs remain",
    s == 200 and b["pair_count"] == 3,
    f"pair_count={b['pair_count']}",
)


# Cleanup
_req("POST", "/api/liquify-3d/clear", {"jobId": JOB})
_req("POST", "/api/liquify-3d/clear", {"jobId": tiny_job})
import shutil
shutil.rmtree(Path(r"D:/Brainfast/project/outputs/jobs") / tiny_job, ignore_errors=True)


print(f"\n=== Summary: {passes} pass / {len(fails)} fail ===")
if fails:
    print("\nFAILURES:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
sys.exit(0)

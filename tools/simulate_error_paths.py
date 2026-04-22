"""Scenario A + B: error paths + file serving sanity checks.

Hit every endpoint with payloads that should fail, and with legitimate
requests that should succeed, and assert the response shape matches the
frontend's expectations. Goal: catch UX bugs where an endpoint "works"
but the status code or payload leaves the UI confused.

Runs in seconds (no heavy compute).
"""

from __future__ import annotations

import json
import sys
from urllib import error, request

BASE = "http://127.0.0.1:8787"

passes: list[str] = []
fails: list[tuple[str, str]] = []
bugs: list[str] = []


def _req(method: str, path: str, body=None, *, timeout=30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8")) if r.headers.get(
                "Content-Type", ""
            ).startswith("application/json") else r.read()
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(body)
        except Exception:
            pass
        return e.code, body


def check(name: str, ok: bool, detail: str = ""):
    if ok:
        passes.append(name)
        print(f"  ✓ {name}" + (f" ({detail})" if detail else ""), flush=True)
    else:
        fails.append((name, detail))
        print(f"  ✗ {name}: {detail}", flush=True)


def section(title: str):
    print(f"\n=== {title} ===", flush=True)


# ---------------------------------------------------------------------------
section("A1. liquify apply / finalize with empty state")
# ---------------------------------------------------------------------------
status, body = _req("POST", "/api/liquify-3d/clear", {"jobId": "empty-job"})
check("clear a job that never had landmarks", status == 200)

status, body = _req("POST", "/api/liquify-3d/apply", {"jobId": "empty-job"})
check(
    "apply with 0 pairs → 400 with clear error",
    status == 400 and isinstance(body, dict) and "landmark" in body.get("error", "").lower(),
    f"status={status} body={body}",
)

status, body = _req("POST", "/api/liquify-3d/finalize", {"jobId": "empty-job"})
check(
    "finalize before Apply → 404 annotation_refined_liquify3d not found",
    status == 404 and "annotation_refined_liquify3d" in str(body.get("error", "")),
    f"status={status} body={body}",
)

# ---------------------------------------------------------------------------
section("A2. undo / pair removal edge cases")
# ---------------------------------------------------------------------------
status, body = _req("DELETE", "/api/liquify-3d/pair/0?job=empty-job")
check(
    "DELETE pair/0 on empty store → 400 IndexError surfaced",
    status == 400 and "out of range" in str(body.get("error", "")).lower(),
    f"status={status} body={body}",
)

status, body = _req("DELETE", "/api/liquify-3d/pair/99?job=35_C0_demo_run2")
check(
    "DELETE pair/99 when only 10 exist → 400",
    status == 400 and "out of range" in str(body.get("error", "")).lower(),
    f"status={status} body={body}",
)

# ---------------------------------------------------------------------------
section("A3. add-pair input validation")
# ---------------------------------------------------------------------------
status, body = _req("POST", "/api/liquify-3d/add-pair", {"jobId": "x"})
check(
    "add-pair with no z/real/atlas → 400",
    status == 400,
    f"status={status}",
)

status, body = _req(
    "POST", "/api/liquify-3d/add-pair",
    {"jobId": "x", "z": "not-int", "real": [1, 2], "atlas": [3, 4]},
)
check(
    "add-pair with z='not-int' → 400",
    status == 400,
    f"status={status}",
)

status, body = _req(
    "POST", "/api/liquify-3d/add-pair",
    {"jobId": "x", "z": 1, "real": [1], "atlas": [3, 4]},  # real short
)
check(
    "add-pair with malformed real coord → 400",
    status == 400,
    f"status={status}",
)

# ---------------------------------------------------------------------------
section("A4. class-prior guardrails")
# ---------------------------------------------------------------------------
status, body = _req(
    "POST", "/api/liquify-3d/class-prior/save",
    {"jobId": "no-pairs-job", "class": "TestClass"},
)
check(
    "save-to-prior with 0 pairs → 400",
    status == 400 and "pair" in str(body.get("error", "")).lower(),
    f"status={status} body={body}",
)

status, body = _req(
    "POST", "/api/liquify-3d/class-prior/save",
    {"jobId": "x"},  # missing class
)
check(
    "save-to-prior without class → 400",
    status == 400 and "class" in str(body.get("error", "")).lower(),
    f"status={status} body={body}",
)

status, body = _req(
    "POST", "/api/liquify-3d/class-prior/apply-warm-start",
    {"jobId": "new-job", "class": "NonexistentClass"},
)
check(
    "warm-start from empty class → 404",
    status == 404,
    f"status={status}",
)

# ---------------------------------------------------------------------------
section("A5. class-registry auto-detect misses")
# ---------------------------------------------------------------------------
status, body = _req("GET", "/api/liquify-3d/class-registry/detect?sampleId=nope_xx")
check(
    "detect on sample not matching any pattern → class=null",
    status == 200 and body.get("class") is None,
    f"status={status} body={body}",
)

status, body = _req("GET", "/api/liquify-3d/class-registry/detect")
check(
    "detect without sampleId → class=null (empty input)",
    status == 200 and body.get("class") is None,
    f"body={body}",
)

# ---------------------------------------------------------------------------
section("A6. wizard error paths")
# ---------------------------------------------------------------------------
status, body = _req("POST", "/api/wizard/inspect-source", {})
check("inspect-source without sourcePath → 400", status == 400)

status, body = _req(
    "POST", "/api/wizard/inspect-source",
    {"sourcePath": "D:/nowhere/never_existed.tif"},
)
check("inspect-source nonexistent → 404", status == 404)

status, body = _req("POST", "/api/wizard/launch", {"sampleId": "x"})
check(
    "launch without inputDir/spacings → 400 with 'missing ...' in error",
    status == 400 and "missing" in str(body.get("error", "")).lower(),
    f"status={status} body={body}",
)

status, body = _req(
    "POST", "/api/wizard/launch",
    {
        "sampleId": "x",
        "inputDir": "D:/nope/not_a_dir",
        "pixelSizeUm": 5.0,
        "zSpacingUm": 25.0,
    },
)
check("launch with nonexistent inputDir → 404", status == 404)

# ---------------------------------------------------------------------------
section("B1. file serving sanity")
# ---------------------------------------------------------------------------
status, body = _req("GET", "/api/outputs/reg-slice-list?job=35_C0_demo_run2")
check(
    "reg-slice-list job-scoped → returns list",
    status == 200 and isinstance(body, dict) and body.get("count", 0) == 111,
    f"count={body.get('count') if isinstance(body, dict) else '?'}",
)

# Fetch actual PNG
status, body = _req(
    "GET", "/api/outputs/reg-slice/slice_0050_overlay.png?job=35_C0_demo_run2"
)
ok = status == 200 and isinstance(body, (bytes, bytearray)) and body[:4] == b"\x89PNG"
check(
    "reg-slice/<file> returns valid PNG bytes",
    ok,
    f"status={status} is_png={ok}",
)

# Hierarchy after finalize — should now have real Allen regions
status, body = _req("GET", "/api/outputs/hierarchy?job=35_C0_demo_run2")
if isinstance(body, bytes):
    text = body.decode("utf-8", errors="replace")
    lines = text.strip().split("\n")
    ok = status == 200 and len(lines) > 100 and "root" in text
    check(
        "hierarchy served after finalize: ≥100 rows with real Allen region names",
        ok,
        f"status={status} lines={len(lines)}",
    )
else:
    check("hierarchy served", False, f"unexpected body type: {type(body)}")

# ---------------------------------------------------------------------------
section("B2. state endpoint coord shape")
# ---------------------------------------------------------------------------
status, body = _req("GET", "/api/liquify-3d/state?job=35_C0_demo_run2")
check(
    "state returns annotation_shape [111, 409, 340]",
    status == 200 and body.get("annotation_shape") == [111, 409, 340],
    f"annotation_shape={body.get('annotation_shape')}",
)
check(
    "state source_annotation_available=True after dual-convention fix",
    body.get("source_annotation_available") is True,
)
check(
    "state refined_annotation_exists=True (we applied earlier)",
    body.get("refined_annotation_exists") is True,
)
check(
    "state guidance=None when all good",
    body.get("guidance") is None,
)

# ---------------------------------------------------------------------------
section("B3. progress endpoint when nothing in flight")
# ---------------------------------------------------------------------------
status, body = _req("GET", "/api/liquify-3d/progress?job=fresh-nothing-running")
check(
    "progress endpoint on idle job → ok=True, no stage/percent keys",
    status == 200 and body.get("ok") and "percent" not in body,
    f"body={body}",
)

# ---------------------------------------------------------------------------
section("B4. QC status")
# ---------------------------------------------------------------------------
status, body = _req("GET", "/api/liquify-3d/qc-status?job=35_C0_demo_run2")
check(
    "qc-status after sign-off → done=True with metrics",
    status == 200 and body.get("done") is True and isinstance(body.get("metrics"), dict),
    f"body={body}",
)

status, body = _req("GET", "/api/liquify-3d/qc-status?job=never-signed-off")
check(
    "qc-status on fresh job → done=False",
    status == 200 and body.get("done") is False,
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n=== Summary: {len(passes)} pass, {len(fails)} fail ===")
if fails:
    print("\nFAILURES:")
    for name, detail in fails:
        print(f"  - {name}: {detail}")
    sys.exit(1)
sys.exit(0)

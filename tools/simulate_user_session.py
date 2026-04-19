"""End-to-end user session simulator — not for production, for QA.

Mimics what a real user does clicking through the 3D Liquify workflow,
including realistic think-time between clicks, reading of response bodies
before the next action, and handling of long-running Apply/Finalize calls
via the progress polling the frontend also uses.

Target: 35_C0_demo_run2 (a completed run with cells_mapped.csv, 111
truth-export overlays, annotation_registered in place).

Run with the Flask server already up on 127.0.0.1:8787.
"""

from __future__ import annotations

import json
import sys
import time
from urllib import error, request

BASE = "http://127.0.0.1:8787"
JOB = "35_C0_demo_run2"
CLASS_NAME_HINT = "ChATe27"


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _get(path: str) -> dict:
    url = BASE + path
    with request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path: str, body: dict | None = None, *, timeout: int = 1800) -> dict:
    data = json.dumps(body or {}).encode("utf-8")
    req = request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _delete(path: str) -> dict:
    req = request.Request(BASE + path, method="DELETE")
    with request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _think(sec: float) -> None:
    """Simulate human think-time between clicks."""
    time.sleep(sec)


def _poll_progress_until(stop_stage: str, *, timeout_s: int = 1800) -> None:
    """Poll /liquify-3d/progress every 2s until the expected 'done' stage."""
    t_start = time.time()
    last_pct = -1
    while True:
        if time.time() - t_start > timeout_s:
            raise TimeoutError(f"progress poll timed out after {timeout_s}s")
        try:
            p = _get(f"/api/liquify-3d/progress?job={JOB}")
        except error.URLError:
            time.sleep(2)
            continue
        pct = p.get("percent")
        stg = p.get("stage") or ""
        if pct is not None and pct != last_pct:
            _log(f"   progress [{pct:>3}%] {stg}: {p.get('message','')}")
            last_pct = pct
        if stg == stop_stage and pct == 100:
            return
        time.sleep(2)


def main() -> int:
    _log("=== Real user session simulator — JOB=" + JOB + " ===")

    # ---- Step 1: Land on 3D Liquify tab ----
    _log("STEP 1: Open 3D Liquify tab (state + class registry)")
    state = _get(f"/api/liquify-3d/state?job={JOB}")
    _log(f"   state: pairs={state['pair_count']} source={state['source_annotation_available']} guidance={state.get('guidance')!r}")
    assert state["source_annotation_available"], "demo_run2 annotation missing — abort"
    ann_shape = state.get("annotation_shape")
    _log(f"   annotation_shape={ann_shape}")
    _think(1.5)

    classes = _get("/api/liquify-3d/class-registry/list")["classes"]
    _log(f"STEP 2: class registry list → {classes}")
    detected = _get(f"/api/liquify-3d/class-registry/detect?sampleId={JOB}")["class"]
    _log(f"   auto-detect for '{JOB}' → {detected}")
    _think(1.0)

    # ---- Step 2: Reload slice list ----
    _log("STEP 3: Reload slice list (Reload button)")
    slices = _get("/api/outputs/reg-slice-list")
    _log(f"   got {slices['count']} overlays available")
    _think(2.0)

    # ---- Step 3: Clear any stale landmarks + add 10 pairs ----
    _log("STEP 4: Clear any prior landmarks for this job")
    _post("/api/liquify-3d/clear", {"jobId": JOB})

    _log("STEP 5: Lay down 10 landmark pairs across 5 z positions")
    # Realistic: small ~8-voxel corrections at 5 z positions (2 per z)
    # image size for 35_C0_demo_run2 is roughly the registered_label size
    # (we let the backend auto-rescale; we post in annotation-grid-native
    # coords directly — no image_dims_yx means 1:1 treat as voxel coords)
    pairs = []
    # image native annotation grid is (409, 340); pick points well inside tissue
    for i, z in enumerate([20, 40, 60, 80, 100]):
        for k, (ay, ax) in enumerate([(200, 170), (240, 180)]):
            ry = ay + 8  # +8 voxel correction in y
            rx = ax + 6  # +6 voxel correction in x
            pairs.append({"z": z, "atlas": [ay, ax], "real": [ry, rx]})
    for i, p in enumerate(pairs):
        resp = _post("/api/liquify-3d/add-pair", {"jobId": JOB, **p})
        _log(f"   [{i+1}/10] z={p['z']} atlas={p['atlas']} real={p['real']} → total={resp['pair_count']}")
        _think(0.4)  # simulate click cadence

    state2 = _get(f"/api/liquify-3d/state?job={JOB}")
    assert state2["pair_count"] == 10, f"expected 10, got {state2['pair_count']}"
    _log(f"STEP 6: verify state → {state2['pair_count']} pairs stored")
    _think(1.0)

    # ---- Step 4: Test Undo ----
    _log("STEP 7: Try Ctrl+Z (undo last pair) via DELETE endpoint")
    undo = _delete(f"/api/liquify-3d/pair/9?job={JOB}")
    assert undo["pair_count"] == 9
    _log(f"   after undo: {undo['pair_count']} pairs left")
    # Re-add the same pair so we have 10 for Apply
    _post("/api/liquify-3d/add-pair", {"jobId": JOB, **pairs[-1]})
    _think(0.8)

    # ---- Step 5: Click Apply — runs Laplacian ----
    _log("STEP 8: Click Apply (runs 3D Laplacian warp, may take several min)")
    t0 = time.time()
    # We fire the POST and simultaneously poll progress (via threads would
    # match the frontend; for a simulator, sequential is fine — Flask dev
    # server is threaded so concurrent requests work).
    import threading
    progress_done = threading.Event()
    def _poll_bg():
        last = -1
        while not progress_done.is_set():
            try:
                p = _get(f"/api/liquify-3d/progress?job={JOB}")
                pct = p.get("percent")
                if pct is not None and pct != last:
                    _log(f"   [apply] [{pct:>3}%] {p.get('stage','')}: {p.get('message','')}")
                    last = pct
            except Exception:
                pass
            time.sleep(1.5)

    t = threading.Thread(target=_poll_bg, daemon=True)
    t.start()
    try:
        apply_resp = _post("/api/liquify-3d/apply", {"jobId": JOB}, timeout=1800)
    finally:
        progress_done.set()
        t.join(timeout=3)
    elapsed = time.time() - t0
    _log(f"   Apply DONE in {elapsed:.0f}s → pair_count={apply_resp['pair_count']} "
         f"max_disp={apply_resp.get('displacement_max_voxels', 0):.2f}vx")
    _log(f"   refined annotation at: {apply_resp.get('output_path')}")

    state3 = _get(f"/api/liquify-3d/state?job={JOB}")
    assert state3["refined_annotation_exists"], "refined annotation didn't get written"
    _log(f"STEP 9: state.refined_annotation_exists={state3['refined_annotation_exists']}")
    _think(2.0)

    # ---- Step 6: Click Finalize ----
    _log("STEP 10: Click Finalize (re-exports truth slices + re-maps cells + re-aggregates)")
    t0 = time.time()
    progress_done.clear()
    t = threading.Thread(target=_poll_bg, daemon=True)
    t.start()
    try:
        fin_resp = _post("/api/liquify-3d/finalize", {"jobId": JOB}, timeout=3600)
    finally:
        progress_done.set()
        t.join(timeout=3)
    elapsed = time.time() - t0
    if not fin_resp.get("ok"):
        _log(f"   ! Finalize failed: {fin_resp.get('error')}")
    else:
        _log(f"   Finalize DONE in {elapsed:.0f}s → mapped_count={fin_resp.get('mapped_count')}")
        _log(f"   new hierarchy: {fin_resp.get('cell_counts_hierarchy_csv')}")
    _think(1.5)

    # ---- Step 7: Verify Results fallback picks up the liquify3d version ----
    if fin_resp.get("ok"):
        _log("STEP 11: Verify /api/outputs/hierarchy now serves liquify3d version")
        # We can't easily read CSV via JSON; just hit HEAD to confirm 200
        url = BASE + "/api/outputs/hierarchy"
        try:
            with request.urlopen(url, timeout=30) as r:
                size = len(r.read())
                _log(f"   GET /api/outputs/hierarchy → {r.status} {size}B")
        except Exception as e:
            _log(f"   hierarchy fetch failed: {e}")
    _think(1.0)

    # ---- Step 8: Save to class prior ----
    _log(f"STEP 12: Save job → class prior '{CLASS_NAME_HINT}'")
    save = _post(
        "/api/liquify-3d/class-prior/save",
        {"jobId": JOB, "class": CLASS_NAME_HINT, "sampleId": JOB},
    )
    _log(f"   prior sample_count={save.get('sample_count')}  entry_count={save.get('entry_count')}  ready={save.get('ready_for_warm_start')}")
    _think(1.0)

    status = _get(f"/api/liquify-3d/class-prior/status?class={CLASS_NAME_HINT}")
    _log(f"STEP 13: class-prior status → {status}")
    _think(1.0)

    coverage = _get(f"/api/liquify-3d/class-prior/coverage?class={CLASS_NAME_HINT}")
    bins = coverage.get("bins", [])
    if bins:
        z_span = f"z=[{min(b['z'] for b in bins)}..{max(b['z'] for b in bins)}]"
        _log(f"STEP 14: coverage heatmap data: {len(bins)} entries, {z_span}, max n={max(b['count'] for b in bins)}")
    else:
        _log("STEP 14: coverage heatmap empty")
    _think(1.0)

    # ---- Step 9: Mark QC done ----
    _log("STEP 15: Mark QC done with metrics")
    qc = _post(
        "/api/liquify-3d/qc-done",
        {
            "jobId": JOB,
            "className": CLASS_NAME_HINT,
            "metrics": {"pair_count": 10, "NCC_est": 0.32, "Dice_est": 0.74},
            "note": "simulated user session end-to-end",
        },
    )
    _log(f"   marker written: {qc.get('marker_path')}")
    _think(0.5)

    qc_status = _get(f"/api/liquify-3d/qc-status?job={JOB}")
    _log(f"STEP 16: QC status → done={qc_status['done']} metrics={qc_status.get('metrics')}")

    _log("=== Session simulator complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tifffile import imwrite

import project.frontend.server_context as ctx
from project.frontend.server import create_app

# Derive project root from this file's location (works in CI and local dev)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"


@pytest.fixture
def client(monkeypatch):
    original_output_dir = ctx.OUTPUT_DIR
    original_project_root = ctx.PROJECT_ROOT
    original_root = ctx.ROOT
    original_run_state = dict(ctx.run_state)
    app = create_app()
    app.testing = True
    with app.test_client() as test_client:
        yield test_client
    ctx.OUTPUT_DIR = original_output_dir
    ctx.PROJECT_ROOT = original_project_root
    ctx.ROOT = original_root
    ctx.run_state.clear()
    ctx.run_state.update(original_run_state)


def test_index_html_has_balanced_interactive_tags():
    html = (_FRONTEND_DIR / "index.html").read_text(
        encoding="utf-8", errors="replace"
    )
    assert html.count("<button") == html.count("</button>")
    assert html.count("<summary") == html.count("</summary>")
    assert html.count("<span") == html.count("</span>")


def test_results_tab_refreshes_outputs_and_file_list():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    assert "if (btn.dataset.tab === 'results') refreshOutputsAndFiles();" in js


def test_index_html_has_whole_brain_3d_sections():
    html = (_FRONTEND_DIR / "index.html").read_text(
        encoding="utf-8", errors="replace"
    )

    for snippet in (
        'id="wholeBrain3dStatusSection"',
        'id="aux2dNotice"',
        'id="wholeBrainStageList"',
        'id="volumeQcSection"',
        'id="volumeQcSummary"',
        'id="sliceInspectorSection"',
        'id="sliceInspectorGrid"',
    ):
        assert snippet in html


def test_app_js_renders_whole_brain_stage_track_and_volume_qc():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    for snippet in (
        "function renderWholeBrain3dStage",
        "function refreshVolumeQcSummary",
        "function refreshSliceInspector",
        "renderWholeBrain3dStage(latestWholeBrainStage)",
        "t('wb3d.status.stage', {",
    ):
        assert snippet in js


def test_refresh_slice_inspector_stays_strictly_on_3d_exports():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    start = js.index("async function refreshSliceInspector()")
    end = js.index(
        "// ================================================================\n// BATCH QC ALL",
        start,
    )
    body = js[start:end]

    assert "fetch('/api/outputs/reg-slice-list')" in body
    assert "qc-list" not in body


def test_slice_inspector_3d_click_enlarges_exported_overlay():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    start = js.index("items.forEach(entry => {")
    end = js.index("sliceInspectorGrid.appendChild(wrap);", start)
    body = js[start:end]

    assert "openLightbox(img.src, entry.name);" in body
    assert "demo-comparison" not in body


def test_index_html_has_manual_count_controls():
    html = (_FRONTEND_DIR / "index.html").read_text(
        encoding="utf-8", errors="replace"
    )

    for snippet in (
        'id="manualCountSourcePath"',
        'id="manualCountLoadBtn"',
        'id="manualCountPalette"',
        'id="manualCountViewport"',
        'id="manualCountImg"',
        'id="manualCountCanvas"',
        'id="manualCountExportBtn"',
    ):
        assert snippet in html


def test_index_html_has_manual_tiff_sidebar_tab():
    html = (_FRONTEND_DIR / "index.html").read_text(
        encoding="utf-8", errors="replace"
    )

    assert 'data-tab="manual-tiff"' in html
    assert 'id="tab-manual-tiff"' in html
    assert 'data-i18n="nav.manualTiff"' in html


def test_app_js_wires_manual_count_viewer():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    assert "manualCountViewport.addEventListener('wheel'" in js
    assert "manualCountPaletteEl.addEventListener('change'" in js
    assert "manualCountCanvas.addEventListener('click'" in js
    assert "manualCountCanvas.addEventListener('contextmenu'" in js
    assert "manualCountExportBtn.onclick" in js
    assert "/api/align/manual-image" in js
    assert "palette: manualCountState.palette" in js


def test_app_js_has_auto_warm_start_flow_for_empty_jobs():
    """Task 3 — app.js must implement a single automatic warm-start flow
    that triggers only when:
      - a class is auto-detected or manually selected
      - liquify state has been refreshed (state.pairs is an array)
      - the job has zero existing landmark pairs
    It must never force-overwrite; jobs with existing pairs get a clear
    "manual overwrite required" banner pointing them at the explicit button.

    Review finding 3 — the fire-once guard must be scoped by (jobId, class),
    NOT a single page-lifetime boolean, so switching jobs or classes
    triggers a fresh auto-apply attempt. Clearing pairs also resets the
    guard so the user sees auto-apply re-fire after a manual Clear.
    """
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    assert "_autoWarmStartIfEmpty" in js, (
        "expected an _autoWarmStartIfEmpty helper driving Task 3 auto-apply"
    )
    # Guard: the auto-apply call MUST use force:false (never clobber)
    auto_fn_start = js.index("async function _autoWarmStartIfEmpty")
    auto_fn_end = js.index("\n  }", auto_fn_start)
    auto_body = js[auto_fn_start:auto_fn_end]
    assert "force: false" in auto_body, "auto warm-start must never force-overwrite"
    assert "state.pairs.length > 0" in auto_body, (
        "auto warm-start must skip jobs that already have manual pairs"
    )
    assert "auto-applied" in auto_body.lower() or "auto-applied" in js, (
        "banner must distinguish the auto-applied state"
    )
    assert "manual overwrite required" in js, (
        "banner must explain why auto-apply was skipped"
    )
    # Review finding 3 — scoped guard, not a lifetime-of-page boolean.
    assert "_autoWarmStartTried" in js, (
        "expected an _autoWarmStartTried guard tracking which (jobId, class) "
        "pairs have already been auto-applied"
    )
    # Must be a Set / Map (scopable), not a naked boolean.
    assert "_autoWarmStartTried = new Set" in js or "_autoWarmStartTried = new Map" in js, (
        "fire-once guard must be a Set/Map keyed by signature; a boolean "
        "never resets on job/class change so the user only ever gets one "
        "auto-apply per page load"
    )
    # Changing job id must retrigger — jobInput listener present
    assert "jobInput?.addEventListener('change'" in js or \
           "jobInput.addEventListener('change'" in js, (
        "jobInput change must re-run _autoWarmStartIfEmpty so switching jobs "
        "doesn't leave the user on a stale page-load-only guard"
    )
    # Clearing pairs must invalidate the tried signature so the user sees a
    # fresh auto-apply after manual Clear. Scan a generous window after the
    # clearBtn handler anchor — the handler contains a nested fetch() block
    # whose own ``});`` would confuse a naive index-based slice.
    clear_start = js.index("clearBtn.addEventListener('click'")
    clear_window = js[clear_start : clear_start + 1200]
    assert "_autoWarmStartTried" in clear_window, (
        "clearBtn handler must reset the auto-warm-start guard so the next "
        "refreshState sees the empty-pairs job as a fresh candidate"
    )


def test_liquify_qc_done_uses_inline_note_input_not_prompt():
    """Browser-hosted UI must not rely on window.prompt for QC sign-off.

    The in-app browser used for release validation does not support prompt(),
    so the QC note must be a normal page input that the click handler reads
    before posting /api/liquify-3d/qc-done.
    """
    html = (_FRONTEND_DIR / "index.html").read_text(
        encoding="utf-8", errors="replace"
    )
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    assert 'id="liq3dQcNote"' in html
    qc_start = js.index("qcDoneBtn?.addEventListener('click'")
    qc_end = js.index("// ------------- #6 Class dropdown", qc_start)
    qc_body = js[qc_start:qc_end]

    assert "prompt(" not in qc_body
    assert "liq3dQcNote" in qc_body
    assert "note," in qc_body


def test_liquify_pair_count_survives_language_application():
    """The pair-count span must not sit inside a translated parent node.

    applyLang() replaces data-i18n element innerHTML, so putting
    #liq3dPairCount inside that same translated element removes it before the
    Liquify tab can update the count in a real browser.
    """
    html = (_FRONTEND_DIR / "index.html").read_text(
        encoding="utf-8", errors="replace"
    )

    marker = 'id="liq3dPairCount"'
    assert marker in html
    before_marker = html[: html.index(marker)]
    h3_start = before_marker.rfind("<h3")
    h3_end = before_marker.rfind("</h3>")
    assert h3_start > h3_end
    h3_open_end = html.index(">", h3_start)
    h3_open = html[h3_start:h3_open_end]

    assert "data-i18n" not in h3_open
    assert '<span data-i18n="liquify3d.pairsTitle">' in html[h3_start : html.index("</h3>", h3_start)]


def test_2d_liquify_batches_pointer_stroke_before_posting():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    assert "liquifyStrokePoints" in js
    assert "sampleLiquifyStrokePoint" in js
    assert "buildLiquifyDragBatch" in js
    assert "async function applyLiquifyDragBatch(drags)" in js
    assert "payload.drags = drags" in js
    assert "const drags = buildLiquifyDragBatch" in js
    assert "drawCanvas.addEventListener('pointerdown'" in js
    assert "drawCanvas.addEventListener('pointermove'" in js
    assert "drawCanvas.addEventListener('pointerup'" in js


def test_liquify3d_has_brush_mode_controls():
    html = (_FRONTEND_DIR / "index.html").read_text(
        encoding="utf-8", errors="replace"
    )
    css = (_FRONTEND_DIR / "styles.css").read_text(
        encoding="utf-8", errors="replace"
    )
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    for snippet in (
        'name="liq3dToolMode"',
        'id="liq3dBrushRadius"',
        'id="liq3dBrushStrength"',
        'id="liq3dStrokeCount"',
        'id="liq3dStrokesBody"',
    ):
        assert snippet in html

    for snippet in (
        ".liq3d-tools",
        ".liq3d-brush-control",
        "#liq3dCanvasWrap",
        "@media (max-width: 800px)",
    ):
        assert snippet in css

    for snippet in (
        "toolMode: 'brush'",
        "postLiquify3dStroke",
        "canvas.addEventListener('pointerdown', liq3dPointerDown)",
        "canvas.addEventListener('pointermove', liq3dPointerMove)",
        "canvas.addEventListener('pointerup', liq3dPointerUp)",
        "/api/liquify-3d/stroke",
        "renderStrokeHistory",
    ):
        assert snippet in js


def test_liquify3d_undo_is_mode_aware():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    assert "undoLastStroke" in js
    assert "/api/liquify-3d/stroke/${lastIdx}" in js or "/api/liquify-3d/stroke/" in js
    assert "state.toolMode === 'brush'" in js


def test_liquify3d_radius_number_input_updates_from_its_own_value():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    assert "function get3dBrushRadius(rawValue = null)" in js
    assert "const v = get3dBrushRadius(brushRadiusNum.value)" in js
    assert "brushRadius.value = String(v)" in js


def test_liquify3d_brush_copy_is_localized():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    for key in (
        "liquify3d.toolBrush",
        "liquify3d.toolLandmark",
        "liquify3d.brushRadius",
        "liquify3d.brushStrength",
        "liquify3d.strokesTitle",
    ):
        assert js.count(key) >= 2
    assert "Brush mode: drag the atlas boundary toward the real anatomy." in js


def test_core_ui_controls_use_design_system_styles():
    """Guard against browser-native controls leaking into release UI."""
    html = (_FRONTEND_DIR / "index.html").read_text(
        encoding="utf-8", errors="replace"
    )
    css = (_FRONTEND_DIR / "styles.css").read_text(
        encoding="utf-8", errors="replace"
    )

    assert '<span class="nav-icon"><i data-lucide="folder"></i></span>' in html
    assert '<span class="nav-icon">📁</span>' not in html
    assert ".form-row" in css and "display: flex" in css
    assert ".btn {" in css
    assert 'input[type="radio"],' in css
    assert 'input[type="checkbox"]' in css


def test_error_panel_is_neutral_until_errors_exist():
    """The sidebar error panel should not look active on a clean page load."""
    css = (_FRONTEND_DIR / "styles.css").read_text(
        encoding="utf-8", errors="replace"
    )
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    assert ".error-panel.has-errors" in css
    error_panel_start = css.index(".error-panel {")
    error_panel_end = css.index(".error-panel.has-errors", error_panel_start)
    empty_panel_css = css[error_panel_start:error_panel_end]

    assert "217, 79, 79" not in empty_panel_css
    assert "state.frontendErrors = loadFrontendErrors()" not in js
    assert "sessionStorage.getItem(FRONTEND_ERRORS_KEY)" not in js
    assert "errorPanel?.classList.toggle('has-errors'" in js


def test_active_output_dir_uses_configured_output_dir_for_run_name(tmp_path):
    original_output_dir = ctx.OUTPUT_DIR
    original_project_root = ctx.PROJECT_ROOT
    original_run_state = dict(ctx.run_state)
    resource_root = tmp_path / "_internal"
    output_root = tmp_path / "writable" / "outputs"
    run_dir = output_root / "run_001"
    run_dir.mkdir(parents=True)
    try:
        ctx.PROJECT_ROOT = resource_root
        ctx.OUTPUT_DIR = output_root
        ctx.run_state.clear()
        ctx.run_state.update({"runName": "run_001", "outputDir": ""})

        assert ctx.active_output_dir() == run_dir
    finally:
        ctx.OUTPUT_DIR = original_output_dir
        ctx.PROJECT_ROOT = original_project_root
        ctx.run_state.clear()
        ctx.run_state.update(original_run_state)


def test_pyinstaller_spec_bundles_pkg_resources_extern():
    spec = (_FRONTEND_DIR / "BrainfastUI.spec").read_text(
        encoding="utf-8", errors="replace"
    )

    assert '"pkg_resources.extern"' in spec


def test_pyinstaller_spec_bundles_scipy_root_extension():
    spec = (_FRONTEND_DIR / "BrainfastUI.spec").read_text(
        encoding="utf-8", errors="replace"
    )

    assert '"scipy._cyutility"' in spec
    assert 'collect_submodules("scipy._lib.array_api_compat")' in spec


def test_desktop_launcher_uses_bundled_resource_root_for_frozen_assets():
    source = (_FRONTEND_DIR / "desktop_app.py").read_text(
        encoding="utf-8", errors="replace"
    )
    normalized = " ".join(source.split())

    assert "def _resource_project_root()" in source
    assert "ensure_atlas_assets(" in normalized
    assert "_resource_project_root()" in normalized
    assert "allow_download=not IS_FROZEN" in normalized
    assert "ensure_atlas_assets(FRONTEND.parent" not in source


def test_guided_tour_is_opt_in_and_targets_visible_oneclick_controls():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")
    tour_start = js.index("// GUIDED TOUR")
    tour_end = js.index("// ==================================================================", tour_start + 1)
    tour_js = js[tour_start:tour_end]

    assert "setTimeout(startTour, 1200)" not in tour_js
    assert "target: '#inputDir'" not in tour_js
    assert "target: '#atlasPath'" not in tour_js
    for target in (
        "target: '#oneClickSourcePath'",
        "target: '#oneClickAtlasVersion'",
        "target: '#oneClickRegMode'",
        "target: '#oneClickStartBtn'",
    ):
        assert target in tour_js
    assert "behavior: 'auto'" in tour_js
    assert "_overlay.addEventListener('click', () => _endTour(false));" in tour_js
    assert "document.addEventListener('keydown', _handleTourKeydown);" in tour_js


def test_app_js_has_readable_manual_tiff_translations():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    assert "'nav.manualTiff': '手动TIFF检查'" in js
    assert "'manualCount.title': '手动TIFF检查'" in js
    assert "'manualCount.desc': '直接打开源TIFF进行人眼检查和手动计数，可用鼠标滚轮切换Z层。'" in js


def test_export_methods_uses_active_output_dir(tmp_path, monkeypatch, client):
    out_dir = tmp_path / "run_001"
    out_dir.mkdir()
    (out_dir / "run_params_20260401_000000.json").write_text(
        (
            '{"alignMode":"affine","pixelSizeUm":"0.65","channels":["red"],'
            '"timestamp":"2026-04-01 10:00:00"}'
        ),
        encoding="utf-8",
    )
    # Main uses job-based output dirs; default job maps to OUTPUT_DIR
    ctx.OUTPUT_DIR = out_dir

    res = client.get("/api/export/methods-text")

    assert res.status_code == 200
    payload = res.get_json()
    assert payload["ok"] is True
    assert "2026-04-01 10:00:00" in payload["text"]
    assert "脑图谱配准" in payload["text"]
    assert "仿射变换" in payload["text"]


def test_demo_best_slice_uses_active_output_dir(tmp_path, monkeypatch, client):
    out_dir = tmp_path / "run_demo"
    out_dir.mkdir()
    Image.new("RGB", (8, 8), (12, 34, 56)).save(out_dir / "demo_best_slice.jpg")
    # Main uses job-based output dirs; default job maps to OUTPUT_DIR
    ctx.OUTPUT_DIR = out_dir

    res = client.get("/api/outputs/demo-best-slice")

    assert res.status_code == 200
    assert res.mimetype == "image/jpeg"


def test_reg_stats_uses_active_output_dir(tmp_path, monkeypatch, client):
    out_dir = tmp_path / "run_stats"
    out_dir.mkdir()
    (out_dir / "slice_registration_qc.csv").write_text(
        "slice_id,best_score,registration_ok\n0,0.81,true\n1,0.65,false\n",
        encoding="utf-8",
    )
    # Main uses job-based output dirs; default job maps to OUTPUT_DIR
    ctx.OUTPUT_DIR = out_dir

    res = client.get("/api/outputs/reg-stats")

    assert res.status_code == 200
    payload = res.get_json()
    assert payload["ok"] is True
    assert payload["mode"] == "slice_qc"
    assert payload["total"] == 2
    assert payload["ok_count"] == 1
    assert payload["mean_score"] == pytest.approx(0.73, rel=1e-3)


def test_named_output_rejects_path_traversal(tmp_path, monkeypatch, client):
    out_dir = tmp_path / "run_safe"
    out_dir.mkdir()
    (out_dir / "safe.txt").write_text("safe", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(ctx, "active_output_dir", lambda: out_dir)

    res = client.get("/api/outputs/named/../../secret.txt")

    assert res.status_code == 404
    assert "secret" not in res.get_data(as_text=True)


def test_status_endpoint_returns_stage_progress(tmp_path, monkeypatch, client):
    out_dir = tmp_path / "run_3d"
    out_dir.mkdir()
    # Main uses job-based state; set up the default job's progress dict
    ctx.OUTPUT_DIR = out_dir
    ctx.run_state["progress"] = {
        "phase": "registration",
        "stepCurrent": 3,
        "stepTotal": 6,
        "slicesDone": 0,
        "slicesTotal": 0,
        "message": "Running SyN",
    }

    res = client.get("/api/status")

    assert res.status_code == 200
    payload = res.get_json()
    assert payload["progress"]["phase"] == "registration"
    assert payload["progress"]["stepCurrent"] == 3
    assert payload["progress"]["stepTotal"] == 6
    assert payload["progress"]["message"] == "Running SyN"


def test_manual_preview_image_serves_png_from_selected_tiff(tmp_path, client):
    tif_path = tmp_path / "real.tif"
    imwrite(str(tif_path), [[0, 100], [200, 300]])

    res = client.get(f"/api/align/manual-image?path={tif_path}")

    assert res.status_code == 200
    assert res.mimetype == "image/png"


def test_manual_preview_image_supports_pseudocolor_palette(tmp_path, client):
    tif_path = tmp_path / "real.tif"
    imwrite(str(tif_path), [[0, 100], [200, 300]])

    res = client.get(f"/api/align/manual-image?path={tif_path}&palette=green")

    assert res.status_code == 200
    rgb = np.asarray(Image.open(io.BytesIO(res.data)).convert("RGB"))
    mask = rgb.max(axis=2) > 0
    assert np.any(rgb[..., 0][mask] != rgb[..., 1][mask])


def test_calibration_save_and_learn_use_the_same_dir_when_rename_fails(
    tmp_path, monkeypatch
):
    """Review finding 2 — on the rename-fail fallback path, ``_save_...`` used
    to write to legacy ``train_data_set/`` while ``_learn_...`` read from the
    new ``outputs/state/calibration/samples/`` because each helper resolved
    the dir independently. A resolver must pin the choice so save + learn
    agree on one location.
    """
    import project.frontend.server_context as ctx_mod

    project_root = tmp_path / "proj"
    project_root.mkdir()
    (project_root / "train_data_set").mkdir()
    (project_root / "train_data_set" / "1_Ori.png").write_bytes(b"legacy")
    monkeypatch.setattr(ctx_mod, "PROJECT_ROOT", project_root)

    # Pin the shared state root somewhere we can observe, then force the
    # rename to fail (as happens on Windows when another process is reading
    # the legacy dir, or on a cross-disk BRAINFAST_STATE_DIR mount).
    state_root = tmp_path / "state"
    monkeypatch.setenv("BRAINFAST_STATE_DIR", str(state_root))
    real_rename = Path.rename

    def _fail_rename(self, target, *a, **kw):
        if self.name == "train_data_set":
            raise OSError("simulated cross-disk rename failure")
        return real_rename(self, target, *a, **kw)

    monkeypatch.setattr(Path, "rename", _fail_rename)

    # Whatever resolver save + learn share must return the SAME dir.
    save_dir = ctx_mod._resolve_calibration_samples_dir()
    learn_dir = ctx_mod._resolve_calibration_samples_dir()
    assert save_dir == learn_dir, (
        "save vs learn must agree on one calibration samples dir after "
        "a rename-failure fallback"
    )
    # Since the rename failed but legacy has data, both must point at the
    # legacy dir (reading the new empty one would make Save Calibration +
    # Learn a silent no-op).
    assert save_dir == project_root / "train_data_set"


def test_extract_preview_frame_reads_single_page_without_full_stack_imread(tmp_path, monkeypatch):
    tif_path = tmp_path / "stack.tif"
    stack = [[[0, 1], [2, 3]], [[10, 11], [12, 13]]]
    imwrite(str(tif_path), stack)

    import project.frontend.blueprints.api_alignment as api_alignment

    # Ensure imread is NOT importable from this module (ruff removed unused
    # import), so _extract_preview_frame can only use page-based TiffFile.
    assert not hasattr(api_alignment, "imread"), (
        "api_alignment should not import imread — _extract_preview_frame must use TiffFile"
    )

    frame = api_alignment._extract_preview_frame(tif_path, z_index=1)

    assert frame.tolist() == [[10, 11], [12, 13]]


def test_runner_forwards_explicit_output_dir(tmp_path, monkeypatch):
    config_path = tmp_path / "cfg.json"
    config_path.write_text("{}", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    original_project_root = ctx.PROJECT_ROOT
    original_output_dir = ctx.OUTPUT_DIR
    original_run_state = dict(ctx.run_state)

    class FakePopen:
        last_cmd = None
        last_cwd = None
        last_env = None

        def __init__(self, cmd, cwd=None, stdout=None, stderr=None, text=None, env=None):
            FakePopen.last_cmd = list(cmd)
            FakePopen.last_cwd = cwd
            FakePopen.last_env = dict(env) if env else {}
            self.stdout = io.StringIO("")

        def wait(self):
            return 0

    monkeypatch.setattr(ctx.subprocess, "Popen", FakePopen)
    ctx.PROJECT_ROOT = tmp_path
    ctx.OUTPUT_DIR = tmp_path / "outputs"
    ctx.OUTPUT_DIR.mkdir(exist_ok=True)

    test_job_id = "test-job-42"

    try:
        ctx._runner(
            str(config_path),
            str(input_dir),
            ["red"],
            run_params={"inputDir": str(input_dir)},
            job_id=test_job_id,
        )
        job_state = ctx.get_job_state(test_job_id)
        recorded_output_dir = job_state["outputs_dir"]
        recorded_env = dict(FakePopen.last_env)
        expected_out = str(ctx._job_output_dir(test_job_id))
    finally:
        ctx.PROJECT_ROOT = original_project_root
        ctx.OUTPUT_DIR = original_output_dir
        ctx.run_state.clear()
        ctx.run_state.update(original_run_state)

    assert recorded_output_dir == expected_out
    # Main passes output dir via env var, not CLI flag
    assert recorded_env.get("BRAINCOUNT_OUTPUT_DIR") == expected_out
    assert recorded_env.get("BRAINCOUNT_JOB_ID") == test_job_id

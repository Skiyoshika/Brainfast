from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tifffile import imwrite

import project.frontend.server_context as ctx
from project.frontend.server import create_app


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
    html = Path(r"D:\Brainfast\project\frontend\index.html").read_text(
        encoding="utf-8", errors="replace"
    )
    assert html.count("<button") == html.count("</button>")
    assert html.count("<summary") == html.count("</summary>")
    assert html.count("<span") == html.count("</span>")


def test_results_tab_refreshes_outputs_and_file_list():
    js = Path(r"D:\Brainfast\project\frontend\app.js").read_text(encoding="utf-8", errors="replace")

    assert "if (btn.dataset.tab === 'results') refreshOutputsAndFiles();" in js


def test_index_html_has_whole_brain_3d_sections():
    html = Path(r"D:\Brainfast\project\frontend\index.html").read_text(
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
    js = Path(r"D:\Brainfast\project\frontend\app.js").read_text(encoding="utf-8", errors="replace")

    for snippet in (
        "function renderWholeBrain3dStage",
        "function refreshVolumeQcSummary",
        "function refreshSliceInspector",
        "renderWholeBrain3dStage(latestWholeBrainStage)",
        "t('wb3d.status.stage', {",
    ):
        assert snippet in js


def test_refresh_slice_inspector_stays_strictly_on_3d_exports():
    js = Path(r"D:\Brainfast\project\frontend\app.js").read_text(encoding="utf-8", errors="replace")

    start = js.index("async function refreshSliceInspector()")
    end = js.index(
        "// ================================================================\n// BATCH QC ALL",
        start,
    )
    body = js[start:end]

    assert "fetch('/api/outputs/reg-slice-list')" in body
    assert "qc-list" not in body


def test_slice_inspector_3d_click_enlarges_exported_overlay():
    js = Path(r"D:\Brainfast\project\frontend\app.js").read_text(encoding="utf-8", errors="replace")

    start = js.index("items.forEach(entry => {")
    end = js.index("sliceInspectorGrid.appendChild(wrap);", start)
    body = js[start:end]

    assert "openLightbox(img.src, entry.name);" in body
    assert "demo-comparison" not in body


def test_index_html_has_manual_count_controls():
    html = Path(r"D:\Brainfast\project\frontend\index.html").read_text(
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
    html = Path(r"D:\Brainfast\project\frontend\index.html").read_text(
        encoding="utf-8", errors="replace"
    )

    assert 'data-tab="manual-tiff"' in html
    assert 'id="tab-manual-tiff"' in html
    assert 'data-i18n="nav.manualTiff"' in html


def test_app_js_wires_manual_count_viewer():
    js = Path(r"D:\Brainfast\project\frontend\app.js").read_text(encoding="utf-8", errors="replace")

    assert "manualCountViewport.addEventListener('wheel'" in js
    assert "manualCountPaletteEl.addEventListener('change'" in js
    assert "manualCountCanvas.addEventListener('click'" in js
    assert "manualCountCanvas.addEventListener('contextmenu'" in js
    assert "manualCountExportBtn.onclick" in js
    assert "/api/align/manual-image" in js
    assert "palette: manualCountState.palette" in js


def test_app_js_has_readable_manual_tiff_translations():
    js = Path(r"D:\Brainfast\project\frontend\app.js").read_text(encoding="utf-8", errors="replace")

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

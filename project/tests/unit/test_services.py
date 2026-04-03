"""Unit tests for frontend/services/*.

All heavy script dependencies are mocked so these tests run without atlas files,
GPU, or any large data files.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# alignment_service
# ---------------------------------------------------------------------------

class TestRenderLandmarkPreview:
    """alignment_service.render_landmark_preview"""

    def _make_pairs_csv(self, tmp_path: Path) -> Path:
        import pandas as pd
        pairs = pd.DataFrame({
            "real_x": [10, 50], "real_y": [20, 60],
            "atlas_x": [12, 52], "atlas_y": [22, 62],
        })
        p = tmp_path / "pairs.csv"
        pairs.to_csv(p, index=False)
        return p

    @patch("scripts.slice_select.select_real_slice_2d")
    @patch("scripts.slice_select.select_label_slice_2d")
    @patch("project.frontend.services.alignment_service.imread")
    def test_writes_side_by_side_png(self, mock_imread, mock_label_sel, mock_real_sel, tmp_path):
        from project.frontend.services.alignment_service import render_landmark_preview

        fake_img = np.zeros((100, 100), dtype=np.uint8)
        mock_imread.return_value = fake_img
        mock_real_sel.return_value = (fake_img, {})
        mock_label_sel.return_value = (fake_img, {})

        pairs_csv = self._make_pairs_csv(tmp_path)
        out = tmp_path / "preview.png"
        n = render_landmark_preview(Path("real.tif"), Path("atlas.tif"), pairs_csv, out)

        assert n == 2
        assert out.exists()
        with Image.open(out) as img:
            assert img.format == "PNG"
            assert img.size == (208, 100)

    @patch("scripts.slice_select.select_real_slice_2d")
    @patch("scripts.slice_select.select_label_slice_2d")
    @patch("project.frontend.services.alignment_service.imread")
    def test_empty_pairs_writes_blank_canvas(self, mock_imread, mock_label_sel, mock_real_sel, tmp_path):
        import pandas as pd
        from project.frontend.services.alignment_service import render_landmark_preview

        fake_img = np.zeros((80, 80), dtype=np.uint8)
        mock_imread.return_value = fake_img
        mock_real_sel.return_value = (fake_img, {})
        mock_label_sel.return_value = (fake_img, {})

        pairs_csv = tmp_path / "empty.csv"
        pd.DataFrame({"real_x": [], "real_y": [], "atlas_x": [], "atlas_y": []}).to_csv(pairs_csv, index=False)
        out = tmp_path / "out.png"
        n = render_landmark_preview(Path("r.tif"), Path("a.tif"), pairs_csv, out)

        assert n == 0
        assert out.exists()
        with Image.open(out) as img:
            assert img.format == "PNG"
            assert img.size == (168, 80)
            assert np.asarray(img).max() == 0

    @patch("scripts.slice_select.select_real_slice_2d")
    @patch("scripts.slice_select.select_label_slice_2d")
    @patch("project.frontend.services.alignment_service.imread")
    def test_writes_browser_readable_png(
        self, mock_imread, mock_label_sel, mock_real_sel, tmp_path
    ):
        from PIL import Image
        from project.frontend.services.alignment_service import render_landmark_preview

        fake_img = np.zeros((32, 32), dtype=np.uint8)
        mock_imread.return_value = fake_img
        mock_real_sel.return_value = (fake_img, {})
        mock_label_sel.return_value = (fake_img, {})

        pairs_csv = self._make_pairs_csv(tmp_path)
        out = tmp_path / "preview.png"

        n = render_landmark_preview(Path("real.tif"), Path("atlas.tif"), pairs_csv, out)

        assert n == 2
        assert Image.open(out).format == "PNG"


class TestProposeAlignmentLandmarks:
    """alignment_service.propose_landmarks"""

    @patch("scripts.ai_landmark.propose_landmarks")
    def test_delegates_kwargs_correctly(self, mock_fn):
        from project.frontend.services.alignment_service import propose_landmarks

        mock_fn.return_value = {"n_pairs": 12}
        result = propose_landmarks(
            Path("r.tif"), Path("a.tif"), Path("out.csv"),
            max_points=20, min_distance=8, ransac_residual=5.0,
        )

        mock_fn.assert_called_once_with(
            Path("r.tif"), Path("a.tif"), Path("out.csv"),
            max_points=20, min_distance=8, ransac_residual=5.0,
        )
        assert result == {"n_pairs": 12}


# ---------------------------------------------------------------------------
# overlay_service
# ---------------------------------------------------------------------------

class TestRenderOverlayFromLabel:
    """overlay_service.render_overlay_from_label"""

    @patch("scripts.overlay_render.render_overlay")
    def test_calls_render_overlay_and_returns_diagnostic(self, mock_render):
        from project.frontend.services.overlay_service import render_overlay_from_label

        mock_render.return_value = (MagicMock(), {"regions": 5})
        diag = render_overlay_from_label(
            Path("real.tif"), Path("label.tif"), Path("out.png"),
            render_kwargs={"alpha": 0.5, "mode": "fill", "return_meta": True},
        )

        mock_render.assert_called_once_with(
            Path("real.tif"), Path("label.tif"), Path("out.png"),
            alpha=0.5, mode="fill", return_meta=True,
        )
        assert diag == {"regions": 5}


class TestApplyLiquifyAndRender:
    """overlay_service.apply_liquify_and_render"""

    @patch("scripts.overlay_render.render_overlay")
    @patch("project.frontend.server_context._apply_liquify_drags")
    @patch("scripts.slice_select.select_label_slice_2d")
    @patch("project.frontend.services.overlay_service.imwrite")
    @patch("project.frontend.services.overlay_service.imread")
    def test_writes_two_label_files_and_returns_diagnostic(
        self, mock_imread, mock_imwrite, mock_sel, mock_liquify, mock_render, tmp_path
    ):
        from project.frontend.services.overlay_service import apply_liquify_and_render

        lbl = np.zeros((64, 64), dtype=np.int32)
        mock_imread.return_value = lbl
        mock_sel.return_value = (lbl, {})
        mock_liquify.return_value = lbl + 1
        mock_render.return_value = (MagicMock(), {"ok": True})

        corrected_path = tmp_path / "corrected.tif"
        hover_path = tmp_path / "hover.tif"
        drags = [{"x1": 10, "y1": 10, "x2": 20, "y2": 20, "radius": 30, "strength": 0.5}]

        result, diag = apply_liquify_and_render(
            Path("base.tif"), drags, corrected_path, hover_path,
            render_kwargs={"real_slice_path": Path("real.tif"), "return_meta": True},
        )

        assert mock_imwrite.call_count == 2
        assert diag == {"ok": True}
        assert result is not None


# ---------------------------------------------------------------------------
# demo_service
# ---------------------------------------------------------------------------

class TestGenerateCellChart:
    """demo_service.generate_cell_chart"""

    @patch("subprocess.run")
    def test_calls_subprocess_with_correct_args(self, mock_run, tmp_path):
        import sys
        from project.frontend.services.demo_service import generate_cell_chart

        mock_run.return_value = MagicMock(returncode=0)
        hier = tmp_path / "hier.csv"
        chart = tmp_path / "chart.png"
        project_root = tmp_path

        generate_cell_chart(hier, chart, project_root)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == sys.executable
        assert str(hier) in args
        assert str(chart) in args

    @patch("subprocess.run", side_effect=Exception("process failed"))
    def test_raises_on_subprocess_error(self, mock_run, tmp_path):
        from project.frontend.services.demo_service import generate_cell_chart

        with pytest.raises(Exception, match="process failed"):
            generate_cell_chart(tmp_path / "h.csv", tmp_path / "c.png", tmp_path)


class TestGenerateDemoComparison:
    """demo_service.generate_demo_comparison — checks FileNotFoundError on missing overlay."""

    def test_raises_when_overlay_missing(self, tmp_path):
        from project.frontend.services.demo_service import generate_demo_comparison

        reg_dir = tmp_path / "registered_slices"
        reg_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "compare.jpg"

        with pytest.raises(FileNotFoundError):
            generate_demo_comparison(0, reg_dir, data_dir, out)


def test_latest_stage_progress_reads_active_output_dir(tmp_path, monkeypatch):
    import project.frontend.server_context as ctx

    out_dir = tmp_path / "run_3d"
    out_dir.mkdir()
    (out_dir / "pipeline_progress.json").write_text(
        (
            '{'
            '"stageName":"Truth Export",'
            '"stageIndex":5,'
            '"stageCount":6,'
            '"percent":82,'
            '"message":"Exporting slices",'
            '"artifacts":{"truth_dir":"truth_export"}'
            "}"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ctx, "active_output_dir", lambda: out_dir)

    progress = ctx.latest_stage_progress()

    assert progress["stageName"] == "Truth Export"
    assert progress["stageIndex"] == 5
    assert progress["stageCount"] == 6
    assert progress["percent"] == 82
    assert progress["artifacts"]["truth_dir"] == "truth_export"


def test_outputs_volume_reg_stats_uses_active_output_dir(tmp_path, monkeypatch):
    import project.frontend.server_context as ctx
    from project.frontend.server import create_app

    out_dir = tmp_path / "run_volume_qc"
    out_dir.mkdir()
    (out_dir / "volume_registration_qc.csv").write_text(
        "metric,value\nNCC,0.912\nSSIM,0.843\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ctx, "active_output_dir", lambda: out_dir)

    app = create_app()
    app.testing = True
    with app.test_client() as client:
        res = client.get("/api/outputs/volume-reg-stats")

    assert res.status_code == 200
    assert res.mimetype in {"text/csv", "application/vnd.ms-excel"}
    assert "NCC,0.912" in res.get_data(as_text=True)


def test_active_output_dir_prefers_recent_progress_markers_over_completed_runs(tmp_path, monkeypatch):
    import project.frontend.server_context as ctx

    project_root = tmp_path
    outputs_root = project_root / "outputs"
    progress_dir = outputs_root / "whole_brain_in_progress"
    completed_dir = outputs_root / "older_completed"
    progress_dir.mkdir(parents=True)
    completed_dir.mkdir(parents=True)

    (progress_dir / "pipeline_progress.json").write_text(
        '{"stageName":"ANTS Registration","stageIndex":3,"stageCount":6,"percent":42}',
        encoding="utf-8",
    )
    (completed_dir / "cell_counts_hierarchy.csv").write_text(
        "region,count\nroot,1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(ctx, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(ctx, "OUTPUT_DIR", outputs_root)
    monkeypatch.setitem(ctx.run_state, "outputDir", "")
    monkeypatch.setitem(ctx.run_state, "runName", "")

    assert ctx.active_output_dir() == progress_dir


def test_active_output_dir_recovers_volume_qc_runs_without_completed_markers(tmp_path, monkeypatch):
    import project.frontend.server_context as ctx

    project_root = tmp_path
    outputs_root = project_root / "outputs"
    volume_qc_dir = outputs_root / "whole_brain_volume_qc"
    completed_dir = outputs_root / "older_completed"
    volume_qc_dir.mkdir(parents=True)
    completed_dir.mkdir(parents=True)

    (volume_qc_dir / "volume_registration_qc.csv").write_text(
        "metric,value\nNCC,0.912\n",
        encoding="utf-8",
    )
    (completed_dir / "cell_counts_hierarchy.csv").write_text(
        "region,count\nroot,1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(ctx, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(ctx, "OUTPUT_DIR", outputs_root)
    monkeypatch.setitem(ctx.run_state, "outputDir", "")
    monkeypatch.setitem(ctx.run_state, "runName", "")

    assert ctx.active_output_dir() == volume_qc_dir

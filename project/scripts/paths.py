"""
Centralized path management for Brainfast pipelines.

Usage:
    from scripts.paths import RunPaths

    paths = RunPaths.from_project_root(project_root, cfg)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    cells_csv = paths.cells_detected
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.asset_bootstrap import default_structure_source
except Exception:
    from asset_bootstrap import default_structure_source


def bootstrap_sys_path() -> Path:
    """Ensure PROJECT_ROOT is on sys.path. Idempotent; frozen-EXE aware.

    Call this at the top of any standalone script that needs to import from
    ``scripts.*`` or ``project.*``.  Returns PROJECT_ROOT.

    Example (at top of script, before any project imports)::

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.paths import bootstrap_sys_path
        PROJECT_ROOT = bootstrap_sys_path()
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        project_root = Path(sys._MEIPASS)
    else:
        # __file__ is scripts/paths.py → parents[1] is project/
        project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def ensure_runtime_cache_dirs(project_root: Path) -> Path:
    """Point runtime caches at a writable project-local directory.

    This avoids sandbox and Windows profile permission issues for libraries
    such as Matplotlib, which otherwise try to write under the user's home
    directory on first import.
    """
    runtime_cache_dir = project_root / "outputs" / ".runtime_cache"
    runtime_cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir = runtime_cache_dir / "matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    return runtime_cache_dir


# ---------------------------------------------------------------------------
# Shared runtime-state layout — survives across jobs and lives OUTSIDE the
# git-tracked source tree. Calibration samples, class priors, and Cellpose
# training pairs all root here. Can be overridden via BRAINFAST_STATE_DIR
# so a deployment can put mutable state on a separate disk.
# ---------------------------------------------------------------------------

STATE_ROOT_ENV: str = "BRAINFAST_STATE_DIR"


def resolve_state_root(project_root: Path) -> Path:
    """Return the canonical shared-state root for learned artifacts.

    Default: ``<project_root>/outputs/state``. Override via the env var
    ``BRAINFAST_STATE_DIR`` (absolute path). Never returns a location inside
    the git-tracked source tree (``train_data_set/``, ``cellpose_training/``).
    """
    override = os.environ.get(STATE_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(project_root) / "outputs" / "state"


def calibration_samples_dir(project_root: Path) -> Path:
    """Per-sample manual-corrected training pairs (ori/show/label triples).

    Previously lived at ``<project_root>/train_data_set/``. Moved under
    ``outputs/state/calibration/samples/`` so the source tree stays clean.
    """
    return resolve_state_root(project_root) / "calibration" / "samples"


def calibration_tuned_json(project_root: Path) -> Path:
    """Output of ``learn_from_trainset.py`` consumed by truth-export."""
    return resolve_state_root(project_root) / "calibration" / "trainset_tuned_params.json"


def class_priors_dir(project_root: Path) -> Path:
    """Per-class running-mean landmark priors aggregated across jobs."""
    return resolve_state_root(project_root) / "class_priors"


def cellpose_training_dir(project_root: Path) -> Path:
    """Detector retraining samples (image + corrected mask pairs)."""
    return resolve_state_root(project_root) / "cellpose_training"


@dataclass
class RunPaths:
    """All file/directory paths for a single pipeline run."""

    # ── Root dirs ─────────────────────────────────────────────────────────────
    project_root: Path
    outputs: Path
    registered_slices: Path
    qc_dir: Path

    # ── Atlas asset ───────────────────────────────────────────────────────────
    annotation_nii: Path
    structure_csv: Path

    # ── Intermediate CSVs ─────────────────────────────────────────────────────
    cells_detected: Path
    cells_dedup: Path
    cells_mapped: Path
    dedup_stats: Path

    # ── Final outputs ─────────────────────────────────────────────────────────
    cell_counts_leaf: Path
    cell_counts_hierarchy: Path
    slice_registration_qc: Path
    slice_qc: Path

    # ── Tuning / training ─────────────────────────────────────────────────────
    tuned_params: Path
    trainset_tuned_params: Path

    # ── Shared state (survives across jobs; NOT under source tree) ────────────
    state_root: Path
    calibration_samples_dir: Path
    calibration_tuned_json: Path
    class_priors_dir: Path
    cellpose_training_dir: Path

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
        cfg: dict[str, Any],
        *,
        outputs_dir: Path | None = None,
    ) -> RunPaths:
        outputs_cfg = cfg.get("outputs", {})
        outputs = Path(outputs_dir) if outputs_dir is not None else (project_root / "outputs")

        leaf_csv = outputs_cfg.get("leaf_csv", "outputs/cell_counts_leaf.csv")
        hierarchy_csv = outputs_cfg.get("hierarchy_csv", "outputs/cell_counts_hierarchy.csv")
        qc_dir_cfg = outputs_cfg.get("qc_dir", "outputs/qc")

        # Resolve relative paths from project root
        def _resolve(p: str) -> Path:
            pp = Path(p)
            if pp.is_absolute():
                return pp
            return project_root / pp

        structure_csv = project_root / "configs" / "allen_mouse_structure_graph.csv"
        structure_csv_fallback = outputs / "registration" / "structure_tree.csv"
        if structure_csv_fallback.exists():
            structure_csv = structure_csv_fallback
        else:
            fallback_structure = default_structure_source(project_root)
            if fallback_structure is not None:
                structure_csv = fallback_structure

        state_root = resolve_state_root(project_root)
        return cls(
            project_root=project_root,
            outputs=outputs,
            registered_slices=outputs / "registered_slices",
            qc_dir=_resolve(qc_dir_cfg),
            annotation_nii=project_root / "annotation_25.nii.gz",
            structure_csv=structure_csv,
            cells_detected=outputs / "cells_detected.csv",
            cells_dedup=outputs / "cells_dedup.csv",
            cells_mapped=outputs / "cells_mapped.csv",
            dedup_stats=outputs / "dedup_stats.csv",
            cell_counts_leaf=_resolve(leaf_csv),
            cell_counts_hierarchy=_resolve(hierarchy_csv),
            slice_registration_qc=outputs / "slice_registration_qc.csv",
            slice_qc=outputs / "slice_qc.csv",
            tuned_params=outputs / "tuned_params.json",
            trainset_tuned_params=outputs / "trainset_tuned_params.json",
            state_root=state_root,
            calibration_samples_dir=state_root / "calibration" / "samples",
            calibration_tuned_json=state_root / "calibration" / "trainset_tuned_params.json",
            class_priors_dir=state_root / "class_priors",
            cellpose_training_dir=state_root / "cellpose_training",
        )

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in (self.outputs, self.registered_slices, self.qc_dir):
            d.mkdir(parents=True, exist_ok=True)

    def registered_slice_overlay(self, idx: int) -> Path:
        return self.registered_slices / f"slice_{idx:04d}_overlay.png"

    def registered_label(self, idx: int) -> Path:
        return self.registered_slices / f"slice_{idx:04d}_registered_label.tif"

    def auto_label(self, idx: int) -> Path:
        return self.registered_slices / f"slice_{idx:04d}_auto_label.tif"

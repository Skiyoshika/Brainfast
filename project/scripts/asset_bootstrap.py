from __future__ import annotations

import os
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from scripts.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_ANNOTATION_NRRD_URL = (
    "http://download.alleninstitute.org/informatics-archive/current-release/"
    "mouse_ccf/annotation/ccf_2017/annotation_25.nrrd"
)
DEFAULT_STRUCTURE_GRAPH_URL = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"

# Zenodo record for CCFv3-BBP atlas (EPFL Blue Brain Project)
ZENODO_CCFV3BBP_RECORD = "15176439"
ZENODO_CCFV3BBP_ANNOTATION_URL = (
    f"https://zenodo.org/records/{ZENODO_CCFV3BBP_RECORD}/files/annotation_25.nrrd"
)
ZENODO_CCFV3BBP_REFERENCE_URL = (
    f"https://zenodo.org/records/{ZENODO_CCFV3BBP_RECORD}/files/average_template_25.nrrd"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def atlas_paths(project_root: Path | None = None) -> dict[str, Path]:
    root = Path(project_root) if project_root is not None else _project_root()
    configs = root / "configs"
    return {
        "project_root": root,
        "configs": configs,
        "annotation_nii": root / "annotation_25.nii.gz",
        "annotation_nrrd": root / "annotation_25.nrrd",
        "structure_graph_csv": configs / "allen_mouse_structure_graph.csv",
        "structure_graph_json": configs / "allen_mouse_structure_graph.json",
        "structure_tree_json": configs / "allen_structure_tree.json",
    }


def default_structure_source(project_root: Path | None = None) -> Path | None:
    paths = atlas_paths(project_root)
    for key in ("structure_graph_csv", "structure_graph_json", "structure_tree_json"):
        candidate = paths[key]
        if candidate.exists():
            return candidate
    return None


def atlas_asset_status(project_root: Path | None = None) -> dict[str, object]:
    paths = atlas_paths(project_root)
    structure_source = default_structure_source(paths["project_root"])
    return {
        "annotationReady": paths["annotation_nii"].exists(),
        "annotationPath": str(paths["annotation_nii"]),
        "annotationNrrdReady": paths["annotation_nrrd"].exists(),
        "annotationNrrdPath": str(paths["annotation_nrrd"]),
        "structureGraphCsvReady": paths["structure_graph_csv"].exists(),
        "structureGraphCsvPath": str(paths["structure_graph_csv"]),
        "structureGraphJsonReady": paths["structure_graph_json"].exists(),
        "structureGraphJsonPath": str(paths["structure_graph_json"]),
        "structureTreeJsonReady": paths["structure_tree_json"].exists(),
        "structureTreeJsonPath": str(paths["structure_tree_json"]),
        "structureReady": structure_source is not None,
        "structurePath": str(structure_source) if structure_source is not None else "",
        "structureFormat": structure_source.suffix.lower() if structure_source is not None else "",
        "allRequiredReady": paths["annotation_nii"].exists() and structure_source is not None,
    }


def _download_to_path(url: str, dest: Path, *, logger: Callable[[str], None]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)
        done = 0
        fd, tmp_name = tempfile.mkstemp(prefix=dest.name, suffix=".tmp", dir=str(dest.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if total > 0:
                        logger(f"downloaded {done / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB")
            tmp_path.replace(dest)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def _convert_with_pynrrd(nrrd_path: Path, nii_path: Path) -> None:
    import nibabel as nib
    import nrrd
    import numpy as np

    data, header = nrrd.read(str(nrrd_path))
    voxel_mm = np.array([0.025, 0.025, 0.025], dtype=float)
    directions = header.get("space directions")
    if directions is not None:
        try:
            norms = []
            for axis in directions:
                if axis is None:
                    norms.append(0.025)
                    continue
                norm = float(np.linalg.norm(np.asarray(axis, dtype=float)))
                if norm > 1.0:
                    norm /= 1000.0
                norms.append(norm)
            if len(norms) >= 3:
                voxel_mm = np.asarray(norms[:3], dtype=float)
        except Exception:
            pass

    img = nib.Nifti1Image(data.astype(np.int32), np.diag([*voxel_mm, 1.0]))
    img.header.set_zooms(tuple(float(v) for v in voxel_mm))
    nib.save(img, str(nii_path))


def _convert_with_sitk(nrrd_path: Path, nii_path: Path) -> None:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(nrrd_path))
    sitk.WriteImage(image, str(nii_path))


def convert_annotation_nrrd_to_nifti(nrrd_path: Path, nii_path: Path) -> None:
    errors: list[str] = []
    for converter in (_convert_with_pynrrd, _convert_with_sitk):
        try:
            converter(nrrd_path, nii_path)
            return
        except Exception as exc:
            errors.append(f"{converter.__name__}: {exc}")
    joined = "; ".join(errors) if errors else "no converters attempted"
    raise RuntimeError(f"failed to convert atlas NRRD to NIfTI ({joined})")


def ensure_atlas_assets(
    project_root: Path | None = None,
    *,
    allow_download: bool = True,
    logger: Callable[[str], None] | None = None,
) -> dict[str, object]:
    root = Path(project_root) if project_root is not None else _project_root()
    paths = atlas_paths(root)
    emit = logger or (lambda _msg: None)

    annotation_url = os.environ.get("BRAINFAST_ATLAS_NRRD_URL", DEFAULT_ANNOTATION_NRRD_URL).strip()
    structure_graph_url = os.environ.get(
        "BRAINFAST_STRUCTURE_GRAPH_URL", DEFAULT_STRUCTURE_GRAPH_URL
    ).strip()

    if not paths["annotation_nii"].exists():
        if not paths["annotation_nrrd"].exists():
            if not allow_download:
                raise FileNotFoundError(f"missing atlas source: {paths['annotation_nrrd']}")
            emit(f"downloading atlas annotation from {annotation_url}")
            _download_to_path(annotation_url, paths["annotation_nrrd"], logger=emit)
        emit("converting annotation_25.nrrd to annotation_25.nii.gz")
        convert_annotation_nrrd_to_nifti(paths["annotation_nrrd"], paths["annotation_nii"])

    if not paths["structure_graph_json"].exists() and allow_download:
        emit(f"downloading structure ontology from {structure_graph_url}")
        _download_to_path(structure_graph_url, paths["structure_graph_json"], logger=emit)

    return atlas_asset_status(root)


# ---------------------------------------------------------------------------
# BrainGlobe Atlas API integration
# ---------------------------------------------------------------------------


def ensure_brainglobe_atlas(
    atlas_name: str = "allen_mouse_25um",
    version: int | None = None,
) -> dict[str, object] | None:
    """Download/cache an atlas via the BrainGlobe Atlas API.

    Parameters
    ----------
    atlas_name : str
        Atlas identifier recognised by ``bg_atlasapi``, e.g.
        ``"allen_mouse_25um"`` or ``"allen_mouse_25um_bbp"``.
    version : int | None
        Specific atlas version.  *None* uses the latest available.

    Returns
    -------
    dict | None
        ``{"annotation_path", "reference_path", "metadata"}`` on success,
        or *None* when ``bg_atlasapi`` is not installed.
    """
    try:
        from bg_atlasapi import BrainGlobeAtlas  # type: ignore[import-untyped]
    except ImportError:
        log.warning(
            "bg-atlasapi is not installed — BrainGlobe atlas support unavailable. "
            "Install with: pip install bg-atlasapi"
        )
        return None

    log.info("Loading BrainGlobe atlas %r (version=%s) ...", atlas_name, version)
    kwargs: dict[str, object] = {"atlas_name": atlas_name}
    if version is not None:
        kwargs["brainglobe_dir"] = None  # use default cache
    atlas = BrainGlobeAtlas(atlas_name)  # downloads on first access

    root_dir = Path(atlas.root_dir)
    annotation_path = root_dir / "annotation.tiff"
    reference_path = root_dir / "reference.tiff"

    metadata = {
        "name": atlas.atlas_name,
        "resolution": atlas.resolution,
        "shape": tuple(atlas.shape) if hasattr(atlas, "shape") else None,
        "orientation": getattr(atlas, "orientation", None),
        "brainglobe_dir": str(root_dir),
    }

    log.info(
        "BrainGlobe atlas ready: %s  resolution=%s  dir=%s",
        atlas_name,
        metadata["resolution"],
        root_dir,
    )
    return {
        "annotation_path": annotation_path,
        "reference_path": reference_path,
        "metadata": metadata,
    }


def ensure_ccfv3bbp_atlas(
    project_root: Path | None = None,
    *,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Ensure the CCFv3-BBP atlas (Blue Brain Project extended annotations).

    Strategy:
    1. Try BrainGlobe Atlas API (``allen_mouse_25um_bbp``).
    2. Fall back to direct Zenodo download of annotation + Nissl template.

    Returns
    -------
    dict
        ``{"annotation_path": Path, "reference_path": Path}``
    """
    emit = logger or (lambda _msg: None)

    # --- attempt 1: BrainGlobe ---
    bg_result = ensure_brainglobe_atlas("allen_mouse_25um_bbp")
    if bg_result is not None:
        return {
            "annotation_path": Path(bg_result["annotation_path"]),
            "reference_path": Path(bg_result["reference_path"]),
        }

    # --- attempt 2: Zenodo download ---
    log.info("Falling back to Zenodo download for CCFv3-BBP atlas")
    root = Path(project_root) if project_root is not None else _project_root()
    bbp_dir = root / "configs" / "ccfv3bbp"
    bbp_dir.mkdir(parents=True, exist_ok=True)

    annotation_nrrd = bbp_dir / "annotation_25.nrrd"
    reference_nrrd = bbp_dir / "average_template_25.nrrd"

    if not annotation_nrrd.exists():
        emit(f"downloading CCFv3-BBP annotation from Zenodo ({ZENODO_CCFV3BBP_RECORD})")
        _download_to_path(ZENODO_CCFV3BBP_ANNOTATION_URL, annotation_nrrd, logger=emit)

    if not reference_nrrd.exists():
        emit(f"downloading CCFv3-BBP Nissl template from Zenodo ({ZENODO_CCFV3BBP_RECORD})")
        _download_to_path(ZENODO_CCFV3BBP_REFERENCE_URL, reference_nrrd, logger=emit)

    return {
        "annotation_path": annotation_nrrd,
        "reference_path": reference_nrrd,
    }


# ---------------------------------------------------------------------------
# Atlas registry
# ---------------------------------------------------------------------------

ATLAS_CHOICES: dict[str, dict[str, object]] = {
    "ccfv3": {
        "label": "Allen CCFv3 (standard)",
        "description": "Original Allen Mouse Brain Common Coordinate Framework v3, 25 um",
        "ensure_fn": ensure_atlas_assets,
        "brainglobe_name": "allen_mouse_25um",
    },
    "ccfv3bbp": {
        "label": "CCFv3-BBP (Blue Brain Project extended)",
        "description": (
            "EPFL Blue Brain Project refined CCFv3 with improved annotations "
            "(olfactory bulb, cerebellum, medulla, cerebellar layers) and "
            "734-brain averaged Nissl template at 10 um resolution"
        ),
        "ensure_fn": ensure_ccfv3bbp_atlas,
        "brainglobe_name": "allen_mouse_25um_bbp",
        "zenodo_record": ZENODO_CCFV3BBP_RECORD,
    },
}

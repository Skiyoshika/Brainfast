"""Phase β: 3D landmark liquify — human-in-the-loop internal structure refinement.

The user picks corresponding (real fluorescence voxel, desired atlas voxel) pairs
in the frontend's 3D viewer. Those pairs are stored per-job and, on demand,
solved into a smooth 3D displacement field via the vendored RegTools Laplacian
solver. The field warps the registered annotation volume so the atlas regions
snap to the user-indicated anatomy.

Design notes
------------
* Coordinates throughout are in **voxel units of the registered-annotation
  grid** (i.e. the moving-space 3D volume ANTs warped Allen into). The frontend
  is responsible for mapping screen clicks to these coordinates before calling
  ``LandmarkStore.add_pair``.
* We **reuse** ``regtools_laplacian.solveLaplacianFromCorrespondences`` — it
  already does Dirichlet-BC CG + Jacobi, handles sparse correspondences,
  supports anisotropic spacing, and was transplanted + tested in Phase α.
* Inverse-warping annotation labels uses nearest-neighbour interpolation to
  preserve discrete label IDs.

See ``docs/superpowers/plans/2026-04-16-internal-alignment-closed-loop-plan.md``
(Phase β) for the full architectural context.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LandmarkPair:
    """One human-provided (real, atlas) voxel correspondence at a given z."""

    z: int
    real: tuple[float, float]  # (y, x) in annotation-grid voxels
    atlas: tuple[float, float]  # (y, x) in annotation-grid voxels


@dataclass(frozen=True)
class LiquifyStroke:
    """One PS-style brush stroke on a single z slice.

    Points are stored as (y, x) coordinates in annotation-grid voxels.
    """

    z: int
    points: list[tuple[float, float]]
    radius: float
    strength: float
    image_dims_yx: tuple[float, float] | None
    created_at: str


class LandmarkStore:
    """CSV-backed append-only store of 3D landmark correspondences for one job.

    Row format::

        z,real_y,real_x,atlas_y,atlas_x

    Rows are written immediately on ``add_pair`` so a crash mid-session does
    not lose user effort. ``remove_pair`` rewrites the file atomically.
    """

    _COLUMNS = ("z", "real_y", "real_x", "atlas_y", "atlas_x")

    def __init__(self, csv_path: Path | str) -> None:
        self._path = Path(csv_path)

    @property
    def path(self) -> Path:
        return self._path

    def list_pairs(self) -> list[LandmarkPair]:
        if not self._path.exists():
            return []
        pairs: list[LandmarkPair] = []
        with self._path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                pairs.append(
                    LandmarkPair(
                        z=int(row["z"]),
                        real=(float(row["real_y"]), float(row["real_x"])),
                        atlas=(float(row["atlas_y"]), float(row["atlas_x"])),
                    )
                )
        return pairs

    def add_pair(
        self,
        z: int,
        real: tuple[float, float],
        atlas: tuple[float, float],
    ) -> LandmarkPair:
        pair = LandmarkPair(z=int(z), real=tuple(real), atlas=tuple(atlas))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._path.exists() or self._path.stat().st_size == 0
        with self._path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if is_new:
                writer.writerow(self._COLUMNS)
            writer.writerow([pair.z, pair.real[0], pair.real[1], pair.atlas[0], pair.atlas[1]])
        return pair

    def remove_pair(self, index: int) -> LandmarkPair:
        pairs = self.list_pairs()
        if not 0 <= index < len(pairs):
            raise IndexError(f"landmark index {index} out of range [0, {len(pairs)})")
        removed = pairs[index]
        remaining = [p for i, p in enumerate(pairs) if i != index]
        self._rewrite(remaining)
        return removed

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()

    def _rewrite(self, pairs: Iterable[LandmarkPair]) -> None:
        # Use mkstemp so each caller gets a unique tmp filename even under
        # concurrent removals (Windows raises PermissionError when two
        # threads collide on the same .tmp name during the final
        # atomic-replace step). Retry the replace a few times with small
        # back-off to survive a brief Windows file-handle hold-over too.
        import os as _os
        import tempfile as _tempfile
        import time as _time

        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = _tempfile.mkstemp(
            prefix=self._path.name + ".",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with _os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(self._COLUMNS)
                for p in pairs:
                    writer.writerow([p.z, p.real[0], p.real[1], p.atlas[0], p.atlas[1]])
            last_error: OSError | None = None
            for attempt in range(8):
                try:
                    tmp_path.replace(self._path)
                    last_error = None
                    break
                except PermissionError as exc:
                    last_error = exc
                    _time.sleep(0.01 * (attempt + 1))
            if last_error is not None:
                raise last_error
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class LiquifyStrokeStore:
    """JSONL-backed store of PS-style brush strokes for one job."""

    def __init__(self, jsonl_path: Path | str) -> None:
        self._path = Path(jsonl_path)

    @property
    def path(self) -> Path:
        return self._path

    def list_strokes(self) -> list[LiquifyStroke]:
        if not self._path.exists():
            return []
        strokes: list[LiquifyStroke] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                pts = [(float(p["y"]), float(p["x"])) for p in row.get("points", [])]
                dims = row.get("image_dims_yx")
                image_dims = None
                if isinstance(dims, (list, tuple)) and len(dims) == 2:
                    image_dims = (float(dims[0]), float(dims[1]))
                strokes.append(
                    LiquifyStroke(
                        z=int(row["z"]),
                        points=pts,
                        radius=float(row.get("radius", 80.0)),
                        strength=float(row.get("strength", 0.72)),
                        image_dims_yx=image_dims,
                        created_at=str(row.get("created_at", "")),
                    )
                )
        return strokes

    def add_stroke(
        self,
        *,
        z: int,
        points: list[dict] | list[tuple[float, float]],
        radius: float,
        strength: float,
        image_dims_yx: tuple[float, float] | None = None,
    ) -> LiquifyStroke:
        parsed: list[tuple[float, float]] = []
        for p in points:
            if isinstance(p, dict):
                parsed.append((float(p["y"]), float(p["x"])))
            else:
                parsed.append((float(p[0]), float(p[1])))
        if len(parsed) < 2:
            raise ValueError("stroke must contain at least two points")
        stroke = LiquifyStroke(
            z=int(z),
            points=parsed,
            radius=float(radius),
            strength=float(strength),
            image_dims_yx=(
                (float(image_dims_yx[0]), float(image_dims_yx[1]))
                if image_dims_yx is not None
                else None
            ),
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(self._stroke_to_row(stroke), separators=(",", ":")) + "\n")
        return stroke

    def remove_stroke(self, index: int) -> LiquifyStroke:
        strokes = self.list_strokes()
        if not 0 <= index < len(strokes):
            raise IndexError(f"stroke index {index} out of range [0, {len(strokes)})")
        removed = strokes[index]
        self._rewrite([s for i, s in enumerate(strokes) if i != index])
        return removed

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()

    def _rewrite(self, strokes: Iterable[LiquifyStroke]) -> None:
        import os as _os
        import tempfile as _tempfile

        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = _tempfile.mkstemp(
            prefix=self._path.name + ".",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as fh:
                for stroke in strokes:
                    fh.write(json.dumps(self._stroke_to_row(stroke), separators=(",", ":")) + "\n")
            tmp_path.replace(self._path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _stroke_to_row(stroke: LiquifyStroke) -> dict:
        return {
            "z": stroke.z,
            "points": [{"y": y, "x": x} for y, x in stroke.points],
            "radius": stroke.radius,
            "strength": stroke.strength,
            "image_dims_yx": list(stroke.image_dims_yx) if stroke.image_dims_yx else None,
            "created_at": stroke.created_at,
        }


def _rescale_stroke_point(
    y: float,
    x: float,
    *,
    image_dims_yx: tuple[float, float] | None,
    annotation_shape: tuple[int, int, int] | None,
) -> tuple[float, float]:
    if not image_dims_yx or not annotation_shape:
        return float(y), float(x)
    img_h, img_w = image_dims_yx
    if img_h <= 0 or img_w <= 0:
        return float(y), float(x)
    _d, ann_h, ann_w = annotation_shape
    return float(y) * (ann_h / img_h), float(x) * (ann_w / img_w)


def strokes_to_landmark_pairs(
    strokes: list[LiquifyStroke],
    *,
    annotation_shape: tuple[int, int, int] | None = None,
    min_segment_px: float = 2.0,
) -> list[LandmarkPair]:
    """Convert brush strokes into source/target correspondences.

    A drag from A to B means the atlas feature currently at A should move
    toward real anatomy at B. Radius and strength create neighbouring controls
    so the Laplacian solve behaves like a soft brush rather than a point pull.
    """

    pairs: list[LandmarkPair] = []
    offsets = (
        (0.0, 0.0),
        (0.35, 0.0),
        (-0.35, 0.0),
        (0.0, 0.35),
        (0.0, -0.35),
        (0.25, 0.25),
        (0.25, -0.25),
        (-0.25, 0.25),
        (-0.25, -0.25),
    )
    for stroke in strokes:
        if len(stroke.points) < 2:
            continue
        radius = max(1.0, float(stroke.radius))
        sigma = max(1.0, radius * 0.46)
        strength = float(np.clip(stroke.strength, 0.05, 1.5))
        for i in range(1, len(stroke.points)):
            y0, x0 = stroke.points[i - 1]
            y1, x1 = stroke.points[i]
            dy = y1 - y0
            dx = x1 - x0
            dist = float(np.hypot(dy, dx))
            if dist < min_segment_px:
                continue
            for off_y_frac, off_x_frac in offsets:
                off_y = off_y_frac * radius
                off_x = off_x_frac * radius
                off_dist = float(np.hypot(off_y, off_x))
                falloff = float(np.exp(-0.5 * (off_dist * off_dist) / (sigma * sigma)))
                pull = falloff * strength
                if pull < 0.04:
                    continue

                atlas_y = y0 + off_y
                atlas_x = x0 + off_x
                real_y = atlas_y + dy * pull
                real_x = atlas_x + dx * pull
                ay, ax = _rescale_stroke_point(
                    atlas_y,
                    atlas_x,
                    image_dims_yx=stroke.image_dims_yx,
                    annotation_shape=annotation_shape,
                )
                ry, rx = _rescale_stroke_point(
                    real_y,
                    real_x,
                    image_dims_yx=stroke.image_dims_yx,
                    annotation_shape=annotation_shape,
                )
                if annotation_shape:
                    _d, ann_h, ann_w = annotation_shape
                    ay = float(np.clip(ay, 0, ann_h - 1))
                    ax = float(np.clip(ax, 0, ann_w - 1))
                    ry = float(np.clip(ry, 0, ann_h - 1))
                    rx = float(np.clip(rx, 0, ann_w - 1))
                pairs.append(
                    LandmarkPair(
                        z=int(stroke.z),
                        real=(float(ry), float(rx)),
                        atlas=(float(ay), float(ax)),
                    )
                )
    return pairs


def combined_landmark_and_stroke_pairs(
    landmarks_csv: str | Path,
    strokes_jsonl: str | Path | None = None,
    *,
    annotation_shape: tuple[int, int, int] | None = None,
    min_segment_px: float = 2.0,
) -> list[LandmarkPair]:
    """Load explicit landmark pairs and append brush-derived correspondences."""

    pairs = LandmarkStore(landmarks_csv).list_pairs()
    if strokes_jsonl is None:
        return pairs
    stroke_pairs = strokes_to_landmark_pairs(
        LiquifyStrokeStore(strokes_jsonl).list_strokes(),
        annotation_shape=annotation_shape,
        min_segment_px=min_segment_px,
    )
    return [*pairs, *stroke_pairs]


# ---------------------------------------------------------------------------
# Warp computation
# ---------------------------------------------------------------------------


def compute_3d_displacement(
    vol_shape: tuple[int, int, int],
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    *,
    spacing: tuple[float, float, float] | None = None,
    rtol: float = 1e-2,
    maxiter: int = 500,
) -> np.ndarray:
    """Solve for a smooth 3D displacement field from sparse landmark pairs.

    Thin wrapper over ``regtools_laplacian.solveLaplacianFromCorrespondences``.
    The return shape is ``(3, n0, n1, n2)``.
    """
    try:
        from scripts.regtools_laplacian import solveLaplacianFromCorrespondences
    except ImportError:
        from regtools_laplacian import solveLaplacianFromCorrespondences

    source = np.asarray(source_pts, dtype=float).reshape(-1, 3)
    target = np.asarray(target_pts, dtype=float).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError(f"source/target shape mismatch: {source.shape} vs {target.shape}")
    if source.size == 0:
        return np.zeros((3, *vol_shape), dtype=np.float32)

    field = solveLaplacianFromCorrespondences(
        vol_shape=vol_shape,
        source_pts=source,
        target_pts=target,
        axes=(0, 1, 2),
        rtol=rtol,
        maxiter=maxiter,
        spacing=spacing,
    )
    return field.astype(np.float32, copy=False)


def landmarks_to_point_arrays(
    pairs: list[LandmarkPair],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert CSV landmark pairs into (source_pts, target_pts) arrays.

    The solver treats *target* as the Dirichlet anchor (where the atlas voxel
    must end up) and *source* = where the real anatomy is. Both are in
    (z, y, x) order for the 3D grid.
    """
    if not pairs:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)
    real = np.array([[p.z, p.real[0], p.real[1]] for p in pairs], dtype=float)
    atlas = np.array([[p.z, p.atlas[0], p.atlas[1]] for p in pairs], dtype=float)
    return real, atlas


# ---------------------------------------------------------------------------
# Warp application (nearest-neighbour for discrete labels)
# ---------------------------------------------------------------------------


def apply_3d_warp_to_annotation(
    annotation: np.ndarray,
    displacement_field: np.ndarray,
) -> np.ndarray:
    """Apply a (3, D, H, W) displacement field to a (D, H, W) label volume.

    For each output voxel *v*, the source location is ``v - displacement[v]``;
    we sample the annotation there via nearest-neighbour. Out-of-bounds
    samples get label 0.
    """
    from scipy.ndimage import map_coordinates

    if annotation.ndim != 3:
        raise ValueError(f"annotation must be 3D, got shape {annotation.shape}")
    if displacement_field.shape != (3, *annotation.shape):
        raise ValueError(
            "displacement_field shape must be (3, D, H, W) matching annotation; "
            f"got {displacement_field.shape} for annotation {annotation.shape}"
        )

    d, h, w = annotation.shape
    iz, iy, ix = np.meshgrid(
        np.arange(d, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    # Target coordinate in input = output coordinate − displacement
    coords = np.stack(
        [
            iz - displacement_field[0],
            iy - displacement_field[1],
            ix - displacement_field[2],
        ],
        axis=0,
    )
    warped = map_coordinates(
        annotation,
        coords,
        order=0,  # nearest-neighbour — preserves discrete labels
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return warped.astype(annotation.dtype, copy=False)


# ---------------------------------------------------------------------------
# End-to-end: annotation + CSV → refined annotation
# ---------------------------------------------------------------------------


def refine_annotation_with_pairs(
    annotation_path: Path | str,
    pairs: list[LandmarkPair],
    output_path: Path | str,
    *,
    spacing: tuple[float, float, float] | None = None,
    rtol: float = 1e-2,
    maxiter: int = 500,
    progress_cb=None,
) -> dict:
    annotation_path = Path(annotation_path)
    output_path = Path(output_path)

    def _emit(stage: str, idx: int, pct: int, msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, idx, 4, pct, msg)
            except Exception:  # noqa: BLE001 - progress is best-effort
                pass

    _emit("load_inputs", 1, 5, "Loading annotation + liquify controls")
    img = nib.load(str(annotation_path))
    ann = np.asarray(img.dataobj, dtype=np.int32)
    source_pts, target_pts = landmarks_to_point_arrays(pairs)

    if spacing is None:
        zooms = img.header.get_zooms()[:3]
        spacing_val = tuple(float(z) for z in zooms) if all(z > 0 for z in zooms) else None
    else:
        spacing_val = spacing

    _emit(
        "solve",
        2,
        15,
        f"Solving 3D Laplacian on {ann.shape} with {len(pairs)} control pair(s)",
    )
    field = compute_3d_displacement(
        vol_shape=tuple(ann.shape),
        source_pts=source_pts,
        target_pts=target_pts,
        spacing=spacing_val,
        rtol=rtol,
        maxiter=maxiter,
    )

    _emit("apply_warp", 3, 80, "Warping annotation with displacement field")
    warped = apply_3d_warp_to_annotation(ann, field)

    _emit("save", 4, 95, "Writing refined annotation NIfTI")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(warped, img.affine, img.header), str(output_path))
    _emit("done", 4, 100, f"Refined annotation saved to {output_path.name}")

    return {
        "pair_count": len(pairs),
        "displacement_max_voxels": float(np.abs(field).max()) if field.size else 0.0,
        "output_path": str(output_path),
    }


def refine_annotation_with_landmarks(
    annotation_path: Path | str,
    landmarks_csv: Path | str,
    output_path: Path | str,
    *,
    spacing: tuple[float, float, float] | None = None,
    rtol: float = 1e-2,
    maxiter: int = 500,
    progress_cb=None,
) -> dict:
    """Load annotation + landmark CSV, solve, warp, save.

    Parameters
    ----------
    annotation_path
        NIfTI of the registered annotation volume (moving-space, integer labels).
    landmarks_csv
        CSV produced by ``LandmarkStore``.
    output_path
        Where to write the refined annotation NIfTI.
    spacing
        Optional physical voxel spacing; if None, uses the input NIfTI header.
    progress_cb
        Optional callable ``progress_cb(stage, index, count, percent, message)``
        invoked at milestone points so a polling frontend can surface
        progress instead of showing a frozen spinner.

    Returns
    -------
    dict
        ``{pair_count, displacement_max_voxels, output_path}`` for logging.
    """
    annotation_path = Path(annotation_path)
    landmarks_csv = Path(landmarks_csv)
    output_path = Path(output_path)
    pairs = LandmarkStore(landmarks_csv).list_pairs()
    return refine_annotation_with_pairs(
        annotation_path=annotation_path,
        pairs=pairs,
        output_path=output_path,
        spacing=spacing,
        rtol=rtol,
        maxiter=maxiter,
        progress_cb=progress_cb,
    )

"""Z-axis AP series continuity analysis for Brainfast.

After the per-slice registration loop, ``smooth_ap_series`` detects AP jumps
between adjacent slices using a monotone cubic spline (PchipInterpolator) and
flags slices where the deviation exceeds ``max_dev`` voxels.

Usage (in main.py, after writing slice_registration_qc.csv)::

    from scripts.z_smoothness import smooth_ap_series, write_smoothness_report
    report = smooth_ap_series(registration_rows, max_dev=8)
    write_smoothness_report(report, outputs_dir / "z_smoothness_report.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:  # pragma: no cover
    PchipInterpolator = None  # type: ignore[assignment,misc]


def smooth_ap_series(
    registration_rows: list[dict[str, Any]],
    *,
    max_dev: int = 8,
) -> dict[str, Any]:
    """Analyse AP (best_z) continuity across slices.

    Parameters
    ----------
    registration_rows:
        List of dicts from the registration loop, each with at least
        ``slice_id`` (int) and ``best_z`` (int) keys.
        Rows where ``registration_ok`` is False are excluded from the spline
        fit (they are reported but not used to shape the curve).
    max_dev:
        Maximum allowed deviation (in atlas voxels) between original and
        smoothed AP before a slice is flagged as an outlier.

    Returns
    -------
    dict with keys:
        ``slice_ids``         – list[int]
        ``original_z``        – list[int]
        ``smoothed_z``        – list[int]
        ``deviation``         – list[float]
        ``is_outlier``        – list[bool]
        ``outlier_count``     – int
        ``max_deviation``     – float
        ``mean_deviation``    – float
        ``monotone``          – bool   (True if smoothed is non-decreasing)
        ``max_dev_threshold`` – int
    """
    ok_rows = [r for r in registration_rows if r.get("registration_ok", True)]

    _empty = lambda ids, zs: {  # noqa: E731
        "slice_ids": ids,
        "original_z": zs,
        "smoothed_z": list(zs),
        "deviation": [0.0] * len(ids),
        "is_outlier": [False] * len(ids),
        "outlier_count": 0,
        "max_deviation": 0.0,
        "mean_deviation": 0.0,
        "monotone": True,
        "max_dev_threshold": int(max_dev),
    }

    all_ids = [int(r["slice_id"]) for r in registration_rows]
    all_zs = [int(r["best_z"]) for r in registration_rows]

    if len(ok_rows) < 2:
        return _empty(all_ids, all_zs)

    df = (
        pd.DataFrame(ok_rows)[["slice_id", "best_z"]]
        .dropna()
        .sort_values("slice_id")
        .reset_index(drop=True)
    )
    xs = df["slice_id"].to_numpy(dtype=float)
    ys = df["best_z"].to_numpy(dtype=float)

    # Use rolling median (window=5) as the reference trend — robust to single outliers.
    # PchipInterpolator passes *through* all points so cannot serve as an outlier detector.
    import pandas as _pd

    window = min(5, max(3, len(xs)))
    smoothed_cont = (
        _pd.Series(ys)
        .rolling(window=window, min_periods=1, center=True)
        .median()
        .to_numpy(dtype=float)
    )

    # For very short series (< 3 points) the rolling median equals the values; skip detection.
    if len(xs) < 3:
        smoothed_int = np.clip(np.round(smoothed_cont).astype(int), 0, 527)
        deviation = np.zeros_like(ys, dtype=float)
    else:
        smoothed_int = np.clip(np.round(smoothed_cont).astype(int), 0, 527)
        deviation = np.abs(ys - smoothed_cont)
    is_outlier = deviation > float(max_dev)

    # Index by slice_id for all-row reporting
    fit_map: dict[int, dict] = {}
    for i, sid in enumerate(df["slice_id"].tolist()):
        fit_map[int(sid)] = {
            "orig": int(df["best_z"].iloc[i]),
            "smoothed": int(smoothed_int[i]),
            "dev": float(deviation[i]),
            "outlier": bool(is_outlier[i]),
        }

    # Failed slices: assign nearest-neighbour smoothed value
    fit_ids = sorted(fit_map.keys())
    full_map: dict[int, dict] = {}
    for r in registration_rows:
        sid = int(r["slice_id"])
        if sid in fit_map:
            full_map[sid] = fit_map[sid]
        else:
            nearest = min(fit_ids, key=lambda k: abs(k - sid))
            full_map[sid] = {
                "orig": int(r["best_z"]),
                "smoothed": fit_map[nearest]["smoothed"],
                "dev": float(abs(r["best_z"] - fit_map[nearest]["smoothed"])),
                "outlier": False,
            }

    ordered = sorted(full_map.keys())
    smoothed_seq = [full_map[i]["smoothed"] for i in ordered]

    return {
        "slice_ids": ordered,
        "original_z": [full_map[i]["orig"] for i in ordered],
        "smoothed_z": smoothed_seq,
        "deviation": [round(full_map[i]["dev"], 2) for i in ordered],
        "is_outlier": [full_map[i]["outlier"] for i in ordered],
        "outlier_count": int(np.sum(is_outlier)),
        "max_deviation": round(float(np.max(deviation)), 2) if len(deviation) else 0.0,
        "mean_deviation": round(float(np.mean(deviation)), 2) if len(deviation) else 0.0,
        "monotone": bool(np.all(np.diff(smoothed_seq) >= 0)),
        "max_dev_threshold": int(max_dev),
    }


def write_smoothness_report(report: dict[str, Any], out_path: Path) -> None:
    """Write the smoothness analysis report to a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

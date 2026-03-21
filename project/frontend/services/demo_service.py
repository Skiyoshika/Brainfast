"""Demo/chart generation helpers for the frontend."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Cell-count chart
# ---------------------------------------------------------------------------

_ANALYSIS_GROUPS: list[tuple[str, tuple[int, ...], str]] = [
    ("Striatum", (477,), "#F06292"),
    ("Pallidum", (803,), "#7E57C2"),
    ("Isocortex", (315,), "#4DB6AC"),
    ("Olfactory Areas", (698,), "#FFB74D"),
    ("Hippocampal Formation", (1089,), "#81C784"),
    ("Thalamus", (549,), "#64B5F6"),
    ("Hypothalamus", (1097,), "#FFD54F"),
    ("Midbrain", (313,), "#BA68C8"),
    ("Pons", (771,), "#90A4AE"),
    ("Medulla", (354,), "#A1887F"),
    ("Cerebellum", (512,), "#4FC3F7"),
    ("Cortical Subplate", (703,), "#AED581"),
    ("Fiber Tracts", (1009,), "#B0BEC5"),
    ("Ventricular System", (73,), "#9575CD"),
]


def _parse_structure_ids(path_text: object, region_id: object = None) -> set[int]:
    ids = {int(m.group(0)) for m in re.finditer(r"\d+", str(path_text or ""))}
    try:
        rid = int(float(region_id))
    except Exception:
        rid = None
    if rid is not None:
        ids.add(rid)
    return ids


def _safe_int(value: object) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def _infer_source_summary(cells) -> dict[str, object]:
    source_paths = [Path(str(v)) for v in cells.get("source_slice_path", []) if str(v).strip()]
    if not source_paths:
        return {
            "sample_name": "Unknown sample",
            "slice_summary": "No source slice metadata",
            "scope_kind": "unknown",
            "scope_label": "Unknown source scope",
            "scope_note": "Source slice paths are missing, so this summary cannot confirm which sample folder was counted.",
            "slice_count": 0,
            "folder_slice_count": None,
        }

    parent_names = [p.parent.name for p in source_paths if p.parent.name]
    sample_name = Counter(parent_names).most_common(1)[0][0] if parent_names else source_paths[0].stem
    common_parent = Counter([str(p.parent) for p in source_paths if str(p.parent)]).most_common(1)[0][0]
    common_parent_path = Path(common_parent)
    unique_paths = {str(p) for p in source_paths}
    slice_count = len(unique_paths)
    scope_kind = "source_folder"
    scope_note = ""
    folder_slice_count: int | None = None
    if sample_name.lower() in {"tmp_channel", "tmp_merged", "registered_slices", "outputs"}:
        sample_name = "Current outputs"
        scope_kind = "working_set"
        scope_note = (
            "This summary reflects the slices currently staged in outputs, not necessarily the full raw sample folder."
        )

    z_values: list[int] = []
    for path in source_paths:
        match = re.search(r"z(\d+)", path.stem, flags=re.IGNORECASE)
        if match:
            z_values.append(int(match.group(1)))
    slice_ids = []
    for value in cells.get("slice_id", []):
        parsed = _safe_int(value)
        if parsed is not None:
            slice_ids.append(parsed)

    if scope_kind != "working_set":
        try:
            tif_count = len(list(common_parent_path.glob("*.tif"))) + len(list(common_parent_path.glob("*.tiff")))
        except Exception:
            tif_count = 0
        folder_slice_count = tif_count or None
        if folder_slice_count and folder_slice_count > slice_count:
            scope_kind = "subset_folder"
            scope_note = (
                "Only part of the source folder is represented in the current outputs. Do not treat this as a full-sample conclusion."
            )

    if folder_slice_count:
        scope_label = f"{slice_count} of {folder_slice_count} slices from {common_parent_path.name}"
    elif scope_kind == "working_set":
        scope_label = f"{slice_count} processed slices in {common_parent_path.name}"
    else:
        scope_label = f"{slice_count} slices from {common_parent_path.name}"

    if z_values:
        slice_summary = f"{scope_label}  |  z{min(z_values):04d}-z{max(z_values):04d}"
    elif slice_ids:
        slice_summary = f"{scope_label}  |  slice_id {min(slice_ids):04d}-{max(slice_ids):04d}"
    else:
        slice_summary = scope_label

    return {
        "sample_name": sample_name,
        "slice_summary": slice_summary,
        "scope_kind": scope_kind,
        "scope_label": scope_label,
        "scope_note": scope_note,
        "slice_count": slice_count,
        "folder_slice_count": folder_slice_count,
    }


def _choose_display_region(row) -> str:
    parent_name = str(row.get("parent_name", "") or "").strip()
    region_name = str(row.get("region_name", "") or "").strip()
    generic = {
        "root",
        "Basic cell groups and regions",
        "Cerebrum",
        "Cerebral cortex",
        "Cortical plate",
        "Cerebral nuclei",
        "Brain stem",
        "fiber tracts",
        "ventricular systems",
    }
    if parent_name and parent_name not in generic and parent_name != region_name:
        return parent_name
    return region_name or "Unknown"


def _load_detection_metadata(base_dir: Path) -> dict[str, object]:
    path = base_dir / "detection_summary.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _detector_summary(cells, detection_meta: dict[str, object]) -> str:
    detector_counts = detection_meta.get("dedup_detector_counts") or detection_meta.get("detector_counts") or {}
    if isinstance(detector_counts, dict) and detector_counts:
        return ", ".join(sorted(str(name) for name, count in detector_counts.items() if int(count or 0) > 0))
    if "detector" not in cells.columns:
        return "Unknown"
    names = sorted({str(v).strip() for v in cells["detector"].fillna("") if str(v).strip()})
    return ", ".join(names) if names else "Unknown"


def _prepare_chart_inputs(hier_path: Path):
    import pandas as pd

    hier = pd.read_csv(hier_path)
    cells_path = hier_path.parent / "cells_mapped.csv"
    detection_meta = _load_detection_metadata(hier_path.parent)
    if cells_path.exists():
        cells = pd.read_csv(cells_path)
    else:
        cells = pd.DataFrame()

    if not cells.empty:
        total_detected = int(len(cells))
        mapping_status = (
            cells["mapping_status"].fillna("").astype(str)
            if "mapping_status" in cells.columns
            else pd.Series([""] * len(cells), index=cells.index, dtype=object)
        )
        mapped_cells = cells[mapping_status == "ok"].copy()
        mapped_count = int(len(mapped_cells))
        outside_count = int(total_detected - mapped_count)
        source_info = _infer_source_summary(cells)
        regions_mapped = int(mapped_cells.get("region_name", pd.Series(dtype=object)).nunique())
        if detection_meta.get("sampling_mode"):
            counting_mode = str(detection_meta.get("sampling_mode"))
        elif "count_sampling_mode" in cells.columns and not cells["count_sampling_mode"].dropna().empty:
            counting_mode = str(cells["count_sampling_mode"].dropna().astype(str).mode().iloc[0])
        else:
            counting_mode = "unknown"
        detector_summary = _detector_summary(cells, detection_meta)

        major_counter: Counter[str] = Counter()
        display_counter: Counter[str] = Counter()
        for row in mapped_cells.to_dict("records"):
            ids = _parse_structure_ids(row.get("structure_id_path"), row.get("region_id"))
            assigned = None
            for label, group_ids, _color in _ANALYSIS_GROUPS:
                if any(group_id in ids for group_id in group_ids):
                    assigned = label
                    break
            if assigned is None:
                assigned = "Other Mapped"
            major_counter[assigned] += 1
            display_counter[_choose_display_region(row)] += 1

        if outside_count > 0:
            major_counter["Outside Atlas"] += outside_count

        major_rows = []
        color_lookup = {label: color for label, _ids, color in _ANALYSIS_GROUPS}
        color_lookup["Outside Atlas"] = "#EF5350"
        color_lookup["Other Mapped"] = "#78909C"
        for label, count in major_counter.items():
            if count <= 0:
                continue
            major_rows.append({"label": label, "count": int(count), "color": color_lookup.get(label, "#78909C")})
        major_df = pd.DataFrame(major_rows).sort_values("count", ascending=False).reset_index(drop=True)

        top_region_df = (
            pd.DataFrame(
                [{"label": label, "count": int(count)} for label, count in display_counter.items() if count > 0]
            )
            .sort_values("count", ascending=False)
            .head(14)
            .reset_index(drop=True)
        )
        return {
            "major_df": major_df,
            "top_region_df": top_region_df,
            "sample_name": source_info["sample_name"],
            "slice_summary": source_info["slice_summary"],
            "scope_kind": source_info["scope_kind"],
            "scope_label": source_info["scope_label"],
            "scope_note": source_info["scope_note"],
            "slice_count": source_info["slice_count"],
            "folder_slice_count": source_info["folder_slice_count"],
            "total_detected": total_detected,
            "mapped_count": mapped_count,
            "outside_count": outside_count,
            "regions_mapped": regions_mapped,
            "counting_mode": counting_mode,
            "detectors": detector_summary,
            "mode_note": "Single-assignment summary from cells_mapped.csv",
        }

    root_rows = hier[hier["depth"] == 0]
    root_count = int(root_rows["count"].iloc[0]) if not root_rows.empty else int(hier["count"].sum())
    major_rows = []
    for label, group_ids, color in _ANALYSIS_GROUPS:
        subset = hier[hier["region_id"].isin(group_ids)]
        if subset.empty:
            continue
        count = int(subset["count"].max())
        if count > 0:
            major_rows.append({"label": label, "count": count, "color": color})
    major_df = pd.DataFrame(major_rows).sort_values("count", ascending=False).reset_index(drop=True)

    top_region_df = hier.copy()
    top_region_df = top_region_df[(top_region_df["count"] > 0) & (top_region_df["depth"] >= 4) & (top_region_df["depth"] <= 7)]
    top_region_df = top_region_df[
        ~top_region_df["region_name"].fillna("").str.contains("layer", case=False, regex=False)
    ]
    top_region_df = top_region_df.sort_values("count", ascending=False).head(14)[["region_name", "count"]]
    top_region_df = top_region_df.rename(columns={"region_name": "label"}).reset_index(drop=True)

    return {
        "major_df": major_df,
        "top_region_df": top_region_df,
        "sample_name": "Current outputs",
        "slice_summary": "Hierarchy fallback view",
        "scope_kind": "fallback",
        "scope_label": "Hierarchy fallback view",
        "scope_note": "cells_mapped.csv is missing, so this summary falls back to hierarchy totals and should not be used for anatomical interpretation.",
        "slice_count": None,
        "folder_slice_count": None,
        "total_detected": root_count,
        "mapped_count": root_count,
        "outside_count": 0,
        "regions_mapped": int(hier["region_name"].nunique()),
        "counting_mode": str(detection_meta.get("sampling_mode") or "unknown"),
        "detectors": ", ".join(
            sorted(str(name) for name, count in (detection_meta.get("detector_counts") or {}).items() if int(count or 0) > 0)
        )
        or "Unknown",
        "mode_note": "Fallback view from cell_counts_hierarchy.csv",
    }


def build_cell_summary(hier_path: Path) -> dict[str, object]:
    chart = _prepare_chart_inputs(hier_path)
    major_df = chart["major_df"]
    top_region_df = chart["top_region_df"]
    total_detected = int(chart["total_detected"])
    mapped_count = int(chart["mapped_count"])
    outside_count = int(chart["outside_count"])
    mapped_total = max(mapped_count, 1)
    major_total = max(int(major_df["count"].sum()) if not major_df.empty else mapped_count, 1)

    warnings: list[str] = []
    scope_note = str(chart.get("scope_note") or "").strip()
    if scope_note:
        warnings.append(scope_note)
    if total_detected > 0 and outside_count / total_detected >= 0.05:
        warnings.append(
            f"{outside_count:,} cells ({outside_count / total_detected * 100:.1f}%) fell outside the registered atlas coverage."
        )
    if str(chart.get("mode_note", "")).startswith("Fallback"):
        warnings.append("This view is using hierarchy fallback totals because cells_mapped.csv is unavailable.")

    major_regions = []
    for row in major_df.to_dict("records"):
        count = int(row["count"])
        major_regions.append(
            {
                "label": str(row["label"]),
                "count": count,
                "pct": count / major_total if major_total else 0.0,
                "color": str(row["color"]),
            }
        )

    top_regions = []
    for row in top_region_df.to_dict("records"):
        count = int(row["count"])
        top_regions.append(
            {
                "label": str(row["label"]),
                "count": count,
                "pct": count / mapped_total if mapped_total else 0.0,
            }
        )

    top_region = top_regions[0] if top_regions else None

    elapsed_s = 0.0
    det_summary_path = hier_path.parent / "detection_summary.json"
    if det_summary_path.exists():
        try:
            det = json.loads(det_summary_path.read_text(encoding="utf-8"))
            elapsed_s = float(det.get("pipeline_elapsed_s", 0))
        except Exception:
            pass

    return {
        "sample_name": chart["sample_name"],
        "slice_summary": chart["slice_summary"],
        "scope_kind": chart.get("scope_kind", "unknown"),
        "scope_label": chart.get("scope_label", chart["slice_summary"]),
        "counting_mode": str(chart.get("counting_mode", "unknown")),
        "detectors": str(chart.get("detectors", "Unknown")),
        "mode_note": chart["mode_note"],
        "total_detected": total_detected,
        "mapped_count": mapped_count,
        "outside_count": outside_count,
        "mapped_pct": (mapped_count / total_detected) if total_detected else 0.0,
        "outside_pct": (outside_count / total_detected) if total_detected else 0.0,
        "regions_mapped": int(chart["regions_mapped"]),
        "major_regions": major_regions,
        "top_regions": top_regions,
        "top_region": top_region,
        "warnings": warnings,
        "pipeline_elapsed_s": elapsed_s,
    }


def generate_cell_chart(hier_path: Path, chart_path: Path, _project_root: Path) -> None:
    """Generate a cell-count summary chart from outputs tables."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    chart = _prepare_chart_inputs(hier_path)
    major_df = chart["major_df"]
    top_region_df = chart["top_region_df"]
    if major_df.empty:
        raise RuntimeError("No chartable region counts found")

    fig = plt.figure(figsize=(16, 11), facecolor="#0e0e16")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.35, 0.95], hspace=0.28, wspace=0.2)
    ax_major = fig.add_subplot(gs[0, 0])
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_top = fig.add_subplot(gs[1, :])

    for ax in (ax_major, ax_pie, ax_top):
        ax.set_facecolor("#141420")
        ax.tick_params(colors="#cccccc", labelsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")

    major_labels = major_df["label"].tolist()
    major_counts = major_df["count"].astype(int).tolist()
    major_colors = major_df["color"].tolist()
    y = np.arange(len(major_df))
    bars = ax_major.barh(y, major_counts, color=major_colors, edgecolor="none", height=0.72)
    ax_major.set_yticks(y)
    ax_major.set_yticklabels(major_labels, color="#d8d8e0", fontsize=10)
    ax_major.invert_yaxis()
    ax_major.set_xlabel("Cell Count", color="#cccccc", fontsize=10)
    ax_major.set_title("Counts by Analysis Region", color="white", fontsize=14, pad=10)
    ax_major.grid(axis="x", color="#232332", linewidth=0.8, alpha=0.6)
    ax_major.set_axisbelow(True)
    max_major = max(major_counts) if major_counts else 1
    ax_major.set_xlim(0, max_major * 1.15)
    for bar, count in zip(bars, major_counts):
        ax_major.text(
            bar.get_width() + max_major * 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(count):,}",
            va="center",
            ha="left",
            fontsize=10,
            color="#d0d0d8",
        )

    total_major = max(sum(major_counts), 1)
    pie_labels = [
        f"{label}\n{count / total_major * 100:.1f}%"
        if count / total_major >= 0.04
        else ""
        for label, count in zip(major_labels, major_counts)
    ]
    ax_pie.pie(
        major_counts,
        colors=major_colors,
        labels=pie_labels,
        labeldistance=1.08,
        startangle=90,
        counterclock=False,
        textprops={"fontsize": 9, "color": "#d8d8e0"},
        wedgeprops={"edgecolor": "#0e0e16", "linewidth": 1.8},
    )
    ax_pie.set_title("Proportion by Analysis Region", color="white", fontsize=14, pad=10)

    top_labels = top_region_df["label"].astype(str).tolist()
    top_counts = top_region_df["count"].astype(int).tolist()
    top_y = np.arange(len(top_region_df))
    top_palette = ["#8ecae6", "#90be6d", "#f9c74f", "#f9844a", "#f94144", "#b388eb"]
    top_colors = [top_palette[i % len(top_palette)] for i in range(len(top_region_df))]
    top_bars = ax_top.barh(top_y, top_counts, color=top_colors, edgecolor="none", height=0.68)
    ax_top.set_yticks(top_y)
    ax_top.set_yticklabels(top_labels, color="#d8d8e0", fontsize=10)
    ax_top.invert_yaxis()
    ax_top.set_xlabel("Cell Count", color="#cccccc", fontsize=10)
    ax_top.set_title("Top Mapped Regions", color="white", fontsize=14, pad=10)
    ax_top.grid(axis="x", color="#232332", linewidth=0.8, alpha=0.6)
    ax_top.set_axisbelow(True)
    max_top = max(top_counts) if top_counts else 1
    ax_top.set_xlim(0, max_top * 1.12)
    for bar, count in zip(top_bars, top_counts):
        ax_top.text(
            bar.get_width() + max_top * 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(count):,}",
            va="center",
            ha="left",
            fontsize=10,
            color="#d0d0d8",
        )

    fig.suptitle("Cell Count Summary", color="white", fontsize=22, fontweight="bold", x=0.02, ha="left", y=0.985)
    subtitle = (
        f"{chart['sample_name']}  |  {chart['slice_summary']}  |  "
        f"detected {int(chart['total_detected']):,}  |  mapped {int(chart['mapped_count']):,}  |  "
        f"outside atlas {int(chart['outside_count']):,}  |  {int(chart['regions_mapped']):,} regions"
    )
    fig.text(0.5, 0.93, subtitle, ha="center", va="top", color="#d0d0d8", fontsize=12)
    fig.text(
        0.02,
        0.015,
        f"Mode: {chart['mode_note']}. This figure is intended for anatomical interpretation; ancestor totals such as CH are not used as pie slices.",
        ha="left",
        va="bottom",
        color="#9aa0b3",
        fontsize=9,
    )

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path, dpi=140, bbox_inches="tight", facecolor="#0e0e16", edgecolor="none")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Detection confidence samples
# ---------------------------------------------------------------------------


def generate_detection_confidence_samples(
    cells_csv: Path,
    out_dir: Path,
    *,
    sample_count: int = 3,
) -> list[dict[str, object]]:
    """Render representative raw-slice overlays from the final deduplicated cell table."""

    import numpy as np
    import pandas as pd
    import tifffile
    from PIL import Image, ImageDraw, ImageFont

    from scripts.image_utils import norm_u8_robust
    from scripts.make_demo_panel import _crop_bounds

    if not cells_csv.exists():
        return []

    cells = pd.read_csv(cells_csv)
    if cells.empty or "source_slice_path" not in cells.columns:
        return []

    cells["source_slice_path"] = cells["source_slice_path"].fillna("").astype(str)
    cells = cells[cells["source_slice_path"].str.strip() != ""].copy()
    if cells.empty:
        return []

    cells["source_exists"] = cells["source_slice_path"].map(lambda p: Path(p).exists())
    cells = cells[cells["source_exists"]].copy()
    if cells.empty:
        return []

    grouped = (
        cells.groupby(["slice_id", "source_slice_path"], as_index=False)
        .agg(
            count=("cell_id", "size"),
            detector_names=("detector", lambda s: ",".join(sorted({str(v) for v in s if str(v)}))),
        )
        .sort_values(["slice_id", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    if grouped.empty:
        return []

    positions = np.linspace(0, len(grouped) - 1, num=min(sample_count, len(grouped)), dtype=int)
    seen: set[tuple[int, str]] = set()
    selected_rows: list[dict[str, object]] = []
    for pos in positions.tolist():
        row = grouped.iloc[int(pos)]
        key = (int(row["slice_id"]), str(row["source_slice_path"]))
        if key in seen:
            continue
        seen.add(key)
        selected_rows.append(row.to_dict())

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    try:
        font_title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 18)
        font_meta = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    except Exception:
        font_title = font_meta = ImageFont.load_default()

    for idx, sample in enumerate(selected_rows, start=1):
        source_path = Path(str(sample["source_slice_path"]))
        raw = tifffile.imread(str(source_path))
        if raw.ndim == 3:
            raw = raw[..., 0]
        gray = norm_u8_robust(raw)
        raw_rgb = np.stack([gray, gray, gray], axis=-1)
        r0, r1, c0, c1 = _crop_bounds(raw_rgb.shape[:2], pad=18, image=raw_rgb)
        crop = raw_rgb[r0 : r1 + 1, c0 : c1 + 1]

        max_w = 560
        if crop.shape[1] > max_w:
            scale = max_w / float(crop.shape[1])
            target_size = (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale))))
            crop = np.array(Image.fromarray(crop).resize(target_size, Image.Resampling.LANCZOS))
            scale_x = target_size[0] / max((c1 - c0 + 1), 1)
            scale_y = target_size[1] / max((r1 - r0 + 1), 1)
        else:
            scale_x = scale_y = 1.0

        canvas = Image.new("RGBA", (crop.shape[1], crop.shape[0] + 52), (18, 18, 24, 255))
        canvas.alpha_composite(Image.fromarray(crop).convert("RGBA"), (0, 52))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.text((12, 10), f"Sample {idx}", fill=(235, 235, 235, 255), font=font_title)
        meta_text = f"{source_path.name}  |  {int(sample['count']):,} cells"
        draw.text((12, 31), meta_text, fill=(160, 168, 180, 255), font=font_meta)

        slice_rows = cells[
            (cells["slice_id"] == int(sample["slice_id"]))
            & (cells["source_slice_path"] == str(source_path))
        ]
        for row in slice_rows.itertuples(index=False):
            x = (float(row.x) - float(c0)) * float(scale_x)
            y = (float(row.y) - float(r0)) * float(scale_y) + 52.0
            if x < 0 or y < 52 or x >= canvas.width or y >= canvas.height:
                continue
            radius = 4
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=(34, 214, 255, 90),
                outline=(255, 255, 255, 235),
                width=1,
            )

        out_name = f"cellcount_sample_{idx:02d}.png"
        out_path = out_dir / out_name
        canvas.convert("RGB").save(str(out_path), quality=95)
        manifest.append(
            {
                "name": out_name,
                "slice_id": int(sample["slice_id"]),
                "source_name": source_path.name,
                "source_path": str(source_path),
                "count": int(sample["count"]),
                "detectors": str(sample.get("detector_names", "")),
            }
        )

    return manifest


# ---------------------------------------------------------------------------
# Slice comparison image
# ---------------------------------------------------------------------------


def generate_demo_comparison(
    slice_idx: int,
    reg_dir: Path,
    data_dir: Path,
    out_path: Path,
) -> None:
    """Generate side-by-side raw vs atlas comparison for a legacy 2D slice."""

    import numpy as np
    import tifffile as tf
    from PIL import Image, ImageDraw, ImageFont

    from scripts.make_demo_panel import _crop_to_brain, _vibrant_recolor

    ov_path = reg_dir / f"slice_{slice_idx:04d}_overlay.png"
    lbl_path = reg_dir / f"slice_{slice_idx:04d}_registered_label.tif"

    if not ov_path.exists():
        raise FileNotFoundError(f"overlay not found: {ov_path}")

    with Image.open(str(ov_path)) as image:
        ov = np.array(image.convert("RGB"))
    lbl = tf.imread(str(lbl_path)) if lbl_path.exists() else np.zeros(ov.shape[:2], dtype=np.int32)

    raw_files = sorted(data_dir.glob("*.tif"))
    raw_file = raw_files[min(slice_idx, len(raw_files) - 1)] if raw_files else None
    if raw_file:
        raw_orig = tf.imread(str(raw_file))
        p1, p99 = np.percentile(raw_orig[raw_orig > 0], [2, 98]) if raw_orig.max() > 0 else (0, 1)
        raw_norm = np.clip(
            (raw_orig.astype(np.float32) - p1) / (p99 - p1 + 1e-6) * 255,
            0,
            255,
        ).astype(np.uint8)
        raw_rgb = np.stack([raw_norm] * 3, axis=-1)
    else:
        raw_rgb = np.zeros_like(ov)

    vibrant = _vibrant_recolor(ov, lbl)
    raw_crop = _crop_to_brain(raw_rgb, pad=25)
    vib_crop = _crop_to_brain(vibrant, pad=25)

    height = max(raw_crop.shape[0], vib_crop.shape[0])

    def _resize_h(arr, target_h):
        img = Image.fromarray(arr)
        scale = target_h / max(img.height, 1)
        return np.array(
            img.resize((max(1, int(img.width * scale)), target_h), Image.Resampling.LANCZOS)
        )

    raw_r = _resize_h(raw_crop, height)
    vib_r = _resize_h(vib_crop, height)
    divider = np.full((height, 6, 3), 35, dtype=np.uint8)
    combined = np.concatenate([raw_r, divider, vib_r], axis=1)

    width = combined.shape[1]
    header = np.full((50, width, 3), 18, dtype=np.uint8)
    body = Image.fromarray(np.concatenate([header, combined], axis=0))
    draw = ImageDraw.Draw(body)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        font_sm = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = font_sm = ImageFont.load_default()

    draw.text((10, 14), "Raw Lightsheet", fill=(200, 200, 200), font=font)
    draw.text((raw_r.shape[1] + 14, 14), "Brainfast - Atlas Registration", fill=(200, 200, 200), font=font)
    draw.text((width - 260, 32), f"Slice {slice_idx} - Allen CCFv3", fill=(100, 100, 100), font=font_sm)
    body.save(str(out_path), quality=92)


# ---------------------------------------------------------------------------
# 3D registration run renders
# ---------------------------------------------------------------------------


def _resolve_run_artifact(run_dir: Path, raw_value: object, fallback_name: str) -> Path:
    value = str(raw_value or "").strip()
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = run_dir / path
        if path.exists():
            return path
    return run_dir / fallback_name


def _load_registration_context(run_dir: Path) -> tuple[dict[str, object], object, object]:
    import nibabel as nib

    meta = json.loads((run_dir / "registration_metadata.json").read_text(encoding="utf-8"))
    brain_path = _resolve_run_artifact(run_dir, meta.get("registered_brain"), "registered_brain.nii.gz")
    annotation_path = _resolve_run_artifact(
        run_dir, meta.get("annotation_fixed_half"), "annotation_fixed_half.nii.gz"
    )
    if not brain_path.exists() or not annotation_path.exists():
        raise FileNotFoundError(f"missing 3D registration artifacts under {run_dir}")
    return meta, nib.load(str(brain_path)), nib.load(str(annotation_path))


def _slice_signal_to_rgb(brain_slice, mask):
    import numpy as np

    x = brain_slice.astype(np.float32, copy=False)
    values = x[mask] if bool(mask is not None and np.any(mask)) else x[x > 0]
    if values.size == 0:
        gray = np.zeros_like(x, dtype=np.uint8)
    else:
        p1 = float(np.percentile(values, 1.0))
        p99 = float(np.percentile(values, 99.5))
        if p99 <= p1:
            p99 = p1 + 1.0
        gray = np.clip((x - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
        gray = np.power(gray, 0.75)
        gray = (gray * 255.0).astype(np.uint8)
        gray = gray.copy()
        if mask is not None:
            gray[~mask] = 0
    return np.repeat(gray[:, :, None], 3, axis=2)


def _slice_alpha(mask):
    import numpy as np
    from scipy.ndimage import gaussian_filter

    alpha = gaussian_filter(mask.astype(np.float32), sigma=1.4)
    return np.clip((alpha - 0.05) / 0.85, 0.0, 1.0).astype(np.float32)


def _select_panel_slices(annotation_volume, n_slices: int) -> list[int]:
    import numpy as np

    coverage = (annotation_volume > 0).reshape(annotation_volume.shape[0], -1).sum(axis=1)
    valid = np.flatnonzero(coverage > max(64, int(coverage.max() * 0.18)))
    if valid.size == 0:
        valid = np.flatnonzero(coverage > 0)
    if valid.size == 0:
        return [annotation_volume.shape[0] // 2]
    positions = np.linspace(0, valid.size - 1, num=min(n_slices, valid.size), dtype=int)
    selected = [int(valid[pos]) for pos in positions]
    return list(dict.fromkeys(selected))


def _build_overlay_panel(brain_slice, label_slice):
    from scripts.make_demo_panel import _apply_tissue_alpha, _crop_to_brain, _vibrant_recolor

    mask = label_slice > 0
    brain_rgb = _slice_signal_to_rgb(brain_slice, mask)
    overlay = _vibrant_recolor(brain_rgb, label_slice.astype("int32"), tissue_mask=mask)
    alpha = _slice_alpha(mask)
    brain_rgb = _apply_tissue_alpha(brain_rgb, alpha)
    overlay = _apply_tissue_alpha(overlay, alpha)
    return (
        _crop_to_brain(brain_rgb, pad=14, mask=mask),
        _crop_to_brain(overlay, pad=14, mask=mask),
    )


def generate_registration_demo_panel(
    run_dir: Path,
    out_path: Path,
    *,
    n_slices: int = 12,
    cols: int = 4,
    thumb_size: int = 380,
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    meta, brain_img, annotation_img = _load_registration_context(run_dir)
    brain = brain_img.get_fdata(dtype=np.float32)
    annotation = np.asarray(annotation_img.dataobj, dtype=np.int32)
    slices = _select_panel_slices(annotation, n_slices)

    rows = (len(slices) + cols - 1) // cols
    gap = 8
    title_h = 50
    panel_w = cols * (thumb_size + gap) + gap
    panel_h = rows * (thumb_size + gap) + gap + title_h
    panel = Image.new("RGB", (panel_w, panel_h), (18, 18, 24))
    draw_panel = ImageDraw.Draw(panel)
    sample_name = Path(str(meta.get("input_source", run_dir.name))).stem

    try:
        font_big = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
        font_sm = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    except Exception:
        font_big = font_sm = ImageFont.load_default()

    draw_panel.text((gap, 10), f"Brainfast - {sample_name} Atlas Registration", fill=(230, 230, 230), font=font_big)

    for pos, z in enumerate(slices):
        brain_crop, overlay_crop = _build_overlay_panel(brain[z], annotation[z])
        height = max(brain_crop.shape[0], overlay_crop.shape[0])

        def _resize_h(arr):
            img = Image.fromarray(arr)
            scale = height / max(img.height, 1)
            return np.array(
                img.resize((max(1, int(img.width * scale)), height), Image.Resampling.LANCZOS)
            )

        brain_r = _resize_h(brain_crop)
        overlay_r = _resize_h(overlay_crop)
        divider = np.full((height, 6, 3), 35, dtype=np.uint8)
        combined = np.concatenate([brain_r, divider, overlay_r], axis=1)

        scale = thumb_size / max(combined.shape[0], combined.shape[1], 1)
        nw = max(1, int(round(combined.shape[1] * scale)))
        nh = max(1, int(round(combined.shape[0] * scale)))
        thumb = Image.fromarray(combined).resize((nw, nh), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (thumb_size, thumb_size), (18, 18, 24))
        cell.paste(thumb, ((thumb_size - nw) // 2, (thumb_size - nh) // 2))

        ap_mm = round(float(z) * 0.025, 2)
        ImageDraw.Draw(cell).text(
            (6, thumb_size - 22),
            f"Slice {z:03d}  atlas_z={z}  AP={ap_mm}mm",
            fill=(235, 235, 235),
            font=font_sm,
        )

        row_i = pos // cols
        col_i = pos % cols
        x = gap + col_i * (thumb_size + gap)
        y = title_h + gap + row_i * (thumb_size + gap)
        panel.paste(cell, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(str(out_path), quality=92)


def generate_registration_best_slice(run_dir: Path, out_path: Path) -> None:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    _, brain_img, annotation_img = _load_registration_context(run_dir)
    brain = brain_img.get_fdata(dtype=np.float32)
    annotation = np.asarray(annotation_img.dataobj, dtype=np.int32)
    coverage = (annotation > 0).reshape(annotation.shape[0], -1).sum(axis=1)
    best_z = int(np.argmax(coverage))
    brain_crop, overlay_crop = _build_overlay_panel(brain[best_z], annotation[best_z])
    height = max(brain_crop.shape[0], overlay_crop.shape[0])

    def _resize_h(arr):
        img = Image.fromarray(arr)
        scale = height / max(img.height, 1)
        return np.array(
            img.resize((max(1, int(img.width * scale)), height), Image.Resampling.LANCZOS)
        )

    brain_r = _resize_h(brain_crop)
    overlay_r = _resize_h(overlay_crop)
    divider = np.full((height, 8, 3), 35, dtype=np.uint8)
    combined = np.concatenate([brain_r, divider, overlay_r], axis=1)

    width = combined.shape[1]
    header = np.full((55, width, 3), 18, dtype=np.uint8)
    body = Image.fromarray(np.concatenate([header, combined], axis=0))
    draw = ImageDraw.Draw(body)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
        font_sm = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except Exception:
        font = font_sm = ImageFont.load_default()

    draw.text((12, 14), "Registered Lightsheet", fill=(200, 200, 200), font=font)
    draw.text((brain_r.shape[1] + 14, 14), "Brainfast - Allen CCFv3 Atlas Registration", fill=(200, 200, 200), font=font)
    draw.text((width - 250, 36), f"Slice {best_z} - fixed space", fill=(100, 100, 100), font=font_sm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body.save(str(out_path), quality=95)


def generate_registration_annotated_slice(
    run_dir: Path,
    out_path: Path,
    *,
    structure_csv: Path,
    top_n: int = 12,
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from scripts.make_demo_panel import (
        _apply_tissue_alpha,
        _combined_structure_lookup,
        _crop_to_brain,
        _select_region_annotations,
        _vibrant_recolor,
    )

    _, brain_img, annotation_img = _load_registration_context(run_dir)
    brain = brain_img.get_fdata(dtype=np.float32)
    annotation = np.asarray(annotation_img.dataobj, dtype=np.int32)
    coverage = (annotation > 0).reshape(annotation.shape[0], -1).sum(axis=1)
    best_z = int(np.argmax(coverage))
    label_slice = annotation[best_z]
    mask = label_slice > 0

    brain_rgb = _slice_signal_to_rgb(brain[best_z], mask)
    annotated = _vibrant_recolor(brain_rgb, label_slice, tissue_mask=mask)
    annotated = _apply_tissue_alpha(annotated, _slice_alpha(mask))
    annotated = _crop_to_brain(annotated, pad=20, mask=mask)
    label_crop = _crop_to_brain(label_slice.astype(np.int32), pad=20, mask=mask)

    target_h = 560
    scale = target_h / max(annotated.shape[0], 1)
    width = max(1, int(annotated.shape[1] * scale))
    ann_r = np.array(Image.fromarray(annotated).resize((width, target_h), Image.Resampling.LANCZOS))
    lbl_r = np.array(
        Image.fromarray(label_crop.astype(np.float32)).resize((width, target_h), Image.Resampling.NEAREST)
    ).astype(np.int32)

    structure_lookup = _combined_structure_lookup(structure_csv)
    region_entries = _select_region_annotations(
        lbl_r,
        structure_lookup=structure_lookup,
        top_n=top_n,
        min_pixels=90,
    )

    result_img = Image.fromarray(ann_r)
    draw = ImageDraw.Draw(result_img)
    try:
        font_sm = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 12)
        font_title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 20)
        font_leg = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
        font_leg_b = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 13)
    except Exception:
        font_sm = font_title = font_leg = font_leg_b = ImageFont.load_default()

    legend_entries = []
    for entry in region_entries:
        cy, cx = int(entry["cy"]), int(entry["cx"])
        acro = str(entry["acro"])
        name = str(entry["name"])
        color = tuple(int(v) for v in entry["color"])
        legend_entries.append((color, acro, name))
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color, outline=(255, 255, 255))
        tx, ty = cx + 7, cy - 8
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            draw.text((tx + dx, ty + dy), acro, fill=(0, 0, 0), font=font_sm)
        draw.text((tx, ty), acro, fill=color, font=font_sm)

    legend_h = 38 + len(legend_entries) * 22
    legend = Image.new("RGB", (width, legend_h), (28, 28, 34))
    leg_draw = ImageDraw.Draw(legend)
    leg_draw.text(
        (12, 8),
        "Dots mark representative anchor points for the labeled regions below.",
        fill=(150, 156, 170),
        font=font_leg,
    )
    for idx, (color, acro, name) in enumerate(legend_entries):
        y = 26 + idx * 22
        leg_draw.rectangle([12, y + 1, 28, y + 15], fill=color)
        leg_draw.text((36, y), acro, fill=color, font=font_leg_b)
        acro_w = leg_draw.textlength(acro, font=font_leg_b) if hasattr(leg_draw, "textlength") else len(acro) * 8
        leg_draw.text((36 + acro_w + 8, y), f"- {name[:38]}", fill=(180, 180, 180), font=font_leg)

    header = Image.new("RGB", (width, 50), (18, 18, 24))
    ImageDraw.Draw(header).text(
        (12, 14),
        "Brainfast - Allen CCFv3 Atlas Registration",
        fill=(200, 200, 200),
        font=font_title,
    )

    out = Image.new("RGB", (width, header.height + result_img.height + legend.height), (18, 18, 24))
    out.paste(header, (0, 0))
    out.paste(result_img, (0, header.height))
    out.paste(legend, (0, header.height + result_img.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(out_path), quality=95)

"""export_paper_report.py — Generate paper-style AAV toolbox result report.

Reads pipeline outputs and produces:
  1. region_count_density.png   — bar chart: cell count + density per region
  2. specificity_summary.csv    — specificity table (per-region, from colocalization_summary)
  3. specificity_summary.html   — same table as HTML
  4. representative_panel.png   — montage of representative-slice overlays (one per key region)
  5. paper_report_summary.txt   — brief methods/results text summary

Usage:
    python export_paper_report.py --outputs-dir project/outputs

Or called from code:
    from scripts.export_paper_report import generate_paper_report
    generate_paper_report(outputs_dir=Path("project/outputs"))
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _save_bar_chart(
    df: pd.DataFrame,
    region_col: str,
    count_col: str,
    density_col: str | None,
    out_png: Path,
    top_n: int = 20,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] matplotlib not available; skipping bar chart")
        return

    df = df.copy().dropna(subset=[count_col]).sort_values(count_col, ascending=False).head(top_n)
    labels = df[region_col].astype(str).tolist()
    counts = df[count_col].astype(float).tolist()

    fig, ax1 = plt.subplots(figsize=(max(8, len(labels) * 0.55), 5))
    x = np.arange(len(labels))

    ax1.bar(x, counts, color="#4878CF", alpha=0.85, label="Cell count")
    ax1.set_ylabel("Cell count", color="#4878CF")
    ax1.tick_params(axis="y", labelcolor="#4878CF")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    if density_col is not None and density_col in df.columns:
        densities = df[density_col].fillna(0).astype(float).tolist()
        ax2 = ax1.twinx()
        ax2.plot(
            x, densities, color="#D65F5F", marker="o", linewidth=1.5, label="Density (cells/mm²)"
        )
        ax2.set_ylabel("Density (cells/mm²)", color="#D65F5F")
        ax2.tick_params(axis="y", labelcolor="#D65F5F")
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=8)

    ax1.set_title(f"Top {len(labels)} regions — cell count & density", fontsize=11)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)


def _save_specificity_table(df: pd.DataFrame, out_csv: Path, out_html: Path) -> None:
    cols = [
        c
        for c in (
            "region_id",
            "region_name",
            "acronym",
            "hemisphere",
            "dtom_pos_count",
            "marker_pos_count",
            "double_pos_count",
            "specificity",
            "sensitivity",
        )
        if c in df.columns
    ]
    sub = df[cols].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    sub.to_html(out_html, index=False, float_format="{:.3f}".format)


def _build_representative_panel(
    summary_df: pd.DataFrame,
    overlay_dir: Path,
    out_png: Path,
    top_n: int = 12,
) -> None:
    """Tile overlay PNGs for top-N representative slices."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("[report] matplotlib / Pillow not available; skipping representative panel")
        return

    if "representative_slice_id" not in summary_df.columns:
        return

    rows = (
        summary_df.dropna(subset=["representative_slice_id"])
        .sort_values("count", ascending=False)
        .head(top_n)
    )

    images = []
    labels = []
    for _, row in rows.iterrows():
        sid = int(row["representative_slice_id"])
        # Match overlay file: slice_<sid:04d>_overlay.png
        candidates = list(overlay_dir.glob(f"slice_{sid:04d}_overlay.png"))
        if not candidates:
            # Also try the qc_overlays copy format
            candidates = list((overlay_dir.parent / "qc_overlays").glob(f"overlay_{sid:03d}.png"))
        if candidates:
            try:
                img = Image.open(str(candidates[0])).convert("RGB")
                images.append(img)
                acronym = row.get("acronym", "")
                labels.append(f"{acronym} (s{sid})")
            except Exception:
                pass

    if not images:
        return

    n = len(images)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    _thumb_size = (200, 200)  # noqa: F841 — reserved for future thumbnail resizing

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for i, (img, lbl) in enumerate(zip(images, labels, strict=False)):
        ax = axes_flat[i]
        ax.imshow(img)
        ax.set_title(lbl, fontsize=7)
        ax.axis("off")
    for j in range(len(images), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Representative section overlays", fontsize=11)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)


def _write_text_summary(
    outputs_dir: Path,
    summary_df: pd.DataFrame | None,
    coloc_df: pd.DataFrame | None,
    out_txt: Path,
) -> None:
    lines = [
        "=== Brainfast — Paper AAV Toolbox Report Summary ===",
        "",
    ]
    if summary_df is not None and not summary_df.empty:
        n_regions = len(summary_df)
        total_cells = int(summary_df["count"].sum()) if "count" in summary_df.columns else "?"
        lines += [
            f"Regions analysed: {n_regions}",
            f"Total mapped cells (representative slices): {total_cells}",
        ]
        if "density_cells_per_mm2" in summary_df.columns:
            top = summary_df.dropna(subset=["density_cells_per_mm2"]).nlargest(
                3, "density_cells_per_mm2"
            )
            if not top.empty:
                lines.append("Top regions by density (cells/mm²):")
                for _, r in top.iterrows():
                    lines.append(
                        f"  {r.get('acronym', r.get('region_name', r['region_id']))}: "
                        f"{r['density_cells_per_mm2']:.2f}"
                    )
    if coloc_df is not None and not coloc_df.empty:
        lines.append("")
        lines.append("Colocalization (top regions by specificity):")
        if "specificity" in coloc_df.columns:
            top_sp = coloc_df.dropna(subset=["specificity"]).nlargest(3, "specificity")
            for _, r in top_sp.iterrows():
                lines.append(
                    f"  {r.get('acronym', r.get('region_name', r['region_id']))}: "
                    f"specificity={r['specificity']:.3f}"
                )
    lines += [
        "",
        "Output files:",
        "  paper_aav_region_summary.csv     — region count / density / representative slice",
        "  region_count_density.png         — bar chart",
        "  specificity_summary.csv/.html    — colocalization metrics",
        "  representative_panel.png         — overlay montage",
    ]
    out_txt.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_paper_report(outputs_dir: Path) -> None:
    """Generate all paper-style report files from pipeline outputs.

    Safe to call even if some intermediate files are missing — those outputs
    will be skipped with a warning.
    """
    outputs_dir = Path(outputs_dir)
    report_dir = outputs_dir / "paper_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _load_if_exists(outputs_dir / "paper_aav_region_summary.csv")
    coloc_df = _load_if_exists(outputs_dir / "colocalization_summary.csv")

    # 1. Count / density bar chart
    if summary_df is not None and not summary_df.empty:
        region_col = "acronym" if "acronym" in summary_df.columns else "region_id"
        _save_bar_chart(
            summary_df,
            region_col=region_col,
            count_col="count",
            density_col="density_cells_per_mm2",
            out_png=report_dir / "region_count_density.png",
        )
        print("[report] region_count_density.png written")
    else:
        print("[report] paper_aav_region_summary.csv not found; skipping bar chart")

    # 2. Specificity / sensitivity table
    if coloc_df is not None and not coloc_df.empty:
        _save_specificity_table(
            coloc_df,
            out_csv=report_dir / "specificity_summary.csv",
            out_html=report_dir / "specificity_summary.html",
        )
        print("[report] specificity_summary.csv / .html written")
    else:
        print("[report] colocalization_summary.csv not found; skipping specificity table")

    # 3. Representative section panel
    if summary_df is not None and not summary_df.empty:
        overlay_dir = outputs_dir / "registered_slices"
        _build_representative_panel(
            summary_df,
            overlay_dir=overlay_dir,
            out_png=report_dir / "representative_panel.png",
        )
        print("[report] representative_panel.png written")

    # 4. Text summary
    _write_text_summary(
        outputs_dir=outputs_dir,
        summary_df=summary_df,
        coloc_df=coloc_df,
        out_txt=report_dir / "paper_report_summary.txt",
    )
    print(f"[report] Report written to {report_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper-style AAV toolbox report")
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Path to pipeline outputs directory (default: outputs)",
    )
    args = parser.parse_args()
    generate_paper_report(Path(args.outputs_dir))

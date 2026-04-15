"""Re-run quantification stage using existing truth export data.

Usage: python scripts/rerun_quantify.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.main import _quantify_against_exported_truth


def main():
    outputs_dir = Path("outputs/ChATe27")
    truth_dir = outputs_dir / "truth_export"
    cfg_path = outputs_dir / "run_config_runtime.json"

    if not cfg_path.exists():
        cfg_path = Path("outputs/run_config_runtime.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))

    # Build truth_rows from existing files
    merged_dir = outputs_dir / "tmp_merged"
    merged_slices = sorted(merged_dir.glob("merged_*.tif"))

    truth_rows = []
    for idx, mp in enumerate(merged_slices):
        label_path = truth_dir / f"slice_{idx:04d}_registered_label.tif"
        overlay_path = truth_dir / f"slice_{idx:04d}_overlay.png"
        if not label_path.exists():
            print(f"WARNING: missing label for slice {idx}")
            continue
        truth_rows.append(
            {
                "slice_id": idx,
                "real_slice_path": str(mp),
                "registered_label_path": str(label_path),
                "overlay_path": str(overlay_path),
            }
        )

    print(f"Re-running quantification on {len(truth_rows)} slices...")
    result = _quantify_against_exported_truth(
        truth_rows=truth_rows,
        cfg=cfg,
        outputs_dir=outputs_dir,
    )
    print(f"Done! Result: {result}")

    # Quick summary
    import pandas as pd

    cells = pd.read_csv(outputs_dir / "cells_mapped.csv")
    mapped = cells[cells["mapping_status"] != "outside_registered_slice"]
    outside = cells[cells["mapping_status"] == "outside_registered_slice"]
    print(f"\nTotal cells: {len(cells)}")
    print(f"Mapped to regions: {len(mapped)} ({100 * len(mapped) / max(len(cells), 1):.1f}%)")
    print(f"Outside atlas: {len(outside)} ({100 * len(outside) / max(len(cells), 1):.1f}%)")

    if len(mapped) > 0:
        top = mapped.groupby("region_name").size().sort_values(ascending=False).head(15)
        print("\nTop 15 brain regions:")
        for name, count in top.items():
            print(f"  {name}: {count}")


if __name__ == "__main__":
    main()

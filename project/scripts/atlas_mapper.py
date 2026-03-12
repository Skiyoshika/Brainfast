from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    import nibabel as nib
except Exception:
    nib = None


def _to_voxel(x_um: float, y_um: float, z_um: float, voxel_um: float) -> tuple[int, int, int]:
    return int(round(x_um / voxel_um)), int(round(y_um / voxel_um)), int(round(z_um / voxel_um))


def map_cells_with_label_volume(
    cells: pd.DataFrame,
    *,
    label_nii: Path | None,
    structure_csv: Path | None,
    pixel_size_um: float,
    slice_spacing_um: float,
    atlas_voxel_um: float = 25.0,
) -> pd.DataFrame:
    """
    Real atlas mapping interface.
    If label volume unavailable, falls back to deterministic placeholder buckets.
    """
    out = cells.copy()
    out["x_um"] = out["x"] * pixel_size_um
    out["y_um"] = out["y"] * pixel_size_um
    out["z_um"] = out["slice_id"] * slice_spacing_um

    if label_nii and label_nii.exists() and nib is not None:
        img = nib.load(str(label_nii))
        data = np.asarray(img.get_fdata(), dtype=np.int32)
        sx, sy, sz = data.shape

        region_ids = []
        for _, r in out.iterrows():
            vx, vy, vz = _to_voxel(float(r.x_um), float(r.y_um), float(r.z_um), atlas_voxel_um)
            if 0 <= vx < sx and 0 <= vy < sy and 0 <= vz < sz:
                region_ids.append(int(data[vx, vy, vz]))
            else:
                region_ids.append(0)
        out["region_id"] = region_ids
    else:
        out["region_id"] = (out["slice_id"] % 3).map({0: 315, 1: 1089, 2: 549})

    if structure_csv and structure_csv.exists():
        st = pd.read_csv(structure_csv)
        key = "id" if "id" in st.columns else "region_id"
        name_col = "name" if "name" in st.columns else ("region_name" if "region_name" in st.columns else None)
        acr_col = "acronym" if "acronym" in st.columns else None
        if name_col:
            st2 = st[[key, name_col] + ([acr_col] if acr_col else [])].copy()
            st2 = st2.rename(columns={key: "region_id", name_col: "region_name"})
            out = out.merge(st2, on="region_id", how="left")
    if "region_name" not in out.columns:
        out["region_name"] = out["region_id"].map({315: "Cortex/MOp", 1089: "Hippocampus/CA1", 549: "Thalamus/VAL"}).fillna("unknown")

    out["hemisphere"] = out["x"].map(lambda x: "left" if x % 2 < 1 else "right")
    return out

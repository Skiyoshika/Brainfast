from pathlib import Path
import pandas as pd
from scripts.align_nonlinear import apply_landmark_nonlinear

project = Path(r"D:\brain-atlas-cellcount-work\project")
outdir = project / "outputs"
real = outdir / "test_real.tif"
atlas = outdir / "test_label.tif"
bad_pairs = outdir / "landmark_pairs_bad.csv"
out = outdir / "aligned_label_nonlinear_bad.tif"

pd.DataFrame([
    {"atlas_x": 10, "atlas_y": 10, "real_x": 12, "real_y": 11},
    {"atlas_x": 30, "atlas_y": 40, "real_x": 33, "real_y": 42},
    {"atlas_x": 80, "atlas_y": 90, "real_x": 81, "real_y": 95},
]).to_csv(bad_pairs, index=False)

try:
    res = apply_landmark_nonlinear(real, atlas, bad_pairs, out)
    print("UNEXPECTED_OK", res)
except Exception as e:
    print("EXPECTED_ERR", str(e))
    print("BAD_PAIRS", bad_pairs)

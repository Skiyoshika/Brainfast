from pathlib import Path
from scripts.align_nonlinear import apply_landmark_nonlinear

project = Path(r"D:\brain-atlas-cellcount-work\project")
real = project / "outputs" / "test_real.tif"
atlas = project / "outputs" / "test_label.tif"
pairs = project / "outputs" / "landmark_pairs.csv"
out = project / "outputs" / "aligned_label_nonlinear.tif"

print(f"REAL_EXISTS {real.exists()}")
print(f"ATLAS_EXISTS {atlas.exists()}")
print(f"PAIRS_EXISTS {pairs.exists()}")
try:
    res = apply_landmark_nonlinear(real, atlas, pairs, out)
    print("RESULT_OK", res)
except Exception as e:
    print("RESULT_ERR", str(e))

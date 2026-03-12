"""
下载 Allen CCFv3 annotation_25.nrrd 并转换为 annotation_25.nii.gz
运行方式: python download_atlas.py
"""
import urllib.request
import os
import sys

URL  = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd"
NRRD = "annotation_25.nrrd"
NII  = "annotation_25.nii.gz"

# ── 1. Download ──────────────────────────────────────────────────────────────
if not os.path.exists(NRRD):
    print(f"Downloading {URL} ...")
    def _progress(block, block_size, total):
        done = block * block_size
        pct  = done / total * 100 if total > 0 else 0
        mb   = done / 1024 / 1024
        tot  = total / 1024 / 1024
        print(f"\r  {pct:5.1f}%  {mb:.1f} / {tot:.1f} MB", end="", flush=True)
    urllib.request.urlretrieve(URL, NRRD, reporthook=_progress)
    print(f"\nSaved → {NRRD}")
else:
    print(f"Already exists: {NRRD}")

# ── 2. Convert NRRD → NIfTI ──────────────────────────────────────────────────
if not os.path.exists(NII):
    print("Converting to NIfTI ...")
    import nrrd
    import nibabel as nib
    import numpy as np

    data, header = nrrd.read(NRRD)
    print(f"  shape: {data.shape}  dtype: {data.dtype}")

    # NRRD space directions → NIfTI affine (voxel size in mm)
    # CCFv3 25µm → spacing = 0.025 mm
    spacings = header.get("space directions", None)
    if spacings is not None:
        spacings = np.array(spacings, dtype=float)
        voxel_mm = np.abs(np.diag(spacings))
    else:
        voxel_mm = np.array([0.025, 0.025, 0.025])

    affine = np.diag(list(voxel_mm) + [1.0])
    img = nib.Nifti1Image(data.astype(np.int32), affine)
    img.header.set_zooms(voxel_mm)
    nib.save(img, NII)
    print(f"Saved → {NII}  ({os.path.getsize(NII)/1024/1024:.1f} MB)")
else:
    print(f"Already exists: {NII}")

print("\nDone! Atlas file ready.")
print(f"Full path: {os.path.abspath(NII)}")

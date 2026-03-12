"""
Thin wrappers for reusing selected logic from
uci-allen-brainrepositorycodegui without pulling full legacy pipeline.
"""

from pathlib import Path


def locate_legacy_repo(base: Path) -> Path:
    repo = base / "repos" / "uci-allen-brainrepositorycodegui"
    if not repo.exists():
        raise FileNotFoundError(f"Legacy repo not found: {repo}")
    return repo


def reuse_targets():
    return {
        "registration_scaffold": "run_registration_cellcounting.py",
        "region_hierarchy": ["cell_regions.py", "count_labels.py"],
        "nii_converter": "tif_to_nii.py",
    }

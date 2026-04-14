"""A/B test: ml_flip=True vs ml_flip=False on real samples.

Picks representative slices from existing data, runs ANTs 3D registration
twice (with/without ML flip), and compares NMI+Dice.
Writes results to stdout and a summary CSV.

Usage:
    # Convenience shortcut (maps to known config + data dir)
    python scripts/ml_flip_ab_test.py --sample 35

    # Explicit config + input (any sample)
    python scripts/ml_flip_ab_test.py --config configs/run_config_35.json --input-dir data/35_C0_test

    # Mix: use --sample for config lookup, override input dir
    python scripts/ml_flip_ab_test.py --sample 35 --input-dir data/35_C0_demo
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Ensure project importability
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root.parent))

# Known sample shortcuts: sample_id -> (config_relative, [input_dir_candidates])
_KNOWN_SAMPLES: dict[int, tuple[str, list[str]]] = {
    35: (
        "configs/run_config_35.json",
        ["data/35_C0_test", "data/35_C0_demo"],
    ),
    41: (
        "configs/run_config_41.json",
        ["data/41_C0_test"],
    ),
    44: (
        "configs/run_config_44.json",
        ["data/44_C0_test"],
    ),
}


def _resolve_sample_shortcut(
    sample_id: int,
    project_root: Path,
) -> tuple[Path, Path]:
    """Return (config_path, input_dir) for a known sample shortcut."""
    if sample_id not in _KNOWN_SAMPLES:
        known = ", ".join(str(k) for k in sorted(_KNOWN_SAMPLES))
        raise SystemExit(
            f"Unknown --sample {sample_id}. Known samples: {known}. "
            f"Use --config and --input-dir for unlisted samples."
        )
    cfg_rel, input_candidates = _KNOWN_SAMPLES[sample_id]
    config_path = project_root / cfg_rel
    input_dir: Path | None = None
    for candidate in input_candidates:
        p = project_root / candidate
        if p.exists():
            input_dir = p
            break
    if input_dir is None:
        tried = ", ".join(input_candidates)
        raise SystemExit(f"No input data found for sample {sample_id}. Tried: {tried}")
    return config_path, input_dir


def _run_pipeline(cfg: dict, input_dir: Path, output_dir: Path) -> dict:
    """Run whole-brain 3D registration pipeline and return metrics."""
    from scripts.main import run_real_input

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_real_input(cfg, input_dir, output_dir=output_dir)

    metrics_csv = output_dir / "ants_registration" / "registration_metrics.csv"
    if not metrics_csv.exists():
        return {"error": "no metrics CSV"}

    metrics = {}
    for line in metrics_csv.read_text().strip().split("\n")[1:]:
        parts = line.split(",")
        if len(parts) == 2:
            metrics[parts[0].strip()] = float(parts[1].strip())
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ml_flip A/B test on real samples",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample number shortcut (e.g. 35). Resolves to a known config "
        "and data directory. Ignored when --config is provided.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to run config JSON (relative to project root or absolute).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to input slice directory (relative to project root or absolute).",
    )
    parser.add_argument(
        "--slices",
        type=int,
        default=5,
        help="Number of slices for quick test",
    )
    args = parser.parse_args()

    project_root = _root

    # --- Resolve config_path and input_dir ---------------------------------
    config_path: Path | None = None
    input_dir: Path | None = None

    if args.config is not None:
        # Explicit config provided
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
        if not input_dir.is_absolute():
            input_dir = project_root / input_dir

    # Fall back to --sample shortcut for any value still missing
    if config_path is None or input_dir is None:
        sample_id = args.sample if args.sample is not None else 35
        shortcut_cfg, shortcut_input = _resolve_sample_shortcut(sample_id, project_root)
        if config_path is None:
            config_path = shortcut_cfg
        if input_dir is None:
            input_dir = shortcut_input

    # Validate paths
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    with open(config_path) as f:
        base_cfg = json.load(f)

    # Derive a label for display
    sample_label = str(args.sample) if args.sample is not None else config_path.stem

    print("=" * 60)
    print(f"  ml_flip A/B Test — {sample_label}")
    print(f"  Config:  {config_path}")
    print(f"  Input:   {input_dir} ({len(list(input_dir.glob('z*.tif')))} slices)")
    print("=" * 60)

    # Per-sample output directory (never overwrites other samples)
    sample_dir = project_root / "outputs" / "ml_flip_ab" / f"sample_{sample_label}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    import datetime

    run_start = datetime.datetime.now().isoformat(timespec="seconds")

    results = {}
    output_dirs: dict[str, Path] = {}
    for flip_val in [False, True]:
        label = f"ml_flip={'True' if flip_val else 'False'}"
        print(f"\n--- Running: {label} ---")
        cfg = json.loads(json.dumps(base_cfg))
        cfg["registration"]["ml_flip"] = flip_val
        output_dir = sample_dir / f"ml_flip_{str(flip_val).lower()}"
        output_dirs[label] = output_dir

        t0 = time.time()
        try:
            metrics = _run_pipeline(cfg, input_dir, output_dir)
        except Exception as exc:
            metrics = {"error": str(exc)}
        elapsed = time.time() - t0

        metrics["elapsed_s"] = round(elapsed, 1)
        results[label] = metrics
        print(f"  {label}: {metrics}")

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for label, m in results.items():
        nmi = m.get("NMI", "N/A")
        dice = m.get("Dice", "N/A")
        ssim = m.get("SSIM", "N/A")
        ncc = m.get("NCC", "N/A")
        print(f"  {label}: NMI={nmi}, Dice={dice}, SSIM={ssim}, NCC={ncc}")

    # Determine winner
    a_nmi = results.get("ml_flip=False", {}).get("NMI", 0)
    b_nmi = results.get("ml_flip=True", {}).get("NMI", 0)
    a_dice = results.get("ml_flip=False", {}).get("Dice", 0)
    b_dice = results.get("ml_flip=True", {}).get("Dice", 0)

    if isinstance(a_nmi, (int, float)) and isinstance(b_nmi, (int, float)):
        winner_nmi = "ml_flip=False" if a_nmi >= b_nmi else "ml_flip=True"
        winner_dice = "ml_flip=False" if a_dice >= b_dice else "ml_flip=True"
        print(
            f"\n  Winner by NMI: {winner_nmi} ({max(a_nmi, b_nmi):.4f} vs {min(a_nmi, b_nmi):.4f})"
        )
        print(
            f"  Winner by Dice: {winner_dice} "
            f"({max(a_dice, b_dice):.4f} vs {min(a_dice, b_dice):.4f})"
        )
    else:
        print("\n  Could not determine winner (errors in one or both runs)")

    # Write per-sample summary CSV
    csv_path = sample_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("variant,NMI,Dice,SSIM,NCC,MSE,PSNR,elapsed_s\n")
        for label, m in results.items():
            row = [
                label,
                str(m.get("NMI", "")),
                str(m.get("Dice", "")),
                str(m.get("SSIM", "")),
                str(m.get("NCC", "")),
                str(m.get("MSE", "")),
                str(m.get("PSNR", "")),
                str(m.get("elapsed_s", "")),
            ]
            f.write(",".join(row) + "\n")

    # Write per-sample manifest
    run_end = datetime.datetime.now().isoformat(timespec="seconds")
    cellpose_used = all(
        "cpsam" in str(m.get("error", ""))
        or m.get("NMI") is not None
        for m in results.values()
    )
    manifest = {
        "sample_label": sample_label,
        "config_path": str(config_path),
        "input_dir": str(input_dir),
        "slice_count": len(list(input_dir.glob("z*.tif"))),
        "run_start": run_start,
        "run_end": run_end,
        "output_dirs": {k: str(v) for k, v in output_dirs.items()},
        "summary_csv": str(csv_path),
        "detector_used": "LoG_fallback" if not cellpose_used else "cpsam",
        "atlas_hemisphere": base_cfg.get("registration", {}).get("atlas_hemisphere", "unknown"),
    }
    manifest_path = sample_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Results saved to: {csv_path}")
    print(f"  Manifest saved to: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A/B test: ml_flip=True vs ml_flip=False on real ChATe27 samples.

Picks 5 representative slices from the existing demo data, runs ANTs
3D registration twice (with/without ML flip), and compares NMI+Dice.
Writes results to stdout and a summary CSV.

Usage:
    python scripts/ml_flip_ab_test.py --sample 35
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
    parser = argparse.ArgumentParser(description="ml_flip A/B test on real samples")
    parser.add_argument("--sample", type=int, default=35, help="Sample number (35/39/41)")
    parser.add_argument("--slices", type=int, default=5, help="Number of slices for quick test")
    args = parser.parse_args()

    project_root = _root
    config_path = project_root / "configs" / "run_config_35.json"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    with open(config_path) as f:
        base_cfg = json.load(f)

    # Use sparse test data (5 slices) for quick comparison
    input_dir = project_root / "data" / "35_C0_test"
    if not input_dir.exists():
        # Fall back to demo data
        input_dir = project_root / "data" / "35_C0_demo"

    print("=" * 60)
    print(f"  ml_flip A/B Test — Sample {args.sample}")
    print(f"  Input: {input_dir} ({len(list(input_dir.glob('z*.tif')))} slices)")
    print("=" * 60)

    results = {}
    for flip_val in [False, True]:
        label = f"ml_flip={'True' if flip_val else 'False'}"
        print(f"\n--- Running: {label} ---")
        cfg = json.loads(json.dumps(base_cfg))
        cfg["registration"]["ml_flip"] = flip_val
        output_dir = project_root / "outputs" / f"ab_test_mlflip_{str(flip_val).lower()}"

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
            f"  Winner by Dice: {winner_dice} ({max(a_dice, b_dice):.4f} vs {min(a_dice, b_dice):.4f})"
        )
    else:
        print("\n  Could not determine winner (errors in one or both runs)")

    # Write CSV
    csv_path = project_root / "outputs" / "ml_flip_ab_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"\n  Results saved to: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

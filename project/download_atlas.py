from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.asset_bootstrap import ensure_atlas_assets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure Brainfast atlas assets are available locally."
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent),
        help="Path to the project/ directory",
    )
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Accepted for compatibility; the script always runs in ensure mode.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only inspect and convert local assets; do not fetch from the network",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final asset status as JSON",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    status = ensure_atlas_assets(
        project_root,
        allow_download=not args.no_download,
        logger=lambda msg: print(f"[atlas] {msg}"),
    )
    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
    else:
        print("[atlas] annotation ready:", status["annotationReady"])
        print("[atlas] structure ready:", status["structureReady"])
        print("[atlas] structure source:", status["structurePath"] or "<missing>")
    return 0 if status["allRequiredReady"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

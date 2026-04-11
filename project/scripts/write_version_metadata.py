from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

DEFAULT_REPOSITORY = "https://github.com/Skiyoshika/Brainfast"
DEFAULT_RELEASES_API = "https://api.github.com/repos/Skiyoshika/Brainfast/releases/latest"
DEFAULT_RELEASES_PAGE = f"{DEFAULT_REPOSITORY}/releases/latest"


def normalize_version(value: str) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("v"):
        return text[1:]
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Brainfast version metadata.")
    parser.add_argument("--version", required=True, help="Semantic version or git tag")
    parser.add_argument("--commit", default="", help="Git commit SHA")
    parser.add_argument("--build-date", default=str(date.today()), help="Build date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "version.json"),
        help="Target version.json path",
    )
    args = parser.parse_args()

    payload = {
        "version": normalize_version(args.version),
        "build_date": str(args.build_date),
        "commit": str(args.commit),
        "repository": DEFAULT_REPOSITORY,
        "releases_api": DEFAULT_RELEASES_API,
        "releases_page": DEFAULT_RELEASES_PAGE,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

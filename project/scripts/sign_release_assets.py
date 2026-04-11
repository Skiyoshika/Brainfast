from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

DEFAULT_TIMESTAMP_URL = "http://timestamp.digicert.com"
DEFAULT_DESCRIPTION = "Brainfast desktop application"
DEFAULT_PUBLISHER_URL = "https://github.com/Skiyoshika/Brainfast"


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def find_signtool() -> Path:
    env_path = os.environ.get("BRAINFAST_SIGNTOOL", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path(r"C:\Program Files (x86)\Windows Kits\10\App Certification Kit\signtool.exe"),
            Path(r"C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe"),
            Path(r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe"),
            Path(r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22000.0\x64\signtool.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("signtool.exe not found; set BRAINFAST_SIGNTOOL or install Windows SDK")


def build_sign_command(
    *,
    signtool: Path,
    pfx_path: Path,
    password: str,
    target: Path,
    timestamp_url: str = DEFAULT_TIMESTAMP_URL,
    description: str = DEFAULT_DESCRIPTION,
    publisher_url: str = DEFAULT_PUBLISHER_URL,
) -> list[str]:
    command = [
        str(signtool),
        "sign",
        "/fd",
        "SHA256",
        "/f",
        str(pfx_path),
        "/p",
        password,
        "/tr",
        timestamp_url,
        "/td",
        "SHA256",
        "/d",
        description,
    ]
    if publisher_url:
        command.extend(["/du", publisher_url])
    command.append(str(target))
    return command


def sign_targets(
    targets: list[Path],
    *,
    pfx_path: Path,
    password: str,
    timestamp_url: str = DEFAULT_TIMESTAMP_URL,
    description: str = DEFAULT_DESCRIPTION,
    publisher_url: str = DEFAULT_PUBLISHER_URL,
) -> None:
    signtool = find_signtool()
    for target in targets:
        if not target.exists():
            raise FileNotFoundError(f"sign target not found: {target}")
        command = build_sign_command(
            signtool=signtool,
            pfx_path=pfx_path,
            password=password,
            target=target,
            timestamp_url=timestamp_url,
            description=description,
            publisher_url=publisher_url,
        )
        subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sign Brainfast desktop release assets with signtool."
    )
    parser.add_argument("--file", action="append", default=[], help="Specific file to sign")
    parser.add_argument("--glob", action="append", default=[], help="Glob under --dist-dir to sign")
    parser.add_argument(
        "--dist-dir",
        default=".",
        help="Base directory used for --glob expansion",
    )
    parser.add_argument("--pfx", default=os.environ.get("BRAINFAST_CODESIGN_PFX", ""))
    parser.add_argument("--password", default=os.environ.get("BRAINFAST_CODESIGN_PASSWORD", ""))
    parser.add_argument(
        "--timestamp-url",
        default=_env_or_default("BRAINFAST_CODESIGN_TIMESTAMP_URL", DEFAULT_TIMESTAMP_URL),
    )
    parser.add_argument(
        "--description",
        default=_env_or_default("BRAINFAST_CODESIGN_DESCRIPTION", DEFAULT_DESCRIPTION),
    )
    parser.add_argument(
        "--publisher-url",
        default=_env_or_default("BRAINFAST_CODESIGN_PUBLISHER_URL", DEFAULT_PUBLISHER_URL),
    )
    args = parser.parse_args()

    if not args.pfx or not args.password:
        raise SystemExit("missing signing credentials: provide --pfx and --password")

    dist_dir = Path(args.dist_dir).resolve()
    targets = [Path(value).resolve() for value in args.file]
    for pattern in args.glob:
        targets.extend(sorted(dist_dir.glob(pattern)))
    unique_targets = []
    seen = set()
    for target in targets:
        key = str(target)
        if key in seen:
            continue
        seen.add(key)
        unique_targets.append(target)
    if not unique_targets:
        raise SystemExit("no signing targets resolved")

    sign_targets(
        unique_targets,
        pfx_path=Path(args.pfx).resolve(),
        password=args.password,
        timestamp_url=args.timestamp_url,
        description=args.description,
        publisher_url=args.publisher_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

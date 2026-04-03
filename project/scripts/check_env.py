from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from project.scripts.asset_bootstrap import atlas_asset_status, default_structure_source
except Exception:
    from scripts.asset_bootstrap import atlas_asset_status, default_structure_source

try:
    from project.scripts.config_validation import load_config, validate_runtime_config
except Exception:
    from scripts.config_validation import load_config, validate_runtime_config

REQUIRED_MODULES = (
    "flask",
    "numpy",
    "pandas",
    "scipy",
    "skimage",
    "tifffile",
    "PIL",
    "nibabel",
    "matplotlib",
    "ants",
)

OPTIONAL_MODULES = (
    "cellpose",
    "SimpleITK",
    "pystray",
    "nrrd",
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _print_status(ok: bool, kind: str, label: str, detail: str = "") -> None:
    state = "OK" if ok else kind
    line = f"[{state}] {label}"
    if detail:
        line = f"{line}: {detail}"
    print(line)


def _resolve_input_dir(raw_value: object, *, project_root: Path, config_path: Path) -> Path | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None

    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate

    for base in (project_root, config_path.parent, Path.cwd()):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved

    return (project_root / candidate).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Brainfast runtime environment")
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent.parent / "configs" / "run_config.template.json"
        ),
        help="Config file to validate",
    )
    parser.add_argument(
        "--require-input-dir",
        action="store_true",
        help="Fail if input.slice_dir is missing or still a placeholder",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    failures = 0

    py_ok = sys.version_info >= (3, 10)
    _print_status(py_ok, "FAIL", "python", f"{sys.version.split()[0]} (need >= 3.10)")
    if not py_ok:
        failures += 1

    for name in REQUIRED_MODULES:
        ok = _module_available(name)
        _print_status(ok, "FAIL", f"python module '{name}'")
        if not ok:
            failures += 1

    for name in OPTIONAL_MODULES:
        ok = _module_available(name)
        _print_status(ok, "WARN", f"optional module '{name}'")

    structure_source = default_structure_source(project_root)
    required_assets = [
        project_root / "annotation_25.nii.gz",
        project_root / "configs" / "allen_structure_tree.json",
        project_root / "frontend" / "index.html",
        project_root / "frontend" / "server.py",
    ]
    for path in required_assets:
        ok = path.exists()
        _print_status(ok, "FAIL", "asset", str(path))
        if not ok:
            failures += 1
    structure_ok = structure_source is not None and structure_source.exists()
    _print_status(
        structure_ok,
        "FAIL",
        "asset",
        str(structure_source) if structure_source is not None else "missing structure source",
    )
    if not structure_ok:
        failures += 1

    status = atlas_asset_status(project_root)
    if not status["annotationReady"] and status["annotationNrrdReady"]:
        _print_status(
            False,
            "WARN",
            "asset",
            "annotation_25.nrrd exists but annotation_25.nii.gz is still missing",
        )

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        _print_status(False, "FAIL", "config", str(cfg_path))
        failures += 1
    else:
        _print_status(True, "OK", "config", str(cfg_path))
        try:
            cfg = load_config(cfg_path)
            issues = validate_runtime_config(cfg, require_input_dir=bool(args.require_input_dir))
            if issues:
                failures += len(issues)
                for issue in issues:
                    _print_status(False, "FAIL", "config", issue)
            else:
                _print_status(True, "OK", "config", "runtime fields validated")
                if args.require_input_dir:
                    input_dir = _resolve_input_dir(
                        cfg.get("input", {}).get("slice_dir"),
                        project_root=project_root,
                        config_path=cfg_path.resolve(),
                    )
                    input_dir_ok = (
                        input_dir is not None and input_dir.exists() and input_dir.is_dir()
                    )
                    _print_status(
                        input_dir_ok,
                        "FAIL",
                        "input.slice_dir",
                        str(input_dir) if input_dir is not None else "missing",
                    )
                    if not input_dir_ok:
                        failures += 1
        except Exception as exc:
            failures += 1
            _print_status(False, "FAIL", "config", str(exc))

    if failures:
        print(f"Environment check finished with {failures} failure(s).")
        return 1

    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

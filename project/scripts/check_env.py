from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_this = Path(__file__).resolve()
sys.path.insert(0, str(_this.parents[2]))  # D:\Brainfast
sys.path.insert(0, str(_this.parents[1]))  # D:\Brainfast\project

try:
    from scripts.asset_bootstrap import atlas_asset_status, default_structure_source
except ImportError:
    from project.scripts.asset_bootstrap import atlas_asset_status, default_structure_source

try:
    from scripts.config_validation import load_config, validate_runtime_config
except ImportError:
    from project.scripts.config_validation import load_config, validate_runtime_config

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
)

OPTIONAL_MODULES = (
    "ants",  # needed for whole-brain 3D registration (pip install -e ".[wholebrain]")
    "cellpose",  # needed for Cellpose detection (pip install -e ".[advanced]")
    "SimpleITK",
    "pystray",
    "nrrd",
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _module_importable(name: str) -> tuple[bool, str]:
    """Try a real import and return (ok, detail).

    ``find_spec`` only checks if the module *name* is visible on sys.path; it
    does not catch ABI mismatches, DLL load failures, or broken transitive
    dependencies that make the actual ``import`` crash.  This function
    performs a real import inside a subprocess-like try/except so that the
    caller gets both the result and a human-readable failure reason.
    """
    try:
        # ANTs has a known issue: ants.__init__ unconditionally imports
        # ants.plotting which depends on mpl_toolkits internals that break
        # with newer matplotlib (>3.9).  The core registration functions work
        # fine.  We stub out the plotting submodule before importing so that
        # check_env tests the *functional* part of the package.
        if name == "ants" and "ants" not in sys.modules:
            import types

            sys.modules.setdefault("ants.plotting", types.ModuleType("ants.plotting"))

        mod = __import__(name)
        ver = getattr(mod, "__version__", "")
        if not ver:
            try:
                from importlib.metadata import version as _pkg_version

                ver = _pkg_version(name.replace(".", "-"))
            except Exception:
                ver = "installed"
        return True, ver
    except Exception as exc:
        return False, str(exc)


# Version boundary checks for the numerical stack.
# These must match pyproject.toml [project].dependencies.
_VERSION_BOUNDS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "numpy": ((1, 26), (3, 0)),  # >=1.26, <3  (tested with 1.26 and 2.4)
    "scipy": ((1, 12), (2, 0)),  # >=1.12, <2  (tested with 1.12 and 1.17)
    "skimage": ((0, 22), (1, 0)),  # >=0.22, <1  (tested with 0.22 and 0.26)
}


def _check_version_bounds(name: str, version_str: str) -> str | None:
    """Return an error string if *version_str* is outside the pinned range."""
    if name not in _VERSION_BOUNDS:
        return None
    lo, hi = _VERSION_BOUNDS[name]
    try:
        parts = tuple(int(x) for x in version_str.split(".")[: len(lo)])
    except (ValueError, TypeError):
        return f"{name} version '{version_str}' cannot be parsed"
    if parts < lo:
        return f"{name}=={version_str} is below minimum {'.'.join(map(str, lo))}"
    if parts >= hi:
        return f"{name}=={version_str} exceeds upper bound <{'.'.join(map(str, hi))}"
    return None


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
        ok, detail = _module_importable(name)
        _print_status(ok, "FAIL", f"python module '{name}'", detail if not ok else detail)
        if not ok:
            failures += 1
        else:
            # Check version bounds for pinned numerical stack
            ver_err = _check_version_bounds(name, detail)
            if ver_err:
                _print_status(False, "FAIL", f"version '{name}'", ver_err)
                failures += 1

    # Determine which optional modules are actually required by the active config
    try:
        cfg_for_deps = load_config(Path(args.config)) if Path(args.config).exists() else {}
    except Exception:
        cfg_for_deps = {}

    needs_ants = (
        cfg_for_deps.get("registration", {}).get("scope") == "whole"
        and cfg_for_deps.get("registration", {}).get("whole_brain_backend") == "miki_3d"
    )
    needs_cellpose = any(
        str(cfg_for_deps.get("detection", {}).get(key, "")).lower()
        in {"cpsam", "sam", "cellpose", "cyto", "cyto2", "cyto3", "nuclei"}
        or str(cfg_for_deps.get("detection", {}).get(key, "")).lower().startswith("cellpose")
        for key in ("primary_model", "secondary_model")
    )

    _config_required_modules = set()
    if needs_ants:
        _config_required_modules.add("ants")
    if needs_cellpose:
        _config_required_modules.add("cellpose")

    for name in OPTIONAL_MODULES:
        ok, detail = _module_importable(name)
        if name in _config_required_modules:
            # Config requires this module — treat as FAIL, not WARN
            _print_status(
                ok, "FAIL", f"module '{name}' (required by active config)", detail if not ok else ""
            )
            if not ok:
                failures += 1
            else:
                ver_err = _check_version_bounds(name, detail)
                if ver_err:
                    _print_status(False, "FAIL", f"version '{name}'", ver_err)
                    failures += 1
        else:
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

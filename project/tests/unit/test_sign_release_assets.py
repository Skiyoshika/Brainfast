from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sign_release_assets import (  # noqa: E402
    DEFAULT_TIMESTAMP_URL,
    _env_or_default,
    build_sign_command,
)


def test_env_or_default_falls_back_for_empty_value(monkeypatch):
    monkeypatch.setenv("BRAINFAST_CODESIGN_TIMESTAMP_URL", "")
    assert (
        _env_or_default("BRAINFAST_CODESIGN_TIMESTAMP_URL", DEFAULT_TIMESTAMP_URL)
        == DEFAULT_TIMESTAMP_URL
    )


def test_build_sign_command_contains_expected_signtool_flags():
    command = build_sign_command(
        signtool=Path("C:/sdk/signtool.exe"),
        pfx_path=Path("C:/certs/brainfast.pfx"),
        password="secret",
        target=Path("D:/Brainfast/project/frontend/dist/BrainfastUI/BrainfastUI.exe"),
    )
    joined = " ".join(command)
    assert command[0].endswith("signtool.exe")
    assert "/fd SHA256" in joined
    assert "/tr http://timestamp.digicert.com" in joined
    assert str(Path("C:/certs/brainfast.pfx")) in joined

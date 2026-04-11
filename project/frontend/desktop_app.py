"""
Brainfast desktop launcher.

- Starts the local Flask backend
- Shows a small splash screen during startup
- Opens the browser automatically
- Lives in the system tray until quit
- Checks atlas assets before backend startup
- Checks GitHub releases in the background
"""

from __future__ import annotations

import atexit
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    FRONTEND = Path(sys._MEIPASS)
else:
    FRONTEND = Path(__file__).resolve().parent

os.environ["BRAINFAST_FRONTEND"] = str(FRONTEND)

from server import app  # noqa: E402

from project.frontend.app_metadata import read_version_info  # noqa: E402
from project.frontend.update_checker import (  # noqa: E402
    check_for_update,
    latest_release_url,
    update_checks_enabled,
)

HOST = "127.0.0.1"
PORT = 8787
LOCK_PORT = 18787
APP_NAME = "Brainfast"

_last_run_done = False
_update_state: dict[str, object] = {
    "checked": False,
    "error": "",
    "has_update": False,
    "latest_version": "",
    "latest_url": "",
}


def _make_icon(size: int = 64, highlight: bool = False):
    from PIL import Image, ImageDraw

    bg = (30, 32, 40, 255)
    accent = (76, 114, 245, 255) if not highlight else (52, 190, 110, 255)
    image = Image.new("RGBA", (size, size), bg)
    draw = ImageDraw.Draw(image)
    margin = size // 8
    draw.ellipse([margin, margin * 2, size - margin, size - margin], fill=accent)
    draw.ellipse([size // 2 - margin, margin, size // 2 + margin, margin * 3], fill=bg)
    draw.line(
        [(size // 2, margin * 2 + 2), (size // 2, size - margin - 2)],
        fill=bg,
        width=max(2, size // 20),
    )
    return image.convert("RGB")


def _show_splash():
    import tkinter as tk

    root = tk.Tk()
    root.overrideredirect(True)
    root.configure(bg="#181a1f")
    root.attributes("-topmost", True)

    width, height = 320, 140
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{width}x{height}+{(screen_w - width) // 2}+{(screen_h - height) // 2}")

    tk.Label(
        root, text="Brainfast", font=("Segoe UI", 15, "bold"), bg="#181a1f", fg="#dde1e9"
    ).pack(pady=(32, 6))
    status = tk.Label(
        root,
        text="Starting backend...",
        font=("Segoe UI", 10),
        bg="#181a1f",
        fg="#878e9e",
    )
    status.pack()
    root.update()
    return root, status


def _single_instance() -> socket.socket | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((HOST, LOCK_PORT))
        return sock
    except OSError:
        return None


def _wait_ready(timeout: float = 12.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((HOST, PORT), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _poll_pipeline(icon):
    global _last_run_done
    import json as json_lib
    import urllib.request

    while True:
        time.sleep(3)
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/api/status", timeout=1) as response:
                data = json_lib.loads(response.read())
            done = bool(data.get("done")) and not bool(data.get("running"))
            if done and not _last_run_done:
                error = data.get("error")
                if error:
                    icon.notify(f"Pipeline error: {error}", APP_NAME)
                else:
                    icon.notify("Pipeline finished successfully.", APP_NAME)
            _last_run_done = done
        except Exception:
            pass


def _check_updates(icon, *, interactive: bool = False):
    global _update_state

    if not update_checks_enabled():
        _update_state = {
            "checked": True,
            "error": "",
            "has_update": False,
            "latest_version": "",
            "latest_url": latest_release_url(FRONTEND.parent),
        }
        if interactive:
            icon.notify("Automatic update checks are disabled.", APP_NAME)
        return

    try:
        result = check_for_update(FRONTEND.parent)
        _update_state = {
            "checked": True,
            "error": "",
            "has_update": bool(result["has_update"]),
            "latest_version": str(result["latest_version"]),
            "latest_url": str(result["latest_url"]),
        }
        if result["has_update"]:
            icon.notify(
                f"Update available: {result['latest_version']} (current {result['current_version']})",
                APP_NAME,
            )
        elif interactive:
            icon.notify(f"Brainfast is up to date ({result['current_version']}).", APP_NAME)
    except Exception as exc:
        _update_state = {
            "checked": True,
            "error": str(exc),
            "has_update": False,
            "latest_version": "",
            "latest_url": latest_release_url(FRONTEND.parent),
        }
        if interactive:
            icon.notify(f"Update check failed: {exc}", APP_NAME)


def _start_update_check(icon) -> None:
    if not update_checks_enabled():
        return
    threading.Thread(target=_check_updates, args=(icon,), daemon=True).start()


def main():
    guard = _single_instance()
    if guard is None:
        webbrowser.open(f"http://{HOST}:{PORT}")
        return
    atexit.register(guard.close)

    splash, splash_status = _show_splash()

    def _update_splash(message: str) -> None:
        splash_status.config(text=message[:56])
        splash.update()

    try:
        from project.scripts.asset_bootstrap import ensure_atlas_assets

        _update_splash("Checking atlas assets...")
        ensure_atlas_assets(FRONTEND.parent, logger=_update_splash)
    except Exception as exc:
        splash.destroy()
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(APP_NAME, f"Atlas bootstrap failed.\n{exc}")
        root.destroy()
        return

    _update_splash("Starting backend...")
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host=HOST,
            port=PORT,
            debug=False,
            use_reloader=False,
            threaded=True,
        ),
        daemon=True,
    )
    flask_thread.start()

    ready = _wait_ready(timeout=15)
    splash.destroy()
    if not ready:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            APP_NAME, "Backend failed to start on port 8787.\nCheck that the port is free."
        )
        root.destroy()
        return

    webbrowser.open(f"http://{HOST}:{PORT}")

    import pystray
    from pystray import Menu
    from pystray import MenuItem as Item

    version_info = read_version_info(FRONTEND.parent)

    def on_open(_icon, _item):
        webbrowser.open(f"http://{HOST}:{PORT}")

    def on_open_releases(_icon, _item):
        webbrowser.open(str(_update_state.get("latest_url") or latest_release_url(FRONTEND.parent)))

    def on_check_updates(_icon, _item):
        threading.Thread(
            target=_check_updates,
            args=(_icon,),
            kwargs={"interactive": True},
            daemon=True,
        ).start()

    def on_quit(_icon, _item):
        _icon.stop()
        os._exit(0)

    icon = pystray.Icon(
        APP_NAME,
        icon=_make_icon(),
        title=f"{APP_NAME} {version_info['version']} running on :{PORT}",
        menu=Menu(
            Item("Open Browser", on_open, default=True),
            Item("Check for Updates", on_check_updates),
            Item("Open Releases Page", on_open_releases),
            Menu.SEPARATOR,
            Item("Quit", on_quit),
        ),
    )

    threading.Thread(target=_poll_pipeline, args=(icon,), daemon=True).start()
    _start_update_check(icon)
    icon.run()


if __name__ == "__main__":
    main()

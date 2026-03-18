"""
Brainfast Desktop Launcher
- System tray icon with right-click menu
- Splash screen while backend starts
- Browser auto-open
- Pipeline-complete notification via tray
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
import atexit
from pathlib import Path

# ── Resolve paths (works both dev and PyInstaller --onedir) ──────────────────
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # PyInstaller onedir: _MEIPASS is the _internal/ directory
    # HTML/CSS/JS are placed directly in _internal/ (see spec)
    FRONTEND = Path(sys._MEIPASS)
else:
    FRONTEND = Path(__file__).resolve().parent

os.environ["BRAINFAST_FRONTEND"] = str(FRONTEND)

from server import app  # noqa: E402 — import after env set

HOST      = "127.0.0.1"
PORT      = 8787
LOCK_PORT = 18787
APP_NAME  = "Brainfast"


# ── Tray icon image (drawn with PIL, no external file needed) ────────────────
def _make_icon(size: int = 64, highlight: bool = False) -> "Image.Image":
    from PIL import Image, ImageDraw
    bg = (30, 32, 40, 255)
    accent = (76, 114, 245, 255) if not highlight else (52, 190, 110, 255)
    img  = Image.new("RGBA", (size, size), bg)
    draw = ImageDraw.Draw(img)
    m = size // 8
    # Brain-like ellipse
    draw.ellipse([m, m * 2, size - m, size - m], fill=accent)
    # Top indent (cerebral notch)
    draw.ellipse([size // 2 - m, m, size // 2 + m, m * 3], fill=bg)
    # Vertical sulcus line
    draw.line([(size // 2, m * 2 + 2), (size // 2, size - m - 2)],
              fill=bg, width=max(2, size // 20))
    return img.convert("RGB")


# ── Splash screen (Tkinter) ──────────────────────────────────────────────────
def _show_splash() -> "Tk":
    import tkinter as tk
    root = tk.Tk()
    root.overrideredirect(True)
    root.configure(bg="#181a1f")
    root.attributes("-topmost", True)

    W, H = 320, 140
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}")

    tk.Label(root, text="🧠", font=("Segoe UI", 36),
             bg="#181a1f", fg="#4c72f5").pack(pady=(18, 4))
    tk.Label(root, text="Brainfast", font=("Segoe UI", 15, "bold"),
             bg="#181a1f", fg="#dde1e9").pack()
    status = tk.Label(root, text="Starting backend…",
                      font=("Segoe UI", 10), bg="#181a1f", fg="#878e9e")
    status.pack(pady=(4, 0))

    root.update()
    return root, status


# ── Single-instance guard ────────────────────────────────────────────────────
def _single_instance() -> "socket.socket | None":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((HOST, LOCK_PORT))
        return s
    except OSError:
        return None


def _wait_ready(timeout: float = 12.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((HOST, PORT), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


# ── Background poller: watch for pipeline completion ────────────────────────
_last_run_done: bool = False

def _poll_pipeline(icon: "pystray.Icon"):
    global _last_run_done
    import urllib.request, json as _json
    while True:
        time.sleep(3)
        try:
            with urllib.request.urlopen(
                f"http://{HOST}:{PORT}/api/status", timeout=1
            ) as r:
                data = _json.loads(r.read())
            done = bool(data.get("done")) and not bool(data.get("running"))
            if done and not _last_run_done:
                err = data.get("error")
                if err:
                    icon.notify(f"Pipeline error: {err}", APP_NAME)
                else:
                    icon.notify("Pipeline finished successfully! ✅", APP_NAME)
            _last_run_done = done
        except Exception:
            pass


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    guard = _single_instance()
    if guard is None:
        # Already running — just open browser
        webbrowser.open(f"http://{HOST}:{PORT}")
        return
    atexit.register(guard.close)

    # Show splash
    splash, splash_status = _show_splash()

    # Start Flask in background
    flask_thread = threading.Thread(
        target=lambda: app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True),
        daemon=True,
    )
    flask_thread.start()

    # Wait for backend
    ready = _wait_ready(timeout=15)
    splash.destroy()

    if not ready:
        import tkinter as tk
        from tkinter import messagebox
        r = tk.Tk(); r.withdraw()
        messagebox.showerror(APP_NAME, "Backend failed to start on port 8787.\nCheck that port is free.")
        r.destroy()
        return

    # Open browser
    webbrowser.open(f"http://{HOST}:{PORT}")

    # Build tray icon
    import pystray
    from pystray import MenuItem as Item, Menu

    def on_open(_icon, _item):
        webbrowser.open(f"http://{HOST}:{PORT}")

    def on_quit(_icon, _item):
        _icon.stop()
        os._exit(0)

    icon = pystray.Icon(
        APP_NAME,
        icon=_make_icon(),
        title=f"{APP_NAME} — running on :{PORT}",
        menu=Menu(
            Item("Open Browser",  on_open, default=True),
            Menu.SEPARATOR,
            Item("Quit",          on_quit),
        ),
    )

    # Pipeline notification poller
    threading.Thread(target=_poll_pipeline, args=(icon,), daemon=True).start()

    icon.run()   # blocks until quit


if __name__ == "__main__":
    main()

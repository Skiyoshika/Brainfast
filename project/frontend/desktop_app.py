from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser
import atexit
from pathlib import Path
from tkinter import Tk, messagebox

from server import app

HOST = "127.0.0.1"
PORT = 8787
LOCK_PORT = 18787


def _show_info(title: str, msg: str):
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title, msg)
    root.destroy()


def _single_instance_guard() -> socket.socket | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((HOST, LOCK_PORT))
        return s
    except OSError:
        return None


def _wait_port_ready(timeout_sec: float = 8.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with socket.create_connection((HOST, PORT), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def run_backend():
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


def _cleanup(guard_sock: socket.socket | None):
    try:
        if guard_sock:
            guard_sock.close()
    except Exception:
        pass


def main():
    guard = _single_instance_guard()
    if guard is None:
        _show_info("IdleBrainUI", "IdleBrainUI is already running.")
        webbrowser.open(f"http://{HOST}:{PORT}")
        return

    atexit.register(_cleanup, guard)

    t = threading.Thread(target=run_backend, daemon=True)
    t.start()

    if not _wait_port_ready():
        _show_info("IdleBrainUI", "Backend failed to start on port 8787.")
        return

    webbrowser.open(f"http://{HOST}:{PORT}")
    _show_info("IdleBrainUI", "Application is running in browser. Close browser tab when done.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if messagebox.askyesno("IdleBrainUI", "Exit IdleBrainUI now?"):
            _show_info("IdleBrainUI", "IdleBrainUI stopped.")


if __name__ == "__main__":
    main()


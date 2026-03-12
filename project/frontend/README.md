# Discord-style Frontend (MVP)

Run full UI + backend bridge:

```bash
cd frontend
python server.py
```

Open: http://127.0.0.1:8787

## Desktop mode (double-click EXE)

```bash
cd frontend
build_desktop.bat
```

Then run:
- `dist/IdleBrainUI.exe`

This launcher starts local backend and auto-opens the UI in browser.

Desktop hardening in current build:
- single-instance guard (prevents double launch)
- port-ready wait before opening browser
- startup failure popup when backend port is unavailable

## Features
- Discord-like dark layout
- Pipeline form + channel selector (red/green/farred)
- Batch all channels mode
- Path validation before run
- Preset save/load (localStorage)
- Live log polling from backend
- Output table snapshot + QC preview

## User guide
- See `../ATLAS_OVERLAY_GUIDE.md` for step-by-step operation and troubleshooting.

## Next
- replace single-process runner with queued job manager
- add cancel button and per-channel progress
- add downloadable reports


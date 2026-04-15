# Brainfast Install Modes

| Mode | Command | Use case |
|------|---------|----------|
| Minimal | `pip install -e ".[dev]"` | 2D development, tests, docs work |
| Default runtime | `pip install -e ".[full,dev]"` | shipped `miki_3d + cpsam` path |
| Desktop build | `pip install -e ".[full,desktop,dev]"` | Windows EXE build and signing |

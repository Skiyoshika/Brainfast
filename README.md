# Brainfast

Chinese documentation index: [project/docs/README.zh-CN.md](project/docs/README.zh-CN.md)

## Index
- [Main Guide](#main-guide)
- [Quick Start](#quick-start)
- [Repository Map](#repository-map)
- [License](#license)

## Main Guide

The main project documentation lives in [project/README.md](project/README.md).

Use that file for:
- installation
- runtime requirements
- the 2D and 3D workflows
- output files
- testing commands

## Quick Start

Validate the environment:

```powershell
python project\scripts\check_env.py --config project\configs\run_config.template.json
```

Start the desktop/web UI:

```powershell
.\Start_Brainfast.bat
```

If you prefer the direct Python entry:

```powershell
python project\frontend\server.py
```

Then open `http://127.0.0.1:8787`.

## Repository Map

- [Main project README](project/README.md)
- [Chinese documentation index](project/docs/README.zh-CN.md)
- [Frontend notes](project/frontend/README.md)
- [Internal 3D sample workflow](project/docs/internal_3d_sample_workflow.md)

## License

Brainfast is distributed under the GNU AGPL-3.0 license. See [LICENSE](LICENSE) for details.

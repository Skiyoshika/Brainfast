"""Stitching module for brain section tile assembly.

Ported from UCI-XuLab-RegTools (github.com/UCI-XuLab/UCI-XuLab-RegTools)
on 2026-04-23. The core stitching algorithms (tile placement, Bezier
distortion correction, multi-band blending, vignetting normalization) are
vendored byte-for-byte from upstream; only ``regtools.stitching.core``
references were rewritten to ``scripts.stitching.core`` so the package
resolves inside Brainfast without the Xu Lab utils tree. Upstream's
``gui.py`` + ``launch.py`` (PyQt5 desktop GUI) are not vendored — Brainfast
surfaces stitching through the web UI / CLI instead.

Structure
---------
- core/       : Stitching algorithms (``stitcher``, ``tile``,
                ``run_tissuecyte_stitching_classic``).
- pipeline.py : CLI ``stitch_pipeline`` function + argparse entry point.

Optional runtime deps (install with ``pip install -e ".[stitching]"``):
``opencv-python``, ``colorama``, ``joblib``.

Usage
-----
    # Programmatic
    from scripts.stitching.pipeline import stitch_pipeline
    stitch_pipeline(input_dir, output_dir)

    # CLI
    python -m scripts.stitching.pipeline --input_dir ... --output_dir ...
"""

from . import core

__all__ = ["core"]

"""Xu Lab desktop-GUI launcher — not ported to Brainfast.

Upstream's ``regtools.stitching.launch`` opens a PyQt5 desktop app via
``AppScaffold`` + ``StitchingTab``. Brainfast deliberately does not vendor
the PyQt5 GUI (we surface stitching through the web UI / CLI instead), so
this module only exists as a discoverable stub that explains where to go.

To run the stitching pipeline in Brainfast use::

    python -m scripts.stitching.pipeline --input_dir ... --output_dir ...

or call ``stitch_pipeline`` programmatically from
``project.scripts.stitching.pipeline``.
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover — intentional stub
    raise NotImplementedError(
        "Brainfast does not vendor the Xu Lab PyQt5 stitching GUI. "
        "Use `python -m scripts.stitching.pipeline` or the Stitching endpoint "
        "in the Brainfast web UI instead."
    )


if __name__ == "__main__":
    main()

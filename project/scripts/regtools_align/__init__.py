"""Vendored axis-alignment utilities from UCI-XuLab-RegTools.

Source: UCI-XuLab/UCI-XuLab-RegTools
Original author: Atchuth Naveen, UC Irvine
Vendored on: 2026-04-16 for Brainfast
Why: Brainfast's ANTs SyN stage was not preceded by any global rigid alignment,
     so SyN had to spend its diffeomorphic capacity correcting roll/pitch of
     the moving volume relative to the template. This is the likely cause of
     residual internal-structure misalignment (NCC -0.16 vs Miki 0.66).
     Miki's pipeline uses a longitudinal-fissure detection + SVD plane fit
     + cross-product rotation stage before SyN, and matches internal CCFv3
     anatomy within Dice 0.75 / NCC 0.66 / PSNR 13.

Usage:
    from project.scripts.regtools_align import compute_longitudinal_fissure_alignment

    # moving: np.ndarray (D, H, W) uint16 fluorescence volume in PIR orientation
    # template: np.ndarray (D, H, W) uint16 Allen template in PIR orientation
    affine_4x4, *_ = compute_longitudinal_fissure_alignment(moving, template)
"""

from .vol2affine import vol2affine as compute_longitudinal_fissure_alignment

__all__ = ["compute_longitudinal_fissure_alignment"]

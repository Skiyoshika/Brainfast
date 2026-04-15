# Brainfast v0.3.0 UX Feedback — Neurobiologist Testing Report

**Tester**: Neuroscience researcher (ChAT cleared-tissue imaging)
**Date**: 2026-04-07
**Samples**: ChAT Sample 35/39/41 (WT/5xFAD/Aged), dual-channel 3D Z-stacks
**Reference**: Miki's ANTs 3D registration results
**Test scenario**: One-Click whole-brain registration with target region (STN + CP)

---

## Test Flow Summary

| Step | Action | Result |
|------|--------|--------|
| Load page | Open Brainfast, check UI | All controls present, 592 regions loaded |
| Enter source | Paste ChAT Sample 35 C0 path | 3D detected (646 slices), scope auto-switched to whole |
| Set hemisphere | Select right_flipped | OK |
| Select regions | Search STN, CP | Tags displayed, AP 151-306 merged |
| Start workflow | Click One-Click Start | Autopick + preview + landmarks completed |
| Review | Check overlay + step collapse | Preview toast error, steps not collapsed |

---

## Issues Found

### RED — Critical (blocks workflow or gives wrong results)

#### R1: Overlay preview image not displayed after registration
- **What happened**: After One-Click workflow finishes, the overlay comparison image exists on disk (`overlay_compare_nonlinear.png`, 366KB, 2726x1636px) and the API endpoint `/api/outputs/overlay-preview` returns 200. But the `<img>` element's `src` is never set in manual review mode.
- **Impact**: User cannot visually verify registration quality — this is the single most important QC step.
- **Suggested fix**: After `runOneClickWorkflow` completes, explicitly set the overlay preview image src: `previewImg.src = withOverlayJobQuery('/api/outputs/overlay-preview', {ts: Date.now()})`.

#### R2: Step panels don't collapse after workflow completes
- **What happened**: All 4 steps + Manual section remain expanded after One-Click finishes. CSS classes `.collapsed` exist in styles.css, but no step card has the `collapsed` class applied.
- **Impact**: Information overload — user doesn't know where to focus. The manual landmark canvases are buried among expanded configuration panels.
- **Suggested fix**: Verify `collapseAllStepsExcept('step3')` is actually called. Check if the step card IDs match (`step1`, `step2`, `step3`, `step4`).

#### R3: pixel_size_um not detected from TIFF metadata
- **What happened**: `/api/slice/info` returns `pixel_size_um: null` for the ChAT TIFF. The UI silently uses the default 0.65 um (microscope resolution). The actual ChAT data has 5 um Z-step.
- **Impact**: Wrong pixel size means wrong atlas scaling, causing registration to fail or produce incorrect alignment. The user (me) didn't even notice until I checked the API response.
- **Suggested fix**:
  1. Parse pixel size from TIFF OME metadata / ImageJ metadata / filename hints (e.g., "z5um" in the filename)
  2. If not found, show a **prominent warning** (not just an empty field) asking the user to enter the pixel size manually
  3. Add a yellow banner: "Pixel size not detected. Registration quality depends on this value."

### YELLOW — Medium (degrades experience)

#### Y1: Region search is substring-only
- **What happened**: Searching "CP" returns CP, scp, mcp, icp, dscp — cerebral peduncle results mixed with cerebellar peduncles. CP (Caudoputamen) is what I want but it's item 1 of 5.
- **Suggested fix**: Sort results with exact acronym match first, then prefix matches, then substring matches.

#### Y2: No confirmation of hemisphere/atlas settings for 3D data
- **What happened**: hemisphere defaults to "auto" every page load. For cleared-tissue samples, you always know which hemisphere it is (it's in the lab notebook). Having to re-select every time is annoying.
- **Suggested fix**: Remember last-used settings in localStorage, or let the user save a "sample profile" with hemisphere, pixel_size, and target regions.

#### Y3: "Preview generation failed" toast still appears
- **What happened**: Even though the overlay endpoint works (200), a transient preview failure during the workflow triggers a persistent error toast.
- **Suggested fix**: Only show preview error if the *final* overlay-preview fetch fails, not intermediate preview attempts during autopick.

#### Y4: Double autopick runs
- **What happened**: Server logs show two `POST /api/atlas/autopick-z` calls (15:33:48 and 15:35:02). The first one is from the initial Start click (which triggers 3D detection and stops), the second from `runOneClickWorkflow()`. Wastes ~25 seconds.
- **Suggested fix**: If the z-slicer is already visible (3D already detected), skip the re-detection step and go straight to autopick.

### GREEN — Positive Feedback

#### G1: Target Region Selector is excellent
- Searching "STN" instantly finds the right region with AP range
- Multi-select with tags is intuitive
- AP range hint updates in real-time ("AP search restricted to slices 151-306")
- **This is the most impactful UX improvement in this session**

#### G2: 3D auto-detection + scope switching works
- Loading a 3D TIFF auto-switches from single to whole mode
- Z-slicer shows correct range (0-645)
- Thumbnail preview loads correctly at mid-Z

#### G3: AP range restriction works
- With STN+CP selected, autopick returned best_z=185 (striatum region) instead of 16 (olfactory bulb)
- Search time reduced from full 528 slices to 155 slices (~3x faster)

#### G4: Atlas version + Registration mode controls
- CCFv3/CCFv3-BBP selector present with clear descriptions
- Auto-enables Nissl mode when BBP selected
- Warning when Nissl selected without BBP atlas

---

## Comparison with Miki's Workflow

| Dimension | Brainfast | Miki (ANTs 3D) |
|-----------|-----------|----------------|
| Registration | 2D per-slice | 3D volumetric |
| AP localization | Autopick (SSIM) + target region filter | 3D volume alignment |
| Deformation model | TPS/BSpline 2D | SyN diffeomorphic 3D |
| Setup time | ~2 min (GUI) | ~10 min (CLI config) |
| Processing time | ~30s per slice | ~30 min whole brain |
| User skill needed | Low (GUI) | High (command line) |
| Best for | Single-slice QC, quick check | Publication-quality 3D |

### Key Architectural Gap
Brainfast's 2D-per-slice approach works well for quick single-section analysis, but for complete cleared-brain Z-stacks (our primary use case), we need either:
1. A batch mode that auto-registers representative slices and interpolates
2. Integration with a 3D registration backend (brainreg/ANTs)

---

## Priority Recommendations for Developers

1. **FIX R1 first** — overlay preview display is the #1 blocker for usability
2. **FIX R3** — pixel size warning prevents silent failure
3. **FIX R2** — step collapse makes the workflow navigable
4. **ENHANCE** — Region search ranking (exact > prefix > substring)
5. **CONSIDER** — 3D batch mode for cleared-tissue Z-stacks

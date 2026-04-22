# Brainfast UX Feedback — Neurobiologist Testing Session
**Date:** 2026-04-08
**Tester:** Brain atlas registration researcher
**Sample:** ChATe27 Sample 35 (ChAT-Cre cleared brain, dual-channel 560nm/640nm, 5µm z-step)
**Brainfast Version:** v0.3.0-desktop
**Workflow tested:** One-Click Mode → Single-layer Registration

---

## Overall Impression

The One-Click workflow *does work* and produced an **Excellent** registration result (SSIM 0.6174 → 0.9893, +60%). The concept is strong — autopick + AI landmark + nonlinear warp in one button is exactly what bench scientists need. However, I encountered **13 usability issues** that would frustrate a first-time user or cause them to give up before seeing the result.

---

## Issues Found (by severity)

### CRITICAL — Blocks core workflow

**Issue #9/Bug: Atlas reference image fails to load (404) in Manual Landmark Correction**
- `/api/outputs/atlas-layer` returns 404
- The manual landmark correction section shows a broken image icon for "Atlas Slice"
- Without the atlas reference image, users **cannot** perform manual landmark correction — a key part of the advertised workflow
- **Impact:** Manual correction is completely non-functional

**Issue #11: Server crash during workflow**
- During the first workflow attempt, the entire Flask server and preview crashed
- Possibly related to memory pressure from the 2.7GB 3D TIFF
- **Impact:** User loses all progress; no error message or recovery option

**Issue #10: Frontend polls non-existent `/api/outputs/volume-reg-stats` endpoint repeatedly**
- Network tab shows 15+ consecutive 404 requests to this endpoint
- Suggests a polling loop with no stop condition for missing endpoints
- **Impact:** Wastes bandwidth, fills server logs with errors, may contribute to performance issues

### HIGH — Significantly degrades experience

**Issue #4: Path handling bug — double backslash on Windows**
- When user enters `D:\Brainfast\...` via `preview_fill`, the path gets double-escaped to `D:\\Brainfast\\...` causing `[Errno 13] Permission denied`
- This appears to be a frontend→backend path escaping issue
- **Impact:** Windows users who type paths manually will hit cryptic errors

**Issue #6: Z-slicer thumbnail shows all-black preview**
- For the ChATe27 3D TIFF (fluorescence data), ALL Z positions show a completely black thumbnail
- The thumbnail likely lacks auto-contrast/brightness normalization
- **Impact:** Users cannot see what's in each Z-layer, making Z selection a blind guess. For researchers choosing a specific brain region (e.g., hippocampus), this is critical

**Issue #5: Error toasts persist and block UI**
- ERR toasts from previous failed attempts stay on screen indefinitely
- They overlap Step 1 title and configuration fields
- No auto-dismiss, and the ✕ button is small and easy to miss
- **Impact:** Stale errors cause confusion — user thinks current attempt is also failing

### MEDIUM — Confusing but workaround exists

**Issue #3: Pixel size warning lacks guidance**
- WARN toast says "Pixel size not detected from image metadata" but doesn't tell user WHERE to enter it manually
- There's no visible pixel-size input field in the One-Click configuration panel
- **Impact:** Researcher knows pixel size (e.g., 5µm) but can't figure out where to input it

**Issue #7: Z-slicer appears in Whole-brain mode**
- When Registration Scope is "Whole-brain Registration", the Z-slicer still appears
- Confusing for users: "I selected whole-brain, why do I need to pick one Z layer?"
- **Impact:** Users may think they need to select a specific layer even for whole-brain processing

**Issue #8: No clear status indication during workflow transitions**
- After autopick → AI registration → overlay generation, the interface doesn't clearly indicate which step completed and which is next
- Toast messages flash briefly but give no persistent status
- **Impact:** Users don't know if the workflow is still running or finished

**Issue #12: Source TIFF path reverts to stale value on page reload**
- After server restart, Source TIFF shows old invalid path `D:\Brainfast\Sample\ChAT\35_C0_56`
- But Registration Scope correctly persisted as "Whole-brain"
- **Impact:** Inconsistent state persistence confuses returning users

**Issue #13: Z-slicer resets to Z=323 after workflow completion**
- User selected Z=290, but after workflow completes, Z-slicer shows Z=323 (the default midpoint)
- **Impact:** Minor but confusing — user can't verify which layer was actually processed

### LOW — Polish items

**Issue #1: Source TIFF field shows a non-existent default path**
- Default value `D:\Brainfast\Sample\ChAT\35_C0_56` does not exist on disk
- Appears to be a leftover from a previous session
- **Impact:** New users may try to start with a non-existent path and hit errors immediately

**Issue #2: Two separate Source input fields (inputDir vs oneClickSourcePath)**
- The Pro mode uses `#inputDir`, One-Click mode uses `#oneClickSourcePath`
- If user fills one and switches modes, their input is lost
- **Impact:** Minor confusion when switching between workflow modes

---

## What Works Well

1. **One-Click concept** — auto-pick atlas + AI registration + manual review in one flow is exactly right
2. **SSIM quality scoring** — "Excellent / Good / Poor" with percentage improvement is very clear and actionable
3. **3D stack auto-detection** — correctly identifies 646 slices and dimensions
4. **Autopick progress modal** — clear progress bar with "Coarse scan: slice N/528" is informative
5. **Registration quality** — SSIM 0.62→0.99 is genuinely excellent; competitive with or better than Miki's brainreg-based results
6. **i18n support** — EN/中文 toggle is appreciated for our bilingual lab

---

## Suggestions for Next Version

1. **Auto-contrast thumbnails** — Apply percentile-based contrast stretch (e.g., 1st–99th percentile) to Z-slicer previews for fluorescence data
2. **Step progress indicator** — A persistent status bar showing "Step 2/4: Auto-picking atlas..." would eliminate confusion
3. **Atlas layer endpoint** — Fix the 404 on `/api/outputs/atlas-layer` to enable manual landmark correction
4. **Smart Z default** — Instead of defaulting to Z=323 (midpoint), auto-detect the layer with maximum tissue signal
5. **Memory-safe TIFF loading** — Use tifffile with `pages=[z]` lazy loading instead of loading the full 2.7GB stack into memory
6. **Toast management** — Auto-dismiss error toasts after 10s, or stack them with "Clear all" button

---

*Tested by a neuroscience researcher using Brainfast for ChAT-Cre cleared brain atlas registration. The tool shows strong potential — the core registration algorithm is excellent. Focus on the critical bugs (#9, #11) first, then polish the UX flow.*

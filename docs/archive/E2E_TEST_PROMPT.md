# Brainfast End-to-End Flow Test Prompt

## Role & Context

You are a neurobiologist testing the Brainfast brain atlas registration tool.
You have cleared-brain fluorescence imaging data and want to perform whole-brain
registration and cell counting. You will systematically test every UI flow,
recording PASS/FAIL for each checkpoint.

**Test Sample:**
```
D:/Brainfast/Sample/ChATe27/35_High_1000ms_560nm_640nm_150W_z5um_Bothlaser - Pos 3 4 [1] 3DMontage_XY1763150824_Z000_T0_C0.tif
```
- Format: 3D TIFF, 646 Z-slices, ~2.7 GB
- Pixel size: 5 µm, Z-step: 5 µm
- Hemisphere: right_flipped (cleared half-brain, lateral side on left)

**Server:** `http://127.0.0.1:8787`

---

## Phase 0: Environment Setup

### 0.1 Server Launch
- [ ] Start Flask server via `preview_start` (name: `brainfast`)
- [ ] Confirm server responds on port 8787
- [ ] Verify no startup errors in server logs (`preview_logs`, level: error)

### 0.2 Page Load
- [ ] Page title contains "Brainfast"
- [ ] No console errors on initial load (`preview_console_logs`, level: error)
- [ ] Network: no failed requests except expected volume-reg-stats 404
- [ ] Language toggle (EN/中文) visible and functional

---

## Phase 1: Input & Configuration

### 1.1 Source Path Input (Fix #1, #12)
- [ ] Placeholder shows generic hint (not a hardcoded Windows path)
- [ ] Type or set the test sample path into `#oneClickSourcePath`
- [ ] Path accepted without permission/encoding errors
- [ ] Reload page → path restored from localStorage
- [ ] Verify: `localStorage.getItem('brainfast.sourcePath')` matches input

### 1.2 3D TIFF Detection (Fix #11)
- [ ] `/api/slice/info` returns `is3d: true`, correct `z_count`
- [ ] Server does NOT crash or OOM (2.7 GB file handled safely)
- [ ] Thumbnail loads at default Z (midpoint ≈ 323), size 299×360+
- [ ] Scope auto-switches to "whole" for 3D stacks

### 1.3 Scope Toggle (Fix #7)
- [ ] In "whole" mode: Z-slicer (`#zSlicerBox`) is hidden
- [ ] Switch to "single": Z-slicer becomes visible
- [ ] Switch back to "whole": Z-slicer hides again
- [ ] Scope hint text updates on each switch

### 1.4 Z-Slicer (Fix #13)
- [ ] Set scope to "single" to reveal Z-slicer
- [ ] Move slider to different Z values (e.g., 100, 300, 500)
- [ ] Thumbnail updates in real-time as Z changes
- [ ] Verify: preview image `src` contains correct `z=` parameter
- [ ] Z number input and slider stay synchronized
- [ ] "Confirm Layer & Continue" button visible

### 1.5 Pixel Size Quick-Input (Fix #3)
- [ ] Pixel size row (`#oneClickPixelSizeRow`) visible in config section
- [ ] Change value → verify it syncs with Step 2 `#pixelSizeUm` field
- [ ] Change Step 2 field → verify it syncs back to quick-input

### 1.6 Configuration Options
- [ ] Hemisphere selector: auto / full / left / right_flipped
- [ ] Atlas version selector present and functional
- [ ] Registration mode selector present
- [ ] Each option shows hint text on change
- [ ] Settings persist to localStorage on change
- [ ] Reload → settings restored from localStorage

### 1.7 Mode Switch Path Sync (Fix #2)
- [ ] Set source path in One-Click tab
- [ ] Switch to Pro/Manual tab → `#inputDir` has same path
- [ ] Change path in Pro tab → switch back → One-Click path updated

---

## Phase 2: One-Click Workflow Execution

### 2.1 Workflow Start (Fix #8)
- [ ] Click `#oneClickStartBtn` ("Start One-Click Workflow")
- [ ] Step indicator activates: Step 1 (Config) → done, Step 2 (Atlas) → active
- [ ] No console errors on workflow start

### 2.2 Atlas Auto-Pick (Fix #4)
- [ ] Autopick modal appears with progress bar
- [ ] Progress updates from 0% to 100% smoothly
- [ ] No path-related errors in server logs (path has spaces + brackets)
- [ ] `/api/atlas/autopick-z` POST returns 200
- [ ] Autopick completes without crash

### 2.3 Overlay Preview Generation
- [ ] After autopick: overlay preview render starts automatically
- [ ] Preview modal shows render progress
- [ ] Render completes → modal auto-closes or shows "Done"
- [ ] Step indicator: Step 2 (Atlas) → done, Step 3 (Register) → active

### 2.4 AI Landmark Registration
- [ ] `/api/align/landmarks` POST returns 200
- [ ] Landmark pairs detected and applied
- [ ] Nonlinear refinement: `/api/align/nonlinear` returns 200
- [ ] Second overlay preview generated with refined alignment
- [ ] Registration quality panel shows before/after scores

### 2.5 Registration Quality Assessment
- [ ] Quality panel visible with numeric scores
- [ ] Before/after comparison meaningful (after > before)
- [ ] Quality rating text displayed (e.g., "Excellent", "Good")
- [ ] Improvement percentage shown

### 2.6 Step Indicator Final State
- [ ] Step 1 Config: done ✓
- [ ] Step 2 Atlas: done ✓
- [ ] Step 3 Register: active or done
- [ ] Step 4 Pipeline: pending or active

---

## Phase 3: Manual Landmark Correction (Fix #9)

### 3.1 Manual Landmark Section
- [ ] Section visible after AI registration completes
- [ ] Real image (overlay preview) loaded with correct dimensions
- [ ] Atlas layer image loaded with matching dimensions
- [ ] Both canvases sized to match their images

### 3.2 Atlas Layer Auto-Generation
- [ ] If atlas-layer PNG doesn't exist: HEAD returns 404
- [ ] POST `/api/overlay/atlas-layer` fires automatically
- [ ] POST uses correct camelCase keys: `labelPath`, `realPath`, `structureCsv`, `pixelSizeUm`
- [ ] POST returns 200 (not 400)
- [ ] Subsequent GET returns 200 with valid PNG
- [ ] Atlas image renders in the manual landmark canvas

### 3.3 Manual Mode Interaction
- [ ] "Enter Manual Mode" button works
- [ ] Can click to place landmark points on real image
- [ ] Can click to place corresponding points on atlas image
- [ ] "Apply Manual Landmarks" button enabled after placing points
- [ ] "Clear Manual Points" removes all placed points
- [ ] "Undo" removes last placed point

---

## Phase 4: Error Handling & Edge Cases

### 4.1 Toast Notifications (Fix #5)
- [ ] Success toasts auto-dismiss after ~4 seconds
- [ ] Info toasts auto-dismiss after ~5 seconds
- [ ] Error toasts auto-dismiss after ≥10 seconds (not stuck forever)
- [ ] All toasts have a manual close (×) button
- [ ] Close button immediately removes the toast

### 4.2 404 Polling Prevention (Fix #10)
- [ ] On page load: `/api/outputs/volume-reg-stats` → 404 (expected)
- [ ] Subsequent polling cycles: NO repeated 404 requests
- [ ] Verify: `_volumeRegStatsAvailable` flag set to `false` after first 404
- [ ] Flag resets when a new workflow generates real stats

### 4.3 Path Normalization (Fix #4)
- [ ] Paths with spaces work: `...Pos 3 4 [1]...`
- [ ] Paths with brackets work: `...[1]...`
- [ ] Mixed separators handled: `D:/path\to/file` → normalized
- [ ] Double-backslashes handled: `D:\\path\\to\\file` → normalized
- [ ] Global `before_request` hook normalizes query-string `path` param
- [ ] Per-endpoint `_normalize_path()` normalizes JSON body paths

### 4.4 Thumbnail Contrast (Fix #6)
- [ ] Thumbnail image is visible (not pure black)
- [ ] Uses percentile-based normalization (1st–99th percentile)
- [ ] Fluorescence signal clearly visible against background
- [ ] Works for both dim and bright Z-slices

### 4.5 Large File Handling (Fix #11)
- [ ] 2.7 GB 3D TIFF: server does not crash
- [ ] Memory usage stays reasonable (no full-file imread)
- [ ] Z-slice extraction uses `TiffFile.pages[z].asarray()` (single page)
- [ ] Multiple rapid Z changes don't cause server instability

---

## Phase 5: Pipeline Execution (if atlas + registration complete)

### 5.1 Pipeline Start
- [ ] Step indicator: Step 4 (Pipeline) → active
- [ ] "Run Pipeline" button clickable
- [ ] Pipeline status updates in real-time
- [ ] Channel selection (Red / Green / Far-Red / All) works

### 5.2 Pipeline Progress
- [ ] `/api/status` polling returns progress info
- [ ] Slice progress bar updates (slicesDone / slicesTotal)
- [ ] No server crashes during cell detection

### 5.3 Pipeline Completion
- [ ] Step indicator: all 4 steps → done
- [ ] Success toast displayed
- [ ] Output files generated in job directory

---

## Phase 6: QC & Results Review

### 6.1 Batch QC Tab
- [ ] Switch to "Batch QC Review" tab
- [ ] Demo panel image loads (if generated)
- [ ] Best slice comparison visible
- [ ] Registration stats summary displayed
- [ ] Lightbox opens on image click

### 6.2 Results Tab
- [ ] Switch to "Results" tab
- [ ] Cell count table loads with region hierarchy
- [ ] Percentage and distribution bars visible
- [ ] Depth filter buttons (1/2/3/4/All) functional
- [ ] "Export CSV" generates downloadable file
- [ ] "Export Methods Text" shows reproducibility info
- [ ] Cell count chart (bar + pie) renders

---

## Phase 7: Cross-Cutting Concerns

### 7.1 i18n
- [ ] Switch to 中文 → all labels update to Chinese
- [ ] Switch back to EN → all labels revert to English
- [ ] Toast messages respect current language
- [ ] Hint texts update in current language

### 7.2 State Persistence
- [ ] Source path persists across reload
- [ ] Scope/hemisphere/atlas/regMode persist across reload
- [ ] Z-slicer state (path + max) re-initialized on reload via `checkSliceIs3D`
- [ ] Job ID survives tab switches within same session

### 7.3 Performance
- [ ] Page load time < 3 seconds
- [ ] No memory leaks from polling intervals
- [ ] Thumbnail generation < 2 seconds per slice
- [ ] Z-slider response feels real-time (< 500ms perceived)

### 7.4 Network Health
- [ ] Check `preview_network(filter: 'failed')` at end of each phase
- [ ] Only expected 404s (volume-reg-stats before first run)
- [ ] No 500 errors
- [ ] No CORS or permission errors

---

## Test Results Template

```
Date:        ____-__-__
Tester:      _______________
Branch:      _______________
Commit:      _______________
Server:      http://127.0.0.1:8787
Sample:      ChATe27 / 35_High... (3D TIFF, 646 slices)

Phase 0: Environment      [ ] / [ ] passed
Phase 1: Input & Config   [ ] / [ ] passed
Phase 2: One-Click Flow   [ ] / [ ] passed
Phase 3: Manual Landmarks [ ] / [ ] passed
Phase 4: Error Handling    [ ] / [ ] passed
Phase 5: Pipeline          [ ] / [ ] passed  (or SKIP if no atlas)
Phase 6: QC & Results      [ ] / [ ] passed  (or SKIP if no pipeline)
Phase 7: Cross-Cutting     [ ] / [ ] passed

Total:   [ ] / [ ] passed
Blockers: _______________
New bugs found: _______________
```

---

## Execution Notes

- **Tools to use:** `preview_start`, `preview_eval`, `preview_snapshot`,
  `preview_click`, `preview_fill`, `preview_console_logs`, `preview_logs`,
  `preview_network`, `preview_inspect`
- **Avoid:** `preview_screenshot` (known 30s timeout on this Windows machine)
- **After each phase:** check `preview_network(filter: 'failed')` and
  `preview_console_logs(level: 'error')` for regressions
- **Between phases:** do NOT reload unless testing persistence explicitly
- **Timing:** Atlas autopick ~30s, overlay render ~20s, AI registration ~30s,
  atlas-layer render ~25s — use `sleep` accordingly before checking results

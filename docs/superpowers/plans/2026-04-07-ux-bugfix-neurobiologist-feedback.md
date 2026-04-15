# Brainfast v0.3.0 UX + Algorithm Improvement Plan (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix remaining 5 UX issues from neurobiologist testing + implement 3 most impactful CCFv3 algorithm improvements.

**Architecture:** UX fixes are frontend-only (app.js + index.html). Algorithm fixes touch core scripts (map_and_aggregate.py, overlay_render.py, atlas_autopick.py).

**Tech Stack:** Vanilla JavaScript, HTML5, Python/Flask, scikit-image, SimpleITK, scipy

---

## Part A: Remaining UX Fixes

### Task A1: Add hemisphere selector to One-Click mode

**Files:**
- Modify: `project/frontend/index.html:170-177`
- Modify: `project/frontend/app.js` (runOneClickWorkflow function + i18n)

Currently the one-click mode has no hemisphere selection. Cleared tissue users (like our ChAT samples) need to specify left/right/right_flipped.

- [ ] **Step 1: Add hemisphere dropdown to HTML**

In `project/frontend/index.html`, after the `oneClickScope` label (around line 177), add:

```html
<label class="field-label">
  <span data-i18n="label.hemisphere">Hemisphere / 半球方向</span>
  <select id="oneClickHemisphere">
    <option value="auto" data-i18n="opt.hemiAuto">Auto-detect (recommended)</option>
    <option value="full" data-i18n="opt.hemiFull">Full brain (both hemispheres)</option>
    <option value="left" data-i18n="opt.hemiLeft">Left hemisphere</option>
    <option value="right_flipped" data-i18n="opt.hemiRightFlipped">Right hemisphere (flipped, lateral=left)</option>
  </select>
  <span class="field-hint" id="oneClickHemiHint" data-i18n="hint.hemiAuto">Auto-detect tries all orientations and picks the best match.</span>
</label>
```

- [ ] **Step 2: Add i18n keys**

In `LANGS.en`:
```javascript
'label.hemisphere': 'Hemisphere',
'opt.hemiAuto': 'Auto-detect (recommended)',
'opt.hemiFull': 'Full brain (both hemispheres)',
'opt.hemiLeft': 'Left hemisphere',
'opt.hemiRightFlipped': 'Right hemisphere (flipped, lateral=left)',
'hint.hemiAuto': 'Auto-detect tries all orientations and picks the best match.',
'hint.hemiFull': 'Register as a complete coronal section with both hemispheres.',
'hint.hemiLeft': 'Left hemisphere only — medial face on right side of image.',
'hint.hemiRightFlipped': 'Right hemisphere, flipped so lateral cortex is on the left side of the image.',
```

In `LANGS.zh`:
```javascript
'label.hemisphere': '半球方向',
'opt.hemiAuto': '自动检测（推荐）',
'opt.hemiFull': '全脑（双半球）',
'opt.hemiLeft': '左半球',
'opt.hemiRightFlipped': '右半球（翻转，外侧在左）',
'hint.hemiAuto': '自动检测会尝试所有朝向并选择最佳匹配。',
'hint.hemiFull': '作为完整冠状切面配准，包含双侧半球。',
'hint.hemiLeft': '仅左半球——内侧面在图像右侧。',
'hint.hemiRightFlipped': '右半球，翻转使外侧皮质在图像左侧。',
```

- [ ] **Step 3: Wire hemisphere into one-click workflow**

In `runOneClickWorkflow()`, after setting alignMode (around the `scope === 'whole'` block), add:

```javascript
const hemiEl = document.getElementById('oneClickHemisphere');
if (hemiEl && hemiEl.value !== 'auto') {
  const flipEl = document.getElementById('flipAtlas');
  if (flipEl) {
    const hemiMap = { 'full': 'none', 'left': 'none', 'right_flipped': 'h' };
    flipEl.value = hemiMap[hemiEl.value] || 'none';
  }
}
```

- [ ] **Step 4: Add hint update handler**

```javascript
const oneClickHemiEl = document.getElementById('oneClickHemisphere');
if (oneClickHemiEl) {
  oneClickHemiEl.onchange = () => {
    const hint = document.getElementById('oneClickHemiHint');
    if (hint) hint.textContent = t('hint.hemi' + oneClickHemiEl.value.charAt(0).toUpperCase() + oneClickHemiEl.value.slice(1).replace('_flipped', 'RightFlipped').replace('full', 'Full').replace('auto', 'Auto'));
  };
}
```

- [ ] **Step 5: Commit**
```bash
git add project/frontend/app.js project/frontend/index.html
git commit -m "feat: add hemisphere selector to one-click registration mode"
```

---

### Task A2: Add multi-channel guidance to One-Click mode

**Files:**
- Modify: `project/frontend/index.html`
- Modify: `project/frontend/app.js`

Users with dual-channel TIFFs (C0 reporter + C1 marker) don't know which file to load or how to process both channels.

- [ ] **Step 1: Add channel info hint below the source TIFF input**

In `project/frontend/index.html`, after the `oneClickSourcePath` input-browse div, add:

```html
<span class="field-hint" data-i18n="hint.channelGuide">For multi-channel data: load the reporter channel (C0) for registration. After registration, use Step 4 "Run Pipeline" to process each channel separately.</span>
```

- [ ] **Step 2: Add i18n keys**

EN: `'hint.channelGuide': 'For multi-channel data: load the reporter channel (C0) for registration. After registration, use Step 4 to process each channel separately.',`
ZH: `'hint.channelGuide': '多通道数据：请加载reporter通道（C0）用于配准。配准完成后，在步骤4"运行流程"中分别处理各通道。',`

- [ ] **Step 3: Commit**
```bash
git add project/frontend/app.js project/frontend/index.html
git commit -m "feat: add multi-channel guidance hint to one-click mode"
```

---

### Task A3: Add TIFF preview thumbnail after loading

**Files:**
- Modify: `project/frontend/index.html`
- Modify: `project/frontend/app.js`

After entering a TIFF path, show a small preview so users can confirm they loaded the correct file.

- [ ] **Step 1: Add thumbnail container in HTML**

After the `zSlicerBox` div in index.html, add:

```html
<div id="oneClickPreviewBox" class="hidden" style="margin-top:0.5rem;text-align:center;">
  <img id="oneClickPreviewImg" style="max-height:180px;max-width:100%;border-radius:6px;border:1px solid #333;" alt="TIFF preview" />
  <div id="oneClickPreviewInfo" class="field-hint" style="margin-top:0.3rem;"></div>
</div>
```

- [ ] **Step 2: Add preview generation on TIFF load**

In app.js, in the `checkSliceIs3D` function, after the pixel size detection block, add:

```javascript
// Show thumbnail preview
const previewBox = document.getElementById('oneClickPreviewBox');
const previewImg = document.getElementById('oneClickPreviewImg');
const previewInfo = document.getElementById('oneClickPreviewInfo');
if (previewBox && previewImg) {
  try {
    const z = res.is3d ? Math.floor((res.z_count || 0) / 2) : 0;
    previewImg.src = `/api/slice/thumbnail?path=${encodeURIComponent(path)}&z=${z}&size=360`;
    previewImg.onload = () => { previewBox.classList.remove('hidden'); };
    previewImg.onerror = () => { previewBox.classList.add('hidden'); };
    if (previewInfo) {
      const dims = res.is3d ? `${res.z_count} slices, ${res.width}×${res.height} px` : `${res.width}×${res.height} px`;
      previewInfo.textContent = dims + (res.pixel_size_um ? `, ${res.pixel_size_um} µm/px` : '');
    }
  } catch {}
}
```

- [ ] **Step 3: Add thumbnail API endpoint**

In `project/frontend/blueprints/api_atlas.py` or a new route, add a `/api/slice/thumbnail` endpoint that reads a single z-layer from the TIFF, normalizes to 8-bit, resizes to `size` px, and returns as JPEG.

- [ ] **Step 4: Commit**
```bash
git add project/frontend/app.js project/frontend/index.html project/frontend/blueprints/
git commit -m "feat: show TIFF preview thumbnail after loading source file"
```

---

### Task A4: Fix QC score always showing 1

**Files:**
- Modify: `project/frontend/blueprints/api_demo.py:141-164`

The `slice_registration_qc.csv` likely has placeholder scores or the scoring doesn't differentiate quality levels.

- [ ] **Step 1: Check the actual QC CSV data**

Read `project/outputs/codex_demo_35_full/slice_registration_qc.csv` to see what scores are stored.

- [ ] **Step 2: If scores are all 1, fix the score writer**

The score comes from `_alignment_quality()` in `overlay_render.py`. If the per-slice QC writer is just writing "1" as a placeholder, fix it to write the actual computed score. Search for where `slice_registration_qc.csv` is written.

- [ ] **Step 3: Add score formatting in frontend**

In app.js where the stats bar is rendered, format scores to 3 decimal places and add a color indicator (green >0.7, yellow 0.4-0.7, red <0.4):

```javascript
const scoreColor = stats.mean_score > 0.7 ? '#5c9' : stats.mean_score > 0.4 ? '#fc5' : '#f55';
```

- [ ] **Step 4: Commit**
```bash
git add -u
git commit -m "fix: compute and display meaningful registration quality scores"
```

---

### Task A5: Make Guide button context-sensitive

**Files:**
- Modify: `project/frontend/index.html:30-46`
- Modify: `project/frontend/app.js`

Currently the guide is static 5-step text. Make it show different content based on current tab.

- [ ] **Step 1: Add per-tab guide content in i18n**

Add keys for each tab's guide content (Registration Workflow, Manual TIFF Check, Batch QC, Results).

- [ ] **Step 2: Update guide button handler**

```javascript
document.getElementById('guideBtn').onclick = () => {
  const activeTab = document.querySelector('.sidebar .active')?.textContent?.trim();
  const guideContent = document.getElementById('guideContent');
  // Map tab to guide key
  const guideMap = {
    'Registration Workflow': 'guide.workflow',
    '配准工作流': 'guide.workflow',
    'Manual TIFF Check': 'guide.manualCheck',
    '手动TIFF检查': 'guide.manualCheck',
    'Batch QC Review': 'guide.batchQc',
    '批量QC审查': 'guide.batchQc',
    'Results': 'guide.results',
    '统计结果': 'guide.results',
  };
  const key = guideMap[activeTab] || 'guide.workflow';
  guideContent.innerHTML = t(key);
  document.getElementById('guideModal').classList.remove('hidden');
};
```

- [ ] **Step 3: Commit**
```bash
git add project/frontend/app.js project/frontend/index.html
git commit -m "feat: make Guide button show context-sensitive help per tab"
```

---

## Part B: CCFv3 Algorithm Improvements (Top 3)

### Task B1: Fix landmark matching — replace index pairing with descriptor matching

**Files:**
- Modify: `project/frontend/services/alignment_service.py` or `project/scripts/ai_landmark.py`

The current Harris corner detection pairs points by array index, which is random. Replace with ORB descriptor matching + RANSAC.

- [ ] **Step 1: Find the landmark detection code**

Search for Harris corner detection and the pairing logic.

- [ ] **Step 2: Replace with ORB matching**

```python
import cv2

def detect_and_match_landmarks(real_img, atlas_img, max_points=30):
    orb = cv2.ORB_create(nfeatures=max_points * 3)
    kp1, des1 = orb.detectAndCompute(real_img, None)
    kp2, des2 = orb.detectAndCompute(atlas_img, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.empty((0, 4))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[:max_points]
    pairs = np.array([[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1],
                       kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]]
                      for m in matches])
    return pairs
```

- [ ] **Step 3: Add RANSAC filtering**

Use the existing RANSAC logic but with properly matched pairs.

- [ ] **Step 4: Commit**
```bash
git add project/scripts/ project/frontend/services/
git commit -m "fix: replace random index-based landmark pairing with ORB descriptor matching"
```

---

### Task B2: Implement per-pixel cell-to-region mapping

**Files:**
- Modify: `project/scripts/map_and_aggregate.py`

Currently cells are mapped by slice_id join. Fix to look up each cell's (x,y) in the warped atlas label map.

- [ ] **Step 1: Modify the mapping function**

```python
import tifffile

def map_cells_to_regions(cells_csv, label_dir, structure_tree):
    """Map each cell to its atlas region by pixel coordinate lookup."""
    cells = pd.read_csv(cells_csv)
    region_ids = []
    for _, cell in cells.iterrows():
        slice_id = cell['slice_id']
        label_path = label_dir / f"slice_{slice_id:04d}_label.tif"
        if label_path.exists():
            label = tifffile.imread(str(label_path))
            x, y = int(round(cell['x'])), int(round(cell['y']))
            if 0 <= y < label.shape[0] and 0 <= x < label.shape[1]:
                region_ids.append(int(label[y, x]))
            else:
                region_ids.append(0)
        else:
            region_ids.append(0)
    cells['region_id'] = region_ids
    # Join with structure tree for names
    cells = cells.merge(structure_tree[['id', 'name', 'acronym', 'structure_id_path']],
                        left_on='region_id', right_on='id', how='left')
    return cells
```

- [ ] **Step 2: Cache label images**

Add LRU cache to avoid re-reading the same label TIFF for every cell on the same slice.

- [ ] **Step 3: Update aggregation**

The `aggregate_counts` function should group by the new per-cell `region_id` and walk up the hierarchy tree.

- [ ] **Step 4: Commit**
```bash
git add project/scripts/map_and_aggregate.py
git commit -m "feat: implement per-pixel cell-to-region mapping instead of slice-level join"
```

---

### Task B3: Add inter-slice AP consistency enforcement

**Files:**
- Modify: `project/scripts/atlas_autopick.py`

After per-slice AP estimation, fit a smooth model and reject outliers.

- [ ] **Step 1: Add consistency enforcement**

After all slices are scored independently, add:

```python
def enforce_ap_consistency(slice_indices, ap_values, scores):
    """Fit linear AP model and reject outliers."""
    import numpy as np
    from scipy import stats as sp_stats

    idx = np.array(slice_indices, dtype=float)
    ap = np.array(ap_values, dtype=float)
    weights = np.array(scores, dtype=float)

    # Weighted linear regression: AP = a * slice_index + b
    slope, intercept, r, p, se = sp_stats.linregress(idx, ap)
    predicted = slope * idx + intercept
    residuals = np.abs(ap - predicted)

    # Reject outliers (> 2 sigma)
    threshold = np.std(residuals) * 2
    inliers = residuals < threshold

    # Refit on inliers
    if np.sum(inliers) >= 3:
        slope, intercept, _, _, _ = sp_stats.linregress(idx[inliers], ap[inliers])

    # Return smoothed AP values
    smoothed = slope * idx + intercept
    return smoothed.astype(int), {'slope': slope, 'intercept': intercept, 'r_squared': r**2}
```

- [ ] **Step 2: Integrate into the autopick pipeline**

After the coarse scan, apply `enforce_ap_consistency` and use smoothed AP values for refinement.

- [ ] **Step 3: Commit**
```bash
git add project/scripts/atlas_autopick.py
git commit -m "feat: enforce inter-slice AP consistency with linear model + outlier rejection"
```

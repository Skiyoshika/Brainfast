# Registration Failure Taxonomy

**Date:** 2026-04-10
**Source:** Sample 35 (cleared tissue, right hemisphere, 111 Z-stack slices) and ChATe27 experience.
**Purpose:** Classify the failure modes encountered during 3D whole-brain registration so that future debugging starts from a known checklist, not from scratch each time.

---

## Category 1: Wrong AP Slice Choice

**What happens:** The atlas coronal slice assigned to a given tissue section is at the wrong anterior-posterior position. Structures visible in the tissue do not match the structures in the assigned atlas slice.

**Root cause in Sample 35:** The default `atlas_autopick` algorithm converged all slices to approximately AP 295, which was incorrect. The fix was to bypass autopick entirely and compute AP directly from the filename Z-index:

```
atlas_z = int(z_filename * -0.2) + 330
```

This required setting `atlas_z_from_filename=True` and `atlas_z_z_scale=0.2` in the config, and disabling AP refinement (`atlas_z_refine_range=0`) because the refinement also converged to the wrong position.

**Detection:** Compare the atlas overlay with the tissue. If major structural boundaries (e.g., hippocampus, striatum, cortex thickness) do not line up at all, AP choice is likely wrong.

**Severity:** Fatal. All downstream region mapping is meaningless if the wrong atlas slice is used.

## Category 2: Hemisphere / Left-Right Inversion

**What happens:** The atlas annotation lands on the wrong side of the tissue, or the left-right orientation of brain regions is flipped.

**Root cause in Sample 35:** The initial configuration used `ml_flip=True`, which caused the annotation to be mirrored. The tissue was a right hemisphere mounted with the lateral side on the left of the image. The fix was to use `atlas_hemisphere="right_flipped"` mode, which correctly places the lateral side on the left and the medial side aligned with the tissue's right edge.

**Root cause in ChATe27:** The pixel size was incorrectly set to 0.65 um instead of the actual 4 um, causing massive scale mismatch that also manifested as apparent L/R confusion.

**Detection:** Check whether cortex, hippocampus, and other asymmetric landmarks are on the correct side. Overlay a few slices and verify that medial structures (midline) align with the tissue's medial edge.

**Severity:** Fatal. Every region label is assigned to the wrong hemisphere.

## Category 3: Scale / Fit Mismatch

**What happens:** The atlas template is significantly larger or smaller than the tissue, or the aspect ratio is wrong. The overlay may cover only a fraction of the tissue, or extend far beyond it.

**Root cause in Sample 35:** The half-hemisphere atlas template was narrower than the tissue by approximately 1.1 mm. This required adjusting the template preparation to account for the actual tissue dimensions.

**Root cause in ChATe27:** Using 0.65 um pixel size (the objective spec) instead of 4 um (the actual scan resolution) made the tissue appear approximately 6x larger than expected, so the atlas template was far too small.

**Detection:** Compare the bounding box of the atlas overlay with the tissue extent. If the overlay covers less than 50% of the tissue width or height, scale is likely wrong.

**Severity:** High. Partial coverage means some tissue regions have no annotation at all, and the regions that are annotated may be stretched or compressed.

## Category 4: Tissue Crop / Padding Mismatch

**What happens:** The registration produces an annotation that technically covers the tissue, but the actual annotation content (non-zero labels) covers only a small fraction of the tissue area.

**Root cause in Sample 35:** ANTs registration with `genericLabel` interpolation for the annotation volume produced only 1.9% annotation coverage -- nearly all pixels were interpolated to zero. Switching to `nearestNeighbor` interpolation for the annotation warp (while keeping the intensity image on linear interpolation) increased coverage to approximately 46%.

**Detection:** Compute the fraction of tissue pixels that have a non-zero annotation label. If this is below 20%, interpolation or padding is likely wrong.

**Severity:** High. The overlay may look roughly correct in shape, but most cells fall in "unlabeled" regions, making the count table nearly empty.

## Category 5: Nonlinear Warp Distortion

**What happens:** The nonlinear (deformable) registration step introduces artifacts: folding, excessive stretching, or anatomically implausible deformations that break the correspondence between atlas regions and tissue.

**Root cause in Sample 35:** The Laplacian PDE refinement step (`laplacian_refine_3d.py`), designed to smooth the warp field, actually degraded results on cross-modality (fluorescence vs. Nissl) data. The refinement assumed similar intensity profiles between moving and fixed images, which does not hold for cleared-tissue fluorescence vs. Allen reference atlas Nissl stains.

**Detection:** Look for:
- Atlas region boundaries that follow tissue texture rather than anatomical boundaries.
- Regions that appear to fold over themselves.
- Dramatically different warp quality between adjacent slices in the same stack.

**Severity:** Medium to high. The gross anatomy may be roughly correct, but fine region boundaries are unreliable.

## Category 6: Output Looks Plausible in One Slice but Inconsistent Across Stack

**What happens:** A spot check of a single slice shows acceptable registration, but reviewing multiple slices reveals that quality varies wildly -- some slices are well-registered while others are clearly wrong, or the registration quality degrades smoothly from one end of the stack to the other.

**Root cause:** 3D registration optimizes a global objective across the entire volume. If the tissue has local damage, tears, folding artifacts, or variable staining quality, the optimizer may sacrifice some slices to improve others. Additionally, if the AP mapping is slightly off, slices near the edges of the stack may be matched to atlas slices that are progressively more incorrect.

**Detection:** Review overlays for at least 5-10 slices spread across the full Z range. Check whether the same anatomical landmark (e.g., a specific cortical layer boundary) is consistently placed across slices.

**Severity:** Medium. Aggregate counts may still be roughly correct if most slices are acceptable, but per-slice region assignments are unreliable for the poor slices.

## Category 7: Cross-Modality Metric Choice

**What happens:** The similarity metric used by the registration algorithm is inappropriate for the image modalities being aligned, leading to convergence to a wrong local minimum or failure to converge at all.

**Relevant context:** The Allen reference atlas is a Nissl-stained brightfield image. Cleared-tissue samples imaged with fluorescence have inverted contrast (bright structures on dark background vs. dark structures on light background) and different tissue contrast patterns. Standard cross-correlation (CC) assumes a linear intensity relationship, which does not hold across modalities. Mutual information (MI) is modality-independent but noisier and slower.

**Detection:** If the registration converges but the overlay is systematically offset or rotated, or if the deformable step makes things worse rather than better, the metric may be inappropriate.

**Severity:** Medium. Can often be addressed by switching metrics or pre-processing (e.g., edge maps, tissue masks) without changing the registration framework.

---

## Debugging Checklist

When a new sample fails registration, work through these categories in order:

1. **AP slice choice:** Is the assigned atlas slice anatomically plausible for this tissue section?
2. **Hemisphere orientation:** Is the atlas on the correct side? Are medial/lateral edges correct?
3. **Scale:** Does the atlas template roughly match the tissue size?
4. **Annotation coverage:** What fraction of tissue pixels have non-zero labels?
5. **Warp quality:** Are region boundaries anatomically plausible, or do they follow image artifacts?
6. **Cross-stack consistency:** Do at least 5 slices across the Z range look acceptable?
7. **Metric appropriateness:** Is the similarity metric suitable for this modality combination?

Fix failures in this order. Do not attempt to tune nonlinear warp parameters (category 5) until categories 1-4 are resolved.

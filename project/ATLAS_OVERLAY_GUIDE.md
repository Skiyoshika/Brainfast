# ATLAS_OVERLAY_GUIDE

## 1) Start app
- Desktop: run `frontend/dist/IdleBrainUI.exe`
- Or dev mode: `cd frontend && python server.py`
- Open: `http://127.0.0.1:8787`

## 2) Fill required paths
- **Input TIFF Folder**: your merged slice folder
- **Atlas annotation_25.nii.gz**: Allen annotation volume path
- **Structure Mapping CSV**: `1_adult_mouse_brain_graph_mapping.csv`
- **Real Slice Path (single TIFF for preview)**: one real slice
- **Atlas Label Slice Path (single TIFF)**: corresponding atlas label slice

## 3) Overlay preview
- Set **Overlay Alpha** (recommended 45)
- Set **Overlay Mode**:
  - `Fill` for full colored region view
  - `Contour` for boundary-only view
- Click **Refresh Preview**

## 4) AI alignment
- Choose **Align Mode** (`Affine` or `Nonlinear`)
- Optional tuning:
  - Max Landmarks: 30~60
  - Min Distance: 8~15
  - RANSAC Residual: 6~10
- Click **AI Landmark Align**
- Check logs for scores:
  - raw score: before -> after
  - edge SSIM: before -> after
- If edge SSIM drops, try:
  - lower `RANSAC Residual`
  - increase `Max Landmarks`

## 5) Landmark quality check
- Click **View Landmarks**
- Confirm real vs atlas point distributions are reasonable

## 6) Run full pipeline
- Select channel (red/green/farred) or **Run All Channels**
- Click **Run Pipeline**
- Monitor logs and progress
- Use **Cancel** to stop if needed

## 7) Outputs
Generated in `project/outputs`:
- `cells_detected.csv`
- `cells_dedup.csv`
- `cells_mapped.csv`
- `cell_counts_leaf.csv`
- `cell_counts_hierarchy.csv`
- `slice_qc.csv`
- `overlay_preview.png`
- `overlay_compare.png`
- `overlay_compare_nonlinear.png`
- `landmark_preview.png`

## 8) Quick troubleshooting
- **Path validation failed**: verify file/folder exists and permissions
- **No preview image**: confirm `real slice` and `atlas label slice` are matching dimensions
- **Alignment poor**: switch to Nonlinear + adjust landmark params
- **Memory issue on large images**: downsample or run tiled preprocessing


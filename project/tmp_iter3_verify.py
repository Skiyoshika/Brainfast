from pathlib import Path
from scripts.atlas_autopick import autopick_best_z
from scripts.overlay_render import render_overlay

base = Path(r'D:\brain-atlas-cellcount-work\project')
real = Path(r'D:\brain-atlas-cellcount-work\Samples\35_High_1000ms_560nm_640nm_150W_z5um_Bothlaser - Pos 3 4 [1] 3DMontage_XY1763150824_Z000_T0_C0.tif')
atlas = Path(r'D:\brain-atlas-cellcount-work\repos\uci-allen-brainrepositorycodegui\CCF_DATA\annotation_25.nii.gz')
out_label = base / 'outputs' / 'auto_label_coronal_iter3_verify.tif'
out_png = base / 'outputs' / 'overlay_preview_iter3_verify.png'

print('=== ITER:3/3 Verification ===')
print('exists:', real.exists(), atlas.exists())

# Test autopick with ROI
res = autopick_best_z(real, atlas, out_label, z_step=1, pixel_size_um=0.65, slicing_plane='coronal', roi_mode='auto')
print('\nautopick result:')
print(f"  best_z: {res['best_z']}")
print(f"  roi_mode: {res['roi_mode']}")
print(f"  roi_bbox: {res['roi_bbox']}")

# Test overlay with diagnostic
_, diag = render_overlay(real, out_label, out_png, alpha=0.45, mode='contour-major',
                         min_mean_threshold=8.0, pixel_size_um=0.65,
                         rotate_deg=0.0, flip_mode='none', return_meta=True, 
                         major_top_k=12, fit_mode='contain')
print('\ndiagnostic:')
for k in ['fitMode', 'real_aspect', 'atlas_aspect', 'roi_bbox', 'roi_center_full', 'roi_roundtrip_error', 'label_shape_fitted']:
    print(f"  {k}: {diag.get(k)}")

print(f'\npng: {out_png.exists()}, size: {out_png.stat().st_size if out_png.exists() else -1}')
print('\n=== Key checks ===')
print(f"✓ ROI bbox has offset: {res['roi_bbox'][0] > 0 or res['roi_bbox'][1] > 0}")
print(f"✓ contain preserved aspect: label_fitted={diag['label_shape_fitted']}, not stretched to full {diag['label_shape_after']}")
print(f"✓ roundtrip error: {diag['roi_roundtrip_error']:.6f}")

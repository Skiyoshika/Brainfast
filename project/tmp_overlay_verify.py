from pathlib import Path
from scripts.atlas_autopick import autopick_best_z
from scripts.overlay_render import render_overlay

base = Path(r'D:\brain-atlas-cellcount-work\project')
real = Path(r'D:\brain-atlas-cellcount-work\Samples\35_High_1000ms_560nm_640nm_150W_z5um_Bothlaser - Pos 3 4 [1] 3DMontage_XY1763150824_Z000_T0_C0.tif')
atlas = Path(r'D:\brain-atlas-cellcount-work\repos\uci-allen-brainrepositorycodegui\CCF_DATA\annotation_25.nii.gz')
out_label = base / 'outputs' / 'auto_label_coronal_verify.tif'
out_png = base / 'outputs' / 'overlay_preview_verify.png'

print('exists', real.exists(), atlas.exists())
res = autopick_best_z(real, atlas, out_label, z_step=1, pixel_size_um=0.65, slicing_plane='coronal')
print('autopick', res)
_, diag = render_overlay(real, out_label, out_png, alpha=0.45, mode='contour-major',
                         min_mean_threshold=8.0, pixel_size_um=0.65,
                         rotate_deg=0.0, flip_mode='none', return_meta=True, major_top_k=12)
print('diagnostic', diag)
print('png', str(out_png), out_png.exists(), out_png.stat().st_size if out_png.exists() else -1)

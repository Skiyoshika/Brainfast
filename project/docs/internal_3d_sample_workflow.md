# Brainfast Internal 3D Sample Workflow

This workflow is for local single-machine validation against the raw 3D TIFF files in `../Sample/`.

## Recommended config

Use one of these configs:

- `configs/run_config_3d_sample.json` for the fast Elastix baseline
- `configs/run_config_3d_ants_sample.json` for the higher-quality `ANTs + Laplacian refinement` path

The ANTs config encodes the current best local defaults learned from `Sample/Miki`:

- `atlas_hemisphere = left`
- `target_um = 25.0`
- `pad_z = 1`
- `pad_y = 50`
- `pad_x = 50`
- `normalize = false`
- `laplacian.enabled = true`
- `laplacian.rtol = 0.01`
- `laplacian.maxiter = 250`

The padding values mirror the extra border used in the `Miki` NIfTI preparation. Brainfast currently recommends the 25um `ANTs + Laplacian` path because it fits local memory while still giving much better registration quality than the earlier Elastix baseline.

## Run a raw Sample TIFF

From `project/`:

```powershell
python scripts/run_3d_registration.py `
  --config configs/run_config_3d_ants_sample.json `
  --input-path "..\Sample\35_High_1000ms_560nm_640nm_150W_z5um_Bothlaser - Pos 3 4 [1] 3DMontage_XY1763150824_Z000_T0_C0.tif" `
  --out-dir outputs\sample35_ch0_3d_internal
```

You can swap `--input-path` to any other raw stack in `../Sample/`.

## Outputs

Each run writes:

- `report.html` and `OPEN_ME_FIRST.txt`
- `brain_25um.nii.gz` or another `brain_<target>.nii.gz`
- `annotation_fixed_half.nii.gz`
- `registered_brain_pre_laplacian.nii.gz`
- `registration_metadata.json`
- `registration_metrics.csv`
- `registration_summary.txt`
- `overview.png`
- `laplacian/parameters/boundary_conditions.csv`
- `laplacian/parameters/fpoints.npy`
- `laplacian/parameters/mpoints.npy`
- `laplacian/parameters/laplacian_deformation_field.npy`

If you are checking results as a user, do this instead of browsing files manually:

1. Open `outputs/index.html`
2. Click the run you care about
3. Open that run's `report.html`

`report.html` shows the before/after overview images, metric deltas, and the few files worth opening. `overview.png` is still the quickest raw visual check, but you should not need to dig through the folder first.

## Current local benchmark

Real sample runs completed successfully on:

- `35 ... C0.tif` with `ANTs + Laplacian`
- `41 ... C0.tif` with `ANTs + Laplacian`

Current measured quality:

- `35/C0`: `NCC 0.6951 -> 0.7297`, `SSIM 0.3558 -> 0.3619`, `Dice 0.8866 -> 0.9128`, `PSNR 13.3658 -> 13.8284`
- `41/C0`: `NCC 0.7176 -> 0.7344`, `SSIM 0.3816 -> 0.3848`, `Dice 0.7184 -> 0.7199`, `PSNR 13.8002 -> 14.0164`

Current quality is good enough for internal single-machine testing. The `ANTs + Laplacian` path is now the default high-quality route intended to close the gap with `Sample/Miki`.

Running native-XY ANTs is supported in code, but it exceeded available memory on this machine during local validation. Treat that mode as experimental unless you have more RAM.

- Current Brainfast high-quality path: ANTs + Laplacian refinement
- `Miki` reference path: ANTs + Laplacian refinement on a slightly wider half-template

Treat this workflow as the internal reproducible baseline, not the final best-quality release pipeline.

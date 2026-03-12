# Claude Review Bundle (pending consolidated review)

## Scope
Continue development while Claude rate-limited; batch review later.

## New/Updated in this batch
1. `scripts/align_ai.py`
   - Landmark-pair affine transform apply
   - Outputs `outputs/aligned_label_ai.tif`

2. `frontend/server.py`
   - Added `/api/align/apply`
   - Computes before/after alignment score
   - Returns transform metadata and scores

## Validation evidence
- `python -m py_compile scripts/align_ai.py frontend/server.py` -> pass
- Frontend server restarted: `python server.py` (active)

## What to ask Claude to focus on
- affine transform direction correctness (`src=atlas`, `dst=real`)
- score function suitability for label-vs-real comparison
- API error handling robustness for missing/invalid pair CSV

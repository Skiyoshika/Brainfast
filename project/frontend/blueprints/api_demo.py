"""api_demo.py — Demo/chart routes."""
from __future__ import annotations

import threading
from pathlib import Path

from flask import Blueprint, jsonify, send_from_directory

import project.frontend.server_context as ctx

bp = Blueprint("api_demo", __name__, url_prefix="/api")


@bp.get("/outputs/demo-best-slice")
def outputs_demo_best_slice():
    """Serve the pre-generated best-slice comparison image."""
    fp = ctx.OUTPUT_DIR / 'demo_best_slice.jpg'
    if not fp.exists():
        return jsonify({"ok": False, "error": "Best-slice image not generated yet"}), 404
    return send_from_directory(str(ctx.OUTPUT_DIR), 'demo_best_slice.jpg')


@bp.get("/outputs/demo-annotated-slice")
def outputs_demo_annotated_slice():
    """Serve the annotated single-slice with region labels."""
    fp = ctx.OUTPUT_DIR / 'demo_annotated_slice.jpg'
    if not fp.exists():
        return jsonify({"ok": False, "error": "Annotated slice not generated yet. Run refresh_demo.py first."}), 404
    return send_from_directory(str(ctx.OUTPUT_DIR), 'demo_annotated_slice.jpg')


@bp.get("/outputs/cell-chart")
def outputs_cell_chart():
    """Generate and serve the cell-count bar+pie chart."""
    import subprocess
    import sys as _sys

    chart_path = ctx.OUTPUT_DIR / 'cell_count_chart.png'
    hier_path  = ctx.OUTPUT_DIR / 'cell_counts_hierarchy.csv'
    if not hier_path.exists():
        return jsonify({"ok": False, "error": "No hierarchy CSV yet"}), 404
    # Regenerate if stale
    if not chart_path.exists() or hier_path.stat().st_mtime > chart_path.stat().st_mtime:
        try:
            script = ctx.PROJECT_ROOT / 'scripts' / 'make_demo_panel.py'
            # inline chart generation to avoid extra script file
            code = r"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, sys
hier = pd.read_csv(sys.argv[1])
d2 = hier[hier['depth']==2].sort_values('count',ascending=False)
d2 = d2[d2['count']>0]
colors=['#E57373','#FF9800','#FFEB3B','#66BB6A','#26C6DA','#5C6BC0','#AB47BC','#EC407A','#26A69A','#8D6E63','#78909C','#FFA726']
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6)); fig.patch.set_facecolor('#0e0e16')
for ax in [ax1,ax2]:
    ax.set_facecolor('#141420'); ax.tick_params(colors='#cccccc',labelsize=9)
    ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
bars=ax1.barh(range(len(d2)),d2['count'].values,color=[colors[i%len(colors)] for i in range(len(d2))],edgecolor='none',height=0.7)
ax1.set_yticks(range(len(d2))); ax1.set_yticklabels([f"{r['acronym']} ({r['region_name'][:20]})" for _,r in d2.iterrows()],fontsize=8.5,color='#cccccc')
ax1.set_xlabel('Cell Count',color='#cccccc',fontsize=10); ax1.set_title('Cell Counts by Major Brain Region',color='white',fontsize=12,pad=10)
[ax1.text(b.get_width()*1.01,b.get_y()+b.get_height()/2,f'{int(r.count):,}',va='center',ha='left',fontsize=8,color='#aaa') for b,r in zip(bars,d2.itertuples())]
total=d2['count'].sum(); pcts=d2['count'].values/total*100
labels=[f"{r['acronym']} {p:.1f}%" if p>3 else '' for (_,r),p in zip(d2.iterrows(),pcts)]
ax2.pie(d2['count'].values,colors=[colors[i%len(colors)] for i in range(len(d2))],labels=labels,labeldistance=1.1,textprops={'fontsize':8,'color':'#cccccc'},startangle=90,wedgeprops={'edgecolor':'#0e0e16','linewidth':1.5})
ax2.set_title('Proportion by Brain Region',color='white',fontsize=12,pad=10)
root=hier[hier['depth']==0]['count'].values[0]
fig.suptitle(f'Sample 35 — Whole Brain Cell Counts  |  Total: {int(root):,} cells  |  {len(hier)} regions mapped',color='white',fontsize=11,y=0.98)
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(sys.argv[2],dpi=120,bbox_inches='tight',facecolor='#0e0e16',edgecolor='none')
"""
            subprocess.run(
                [_sys.executable, '-c', code, str(hier_path), str(chart_path)],
                cwd=str(ctx.PROJECT_ROOT), timeout=60, check=True, capture_output=True
            )
        except Exception as e:
            return jsonify({"ok": False, "error": f"Chart generation failed: {e}"}), 500
    if not chart_path.exists():
        return jsonify({"ok": False, "error": "Chart not found"}), 404
    return send_from_directory(str(ctx.OUTPUT_DIR), 'cell_count_chart.png')


@bp.get("/outputs/demo-comparison/<int:slice_idx>")
def outputs_demo_comparison(slice_idx: int):
    """Generate and serve a side-by-side raw vs atlas comparison for a given slice index."""
    import sys as _sys
    import numpy as _np
    import tifffile as _tf

    reg_dir = ctx.OUTPUT_DIR / 'registered_slices'
    data_dir = ctx.PROJECT_ROOT / 'data' / '35_C0_demo'
    ov_path = reg_dir / f'slice_{slice_idx:04d}_overlay.png'
    lbl_path = reg_dir / f'slice_{slice_idx:04d}_registered_label.tif'
    if not ov_path.exists():
        return jsonify({"ok": False, "error": f"slice {slice_idx} not found"}), 404
    out_path = ctx.OUTPUT_DIR / f'compare_{slice_idx:04d}.jpg'
    try:
        from scripts.make_demo_panel import _vibrant_recolor, _crop_to_brain
        from PIL import Image as _Im, ImageDraw as _ID, ImageFont as _IF

        ov  = _np.array(_Im.open(str(ov_path)).convert('RGB'))
        lbl = _tf.imread(str(lbl_path)) if lbl_path.exists() else _np.zeros(ov.shape[:2], dtype=_np.int32)
        raw_files = sorted(data_dir.glob('*.tif'))
        raw_file = raw_files[min(slice_idx, len(raw_files)-1)] if raw_files else None

        if raw_file:
            raw_orig = _tf.imread(str(raw_file))
            p1, p99 = _np.percentile(raw_orig[raw_orig>0], [2,98]) if raw_orig.max()>0 else (0,1)
            raw_norm = _np.clip((raw_orig.astype(_np.float32)-p1)/(p99-p1+1e-6)*255,0,255).astype(_np.uint8)
            raw_rgb = _np.stack([raw_norm]*3, axis=-1)
        else:
            raw_rgb = _np.zeros_like(ov)

        vibrant  = _vibrant_recolor(ov, lbl)
        raw_crop = _crop_to_brain(raw_rgb, pad=25)
        vib_crop = _crop_to_brain(vibrant, pad=25)
        H = max(raw_crop.shape[0], vib_crop.shape[0])
        def _rh(arr, h):
            img = _Im.fromarray(arr)
            sc = h / img.height
            return _np.array(img.resize((int(img.width*sc), h), _Im.LANCZOS))
        raw_r = _rh(raw_crop, H)
        vib_r  = _rh(vib_crop, H)
        div = _np.full((H, 6, 3), 35, dtype=_np.uint8)
        combined = _np.concatenate([raw_r, div, vib_r], axis=1)
        W = combined.shape[1]
        header = _np.full((50, W, 3), 18, dtype=_np.uint8)
        body = _Im.fromarray(_np.concatenate([header, combined], axis=0))
        draw = _ID.Draw(body)
        try:
            font = _IF.truetype('C:/Windows/Fonts/arial.ttf', 22)
            sm   = _IF.truetype('C:/Windows/Fonts/arial.ttf', 14)
        except Exception:
            font = sm = _IF.load_default()
        draw.text((10, 14), 'Raw Lightsheet', fill=(200,200,200), font=font)
        draw.text((raw_r.shape[1]+14, 14), 'Brainfast — Atlas Registration', fill=(200,200,200), font=font)
        draw.text((W-260, 32), f'Slice {slice_idx} · Allen CCFv3', fill=(100,100,100), font=sm)
        body.save(str(out_path), quality=92)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return send_from_directory(str(ctx.OUTPUT_DIR), out_path.name)


@bp.get("/outputs/demo-panel")
def outputs_demo_panel():
    """Serve the demo panel image, generating it on-demand if needed."""
    import subprocess
    import sys

    panel_path = ctx.OUTPUT_DIR / 'demo_panel.jpg'
    reg_dir = ctx.OUTPUT_DIR / 'registered_slices'
    # Auto-regenerate if stale or missing
    if not panel_path.exists() or (
        reg_dir.exists() and
        any(p.stat().st_mtime > panel_path.stat().st_mtime for p in reg_dir.glob('slice_*_overlay.png'))
    ):
        try:
            script = ctx.PROJECT_ROOT / 'scripts' / 'make_demo_panel.py'
            subprocess.run(
                [sys.executable, str(script),
                 '--reg_dir', str(reg_dir),
                 '--out', str(panel_path),
                 '--n', '12', '--cols', '4', '--size', '380'],
                cwd=str(ctx.PROJECT_ROOT), timeout=120, check=True,
                capture_output=True
            )
        except Exception as e:
            return jsonify({"ok": False, "error": f"Panel generation failed: {e}"}), 500
    if not panel_path.exists():
        return jsonify({"ok": False, "error": "Panel not found"}), 404
    return send_from_directory(str(ctx.OUTPUT_DIR), 'demo_panel.jpg')


@bp.post("/outputs/refresh-demo")
def outputs_refresh_demo():
    """Run refresh_demo.py to regenerate all demo visuals (panel, annotated slice, chart)."""
    import subprocess
    import sys

    script = ctx.PROJECT_ROOT / 'scripts' / 'refresh_demo.py'
    if not script.exists():
        return jsonify({"ok": False, "error": "refresh_demo.py not found"}), 404

    def _run():
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(ctx.PROJECT_ROOT),
                timeout=180,
                capture_output=True,
                text=True,
            )
            ctx._append_log(f"[refresh_demo] {result.stdout.strip()}")
            if result.returncode != 0:
                ctx._append_log(f"[refresh_demo] ERROR: {result.stderr.strip()}")
        except Exception as e:
            ctx._append_log(f"[refresh_demo] exception: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "refresh_demo.py started in background"})


@bp.get("/outputs/reg-stats")
def outputs_reg_stats():
    """Return registration quality summary for display."""
    import csv

    qc_path = ctx.OUTPUT_DIR / 'slice_registration_qc.csv'
    if not qc_path.exists():
        return jsonify({"ok": False, "error": "No registration QC data yet"})
    try:
        rows = list(csv.DictReader(open(qc_path)))
        scores = [float(r['best_score']) for r in rows if r.get('best_score')]
        ok_count = sum(1 for r in rows if r.get('registration_ok', '').lower() == 'true')
        return jsonify({
            "ok": True,
            "total": len(rows),
            "ok_count": ok_count,
            "mean_score": round(sum(scores)/len(scores), 3) if scores else 0,
            "min_score": round(min(scores), 3) if scores else 0,
            "max_score": round(max(scores), 3) if scores else 0,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

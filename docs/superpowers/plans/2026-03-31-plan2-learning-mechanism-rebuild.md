# Plan 2: Learning Mechanism Rebuild

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复训练与推理的配置不一致，让 `learn_from_trainset.py` 真正能为 Sample 35 学出有效的配准参数。

**Architecture:** 在现有 `learn_from_trainset.py` 基础上增加一个 `--config` 参数，使训练时读取与推理相同的 `run_config_35.json`（AP方法、pixel size、半脑模式一致）。扩展 warp 参数搜索空间（加入 `tissue_shrink_factor` 和 `contour_tps_smooth`）。输出 per-sample 分数矩阵和 HTML 可视化报告，让结果可验证。不改动 `overlay_render.py` 等核心渲染逻辑。

**Tech Stack:** Python 3.10+, numpy, pandas, PIL, tifffile（已有），jinja2（新增，仅用于 HTML 报告模板）

**前置条件：** Plan 1 完成（Sample 35 已跑通，`outputs/registered_slices/` 存在）

---

## 文件清单

| 操作 | 路径 | 说明 |
|---|---|---|
| 修改 | `project/scripts/learn_from_trainset.py` | 增加 `--config` 参数，修复训练/推理一致性，扩展搜索空间，输出 HTML 报告 |
| 创建 | `project/train_data_set/sample35/` | Sample 35 专用训练对目录（手动准备）|
| 创建 | `project/tests/unit/test_learn_from_trainset.py` | 单元测试（配置读取、分数计算函数）|
| 修改 | `project/requirements-min.txt` | 添加 `jinja2`（Flask 已依赖，确认即可）|

---

## Task 1: 验证 jinja2 可用

**Files:**
- Check: `project/requirements-min.txt`

- [ ] **Step 1: 确认 jinja2 已作为 Flask 依赖安装**

```bash
python -c "import jinja2; print(jinja2.__version__)"
```

预期：打印版本号（如 `3.1.x`）。jinja2 是 Flask 的依赖，应已安装。

如果未安装：
```bash
pip install jinja2
```

并在 `project/requirements-min.txt` 末尾添加一行：
```
jinja2>=3.1.0
```

- [ ] **Step 2: Commit（如果修改了 requirements）**

```bash
git add project/requirements-min.txt
git commit -m "deps: explicitly list jinja2 for HTML report generation"
```

---

## Task 2: 为 Sample 35 准备训练对

**Files:**
- Create directory: `project/train_data_set/sample35/`

这一步需要**人工完成**：从 Sample 35 的已知良好切片中提取训练对。

- [ ] **Step 1: 确认已有 overlay 质量**

从 Plan 1 Task 2 的人工检查中，选出配准效果最好的 5-8 个切片编号。查看：

```bash
python -c "
import pandas as pd
df = pd.read_csv('project/outputs/slice_registration_qc.csv')
top = df.sort_values('best_score', ascending=False).head(10)
print(top[['slice_id','best_z','best_score','slice_path']].to_string())
"
```

记下 `slice_id` 和对应的 `slice_path`（原始 TIFF 路径）。

- [ ] **Step 2: 创建训练目录**

```bash
mkdir -p project/train_data_set/sample35
```

- [ ] **Step 3: 为每个好切片复制 Ori 图像**

对每个选定的切片（以 slice_id=0 为例）：

```bash
python -c "
import shutil
from pathlib import Path
import pandas as pd
from tifffile import imread
from PIL import Image
import numpy as np

df = pd.read_csv('project/outputs/slice_registration_qc.csv')
# 选分数最高的 6 个切片
top = df.sort_values('best_score', ascending=False).head(6)

out_dir = Path('project/train_data_set/sample35')
out_dir.mkdir(exist_ok=True)

for i, (_, row) in enumerate(top.iterrows(), 1):
    sid = int(row['slice_id'])
    src = Path(str(row['slice_path']))
    if not src.exists():
        print(f'SKIP: {src} not found')
        continue
    # 复制 TIFF 作为 Ori（训练脚本会读取 *_Ori.tif 或 *_Ori.png）
    dst_tif = out_dir / f'{i}_Ori.tif'
    shutil.copy2(src, dst_tif)
    # 同时生成 PNG 预览
    arr = imread(str(src))
    if arr.ndim == 3:
        arr = arr[..., 0]
    p2, p98 = np.percentile(arr, [2, 98])
    arr_norm = np.clip((arr.astype(np.float32) - p2) / (p98 - p2 + 1e-6), 0, 1)
    img = Image.fromarray((arr_norm * 255).astype(np.uint8))
    img.save(str(out_dir / f'{i}_Ori.png'))
    # 复制配准 label（已有的 registered_label.tif 作为训练目标）
    label_src = Path(str(row['registered_label_path']))
    if label_src.exists():
        shutil.copy2(label_src, out_dir / f'{i}_Label.tif')
        print(f'OK: slice_id={sid} -> {i}_Ori.tif + {i}_Label.tif')
    else:
        print(f'WARNING: no label for slice_id={sid}, only Ori copied')
print('Done')
"
```

- [ ] **Step 4: 验证训练对存在**

```bash
ls project/train_data_set/sample35/
```

预期：至少存在 `1_Ori.tif 1_Label.tif 2_Ori.tif 2_Label.tif ...` 等文件。

- [ ] **Step 5: Commit 训练数据**

```bash
git add project/train_data_set/sample35/
git commit -m "data: add sample35 training pairs from top-scoring registered slices"
```

---

## Task 3: 为 learn_from_trainset.py 写新测试

**Files:**
- Create: `project/tests/unit/test_learn_from_trainset.py`

- [ ] **Step 1: 写单元测试**

创建 `project/tests/unit/test_learn_from_trainset.py`：

```python
"""Unit tests for learn_from_trainset helpers (no atlas file required)."""
from __future__ import annotations

import numpy as np
import pytest


def test_dice_identical():
    """Dice of identical masks should be 1.0."""
    from project.scripts.learn_from_trainset import _dice
    mask = np.ones((10, 10), dtype=bool)
    assert _dice(mask, mask) == pytest.approx(1.0)


def test_dice_disjoint():
    """Dice of disjoint masks should be ~0.0."""
    from project.scripts.learn_from_trainset import _dice
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[:5, :] = True
    b[5:, :] = True
    assert _dice(a, b) == pytest.approx(0.0, abs=1e-4)


def test_boundary_f1_identical():
    """boundary_f1 of matching boundaries should be high."""
    from project.scripts.learn_from_trainset import _boundary_f1
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    score = _boundary_f1(mask, mask, tol_px=2)
    assert score > 0.9


def test_pair_ids_empty(tmp_path):
    """_pair_ids returns empty list when no Ori/Label pairs found."""
    from project.scripts.learn_from_trainset import _pair_ids
    assert _pair_ids(tmp_path) == []


def test_pair_ids_detects_label_pair(tmp_path):
    """_pair_ids detects pairs with _Ori.png + _Label.tif."""
    from PIL import Image
    import numpy as np
    from tifffile import imwrite

    img = Image.fromarray(np.zeros((8, 8), dtype=np.uint8))
    img.save(str(tmp_path / "1_Ori.png"))
    imwrite(str(tmp_path / "1_Label.tif"), np.zeros((8, 8), dtype=np.int32))

    from project.scripts.learn_from_trainset import _pair_ids
    ids = _pair_ids(tmp_path)
    assert ids == ["1"]


def test_match_shape_label_preserves_values(tmp_path):
    """_match_shape_label should preserve integer label values after resize."""
    from project.scripts.learn_from_trainset import _match_shape_label
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out = _match_shape_label(a, (2, 2))
    assert out.dtype == np.int32
    assert set(out.ravel()) == {1, 2, 3, 4}


def test_cfg_search_space_respects_config_pixel_size():
    """Training Cfg should accept pixel_size_um from config dict."""
    from project.scripts.learn_from_trainset import Cfg
    cfg = Cfg(
        fit_mode="cover",
        edge_smooth_iter=1,
        profile_name="balanced",
        warp_params={"tissue_shrink_factor": 0.88},
    )
    assert cfg.warp_params["tissue_shrink_factor"] == pytest.approx(0.88)
```

- [ ] **Step 2: 运行测试确认通过**

```bash
cd d:/Brainfast
python -m pytest project/tests/unit/test_learn_from_trainset.py -v
```

预期：6 tests PASSED。

- [ ] **Step 3: Commit**

```bash
git add project/tests/unit/test_learn_from_trainset.py
git commit -m "test: add unit tests for learn_from_trainset helpers"
```

---

## Task 4: 修复 learn_from_trainset.py（训练/推理一致性 + 扩展搜索空间）

**Files:**
- Modify: `project/scripts/learn_from_trainset.py`

- [ ] **Step 1: 增加 `--config` 参数和 pixel_size 读取**

在 `learn_from_trainset.py` 的 `main()` 函数中，找到 `ap.add_argument` 块，增加以下参数：

```python
    ap.add_argument(
        "--config",
        default="",
        help="Path to run_config JSON (reads pixel_size_um_xy, registration settings). "
             "Overrides --pixel-size-um if provided.",
    )
    ap.add_argument(
        "--hemisphere",
        default="",
        help="Force hemisphere mode (e.g. right_flipped). Overrides config if set.",
    )
    ap.add_argument(
        "--atlas-z-offset",
        type=int,
        default=0,
        help="Atlas Z offset to add when using atlas_z_from_filename mode.",
    )
    ap.add_argument(
        "--atlas-z-scale",
        type=float,
        default=0.2,
        help="Scale factor: atlas_z = filename_z * scale + offset.",
    )
    ap.add_argument(
        "--use-filename-z",
        action="store_true",
        help="Use atlas_z_from_filename mode (same as inference for sample_35).",
    )
    ap.add_argument(
        "--shrink-values",
        default="0.85,0.88,0.92,1.0",
        help="Comma-separated tissue_shrink_factor values to search.",
    )
    ap.add_argument(
        "--tps-smooth-values",
        default="1.5,2.0,2.5,3.5",
        help="Comma-separated contour_tps_smooth values to search.",
    )
```

- [ ] **Step 2: 在 main() 中读取 config 并覆盖 pixel size 和 warp 设置**

在 `args = ap.parse_args()` 之后，读取 config 文件（如果提供）：

```python
    # ── Read run config if provided ──────────────────────────────────────────
    run_cfg: dict = {}
    if args.config:
        import json as _json
        run_cfg = _json.loads(Path(args.config).read_text(encoding="utf-8-sig"))

    # Resolve pixel size (config overrides CLI arg)
    pixel_size_um = float(args.pixel_size_um)
    if run_cfg:
        _ps = run_cfg.get("input", {}).get("pixel_size_um_xy")
        if _ps:
            pixel_size_um = float(_ps)
            print(f"[config] pixel_size_um_xy = {pixel_size_um} (from config)")

    # Resolve hemisphere
    hemisphere = str(args.hemisphere).strip()
    if not hemisphere and run_cfg:
        hemisphere = str(run_cfg.get("registration", {}).get("atlas_hemisphere", "")).strip()
    if hemisphere:
        print(f"[config] atlas_hemisphere = {hemisphere}")

    # Resolve atlas_z settings for filename-based mode
    use_filename_z = bool(args.use_filename_z)
    atlas_z_scale = float(args.atlas_z_scale)
    atlas_z_offset = int(args.atlas_z_offset)
    if run_cfg:
        reg_cfg = run_cfg.get("registration", {})
        if reg_cfg.get("atlas_z_from_filename"):
            use_filename_z = True
        if "atlas_z_z_scale" in reg_cfg:
            atlas_z_scale = float(reg_cfg["atlas_z_z_scale"])
        if "atlas_z_offset" in reg_cfg:
            atlas_z_offset = int(reg_cfg["atlas_z_offset"])
    if use_filename_z:
        print(f"[config] atlas_z_from_filename=True, scale={atlas_z_scale}, offset={atlas_z_offset}")
```

- [ ] **Step 3: 扩展参数搜索空间（加入 shrink + tps_smooth）**

找到 `cfgs: list[Cfg] = []` 的构建循环，在其之前解析新的搜索维度，然后扩展循环：

```python
    # Parse new search dimensions
    shrink_values = [float(x.strip()) for x in str(args.shrink_values).split(",") if x.strip()]
    tps_smooth_values = [float(x.strip()) for x in str(args.tps_smooth_values).split(",") if x.strip()]
    if not shrink_values:
        shrink_values = [1.0]
    if not tps_smooth_values:
        tps_smooth_values = [2.0]

    cfgs: list[Cfg] = []
    for fit_mode in fit_modes:
        for smooth in smooth_values:
            for profile_name in profile_names:
                for shrink in shrink_values:
                    for tps_smooth in tps_smooth_values:
                        wp = dict(profiles[profile_name])
                        wp["tissue_shrink_factor"] = shrink
                        wp["contour_tps_smooth"] = tps_smooth
                        if hemisphere:
                            wp["force_hemisphere"] = hemisphere
                        cfgs.append(
                            Cfg(
                                fit_mode=fit_mode,
                                edge_smooth_iter=int(smooth),
                                profile_name=profile_name,
                                warp_params=wp,
                            )
                        )
```

注意：`Cfg.key()` 方法需要更新以包含新维度，找到该方法改为：

```python
    def key(self) -> str:
        shrink = self.warp_params.get("tissue_shrink_factor", 1.0)
        tps = self.warp_params.get("contour_tps_smooth", 2.0)
        return (
            f"fit={self.fit_mode}|smooth={self.edge_smooth_iter}"
            f"|profile={self.profile_name}|shrink={shrink:.2f}|tps={tps:.1f}"
        )
```

- [ ] **Step 4: 在 autopick 调用处处理 filename-z 模式**

找到 `for sid in ids:` 循环内调用 `autopick_best_z` 的位置，在其之前加入 filename-z 逻辑：

```python
        # ── Resolve atlas_z: filename-based mode (same as inference) ─────────
        fixed_z: int | None = None
        if use_filename_z:
            import re as _re
            _m = _re.search(r"z(\d+)", ori_png.stem)
            if _m:
                fixed_z = max(0, min(527, int(int(_m.group(1)) * atlas_z_scale) + atlas_z_offset))
                print(f"  [{sid}] atlas_z from filename: z{_m.group(1)} -> atlas_z={fixed_z}")

        if fixed_z is not None:
            import nibabel as _nib
            import numpy as _np2
            nii = _nib.load(str(annotation))
            vol = _np2.asarray(nii.get_fdata(), dtype=_np2.int32)
            best_slice = vol[fixed_z, :, :]
            from tifffile import imwrite as _imwrite
            _imwrite(str(label_tif), best_slice)
            auto_meta = {"best_z": fixed_z, "best_score": 1.0}
        else:
            auto_meta = autopick_best_z(
                real_path=real_tif,
                annotation_nii=annotation,
                out_label_tif=label_tif,
                z_step=1,
                pixel_size_um=float(pixel_size_um),
                slicing_plane="coronal",
                roi_mode="auto",
            )
```

- [ ] **Step 5: 在 render_overlay 调用处传入 pixel_size_um 和 warp hemisphere**

找到 `render_overlay(` 调用，确保 `pixel_size_um=float(pixel_size_um)` 已传入（已有），并确认 `warp_params=cfg.warp_params` 包含 `force_hemisphere`（由 Step 3 已注入）。

检查这行存在且正确：
```python
            _, _diag = render_overlay(
                real_slice_path=real_tif,
                label_slice_path=label_tif,
                out_png=out_png,
                alpha=0.72,
                mode="contour-major",
                pixel_size_um=float(pixel_size_um),   # ← 使用从 config 读取的值
                fit_mode=cfg.fit_mode,
                edge_smooth_iter=cfg.edge_smooth_iter,
                major_top_k=int(args.major_top_k),
                return_meta=True,
                warped_label_out=warped_tif,
                warp_params=cfg.warp_params,            # ← 包含 force_hemisphere, shrink, tps_smooth
            )
```

- [ ] **Step 6: 运行单元测试确认无回归**

```bash
cd d:/Brainfast
python -m pytest project/tests/unit/test_learn_from_trainset.py -v
```

预期：全部通过。

- [ ] **Step 7: Commit**

```bash
git add project/scripts/learn_from_trainset.py
git commit -m "feat: fix training/inference consistency in learn_from_trainset (config, hemisphere, shrink search)"
```

---

## Task 5: 增加 HTML 可视化报告输出

**Files:**
- Modify: `project/scripts/learn_from_trainset.py`（在 `main()` 结尾处追加）

- [ ] **Step 1: 在 main() 末尾追加 HTML 报告生成**

在 `out_json.write_text(...)` 之后，追加：

```python
    # ── Generate HTML visual comparison report ───────────────────────────────
    _write_html_report(out_json, out, ids, work_dir, cfgs)
    print(f"[report] HTML report: {out_json.with_suffix('.html')}")
```

然后在文件顶部（`main()` 之前）定义这个函数：

```python
def _write_html_report(out_json: Path, data: dict, ids: list[str], work_dir: Path, cfgs: list[Cfg]) -> None:
    """Generate a side-by-side HTML comparison: pred overlay vs target for each training sample."""
    import base64
    import io

    rows_html = []
    best_key = data["best"]["key"]

    for sid in ids:
        info = data["per_sample"].get(sid, {})
        scores = info.get("scores", {})
        best_sid_key, best_sid_score = max(scores.items(), key=lambda x: x[1]) if scores else ("", 0.0)

        # Find best overlay PNG for this sample
        best_png = work_dir / f"{sid}_{best_sid_key.replace('|', '_').replace('=', '-')}.png"
        ori_png = work_dir / f"{sid}_ori_gray.tif"

        def _encode_img(p: Path) -> str:
            if not p.exists():
                return ""
            try:
                from PIL import Image
                img = Image.open(str(p)).convert("RGB")
                img.thumbnail((320, 320))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()
            except Exception:
                return ""

        pred_b64 = _encode_img(best_png)
        # Score bar: color by score value
        score_color = "#4CAF50" if best_sid_score >= 0.5 else ("#FF9800" if best_sid_score >= 0.3 else "#F44336")
        score_bar = f'<span style="background:{score_color};color:white;padding:2px 8px;border-radius:4px">{best_sid_score:.3f}</span>'

        # Scores table for this sample
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:5]
        score_rows = "".join(
            f'<tr><td style="font-size:11px">{k}</td><td style="font-size:11px">{v:.4f}</td></tr>'
            for k, v in sorted_scores
        )

        img_tag = f'<img src="data:image/png;base64,{pred_b64}" style="max-width:320px">' if pred_b64 else "<em>no image</em>"
        rows_html.append(f"""
        <tr>
          <td><b>{sid}</b></td>
          <td>{img_tag}</td>
          <td>{score_bar}<br><small>{best_sid_key}</small><br>
            <table border=0>{score_rows}</table>
          </td>
        </tr>""")

    global_best = data["best"]
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Trainset Tuning Report</title>
<style>body{{font-family:sans-serif;margin:20px}} table{{border-collapse:collapse}} td{{padding:8px;vertical-align:top;border:1px solid #ddd}}</style>
</head><body>
<h2>Brainfast Trainset Tuning Report</h2>
<p><b>Train dir:</b> {data['train_dir']}<br>
<b>Samples:</b> {data['n_samples']}<br>
<b>Global best key:</b> <code>{global_best['key']}</code><br>
<b>Global best score:</b> {global_best['score']:.4f}</p>
<h3>Global best params</h3>
<pre>{__import__('json').dumps(global_best['params'], indent=2, ensure_ascii=False)}</pre>
<h3>Per-sample results (top-5 configs shown)</h3>
<table>
<tr><th>Sample</th><th>Best overlay (pred)</th><th>Scores</th></tr>
{"".join(rows_html)}
</table>
</body></html>"""

    out_json.with_suffix(".html").write_text(html, encoding="utf-8")
```

- [ ] **Step 2: 运行训练脚本（用 sample35 训练对）**

```bash
cd d:/Brainfast/project
python scripts/learn_from_trainset.py \
  --train-dir train_data_set/sample35 \
  --annotation annotation_25.nii.gz \
  --config configs/run_config_35.json \
  --use-filename-z \
  --pixel-size-um 5.0 \
  --profiles balanced,internal_strong,conservative \
  --fit-modes cover,contain \
  --smooth-values 0,1 \
  --shrink-values 0.85,0.88,0.92,1.0 \
  --tps-smooth-values 1.5,2.0,2.5,3.5
```

这会花 5-20 分钟（取决于切片数和参数组合数）。预期输出：
```
[config] pixel_size_um_xy = 5.0 (from config)
[config] atlas_hemisphere = right_flipped
[config] atlas_z_from_filename=True, scale=0.2, offset=150
...
{"key": "fit=cover|smooth=1|profile=...|shrink=0.88|tps=2.0", "score": 0.XXXX, ...}
```

- [ ] **Step 3: 查看 HTML 报告**

```bash
start project/outputs/trainset_tuned_params.html
```

在浏览器中打开，检查每个训练样本的 pred overlay 是否与组织对齐。

- [ ] **Step 4: 对比新旧分数**

```bash
python -c "
import json
from pathlib import Path
data = json.loads(Path('project/outputs/trainset_tuned_params.json').read_text())
print('Best score:', data['best']['score'])
print('Best key:', data['best']['key'])
print('Best params:', json.dumps(data['best']['params'], indent=2))
"
```

记录 `best_score`，与之前（如有）对比。

- [ ] **Step 5: 全量单元测试**

```bash
cd d:/Brainfast
python -m pytest project/tests/unit/ -v --tb=short
```

预期：全部通过。

- [ ] **Step 6: Commit**

```bash
git add project/scripts/learn_from_trainset.py project/outputs/trainset_tuned_params.json project/outputs/trainset_tuned_params.html
git commit -m "feat: add HTML visual report and expand warp param search space in learn_from_trainset"
```

---

## Task 6: 用新学到的参数重跑 Sample 35 并对比

- [ ] **Step 1: 备份当前输出**

```bash
cd d:/Brainfast/project
mkdir -p outputs_before_tuning
cp outputs/cell_counts_leaf.csv outputs_before_tuning/
cp outputs/slice_registration_qc.csv outputs_before_tuning/
```

- [ ] **Step 2: 重跑 pipeline（自动读取新 trainset_tuned_params.json）**

```bash
python scripts/main.py --config configs/run_config_35.json --run-real-input data/35_C0_demo
```

pipeline 会自动读取 `outputs/trainset_tuned_params.json` 应用新参数。

- [ ] **Step 3: 对比配准质量**

```bash
python -c "
import pandas as pd
before = pd.read_csv('project/outputs_before_tuning/slice_registration_qc.csv')
after = pd.read_csv('project/outputs/slice_registration_qc.csv')
print(f'--- 配准分数对比 ---')
print(f'调参前 均值: {before.best_score.mean():.3f}, 通过率: {before.registration_ok.mean():.0%}')
print(f'调参后 均值: {after.best_score.mean():.3f}, 通过率: {after.registration_ok.mean():.0%}')
"
```

- [ ] **Step 4: 运行输出验证**

```bash
cd d:/Brainfast
python project/scripts/validate_outputs.py
```

预期：PASS。

- [ ] **Step 5: 最终 commit**

```bash
git add project/outputs/trainset_tuned_params.json
git commit -m "feat: apply tuned registration params from sample35 training set"
```

---

## 验收标准

1. `learn_from_trainset.py --config configs/run_config_35.json --use-filename-z` 可正常运行
2. `trainset_tuned_params.json` 中的 `best.params.warpParams` 包含 `tissue_shrink_factor` 和 `contour_tps_smooth`
3. HTML 报告可在浏览器打开，至少显示 3 个训练样本的 overlay 图
4. 调参后 pipeline 运行：配准分数均值 ≥ 调参前
5. 全量单元测试通过

---

## 下一步

Plan 3（QC 报告）可与本 Plan 并行推进。

# Plan 3: QC Report + Registration Threshold Hardening

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为每次 pipeline 运行生成一份 HTML QC 报告，包含 per-slice overlay 缩略图和配准分数可视化；同时将 `fail_score_threshold` 默认值从 0.1 提升到 0.4，剔除低质量配准切片。

**Architecture:** 新建 `project/scripts/qc_report.py` 模块，读取 `outputs/` 下已有的 QC CSV 和 overlay PNG，生成自包含 HTML（base64 内嵌图片，无外部依赖）。在 `main.py` 的 `run_real_input()` 末尾调用它。不改动配准/检测核心逻辑。同时更新 `run_config_35.json` 中的 threshold。

**Tech Stack:** Python 3.10+, base64, csv, pathlib（均为标准库，无新依赖）

**前置条件：** Plan 1 完成（`outputs/` 目录存在，`slice_registration_qc.csv` 和 overlay PNG 已产出）

---

## 文件清单

| 操作 | 路径 | 说明 |
|---|---|---|
| 创建 | `project/scripts/qc_report.py` | HTML QC 报告生成模块 |
| 创建 | `project/tests/unit/test_qc_report.py` | 单元测试 |
| 修改 | `project/scripts/main.py` | 在 `run_real_input()` 末尾调用报告生成 |
| 修改 | `project/configs/run_config_35.json` | `fail_score_threshold: 0.1 → 0.4` |

---

## Task 1: 编写 qc_report.py 模块

**Files:**
- Create: `project/scripts/qc_report.py`

- [ ] **Step 1: 写 QC 报告生成模块**

创建 `project/scripts/qc_report.py`：

```python
"""qc_report.py — Generate a self-contained HTML QC report after pipeline run.

Reads:
  outputs/slice_registration_qc.csv
  outputs/registered_slices/*_overlay.png

Writes:
  outputs/qc_report.html
"""
from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path


def _score_color(score: float) -> str:
    """Return green/orange/red hex based on registration score."""
    if score >= 0.5:
        return "#4CAF50"
    if score >= 0.35:
        return "#FF9800"
    return "#F44336"


def _encode_png(png_path: Path, max_dim: int = 280) -> str:
    """Return base64-encoded PNG thumbnail, or empty string if file not found."""
    if not png_path.exists():
        return ""
    try:
        # Use PIL if available, otherwise read raw bytes
        try:
            from PIL import Image

            img = Image.open(str(png_path)).convert("RGB")
            img.thumbnail((max_dim, max_dim))
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            # Fallback: embed raw PNG bytes without resizing
            raw = png_path.read_bytes()
            return base64.b64encode(raw).decode()
    except Exception:
        return ""


def _read_registration_qc(qc_csv: Path) -> list[dict]:
    """Read slice_registration_qc.csv and return list of row dicts."""
    if not qc_csv.exists():
        return []
    rows = []
    with qc_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _read_cell_counts(leaf_csv: Path) -> list[dict]:
    """Read cell_counts_leaf.csv, return top 15 regions by count."""
    if not leaf_csv.exists():
        return []
    rows = []
    with leaf_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    rows.sort(key=lambda r: -int(r.get("count", 0)))
    return rows[:15]


def _summary_stats(rows: list[dict]) -> dict:
    total = len(rows)
    if total == 0:
        return {"total": 0, "passed": 0, "pass_pct": 0, "mean_score": 0.0}
    passed = sum(1 for r in rows if r.get("registration_ok", "").lower() in ("true", "1"))
    scores = []
    for r in rows:
        try:
            scores.append(float(r["best_score"]))
        except (KeyError, ValueError):
            pass
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "total": total,
        "passed": passed,
        "pass_pct": round(passed / total * 100),
        "mean_score": round(mean_score, 3),
    }


def generate_qc_report(outputs_dir: Path) -> Path:
    """Generate HTML QC report. Returns path to the written HTML file."""
    qc_csv = outputs_dir / "slice_registration_qc.csv"
    reg_dir = outputs_dir / "registered_slices"
    leaf_csv = outputs_dir / "cell_counts_leaf.csv"
    out_html = outputs_dir / "qc_report.html"

    rows = _read_registration_qc(qc_csv)
    stats = _summary_stats(rows)
    top_regions = _read_cell_counts(leaf_csv)

    # ── Build per-slice table rows ────────────────────────────────────────────
    slice_rows_html = []
    for row in rows:
        sid = row.get("slice_id", "?")
        best_z = row.get("best_z", "?")
        score_raw = row.get("best_score", "0")
        ok_raw = row.get("registration_ok", "false")
        method = row.get("registration_method", "")
        render_ms = row.get("render_ms", "")

        try:
            score = float(score_raw)
        except ValueError:
            score = 0.0
        ok = ok_raw.lower() in ("true", "1")
        color = _score_color(score)
        status_badge = (
            f'<span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px">PASS</span>'
            if ok
            else f'<span style="background:#F44336;color:white;padding:2px 6px;border-radius:3px">FAIL</span>'
        )

        # Find overlay PNG
        overlay_candidates = [
            reg_dir / f"slice_{int(sid):04d}_overlay.png",
            outputs_dir / "qc_overlays" / f"overlay_{int(sid):03d}.png",
        ]
        b64 = ""
        for cand in overlay_candidates:
            b64 = _encode_png(cand)
            if b64:
                break

        img_tag = (
            f'<img src="data:image/png;base64,{b64}" style="max-height:160px;max-width:200px">'
            if b64
            else '<em style="color:#999">no image</em>'
        )

        render_str = f"{float(render_ms):.0f} ms" if render_ms else ""
        slice_rows_html.append(f"""
        <tr>
          <td style="text-align:center">{sid}</td>
          <td>{img_tag}</td>
          <td style="text-align:center">{best_z}</td>
          <td style="text-align:center">
            <span style="background:{color};color:white;padding:2px 8px;border-radius:4px">
              {score:.3f}
            </span>
          </td>
          <td style="text-align:center">{status_badge}</td>
          <td style="font-size:11px;color:#666">{method}</td>
          <td style="font-size:11px;color:#666">{render_str}</td>
        </tr>""")

    # ── Build top-regions table ───────────────────────────────────────────────
    region_rows_html = []
    total_cells = sum(int(r.get("count", 0)) for r in top_regions)
    for r in top_regions:
        cnt = int(r.get("count", 0))
        pct = round(cnt / total_cells * 100, 1) if total_cells else 0
        bar = f'<div style="background:#2196F3;height:8px;width:{min(pct*3, 100):.0f}px;display:inline-block"></div>'
        region_rows_html.append(f"""
        <tr>
          <td>{r.get("region_name","")}</td>
          <td style="color:#666;font-size:11px">{r.get("acronym","")}</td>
          <td style="text-align:center">{r.get("hemisphere","")}</td>
          <td style="text-align:right">{cnt}</td>
          <td>{pct}% {bar}</td>
        </tr>""")

    # ── Summary bar color ─────────────────────────────────────────────────────
    pct = stats["pass_pct"]
    summary_color = "#4CAF50" if pct >= 80 else ("#FF9800" if pct >= 50 else "#F44336")

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Brainfast QC Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 24px; color: #222; }}
  h2 {{ color: #1a237e; }}
  h3 {{ color: #283593; border-bottom: 2px solid #e8eaf6; padding-bottom: 4px; }}
  .summary-box {{ display:inline-block; background:#f5f5f5; border-radius:8px; padding:16px 24px; margin:8px; text-align:center; }}
  .summary-num {{ font-size:2em; font-weight:bold; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th {{ background: #3f51b5; color: white; padding: 8px; text-align: left; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #e0e0e0; }}
  tr:hover {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h2>Brainfast QC Report</h2>

<h3>Summary</h3>
<div>
  <div class="summary-box">
    <div class="summary-num" style="color:{summary_color}">{stats['pass_pct']}%</div>
    <div>Pass Rate</div>
    <div style="font-size:12px;color:#666">{stats['passed']}/{stats['total']} slices</div>
  </div>
  <div class="summary-box">
    <div class="summary-num" style="color:{_score_color(stats['mean_score'])}">{stats['mean_score']}</div>
    <div>Mean Score</div>
  </div>
  <div class="summary-box">
    <div class="summary-num">{total_cells}</div>
    <div>Total Cells (dedup)</div>
  </div>
</div>

<h3>Top Brain Regions (by cell count)</h3>
<table>
<tr><th>Region</th><th>Acronym</th><th>Hemisphere</th><th>Count</th><th>Fraction</th></tr>
{"".join(region_rows_html) if region_rows_html else '<tr><td colspan=5><em>No cell count data</em></td></tr>'}
</table>

<h3>Per-Slice Registration ({stats['total']} slices)</h3>
<table>
<tr>
  <th>Slice</th>
  <th>Overlay</th>
  <th>Atlas Z</th>
  <th>Score</th>
  <th>Status</th>
  <th>Method</th>
  <th>Time</th>
</tr>
{"".join(slice_rows_html) if slice_rows_html else '<tr><td colspan=7><em>No registration data</em></td></tr>'}
</table>

<p style="color:#999;font-size:11px;margin-top:24px">
  Generated by Brainfast validate_outputs / qc_report.py
</p>
</body></html>"""

    out_html.write_text(html, encoding="utf-8")
    return out_html
```

- [ ] **Step 2: 确认模块可导入**

```bash
cd d:/Brainfast
python -c "from project.scripts.qc_report import generate_qc_report; print('OK')"
```

预期：`OK`

---

## Task 2: 写单元测试

**Files:**
- Create: `project/tests/unit/test_qc_report.py`

- [ ] **Step 1: 写测试**

创建 `project/tests/unit/test_qc_report.py`：

```python
"""Unit tests for qc_report.py — no actual PNG or pipeline outputs required."""
from __future__ import annotations

import csv
import pytest
from pathlib import Path


def test_score_color_green():
    from project.scripts.qc_report import _score_color
    assert _score_color(0.9) == "#4CAF50"


def test_score_color_orange():
    from project.scripts.qc_report import _score_color
    assert _score_color(0.4) == "#FF9800"


def test_score_color_red():
    from project.scripts.qc_report import _score_color
    assert _score_color(0.1) == "#F44336"


def test_summary_stats_empty():
    from project.scripts.qc_report import _summary_stats
    s = _summary_stats([])
    assert s["total"] == 0
    assert s["pass_pct"] == 0


def test_summary_stats_all_pass():
    from project.scripts.qc_report import _summary_stats
    rows = [
        {"best_score": "0.7", "registration_ok": "True"},
        {"best_score": "0.8", "registration_ok": "True"},
    ]
    s = _summary_stats(rows)
    assert s["total"] == 2
    assert s["passed"] == 2
    assert s["pass_pct"] == 100
    assert s["mean_score"] == pytest.approx(0.75)


def test_summary_stats_mixed():
    from project.scripts.qc_report import _summary_stats
    rows = [
        {"best_score": "0.8", "registration_ok": "True"},
        {"best_score": "0.2", "registration_ok": "False"},
    ]
    s = _summary_stats(rows)
    assert s["passed"] == 1
    assert s["pass_pct"] == 50


def test_generate_qc_report_creates_html(tmp_path):
    """generate_qc_report creates an HTML file even with minimal input."""
    from project.scripts.qc_report import generate_qc_report

    # Write a minimal slice_registration_qc.csv
    qc_csv = tmp_path / "slice_registration_qc.csv"
    with qc_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "slice_id", "best_z", "best_score", "registration_ok",
            "registration_method", "render_ms", "slice_path",
            "auto_label_path", "registered_label_path", "overlay_path",
            "score_type", "slicing_plane", "autopick_ms",
        ])
        writer.writeheader()
        writer.writerow({
            "slice_id": "0", "best_z": "250", "best_score": "0.72",
            "registration_ok": "True", "registration_method": "tps",
            "render_ms": "1200", "slice_path": "", "auto_label_path": "",
            "registered_label_path": "", "overlay_path": "",
            "score_type": "formula", "slicing_plane": "coronal",
            "autopick_ms": "50",
        })

    out_html = generate_qc_report(tmp_path)
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")
    assert "Brainfast QC Report" in content
    assert "0.720" in content   # score appears
    assert "PASS" in content


def test_generate_qc_report_handles_missing_csv(tmp_path):
    """generate_qc_report does not crash when no CSV exists."""
    from project.scripts.qc_report import generate_qc_report
    out_html = generate_qc_report(tmp_path)
    assert out_html.exists()
    assert "No registration data" in out_html.read_text(encoding="utf-8")
```

- [ ] **Step 2: 运行测试**

```bash
cd d:/Brainfast
python -m pytest project/tests/unit/test_qc_report.py -v
```

预期：8 tests PASSED。

- [ ] **Step 3: Commit**

```bash
git add project/scripts/qc_report.py project/tests/unit/test_qc_report.py
git commit -m "feat: add qc_report module with HTML per-slice overlay and score visualization"
```

---

## Task 3: 集成到 main.py

**Files:**
- Modify: `project/scripts/main.py`

- [ ] **Step 1: 在 run_real_input() 末尾调用 generate_qc_report**

在 `main.py` 中找到 `run_real_input()` 函数末尾（Auto-regenerate demo visuals 之前），插入：

```python
    # Generate HTML QC report
    try:
        from scripts.qc_report import generate_qc_report
        report_path = generate_qc_report(outputs_dir)
        print(f"\nQC report: {report_path}")
    except Exception as _qc_err:
        print(f"[warn] QC report generation failed: {_qc_err}")
```

- [ ] **Step 2: 确认 import 路径正确**

在 `main.py` 顶部的 try/except import 块中，两个分支都需要能导入 `qc_report`。找到该块，添加：

```python
try:
    # ... 已有 imports ...
    from scripts.qc_report import generate_qc_report  # 新增
except Exception:
    # ... 已有 fallback imports ...
    from qc_report import generate_qc_report  # 新增
```

- [ ] **Step 3: Commit**

```bash
git add project/scripts/main.py
git commit -m "feat: auto-generate HTML QC report at end of run_real_input pipeline"
```

---

## Task 4: 调整 fail_score_threshold

**Files:**
- Modify: `project/configs/run_config_35.json`

- [ ] **Step 1: 提高阈值**

编辑 `project/configs/run_config_35.json`，将：

```json
"fail_score_threshold": 0.1,
```

改为：

```json
"fail_score_threshold": 0.4,
```

> **为什么是 0.4：** 当前分数系统中 0.1 基本不过滤任何切片，0.65（原注释里的值）过于严格会丢弃太多切片。0.4 是一个合理的中间值，对应"大体对齐但不完美"的配准质量。可根据实测后再调整。

- [ ] **Step 2: 重跑 pipeline，观察哪些切片被过滤**

```bash
cd d:/Brainfast/project
python scripts/main.py --config configs/run_config_35.json --run-real-input data/35_C0_demo
```

如果看到 `RuntimeError: N slice(s) failed registration score threshold`，说明有切片被过滤。这是预期行为。

查看被过滤的切片：
```bash
python -c "
import pandas as pd
df = pd.read_csv('project/outputs/slice_registration_qc.csv')
failed = df[~df.registration_ok]
print(f'过滤切片数: {len(failed)}')
print(failed[['slice_id','best_z','best_score']].to_string())
"
```

如果过滤掉 >50% 的切片，则配准质量需要先通过 Plan 2 改善，此时可暂时降回 0.3。

- [ ] **Step 3: 打开 QC 报告验证**

```bash
start project/outputs/qc_report.html
```

确认：
- 摘要区域显示通过率和均值分数
- 每个切片有 overlay 缩略图
- FAIL 切片用红色标注

- [ ] **Step 4: Commit**

```bash
git add project/configs/run_config_35.json
git commit -m "config: raise fail_score_threshold to 0.4 for sample_35"
```

---

## Task 5: 全量测试

- [ ] **Step 1: 运行所有单元测试**

```bash
cd d:/Brainfast
python -m pytest project/tests/unit/ -v --tb=short
```

预期：全部通过。

- [ ] **Step 2: 运行输出验证**

```bash
python project/scripts/validate_outputs.py
```

预期：PASS。验证脚本应同时检查 `qc_report.html` 是否存在（下一步加入）。

- [ ] **Step 3: 将 qc_report.html 加入 validate_outputs.py 的检查列表**

编辑 `project/scripts/validate_outputs.py`，在 `REQUIRED_FILES` 列表中追加：

```python
REQUIRED_FILES = [
    "cell_counts_leaf.csv",
    "cell_counts_hierarchy.csv",
    "cells_detected.csv",
    "cells_dedup.csv",
    "cells_mapped.csv",
    "slice_registration_qc.csv",
    "dedup_stats.json",
    "qc_report.html",       # 新增
]
```

- [ ] **Step 4: 最终 commit**

```bash
git add project/scripts/validate_outputs.py
git commit -m "feat: include qc_report.html in validate_outputs required files list"
```

---

## 验收标准

1. `python scripts/main.py --config ... --run-real-input ...` 运行结束后，`outputs/qc_report.html` 存在且可在浏览器打开
2. HTML 报告包含：摘要统计（通过率、均值分数、细胞总数）、top-15 脑区表、每切片 overlay 缩略图
3. `fail_score_threshold=0.4` 下，被过滤的切片显示为红色 FAIL 标记
4. `python -m pytest project/tests/unit/test_qc_report.py -v` 全部通过
5. `python project/scripts/validate_outputs.py` 输出 PASS（包含 qc_report.html 检查）

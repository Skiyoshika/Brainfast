# 工作交接文档（Claude → Codex）

**时间：** 2026-04-01（更新于同日，完整版）  
**主题：** AAV Toolbox 论文复现 — Phase 1-5 全部完成，进入人工实测阶段  
**接手方：** Codex（后续修复实测中发现的问题）

---

## 本轮完成的工作总览

| Phase | 内容 | 状态 | 新增/修改文件 |
|-------|------|------|--------------|
| Phase 1 | sagittal / parasagittal 主流程打通 | ✅ | main.py, run_config.template.json |
| Phase 2 | reporter-positive 检测模式 | ✅ | detect.py, run_config.template.json |
| Phase 3 | 区域面积 + density + 代表切片 | ✅ | map_and_aggregate.py, main.py, paper_aav_summary.py（新） |
| Phase 4 | 共定位 specificity / sensitivity | ✅ | colocalization.py（新）, main.py |
| Phase 5 | 论文风格报告输出 | ✅ | export_paper_report.py（新）, main.py |

---

## Phase 1：sagittal 主流程打通

**改动：**

- [project/scripts/main.py](../project/scripts/main.py)：`run_real_input()` 从 `cfg["input"]["slicing_plane"]` 读取（默认 `"coronal"`），两处硬编码替换为变量
- [project/configs/run_config.template.json](../project/configs/run_config.template.json)：`input` 段新增 `"slicing_plane": "coronal"`

**用法：**
```json
"input": { "slicing_plane": "sagittal" }
```

**注意：** `atlas_autopick.py` 对未知 plane 会 raise ValueError。目前只支持 `"coronal"` 和 `"sagittal"`。

---

## Phase 2：reporter-positive 检测模式

**改动：**

- [project/scripts/detect.py](../project/scripts/detect.py)：新增 `detect_cells_reporter_positive()`；`detect_cells()` 入口检查 `detection.mode == "reporter_positive"`
- [project/configs/run_config.template.json](../project/configs/run_config.template.json)：新增 `mode`, `reporter_intensity_threshold_pct`, `reporter_min_area_px`, `reporter_max_area_px`

**检测逻辑：** robust percentile 归一化 → 阈值化（默认取 95 百分位）→ connected component → 面积过滤 → 质心

**用法：**
```json
"detection": {
  "mode": "reporter_positive",
  "reporter_intensity_threshold_pct": 95.0,
  "reporter_min_area_px": 8.0,
  "reporter_max_area_px": 2000.0
}
```

**不破坏现有流程：** `mode` 缺失或为 `"cellpose"` 时走原有路径。

---

## Phase 3：区域面积 + density + 代表切片

**改动：**

- [project/scripts/map_and_aggregate.py](../project/scripts/map_and_aggregate.py)：新增 `compute_region_areas_from_label_tif(label_tif, slice_id, pixel_size_um)` — 统计每个 region_id 的像素数和 mm² 面积
- [project/scripts/main.py](../project/scripts/main.py)：每轮循环后调用此函数，收集到 `region_area_rows`；循环结束后写 `outputs/region_areas.csv`；然后调用 `paper_aav_summary`
- [project/scripts/paper_aav_summary.py](../project/scripts/paper_aav_summary.py)（新）：`generate_paper_aav_summary(cells_mapped_csv, region_areas_csv, out_csv)` — 选代表切片（count 最高）+ 计算 density，输出 `paper_aav_region_summary.csv`

**输出文件：**
- `outputs/region_areas.csv` — 列：`slice_id, region_id, area_px, area_mm2`
- `outputs/paper_aav_region_summary.csv` — 列：`region_id, region_name, acronym, hemisphere, representative_slice_id, count, area_mm2, density_cells_per_mm2`

---

## Phase 4：共定位分析

**改动：**

- [project/scripts/colocalization.py](../project/scripts/colocalization.py)（新）：`run_colocalization(cells_mapped_csv, marker_tifs, out_dir, marker_intensity_threshold_pct)` — 逐细胞判断 marker+，输出 per-cell 和 per-region 统计
- [project/scripts/main.py](../project/scripts/main.py)：如果 `detection.marker_channel` 非空，提取 marker channel + merge，构建 `{sid: tif_path}` 映射，pipeline 结束后调用 `run_colocalization`
- [project/configs/run_config.template.json](../project/configs/run_config.template.json)：新增 `marker_channel: ""`, `marker_intensity_threshold_pct: 95.0`

**用法（以 dTom + PV 双标为例）：**
```json
"detection": {
  "marker_channel": "green",
  "marker_intensity_threshold_pct": 95.0
}
```

**输出文件：**
- `outputs/cells_colocalization.csv` — 列：`cell_id, slice_id, x, y, region_id, dtom_pos, marker_pos, double_pos`
- `outputs/colocalization_summary.csv` — 列：`region_id, ..., dtom_pos_count, marker_pos_count, double_pos_count, specificity, sensitivity`

**已知限制：**
- `sensitivity` 目前输出 NaN + 说明文字（`sensitivity_note` 列）。原因：sensitivity 需要区域内全部 marker+ 细胞数（包括非 dTom 细胞），当前 pipeline 只处理了 dTom 检测通道，marker+ total 无法从 cells_mapped.csv 推算。**要完整计算 sensitivity，需要对 marker channel 也跑一遍细胞检测，再与 dTom+ 做比对。**

---

## Phase 5：论文风格报告输出

**改动：**

- [project/scripts/export_paper_report.py](../project/scripts/export_paper_report.py)（新）：`generate_paper_report(outputs_dir)` — 读取 pipeline 输出，生成报告
- [project/scripts/main.py](../project/scripts/main.py)：pipeline 末尾调用 `generate_paper_report`

**生成内容（写入 `outputs/paper_report/`）：**
- `region_count_density.png` — 双轴柱状图（count + density/mm²），top-20 regions
- `specificity_summary.csv` + `specificity_summary.html` — colocalization 指标表
- `representative_panel.png` — 代表切片 overlay 缩略图拼图（top-12 regions）
- `paper_report_summary.txt` — 纯文字结果摘要

**单独运行：**
```bash
python project/scripts/export_paper_report.py --outputs-dir project/outputs
```

---

## 遗留问题 / 需要人工实测验证的点

### 🔴 必须验证（阻断复现能力）

1. **Sagittal atlas 坐标系**
   - `atlas_autopick.py` 中 sagittal 切片用 `vol[:, y, :]`（axis 1 = ML）
   - **需要验证** `atlas_mapper.py` 在 sagittal 模式下注册 label 的坐标是否与 Allen atlas sagittal orientation 对齐
   - 如果发现 LR 翻转 / DV 反转，需要在 `atlas_mapper.py` 里加坐标修正

2. **reporter_positive 阈值 95% 是初始值**
   - 在论文样本上 spot-check：抽几张切片用 `detect_cells_reporter_positive()` 检测，目视核对是否遗漏或误判
   - 根据结果调整 `reporter_intensity_threshold_pct` / `reporter_min_area_px` / `reporter_max_area_px`

3. **Sensitivity 计算不完整**
   - 目前 `colocalization_summary.csv` 里 sensitivity 全为 NaN
   - 要实现完整的 sensitivity，需要在 marker channel 上也跑细胞检测，拿到 marker+ total per region
   - 方案：在 `colocalization.py` 里新增 `detect_marker_positive_per_region()` 函数，类似 `detect_cells_reporter_positive`，但只统计 per-region count，不输出 per-cell 表

### 🟡 建议验证（影响数值精度）

4. **region_areas.csv 坐标系单位**
   - `compute_region_areas_from_label_tif` 用的是注册后 label TIF 的像素坐标
   - label TIF 的像素分辨率是 atlas 25µm / 还是与原图对齐后的 25µm？
   - 如果 label TIF 已经 warp 到与原图同分辨率，`pixel_size_um` 应该传入原图的 `px_um`（当前实现正确）
   - 如果 label TIF 是原始 atlas 分辨率（25µm/voxel），pixel_size_um 应传 25.0

5. **representative_panel.png 路径匹配逻辑**
   - `export_paper_report.py` 先在 `registered_slices/` 找 `slice_{sid:04d}_overlay.png`，再在 `qc_overlays/` 找 `overlay_{sid:03d}.png`
   - 实测确认 sid 编号是否与文件名一致

6. **marker channel merge 编号对齐**
   - `_merged_marker_files[sid]` 假设 merged marker 的顺序与 merged active channel 完全一致
   - 如果 `merge_every_n_slices` 对 marker 和 active channel 产生的文件数不同，会导致错位
   - 实测：检查 len(_merged_marker_files) == len(merged_files)

---

## 新文件清单

```
project/scripts/paper_aav_summary.py      — Phase 3: 代表切片 + density
project/scripts/colocalization.py         — Phase 4: 共定位分析
project/scripts/export_paper_report.py    — Phase 5: 报告输出
```

## 修改文件清单

```
project/scripts/main.py                   — Phase 1-5: 全部接入
project/scripts/detect.py                 — Phase 2: reporter_positive mode
project/scripts/map_and_aggregate.py      — Phase 3: compute_region_areas_from_label_tif
project/configs/run_config.template.json  — Phase 1-4: 新配置字段
```

---

## 人工实测建议顺序

1. 用现有 Sample 35 数据（coronal）先跑一遍，确认 Phase 3/5 新输出正确生成
2. 用一份 sagittal 样本确认 Phase 1 端到端打通
3. 用 `mode: reporter_positive` 配置跑 Phase 2，目视 spot-check
4. 用双通道样本（有 marker channel）验证 Phase 4，人工数几个细胞对比

---

*本交接文档由 Claude Sonnet 4.6 生成，时间 2026-04-01*

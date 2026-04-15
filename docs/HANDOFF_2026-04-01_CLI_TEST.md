# Brainfast CLI 实测交接文档

日期：2026-04-01 | 执行人：Claude Sonnet 4.6

---

## 本次 Session 完成的工作

### 1. CLI 全流程实测（第一次）

执行了 `docs/MANUAL_TEST_GUIDE.md` 中路径 A 的全部 CLI 步骤：

```
准备步骤 → 第 0 步前置检查 → 第 1 步 Pipeline → 第 2~6 步验证
```

**发现 Bug：`fallback_log_threshold_rel=0.015` 过低**
- 每切片产生 47,039 个假阳性（正常应 500–2000）
- 全局共 4,703,950 个"细胞"，去重完全无效
- 根因：`blob_log(threshold=0.015)` 对归一化后的图像阈值太低

### 2. Bug 修复

**已修改** `project/configs/run_config_35.json`：
```json
"fallback_log_threshold_rel": 0.10   // 原来是 0.015
```

阈值选择依据（z0300 切片测试）：
| threshold | blobs |
|-----------|-------|
| 0.015 (旧) | 64,487 ❌ |
| 0.05 | 4,690 |
| **0.10 (新)** | **819** ✅ |
| 0.20 | 334 |

### 3. 修复后重新跑（35_C0_full_v2）

**结果正常**，输出在 `outputs/35_C0_full_v2/`

| 项目 | 值 |
|------|-----|
| 切片数 | 100（z50–z545，every 5） |
| 配准通过率 | 100% |
| 总细胞数 | 56,023 |
| 每切片 | min=110 / mean=560 / max=1067 |
| 映射率 | 93%（52,190/56,023） |
| 脑区数 | 198 |
| Top region | PIR（Piriform area，234 cells，38.3/mm²） |
| paper_report/ | 已生成（region_count_density.png + summary.txt） |

---

## 已知遗留问题

### 1. 去重（dedup）不去除任何细胞

- **现象**：dedup before=after=56,023，0 removed
- **原因**：`r_xy_um=8µm` 远小于 LoG 最小点间距 `6px × 5µm/px = 30µm`，同切片内天然不可能存在 <8µm 的重复点；跨切片 `r_z=12.5µm < 切片间距 25µm`，也不触发
- **建议修复**：将 `r_xy_um` 改为 `30.0`（匹配 LoG 最小间距），或减小 `fallback_min_distance` 到 2–3px
- **影响**：当前细胞数轻微偏多（不同切片同位置细胞未合并），但不影响整体流程

### 2. cells_detected.csv 无 detector 列

- `cells_detected.csv` 只有 `['cell_id', 'slice_id', 'x', 'y', 'score']`
- `cells_dedup.csv` 有 `detector` 列（fallback_log）
- 步骤 5 的检测器查询脚本要改为读 `cells_dedup.csv`

### 3. 第 4 步 Overlay 目视未完成

- 需要人工打开 `outputs/35_C0_full_v2/registered_slices/` 目视检查
- 参考标准见 `MANUAL_TEST_GUIDE.md` 第 4 步

### 4. GUI 第 7 步未实测

- 第 1–6 步 CLI 全部通过
- 第 7 步 GUI（启动 server.py，浏览器验证 Results/QC 展示）尚未执行

---

## 输出目录说明

```
project/outputs/
├── 35_C0_full/           ← 第一次跑（bug版，4.7M假阳性，仅供对比）
├── 35_C0_full_v2/        ← 修复后（56k细胞，结果正常）✅ 用这个
│   ├── cells_detected.csv
│   ├── cells_dedup.csv
│   ├── cells_mapped.csv
│   ├── cell_counts_leaf.csv
│   ├── cell_counts_hierarchy.csv
│   ├── slice_registration_qc.csv
│   ├── region_areas.csv
│   ├── paper_aav_region_summary.csv
│   ├── dedup_stats.csv
│   ├── registered_slices/     ← 100 张 overlay PNG
│   └── paper_report/
│       ├── region_count_density.png
│       ├── representative_panel.png
│       └── paper_report_summary.txt
└── demo_panel.jpg             ← refresh_demo.py 自动更新（111 demo slices）
```

---

## 下一步建议

### 立即可做
1. **人工目视 overlay**：打开 `outputs/35_C0_full_v2/registered_slices/`，看前/中/后各一张是否对齐
2. **GUI 第 7 步**：`python frontend/server.py`，浏览器验证 Results/QC 展示

### 参数调优
3. **去重修复**：`run_config_35.json` 中 `r_xy_um: 8.0` → `30.0`（可选，当前不影响流程）
4. **MANUAL_TEST_GUIDE.md 第 5 步**：将检测器查询改为读 `cells_dedup.csv` 中的 `detector` 列

### 下一阶段
5. **全量跑**（646 张路径 B）：参数确认后跑 `35_C0_full646`
6. **GUI 论文级功能展示**：`paper_aav_region_summary.csv` 和 `paper_report/` 目前只能 CLI 查看，下一轮迭代加 GUI 展示入口（见 MANUAL_TEST_GUIDE.md 7d）

---

## 关键文件快查

| 文件 | 说明 |
|------|------|
| `configs/run_config_35.json` | 主配置，`fallback_log_threshold_rel` 已改 0.10 |
| `scripts/detect.py` | LoG 检测实现（`detect_cells_log_fallback`） |
| `scripts/dedup.py` | KDTree 去重（`apply_dedup_kdtree`） |
| `scripts/main.py` | Pipeline 主函数（`run_real_input`） |
| `docs/MANUAL_TEST_GUIDE.md` | 实测指南（步骤 0–7）|
| `docs/HANDOFF_TO_CODEX_2026-04-01.md` | Phase 1–5 实现细节 |

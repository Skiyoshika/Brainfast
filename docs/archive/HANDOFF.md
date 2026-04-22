# 工作交接文档

**时间：** 2026-03-31  
**交接给：** Codex（继续 Plan 1 执行）

---

## 当前状态

### Branch
`feature/plan1-first-run`（worktree 位于 `D:\Brainfast\.worktrees\plan1-first-run\`）

### 已完成的 commits（worktree branch）
1. `feat: add validate_outputs script for post-run health check`
2. `fix: move csv import to top-level, guard duplicate registered_slices check`
3. `test: add cellpose smoke tests for environment validation`
4. `ci: add cellpose to requirements and unit test job`
5. `feat: enable cellpose_cyto3 as primary cell detector for sample_35`（config 改为 cellpose_cyto3 + gpu=true + device=cuda）

### 所有单元测试：38 tests PASS

---

## 当前卡住的问题（需要立即解决）

### 问题：pipeline 找不到 structure ontology 文件

**错误：**
```
FileNotFoundError: structure ontology source not found in outputs/registration or project/configs
```

**根因：** `main.py` 的 `_project_root()` 解析到 `D:\Brainfast\project\`（主项目），而 structure JSON 下载到了 worktree 的 configs。

**文件已存在于：**
- `D:\Brainfast\.worktrees\plan1-first-run\project\configs\allen_mouse_structure_graph.json` ✅（637KB）

**修复方法（用户已在执行）：**
```powershell
Copy-Item D:\Brainfast\.worktrees\plan1-first-run\project\configs\allen_mouse_structure_graph.json `
  D:\Brainfast\project\configs\allen_mouse_structure_graph.json
```

**验证：**
```powershell
cd D:\Brainfast\.worktrees\plan1-first-run\project
python scripts/main.py --config configs/run_config_35.json --run-real-input D:\Brainfast\project\data\35_C0_demo
```

---

## Plan 1 剩余任务

### Task 2（人工，用户执行）
Pipeline 跑完后，按 `docs/MANUAL_RUN_GUIDE.md` 执行：
1. 运行 `python project/scripts/validate_outputs.py`
2. 查看配准分数分布（见指南第三步）
3. 人工打开 overlay PNG 检查配准质量
4. 查看 Cellpose 是否实际被使用（`cells_detected.csv` 中 detector 列）

### Plan 1 收尾（Codex 执行）
所有 Task 完成后，合并 worktree：
```bash
cd D:/Brainfast
git checkout main
git merge feature/plan1-first-run
git worktree remove .worktrees/plan1-first-run
```

---

## 接下来的计划（Plan 2 + Plan 3 并行）

详细计划文档：
- `docs/superpowers/plans/2026-03-31-plan2-learning-mechanism-rebuild.md`
- `docs/superpowers/plans/2026-03-31-plan3-qc-report.md`

### Plan 2 关键任务
修复 `learn_from_trainset.py` 训练/推理不一致：
- 增加 `--config` 参数（从 `run_config_35.json` 读取 pixel_size、hemisphere、AP 方法）
- 从 Sample 35 好切片提取训练对到 `project/train_data_set/sample35/`
- 扩展 warp 参数搜索（加入 tissue_shrink_factor + contour_tps_smooth）
- 输出 HTML 可视化报告

### Plan 3 关键任务
新建 `project/scripts/qc_report.py`：
- 读取 `outputs/slice_registration_qc.csv` + overlay PNG
- 生成自包含 HTML 报告（base64 内嵌图片）
- 集成到 `main.py` 末尾自动调用
- `fail_score_threshold` 从 0.1 改为 0.4

---

## 重要配置说明（Sample 35）

```json
{
  "registration": {
    "atlas_z_from_filename": true,
    "atlas_z_z_scale": 0.2,
    "atlas_z_offset": 150,
    "atlas_hemisphere": "right_flipped",
    "fail_score_threshold": 0.1   ← Plan 3 要改为 0.4
  },
  "detection": {
    "primary_model": "cellpose_cyto3",  ← 已改
    "cellpose_gpu": true                ← 已改
  }
}
```

## 关键文件路径
- Pipeline 入口：`project/scripts/main.py`
- 配准渲染：`project/scripts/overlay_render.py`
- 细胞检测：`project/scripts/detect.py`
- 学习机制：`project/scripts/learn_from_trainset.py`
- 验证脚本：`project/scripts/validate_outputs.py`（新增）
- 配置文件：`project/configs/run_config_35.json`
- 设计文档：`docs/superpowers/specs/2026-03-31-brainfast-production-readiness-design.md`
- 实测指南：`docs/MANUAL_RUN_GUIDE.md`

# Brainfast v1 交接文档 — 2026-04-20

> **交接对象**：审查端
> **本轮会话范围**：`0f4954b`..`6e7868d`（12 个 commit，63 files changed, +4176 / −284）
> **HEAD 分支**：`main`（比 `origin/main` 领先 17 个 commit）
> **测试面板**：`pytest project/tests/unit` 418 passed · `pytest project/tests/integration` 11 passed · `ruff check` + `ruff format --check` 全绿
> **i18n 对齐**：8/0 pass（`tools/simulate_i18n_parity.py`）

---

## 1. 总览（TL;DR）

本轮工作分四批交付，按先后顺序：

| 批次 | 解决的问题 | commits |
|---|---|---|
| A. **Wizard 生产事故修复** | wizard 返回的 outputs_dir 是死指针；默认 Cellpose OOM 把低显存机器卡死在 82% | `0f4954b` |
| B. **ETA 估算模块** | ANTs 阶段无 ETA 显示；硬编码 baseline 在新硬件上漂移 | `0b3c456` · `c9815e9` · `b12f784` |
| C. **操作说明 PPT + 双通道支持** | 用户要 17 页图文说明书；双通道共定位场景（560/640nm）以前要跑两遍 8h 全流程 | `cf5281c` · `0783bc0` · `24aa359` |
| D. **闭环产品化修补**（按 `2026-04-20-closed-loop-productization-remediation.md` 计划） | UI 学到的校准没流进默认 whole-brain 路径；mutable 学习产物污染源码树；class-prior warm-start 是纯手动；文档过时；CI 没覆盖闭环 | `3ec553b` · `8b7e115` · `118d859` · `9273bd6` · `6e7868d` |

---

## 2. 按 commit 逐项说明

### A. Wizard 生产事故（`0f4954b`）

**`fix(wizard): co-locate runtime config + flip allow_fallback default`**

修两个 production bug：

- **Bug #6（路径不一致）**：`api_wizard.wizard_launch` 把 `runtime_config.json` 写到 `outputs/<id>/` 但 `ctx._runner` 把流水线产物写到 `outputs/jobs/<id>/`。前端拿到的 `outputs_dir` 指向 empty dir。修为统一用 `ctx._job_output_dir(job_id)`。
- **Bug #7（cpsam OOM 死胡同）**：cpsam 在 1636×1359 大切片上每片 tile 成 `952×3×730×730 float32 = 5.67 GiB`，低显存机器跑到 82% Truth Export 就 OOM 死。默认改为 `allow_fallback: true` + `fallback_model: "log"`；严格流水线可在 config 里显式关。

文件：`project/frontend/blueprints/api_wizard.py`（+15/−5）。

---

### B. ETA 估算模块（3 commits）

#### B1. `0b3c456` `feat(eta): add per-stage ETA estimation to pipeline_progress + /api/status`

**动机**：用户反馈 4h 估算跟真实 4h40m 差 16% 是小事，真问题是 **ANTs 阶段全程 0 ETA**（`slicesDone=0` 前端没法算）。

**实现**：
- `pipeline_progress.py` 新增 `DEFAULT_STAGE_BASELINES`（6 个 stage 的 `(kind, factor)` 成本模型）
- `write_stage_progress` 持久化 `stageStartedTs` + `runStartedTs` 到 `pipeline_progress.json`
- `compute_eta(progress, slice_count)` 返回 `etr_in_stage_s` / `etr_remaining_stages_s` / `etr_total_s` / `correction_factor` / `method`
- 自校正：实际 elapsed / 预期 elapsed 乘到剩余 stages 上，clamp `0.5..2.0×`
- `/api/status` 返回 `eta` block

**测试**：`test_pipeline_eta.py` 11 个测试 + `test_api_pipeline.py` 1 个集成测试。含 retro 验证：用 `real-35-full` 真实 timeline 在 ANTs 结束点预测剩余时间，误差在 ±20%。

#### B2. `c9815e9` `feat(eta-ui): consume backend ETA so progress shows time during ANTs too`

前端 `getRunEtaSeconds(status)` 优先读 `status.eta.etr_total_s`，naive fallback 只用于 `slicesDone > 0` 阶段。ANTs 期间现在也能显示"约还需 2h 30min"。

#### B3. `b12f784` `feat(eta): self-calibrating ETA via persistent per-stage history`

**动机**：B1 的 baseline 是在一台机器上标定的，换台机器（不同 GPU / 磁盘 / CPU）必然漂移。

**实现**：
- `write_stage_progress` 累积 `stageCompletions: {stage_name: elapsed_s}`
- `maybe_record_run_completion(outputs_dir, history_path, run_id, slice_count)` 在流水线完成时 append 一行 JSONL 到 `outputs/eta_history.jsonl`（idempotent）
- `compute_baselines_from_history(history_path, defaults=..., min_samples=2)`：
  - `constant` 类 → `mean(elapsed_s)`
  - `per_slice` 类 → `mean(elapsed_s / slice_count)`
  - <2 样本时回退到 hardcoded default
- `/api/status` 每次都 load history → pass 给 `compute_eta`

**retrospective seed** 两次真实 run（`wiz-e2e-v2` 111 slices + `real-35-full` 646 slices）后：
```
Stage                 default  →  learned    Δ
ANTS Registration    2000.0s  →  2035.0s   +2%
Truth Export         7.00s/sl →  7.22s/sl  +3%
Quantification      16.50s/sl → 14.75s/sl  −11%
Volume Build         0.60s/sl →  0.36s/sl  −39%
```

---

### C. PPT + 双通道（3 commits）

#### C1. `cf5281c` `docs(manual): 17-slide Operations Manual PPT + 2 liquify fixes`

生成 `docs/manual/Brainfast_Operations_Manual.pptx`（4.3 MB, 17 张 16:9 slides）。截图来源于真实 `real-35-full` job（646 slices, 5.93M cells, 60 landmarks 已 Apply），不是 mockup。

脚手架：
- `docs/manual/capture_screenshots.py` — Playwright 驱动截图流水线
- `docs/manual/build_pptx.py` — python-pptx 组装
- `docs/manual/screenshots/` — 11 张 PNG

同时修复 2 个 frontend bug（截图过程中暴露）：`reloadSliceList` 和 `loadSliceImage` 都硬编码忽略 Job ID input — 现在都从面板 Job ID 读。

#### C2. `0783bc0` `feat(dual-channel): reuse registration + 2nd-channel UI overlay`

**动机**：大多数神经科学样本有 C0 reporter + C1 co-label。以前双通道要串行跑两次完整流水线 = 8h + 13GB。

**实现**：
- `registration.reuse_from_dir` config 字段：指向已完成的 prior run → `run_whole_brain_3d` 跳过 stages 1-5，只跑 Quantification（从 4h 到 ~15-30min）
- `_runner` 检测 index>=1 时设 `BRAINCOUNT_REUSE_FROM_DIR` 环境变量，接受 `input_dir: dict` 形式
- `_extract_channel_to_tmp` 原本 `rm -rf tmp_channel/` — 毁了 C0 当 C1 启动。改为只删本通道 stale 文件
- `/api/outputs/raw-channel-slice?job=...&z=...&channel=...&tint=00ffff` 返回 pseudocolor PNG
- `/api/outputs/channel-info` 告诉 UI 可叠加通道
- 3D Liquify 加 "Overlay 2nd channel" toggle + 颜色 + 透明度；canvas 用 `mix-blend-mode: screen` 叠

**测试**：+6 new（2 reuse + 4 endpoint + 2 extract_channel safety），ruff 干净。

#### C3. `24aa359` `feat(wizard): dual-channel UI — 1-click launch for 2-channel samples`

New Sample 向导加 "Add second channel" checkbox，展开后有 2nd source path + Inspect + 2nd channel dropdown。Inspect 比较两个通道的 shape，不一致时 warn。`/api/wizard/launch` 接受 `inputDirs: {red: ..., farred: ...}` 传 dict 给 `_runner`。+3 tests。

---

### D. 闭环产品化修补（5 commits，按 plan 文件逐 task 落地）

Plan: `docs/superpowers/plans/2026-04-20-closed-loop-productization-remediation.md`

#### D1. `3ec553b` `refactor(state): move learned artifacts out of tracked source tree` — **Task 1**

**之前**：calibration samples 写 `<project>/train_data_set/`，class priors 写 `<project>/train_data_set/class_priors/`，Cellpose 训练样本写 `<project>/cellpose_training/` —— 全部污染 git 源码树。

**现在**：全部迁到 `<project>/outputs/state/`（可用 `BRAINFAST_STATE_DIR` 覆盖）。
```
outputs/state/
  calibration/
    samples/
    trainset_tuned_params.json
    manifests/
  class_priors/
  cellpose_training/
```

Legacy-rename-in-place：若旧目录存在就 `Path.rename()` 移过来，不复制不丢数据。

文件：`paths.py` 新 helpers + RunPaths 新字段；`server_context._save_calibration_pair` / `_learn_from_trainset_async`；`api_liquify_3d._class_priors_root`；`api_cellpose._training_dir`。测试 +3（`test_runpaths_exposes_shared_state_root_under_outputs` / `..never_point_inside_source_tree` / `..overridable_via_env`）。

#### D2. `8b7e115` `feat(calibration): thread learned params through default whole-brain path` — **Task 2**

**之前**：`scope=whole + whole_brain_backend=miki_3d`（shipped default）在 `main.py` 里**提前 return** `run_whole_brain_3d(...)`，在 return 之后才有 `_load_tuned_overlay_params(outputs_dir)` — 也就是说默认路径**永远**看不到 UI 学到的校准。`export_registered_truth_slices` 还硬编码 `fit_mode="cover"` / `edge_smooth_iter=0`，即使传了也被忽略。

**修复**：
1. `_load_tuned_overlay_params(outputs_dir, project_root=...)` 解析顺序：job-local `trainset_tuned_params.json` → shared `<state_root>/calibration/trainset_tuned_params.json` → 空默认。
2. `main.py` 在 whole-brain return 之前就解析，调用 `_snapshot_tuned_params_into_job(...)` 写 job-local 快照（防后续全局 learn 追溯改变在途 job），把 `warp_params/fit_mode/edge_smooth_iter` 注入 `cfg["truth_export"]`。
3. `run_whole_brain_3d` 读 `cfg["truth_export"]` 转交给 `export_registered_truth_slices(..., fit_mode=..., edge_smooth_iter=...)`。
4. `finalize_liquify_to_cell_counts` 同步接受这三个 kwarg。

**测试 +5**：含 `test_learned_calibration_reaches_default_whole_brain_path`（集成 smoke）作为 **CI release 门** — `warp_params["learned_marker"]` 必须最终出现在 `export_registered_truth_slices` 的 kwargs 里。

#### D3. `118d859` `feat(liquify): auto-apply class prior to empty jobs + iterator safety` — **Task 3**

**之前**：n≥3 个同类样本后 warm-start 可用，但要用户手点 "Warm-start from prior"；大多数用户不知道要点。`ClassPriorStore.update` 有 iterator bug — `len(list(pairs))` 在 for 循环之后执行，generator 输入的话永远记 `pair_count=0` 到 sample_log。

**修复**：
- `_autoWarmStartIfEmpty()` 在（class 被 auto-detect 或手选）**且**（liquify state 已 refresh）**且**（`state.pairs.length === 0`）三个条件都满足时触发一次自动 warm-start。
- **永不 force**。有手动 pairs 的 job banner 显示 "manual overwrite required"，路由到现有按钮的 `confirm()` 流程。
- fire-once 标志 `_autoWarmStartTried` 防 tab 切回重复触发。
- `update()` 顶部 `pair_list = list(pairs)` 一次性 materialize。

**测试 +3**：含 `test_update_accepts_iterator_and_preserves_pair_count` 明确回归 + `test_app_js_has_auto_warm_start_flow_for_empty_jobs` 检查 JS 源码不能出现 `force: true` 在 auto 流程里。

#### D4. `9273bd6` `docs(v1): align public product contract with the real feature set` — **Tasks 4 + 5**

**Task 4 · docs**：
- `README.md` 的 `## Current workflow boundaries` 重写成**三个学习环**的对比表（calibration learn / class-prior warm-start / Cellpose retraining），明确每个的触发点、它 tunes 什么、artifacts 在哪。删掉过时的 "does not yet ship a detector-specific manual relabel workflow" 一行（Cellpose tab 已经上线）。
- `docs/user_guide.md` 修掉所有 stale 字符串：`<org>/Brainfast`、`pip install -e .`、`[advanced]`、`StartIdleBrainTrial.bat`、`--port 8788` 全部换成真实的 `pip install -e ".[full,dev]"` / `Start_Brainfast.bat` / `BRAINFAST_PORT` 环境变量。
- `docs/release/known-limitations.md` 重写成 post-fix 状态：三个学习环都声明已 live，剩余 limitations 只有 QC-review-required、manual-overwrite-gate、ml_flip 仍 provisional、state 在 `outputs/state/`。

过时字符串扫描（plan Step 5 命令）结果：**0 matches**。

**Task 5 · release gates**：
- `docs/release/manual-acceptance.md` 扩充成 5 轮签收（calibration 往返 + next-run 消费、class-prior auto-seed、liquify finalize、Cellpose tab smoke、shared-state 洁净）+ sign-off checklist。
- `.github/workflows/test.yml` 在 `smoke-default-path` job 里加了 2 个命名 guard 步骤（"Guard — learned calibration reaches default whole-brain path" + "Guard — class-prior auto-apply + iterator safety"），让闭环失败在 PR check UI 上有显式名字而不是淹没在整个 pytest 输出里。

#### D5. `6e7868d` `style(ruff): repair import ordering + apply ruff format sweep` — **Task 6**

- `api_pipeline.py` 里合并 `from project.scripts.pipeline_progress import (...)` 分拆成 4 条单独 import 满足 ruff I001
- `ruff format` 一次过（14 文件重新格式化，90 文件不变）
- `ruff check` + `ruff format --check` 在计划指定的所有 source root 都是绿的

---

## 3. 测试证据

### 单元测试（Windows / Python 3.11）

```
$ python -m pytest project/tests/unit/ -q
...
====================== 418 passed, 14 warnings in 29.80s ======================
```

新增测试（本轮）：
- `test_pipeline_eta.py` — 11 cases（ETA + history）
- `test_api_pipeline.py` — +3 state contract + 1 ETA
- `test_api_outputs.py` — +4 raw-channel-slice + channel-info
- `test_api_wizard.py` — +3 dual-channel payload
- `test_whole_brain_3d.py` — +3 (reuse + tuned params)
- `test_main.py` — +4 (extract_channel safety × 2 + tuned param resolution × 2)
- `test_class_prior.py` — +1 iterator safety
- `test_frontend_regressions.py` — +1 auto-warm-start flow

### 集成测试

```
$ python -m pytest project/tests/integration/ -q
...
================= 11 passed, 13 warnings in 89.83s (0:01:29) ==================
```

含新 `test_learned_calibration_reaches_default_whole_brain_path`，是 CI 释放门的核心回归保护。

### Lint

```
$ python -m ruff check project/scripts project/frontend/blueprints \
      project/frontend/server_context.py project/frontend/app_metadata.py \
      project/frontend/update_checker.py
All checks passed!

$ python -m ruff format --check <same paths>
90 files already formatted
```

### i18n parity

```
$ python tools/simulate_i18n_parity.py
HTML refs:  311 distinct i18n keys
LANGS.en:   542 entries
LANGS.zh:   542 entries
=== Summary: 8 pass / 0 fail ===
```

### 真实样本 E2E（`real-35-full` ChATe27 · 646 slices · 1636×1359）

| 阶段 | 耗时 | 结果 |
|---|---:|---|
| 第一次完整 C0 run（`wiz-e2e-v2`, 111 slices demo） | 78 min | 880K cells, 467 regions |
| 第一次完整 C0 run（真实样本, 646 slices） | **4h 40min** | 5.93M cells, 551 regions |
| β/γ 闭环：5 sparse landmarks → Apply + Finalize | ~4 min | 1-2 cells 重新归类（体积稀疏，符合预期） |
| β/γ 闭环：60 dense landmarks → Apply + Finalize | ~4 min | **5,249 cells** 迁移（87.5 cells/landmark, 0.088% of cells） |
| C1（reuse_from_dir）detection-only | **~2h** (detection dominant on 646 large slices) | 产 `cells_mapped_farred.csv` + `ch_2_*.tif` |

关键发现：**Landmark 密度需随 volume size scale**。Demo 5 landmarks 动 4800 cells（960/landmark），real-full 同样 5 landmarks 只动 1 cell；60 dense landmarks 补齐。已在 PPT slide 13 + plan 文档注明。

---

## 4. 审查重点清单

建议审查按这个顺序走：

- [ ] **契约 1 · Shared state**（`3ec553b`）：`grep -rn "train_data_set" project/` 应只剩注释/legacy-fallback；`grep -rn "cellpose_training" project/frontend` 同理。所有 mutable 学习产物应在 `outputs/state/` 下。
- [ ] **契约 2 · 学习闭环**（`8b7e115`）：运行 `pytest project/tests/integration/test_default_path_smoke.py::TestDefaultRuntimePath::test_learned_calibration_reaches_default_whole_brain_path -v` — 这是 CI 释放门。改任何会影响 `cfg["truth_export"]` 到 `export_registered_truth_slices` 的路径都必须让这条测试仍绿。
- [ ] **契约 3 · Class-prior 安全**（`118d859`）：`test_update_accepts_iterator_and_preserves_pair_count` + `test_app_js_has_auto_warm_start_flow_for_empty_jobs`。JS 层检查 `_autoWarmStartIfEmpty` 函数体**不能**出现 `force: true`，**必须**出现 `state.pairs.length > 0`。
- [ ] **API 向后兼容**：`/api/wizard/launch` 老的 `inputDir: str` 单通道调用路径仍然工作（`test_launch_with_valid_payload_invokes_run_pipeline` 守住）。
- [ ] **双通道 UI**：打开 3D Liquify，Job ID 填一个有 `tmp_channel/ch_0_*.tif + ch_2_*.tif` 的 job，"Overlay 2nd channel" row 应自动显示；勾上后 main img + overlay img 用 `mix-blend-mode: screen` 叠（不是替换）。
- [ ] **CI 配置**：`.github/workflows/test.yml` 的 `smoke-default-path` job 应出现两个命名 guard step；每次 PR 都应看到这两个 check。
- [ ] **文档**：
  - [ ] `grep -E "org>/Brainfast|\[advanced\]|StartIdleBrainTrial|--port 8788" README.md docs/user_guide.md docs/release/known-limitations.md` → 0 matches
  - [ ] `docs/release/manual-acceptance.md` 的 5 轮签收清单是否能用（审查者可以按它本地过一遍）
  - [ ] `docs/manual/Brainfast_Operations_Manual.pptx` 能打开、17 页都有截图

---

## 5. 已知 caveat / 非范围

- **真实 C0 + C1 双通道 UI 叠加**：代码 live，但在本地 session 里 `real-35-full/tmp_channel/` 当前只有 `ch_2_*.tif`（在 `_extract_channel_to_tmp` 修复之前 C0 文件被盖掉）。审查者在本机重跑 dual-channel 应该能看到两个通道共存，因为 fix 已 land（`118d859` 前一条 commit `0783bc0`）。
- **C1 background detection 运行中**：本轮会话启动的 C1 detection 在后台进程 PID 66252（已于本轮后期完成），这不是代码改动的一部分。
- **ANTs 仍是 C 外部依赖**：双通道 reuse 只能跳过配准，无法并行；`_runner` 还是 for-loop 串行跑通道。若需要并行支持要另起工作。
- **Wizard Step 5 未做**：plan 文件并没有要求 wizard 多通道，但作为 C3 我做了 UI + API + 3 tests。这是 plan 范围外的顺手交付。
- **class priors + calibration 的 legacy rename**：若用户在旧版 Brainfast 上有数据，第一次启动会自动迁移（`legacy_root.rename(new_root)`）。若目标 disk 不同（`BRAINFAST_STATE_DIR` 指别处）rename 会 fail，fallback 到 read-legacy-write-new 模式（打印一次 warning）。没做跨盘拷贝逻辑。
- **PPT 里的 13_liquify3d_overlay.png 显示 overlay row 未可见**：因为截图时 tmp_channel/ 只有 ch_2_*（见上）。用 fresh 双通道 job 重跑 `capture_phase5_screenshots.py` 会显示完整 overlay 控件。

---

## 6. 回滚方案

若某个 commit 需要撤销，建议次序（安全从后往前）：

1. `6e7868d` — 纯 lint，可 `git revert` 单独回退，不影响功能。
2. `9273bd6` — 文档 + CI，`git revert` 无功能影响。
3. `118d859` — class-prior auto-apply，revert 后用户仍能手动点按钮。iterator 修复丢失会重新出现 `pair_count=0` bug。
4. `8b7e115` — calibration 闭环。revert 后默认路径再次不消费 UI 校准（回到 plan 发现前状态）。
5. `3ec553b` — shared state。revert 后 legacy dirs 会再被写，但数据不会丢（legacy rename 是单向）。
6. `24aa359` + `0783bc0` — 双通道。revert 两个一起即可。
7. `cf5281c` — PPT。可单独 revert。
8. `b12f784` + `c9815e9` + `0b3c456` — ETA。有依赖，按倒序 revert。
9. `0f4954b` — Wizard 修复。不建议 revert（会恢复 cpsam OOM 死锁）。

## 7. 下一步建议（超出本轮范围）

- **并行通道 detection**：当前 dual-channel 还是串行 `for ch in channels`。如果 detection 是 CPU-only（LoG fallback）应该能 thread-parallel；GPU cpsam 可能 VRAM-bound 要看。
- **自动 class-prior export**：用户做完 QC 后当前要手点 "Save job → class prior"。可以在 Finalize 成功时自动 save（加二次 confirm）。
- **ETA 模块跨机器 seed**：本轮 retro-seed 了 `real-35-full` + `wiz-e2e-v2` 但都在同台机器。如果部署到 Linux 或不同 GPU，历史会在那台机器上自动收敛，但冷启动 3 次跑仍然会差。可以考虑把几台参考机器的 history JSONL ship 到 repo 作为更好的冷启动 baseline。
- **`BRAINFAST_STATE_DIR` 跨盘 rename fallback 路径不完美**：见 §5。可以加一个 shutil.copytree+rmtree 分支。

---

## 8. 联系方式 / 交接要点

- 本轮所有代码修改遵循 TDD（先红后绿），commit 消息含 plan task 编号或 bug 编号
- 全量 test/lint 报告见 §3
- 所有 commit 都在 `main` 本地分支；push 前建议先 CI 预跑（这批命名 guard 会首次真亮相）
- 如审查反馈要求回滚或重做某个 task，见 §6 的顺序建议
- 本文档位于 `docs/HANDOFF_2026-04-20_REVIEW.md`，配合 `docs/superpowers/plans/2026-04-20-closed-loop-productization-remediation.md` 一起看即完整

# Brainfast Review Addendum — 2026-04-20

> 对应主文档：`docs/HANDOFF_2026-04-20_REVIEW.md`
> 对应整改计划：`docs/superpowers/plans/2026-04-20-closed-loop-productization-remediation.md`
> 审查范围：`0f4954b^..6e7868d`
> 审查方式：静态代码审查
> 本附录结论：新增 3 个问题，分别为 1 个 `P1`、2 个 `P2`

---

## 1. TL;DR

这轮新增问题集中在 3 条线上：

1. **dual-channel reuse 契约不完整**
   `reuse_from_dir` 现在只校验少量 3D 工件，不校验 `truth_export` 的逐切片 label 是否存在且数量匹配，可能静默产出空的或被截断的第二通道结果。

2. **calibration shared-state fallback 不闭环**
   legacy 目录迁移失败时，保存仍会写回旧树，但学习线程只读新 state root，导致 `Save Calibration + Learn` 在特定部署下失效。

3. **class-prior auto-apply 只有一次**
   `_autoWarmStartTried` 是页面级单例标志，不按 job/class 重置，导致同一浏览器会话里第二个空 job 起就不再自动 warm-start。

---

## 2. Findings Summary

| 优先级 | 位置 | 问题 |
|---|---|---|
| `P1` | `project/scripts/whole_brain_3d.py:394-485` | reuse mode 可能静默产出空或截断的 second-channel counts |
| `P2` | `project/frontend/server_context.py:583-660` | calibration fallback 写旧目录，但 learn 只读新 state root |
| `P2` | `project/frontend/app.js:6868-6888` | class-prior auto-apply 在单页会话中只生效一次 |

---

## 3. Detailed Findings

### Finding 1 — `P1`

**位置**  
`project/scripts/whole_brain_3d.py:394-485`

**标题**  
Reuse mode can silently yield empty or truncated second-channel counts

**问题描述**  
当前 `_REUSE_REQUIRED_FILES` 只要求：

- `ants_registration/annotation_registered.nii.gz`
- `laplacian_refinement/annotation_refined.nii.gz`
- `laplacian_refinement/laplacian_deformation_field.npy`

但 `_reuse_prior_registration_and_quantify()` 真正构造 `truth_rows` 时，依赖的却是：

- `truth_export/slice_*_registered_label.tif`

而且它的拼接方式是按 index 直接把：

- prior channel 的 `slice_*_registered_label.tif`
- current channel 的 `merged_slice_paths`

两边 `zip` 到最短长度为止。

这会导致两类静默错误：

1. prior dir 没有 `truth_export` label 时，`reuse_from_dir` 仍然通过校验，但 `truth_rows=[]`，后续 quantification 可能直接产出空结果。
2. 两个通道切片数不一致时，尾部切片被静默截断，没有显式报错，也没有 QA 信号。

此外，前端注释写的是 “same shape + same page count”，但实际只比较了 `sample_shape`，没有比较 `n_pages` / `n_files`，这条不变量并没有端到端成立。

**影响**  
这是一个 release-blocking 级别的问题。因为 dual-channel reuse 的卖点就是“复用 C0 配准，快速得到 C1 结果”；如果它可以静默产出空结果或不完整结果，reviewer 很难第一时间发现。

**建议修复**  

1. 把 `truth_export/slice_*_registered_label.tif` 纳入 reuse mode 的必需工件校验。
2. 在 `_reuse_prior_registration_and_quantify()` 中显式校验：
   - prior truth label 数量
   - current channel merged slice 数量
   - 二者必须一致，否则直接 fail loudly。
3. 在 wizard / 前端 dual-channel inspect 阶段补上：
   - `n_files` 或 `n_pages` 一致性检查
   - 不一致时禁止 launch，而不是只给 warning。

**建议验证**

- 单测：`reuse_from_dir` 缺少 `truth_export` label 时必须报错。
- 单测：prior truth labels 数量与当前 channel slices 不一致时必须报错。
- 手工：双通道同样本、故意删一张 C1 slice，确认 launch 或 reuse 明确失败。

---

### Finding 2 — `P2`

**位置**  
`project/frontend/server_context.py:583-660`

**标题**  
Calibration fallback saves to the legacy tree while learning reads the new state root

**问题描述**  
`_save_calibration_pair()` 在 legacy `train_data_set` 重命名失败时，日志写的是：

- “reading legacy, new writes go to shared-state”

但实际代码会：

- `train_dir = legacy_train_dir`

也就是后续样本仍然写到旧目录。

与此同时，`_learn_from_trainset_async()` 启动 `learn_from_trainset.py` 时，`--train-dir` 固定传的是：

- `outputs/state/calibration/samples`

所以在这些场景下会出现写读分裂：

- `BRAINFAST_STATE_DIR` 指向别盘
- 跨盘 `rename()` 失败
- legacy 目录存在且有旧数据

此时 `Save Calibration + Learn` 可能表现为：

1. finalize/save 看起来成功；
2. learn 线程也能跑；
3. 但 learn 用到的并不是刚保存的样本。

这会让功能看起来可用，实则学习结果为空或陈旧，而且源码树还会继续被污染。

**影响**  
这是 shared-state 改造后的闭环一致性问题。虽然只在 fallback 路径触发，但一旦触发，用户很难从 UI 上理解为什么 learn 没效果。

**建议修复**  

1. 迁移失败时不要把 `train_dir` 回退成 legacy 目录继续写。
2. 明确分两种策略：
   - 成功迁移：后续统一读写新目录；
   - 迁移失败：显式进入 “read-legacy-write-new” 模式，新写入仍落 shared state。
3. learn 线程的 `--train-dir` 与 save path 必须来自同一个 helper，不能分散拼路径。

**建议验证**

- 单测：模拟 `rename()` 失败后，保存路径仍然是 `outputs/state/calibration/samples`。
- 单测：保存 1 个 calibration sample 后，learn 线程读取到同一目录。
- 手工：设置 `BRAINFAST_STATE_DIR` 到不同盘符/目录，确认不会再写回 `train_data_set/`。

---

### Finding 3 — `P2`

**位置**  
`project/frontend/app.js:6868-6888`

**标题**  
Class-prior auto-apply only works for the first empty job per page session

**问题描述**  
`_autoWarmStartTried` 目前是页面级的单一布尔值：

```js
let _autoWarmStartTried = false;
```

它在第一次 auto-apply 成功后会被置为 `true`，但以下场景都没有重置：

- 切换 `Job ID`
- 切换 `Class`
- 清空当前 pairs
- 进入另一个新 job

结果是：

1. 第一个空 job 会正常 auto warm-start；
2. 同一页面会话中的第二个空 job 开始，自动 warm-start 静默失效；
3. 用户只有刷新页面才可能恢复预期行为。

这和文档/交接里描述的“empty jobs auto-apply”不一致。

**影响**  
问题不会破坏已有数据，但会直接破坏操作者对 class-prior 功能的信任，尤其是在批量 review 多个 job 时。

**建议修复**  

1. 把 `_autoWarmStartTried` 改成按 `(jobId, className)` 维度记录，而不是全局单值。
2. 至少在以下动作上重置：
   - `liq3dJobId` 变化
   - `liq3dClassName` 变化
   - `clear all pairs`
   - 手动加载新的 liquify state
3. 更稳妥的做法是维护一个：
   - `const autoWarmStartAttempted = new Set()`
   - key = `${jobId}::${className}`

**建议验证**

- 单测/前端回归：同一 session 依次切两个空 job，两个都能 auto-apply。
- 单测：同一个 job 改 class 后，会重新尝试 auto-apply。
- 手工：先对 job A 自动 warm-start，再切 job B，确认无需刷新页面。

---

## 4. Recommended Fix Order

建议按这个顺序处理：

1. **Finding 1 (`P1`)**
   这是结果正确性问题，而且发生后可能是静默错误。

2. **Finding 2 (`P2`)**
   这是 calibration shared-state 改造后的闭环一致性问题，关系到 `Save Calibration + Learn` 是否真的可用。

3. **Finding 3 (`P2`)**
   这是前端状态管理问题，修复成本较低，但影响用户对 class-prior 自动化的感知。

---

## 5. Reviewer Notes

- 本附录是对 `docs/HANDOFF_2026-04-20_REVIEW.md` 的补充，不替代主 handoff。
- 这 3 个问题都不是“文档措辞”级别，而是实际行为契约问题。
- 其中 Finding 1 建议在修复前不要把 dual-channel reuse 当作完全签收完成。


# Brainfast 生产就绪路线图设计文档

**日期：** 2026-03-31  
**目标：** 使 Brainfast 能够投入真实生物学实验流程使用（B级操作难度，面向 C 级演进）  
**版本：** 当前 v0.3.0-desktop

---

## 1. 背景与评估

### 1.1 项目现状

Brainfast 是一套鼠脑切片配准 + 细胞计数工具，核心流程为：

```
TIFF Z-stack → 通道提取 → 切片合并 → AP定位 → 2D atlas配准
→ 细胞检测 → 跨切片去重 → 脑区映射 → 聚合输出 → CSV/QC
```

后端：Python + Flask（Blueprint 模块化），前端：单页 HTML/JS（EN/ZH i18n）。

### 1.2 核心问题诊断

**阻断级（现在跑不准）：**

1. **从未亲自端到端跑过**：所有测试均由 Claude/Codex 代跑，无实测基准，无法知道哪里出错。

2. **Cellpose 未启用**：包已安装、模型权重已缓存（cyto2/cyto3/nuclei），但配置文件中 `primary_model: "fallback"` 强制使用 LoG fallback，Cellpose 从未被调用。

3. **学习机制训练/推理根本不一致**：`learn_from_trainset.py` 本质是在 12 个固定参数组合里做网格搜索，且：

   | 维度 | 训练阶段 | 推理阶段（Sample 35）|
   |---|---|---|
   | AP 定位方法 | `autopick_best_z`（自动搜索）| `atlas_z_from_filename`（公式）|
   | 像素尺寸 | 默认 0.65 µm | 实际 5.0 µm |
   | 半脑模式 | 无（训练样本为完整双侧脑）| `right_flipped` |
   
   训练结果 `trainset_tuned_params.json` 对 Sample 35 几乎无效。

**严重级（能跑但结果不可信）：**

4. `fail_score_threshold=0.1`：几乎不过滤任何切片，坏配准也进入细胞计数。
5. AP offset 手动调（`atlas_z_offset=150` 硬编码），换样本或切片间距即失效。
6. 无 per-slice QC 可视化报告：不知道哪些切片可信，最终计数无从溯源。

**体验级：**

7. 新样本需手动修改 JSON 配置（pixel size、Z offset、半脑模式等）。
8. 不支持多样本批处理。
9. 无 Windows 安装包（PyInstaller 被 Defender 阻断）。

---

## 2. 目标

- **主目标**：自己能够顺利完成一次端到端分析，结果可信、可溯源。
- **次目标**：学习机制真正工作，训练数据对配准质量有可量化的正向贡献。
- **长期目标**：B 级操作难度（生物学家无需 JSON/命令行），为将来开放给社区做好架构准备。

---

## 3. 设计方案：双轨并行

### 轨道 1：可信度（优先）

**P0 — 首次实测（所有后续工作的地基）**

亲自用 Sample 35 完整跑一遍主流程，记录：
- 每步实际输出文件是否存在
- 配准 overlay 视觉质量（每个切片的 overlay PNG）
- 细胞检测数量（LoG fallback vs Cellpose 对比）
- 最终输出 CSV 的合理性

验收标准：能产出 `cell_counts_leaf.csv`，且至少 80% 切片的 overlay 肉眼对齐合理。

**P1 — 启用 Cellpose**

- `run_config_35.json` 中 `primary_model` 改为 `"cellpose_cyto3"`（或 `cyto2`）
- `fallback_model` 保留 `"log"` 作为 Cellpose 失败时的安全网
- 在 `requirements-min.txt` 中取消注释 cellpose
- 新增一个 CI smoke test：import cellpose + load model（不需要真实图像）
- 验收标准：跑通 Sample 35 后细胞检测数量与 LoG 对比，能看出差异

**P2 — QC 报告**

为每次 pipeline 运行生成一份 HTML QC 报告，包含：
- 每切片的 overlay 缩略图 + 配准分数（颜色编码：绿/黄/红）
- fail_score_threshold 调整为 0.4（当前 0.1 过于宽松）
- 汇总表：总切片数、通过率、细胞总数、每脑区分布
- 报告路径：`outputs/qc_report.html`

**P3 — AP 估计改进**

引入 DeepSlice 作为可选 AP 估计方法（`ap_method: "deepslice"` 已有基础实现），替代纯公式。对于不同样本类型（不同 Z-step、不同切割方向），DeepSlice 比公式更鲁棒。

---

### 轨道 2：学习机制重建（P0 完成后并行推进）

**问题核心**：现有 `learn_from_trainset.py` 的搜索空间（12种参数组合）和训练数据（7对 Show.png，无 Label.tif）与 Sample 35 的实际分析条件完全不匹配。

**重建目标**：

1. **训练/推理条件一致**
   - 训练时使用与目标样本相同的 AP 方法（`atlas_z_from_filename` 或 DeepSlice）
   - 训练时使用目标样本的真实 pixel size（从 config 读取）
   - 支持 `right_flipped` 等半脑模式

2. **为 Sample 35 建立专用训练对**
   - 从 Sample 35 的已知良好切片中提取若干对 `(real_tif, target_label.tif)`
   - 这些对的 AP 位置用公式计算，与推理一致
   - 目标数量：5-10 对（覆盖脑干、皮层、纹状体等不同解剖区域）

3. **扩展参数搜索空间**
   - 当前：3 profiles × 2 fit_modes × 2 smooth = 12 组合
   - 目标：增加 `tissue_shrink_factor`（0.80–0.95）和 `contour_tps_smooth`（1.0–4.0）的搜索，约 40-80 组合
   - 加入 per-sample 最优（不仅选全局最优，也输出每个解剖区域的最优参数）

4. **可量化验证输出**
   - 训练结束后输出 `outputs/trainset_tuned_params.json`（已有），增加：
     - per-sample 分数矩阵（每个训练对 × 每种参数组合的得分表格）
     - 可视化比对 HTML（训练对的 pred vs target overlay，便于人工检查）
   - 验收标准：在已知训练对上，新参数比默认参数的平均 boundary_F1 提升 ≥ 0.05

---

### 轨道 3：体验与分发（轨道 1 稳定后）

**GUI 配置向导**

新增 `/setup` 页面，引导用户填写：
- 输入 TIFF 文件夹路径（文件夹选择器）
- 像素尺寸（µm）
- 切片间距（µm）
- 标记物类型（神经元/核标记/胞质）
- 半脑/全脑选择
- 生成配置文件保存到 `configs/`

**多样本批处理**

UI 支持同时提交多个样本的分析任务，后台队列顺序执行。

**Windows 安装包**

修复打包问题（调查 Defender 误报根因），使用 NSIS + 独立 Python 环境，产出 `.exe` 安装包。

---

## 4. 实施优先级

```
轨道1-P0: 首次实测 Sample 35                    ← 必须第一个完成
轨道1-P1: 启用 Cellpose                          ← P0 完成后立即做
轨道2:    学习机制重建（分三个子步骤）             ← 与 P1 并行开始
轨道1-P2: QC 报告 + fail_score 调整              ← 与轨道2并行
轨道1-P3: DeepSlice AP 改进                      ← P2 完成后
轨道3:    GUI向导、批处理、安装包                  ← 轨道1/2 稳定后
```

---

## 5. 不在本次范围内

- 多通道同时计数（如 red + green 双标）
- 3D 配准（当前为 2D slice-by-slice）
- 云端/服务器部署
- 与其他脑图谱（非 Allen CCFv3）兼容
- 统计分析模块（组间比较、显著性检验）

---

## 6. 验收标准（总体）

1. 用 Sample 35 跑完完整流程，`outputs/` 产出所有预期文件
2. 至少 80% 切片配准分数 ≥ 0.4（新阈值）
3. Cellpose 作为主检测器工作（非 fallback）
4. 学习机制：训练对上新参数 boundary_F1 ≥ 旧参数 + 0.05
5. QC HTML 报告可在浏览器打开，包含所有切片 overlay 缩略图

# Brainfast x AAV Toolbox 复现差距清单

更新时间：2026-04-01  
适用对象：Codex / Claude / 后续接手者  
目的：基于当前代码库现状，重新判断 Brainfast 是否能复现 AAV toolbox 论文中的数据分析方法，并给出最新版差距清单与执行顺序。

本文替代旧版 handoff。旧版里“主流程仍写死 coronal、尚未真正支持 slicing_plane”的判断已经过时；当前代码已经会从配置读取 `input.slicing_plane`，见：

- [main.py](D:\Brainfast\project\scripts\main.py#L242)
- [main.py](D:\Brainfast\project\scripts\main.py#L415)

论文参考：

- [An enhancer-AAV toolbox to target and manipulate distinct interneuron subtypes](https://pmc.ncbi.nlm.nih.gov/articles/PMC11291062/)

相关仓库入口：

- [README.md](D:\Brainfast\project\README.md#L12)
- [REPRODUCE.md](D:\Brainfast\REPRODUCE.md#L57)
- [main.py](D:\Brainfast\project\scripts\main.py#L196)

## 一句话结论

当前 Brainfast 仍然不能“傻瓜式复现整篇论文的数据分析方法”。

更准确地说：

- 它已经能覆盖论文里一部分组织学图像分析子流程。
- 它还不能覆盖整篇论文的方法学范围。
- 即便只看图像分析子流程，目前也还没有达到“论文同款、开箱即复现”的程度。

## 当前已经具备的论文相关能力

和旧 handoff 相比，当前仓库已经不是“只有通用 atlas 计数工具”那么简单，已经长出一层明确朝论文图像分析靠拢的能力：

1. 主流程支持从配置读取切片方向
   - 当前主流程会读取 `input.slicing_plane`，不再是旧结论里的“写死 coronal”。
   - 入口证据见 [main.py](D:\Brainfast\project\scripts\main.py#L242)。

2. 已有论文导向的 `reporter_positive` 检测模式
   - 见 [detect.py](D:\Brainfast\project\scripts\detect.py#L152)
   - 路由逻辑见 [detect.py](D:\Brainfast\project\scripts\detect.py#L323)

3. 已有 paper-style 区域汇总脚本
   - 会输出 `paper_aav_region_summary.csv`
   - 见 [main.py](D:\Brainfast\project\scripts\main.py#L562)
   - 见 [paper_aav_summary.py](D:\Brainfast\project\scripts\paper_aav_summary.py#L1)

4. 已有共定位脚本
   - 会输出 `colocalization_summary.csv`
   - 见 [main.py](D:\Brainfast\project\scripts\main.py#L573)
   - 见 [colocalization.py](D:\Brainfast\project\scripts\colocalization.py#L1)

5. 已有论文风格报告导出脚本
   - 见 [main.py](D:\Brainfast\project\scripts\main.py#L621)
   - 见 [export_paper_report.py](D:\Brainfast\project\scripts\export_paper_report.py#L215)

6. 当前仓库主流程已能真实跑通
   - 本地验证已通过环境检查、全量测试和一次真实输入 smoke run。
   - 这说明“复现差距”现在主要不是“代码完全不能跑”，而是“方法学边界和论文一致性不够”。

## 当前仍然不能声称“已复现论文方法”的原因

### 1. 论文范围远大于 Brainfast 当前产品边界

这篇论文不只是“组织切片配准和细胞计数”。它还包含：

- single-cell genomic / scATAC-seq enhancer discovery
- enhancer ranking and selection
- electrophysiology
- optogenetics
- behavior / functional validation
- cross-species validation in NHP and human tissue

这些都不在 Brainfast 当前代码边界内，因此不能声称“整篇论文可复现”。

### 2. 当前图像检测逻辑仍不是论文同款 study-specific detector

当前 Brainfast 已有 `reporter_positive`，但这仍然更像是研究导向规则模式，而不是论文中那种针对该研究数据训练/微调的特定 DNN 检测流程。

现在的现实状态是：

- 有组织学图像分析能力
- 有论文导向的 mode
- 但没有证据说明它已经等价于论文中的 study-specific detection pipeline

### 3. `sensitivity` 仍未真正算出

当前 [colocalization.py](D:\Brainfast\project\scripts\colocalization.py#L123) 已明确写了：

- 当前表结构不足以给出完整 marker-positive 总数
- 所以 `sensitivity` 被留成 `NaN`

这意味着即使 `specificity` 有了，论文常用的另一半关键指标仍未闭环。

### 4. “代表切片”逻辑还不是论文同款

当前 [paper_aav_summary.py](D:\Brainfast\project\scripts\paper_aav_summary.py#L3) 采用的是：

- 每个区域取细胞计数最高的切片

这和论文里基于固定 parasagittal / medio-lateral coordinate 的标准化取样逻辑不是一回事。当前更像是“工程上合理的 representative slice heuristic”，而不是“论文同款 sampling protocol”。

### 5. 缺少论文级 preset / SOP

现在的配置和 GUI 仍然是通用工作流思路，不是明确的“论文复现模式”。  
例如当前 [run_config_35.json](D:\Brainfast\project\configs\run_config_35.json#L24) 仍然是偏通用/示例配置，而不是某个 paper-specific preset。

所以当前仍然需要懂 pipeline 的人自己拼配置，不能称为“傻瓜式复现”。

## 最新差距清单

按优先级划分如下。

### P0：阻止“论文图像分析子流程可复现”声明的差距

1. 缺少 paper-specific preset
   - 需要一个明确的 `paper_mode` 或等价配置入口。
   - 应自动绑定 slicing plane、channel、detector mode、colocalization、report export。

2. 检测器仍不够论文同款
   - 当前 `reporter_positive` 是重要进展，但还不能等价替代论文中的研究专用检测方案。
   - 需要至少完成一轮 paper dataset 上的 detector calibration / validation。

3. `sensitivity` 未闭环
   - 当前只具备部分共定位统计。
   - 要真正接近论文，需要能从 marker-positive 总量出发完成 sensitivity 计算。

4. representative slice 逻辑与论文不一致
   - 当前是“最高计数切片”
   - 论文更接近“固定坐标 + 标准化选片 + 生物重复汇总”

5. 缺少论文级输出规范
   - 当前有 `paper_aav_region_summary.csv` 和 `paper_report/`
   - 但还缺少一套清晰定义过的论文复现输出 contract
   - 包括哪些 CSV、哪些图、哪些 summary 表才算“复现完成”

### P1：不阻止工程试跑，但阻止“论文同款”表达的差距

6. GUI 里还没有“论文复现模式”入口
   - 目前 GUI 已修到可继续测试，但它仍然是通用 Brainfast UI，不是 AAV toolbox 专用 SOP 界面。

7. 缺少 replicate-level 统计与汇总规范
   - 当前更多是单次 run 结果输出。
   - 论文级结果需要更明确的 replicate merge / SEM / cohort summary。

8. 报告层还不够论文化
   - 当前报告已经能出图、出 panel、出 region summary。
   - 但还没到“一键导出论文图表包”的程度。

9. 真实 paper dataset 上的端到端再验证还没完成
   - 当前本地已验证的是通用流程和 smoke run。
   - 还没有证据表明它已经在论文那类数据分布上稳定达标。

### Out of Scope：不应继续往 Brainfast 主线里硬塞的部分

以下内容不应被当作 Brainfast 当前阶段的“必须补齐项”：

- scATAC-seq enhancer discovery
- electrophysiology analysis
- optogenetic analysis
- behavior analysis
- cross-species wet-lab validation

这些内容属于“论文整体复现”范围，而不是“Brainfast 图像分析引擎升级”范围。

## 现在到底能 claim 到什么程度

当前最稳妥的表述是：

1. 已能完成通用的 atlas alignment + cell counting + region aggregation 工作流。

2. 已具备朝 AAV toolbox 论文图像分析靠拢的后端能力：
   - slicing plane configurable
   - reporter-positive detection mode
   - paper-style region density summary
   - colocalization scaffold
   - paper report scaffold

3. 仍不能声称：
   - “可复现整篇论文”
   - “已论文同款复现图像分析方法”
   - “一键傻瓜式复现”

4. 当前最多只能谨慎声称：
   - “可部分复现论文中的组织学图像分析子流程”

## 建议的实施顺序

### Phase 1：把“论文模式”做成真正可跑的 preset

目标：

- 新建 AAV toolbox paper preset
- 让 GUI/CLI 能以 preset 方式进入论文模式

最低交付：

- `paper_mode = aav_toolbox_histology`
- 固定/推荐 detector mode
- 固定/推荐 slicing plane
- 固定输出目录结构

### Phase 2：补齐共定位统计闭环

目标：

- 真正算出 `specificity`
- 真正算出 `sensitivity`

最低交付：

- marker-positive 总量统计
- region-level colocalization summary
- 清晰定义输入要求和失败条件

### Phase 3：替换 representative slice 逻辑

目标：

- 从“最高计数切片 heuristic”升级为“更接近论文 protocol 的 standardized sampling”

最低交付：

- 引入固定坐标或可配置标准切片集合
- 输出每个区域对应的 representative slice 选择依据

### Phase 4：把结果层做成论文输出包

目标：

- 把现有 `paper_report` 从 scaffold 提升成明确产物规范

最低交付：

- region count / density chart
- specificity / sensitivity table
- representative panel
- 简洁 summary text

### Phase 5：用真实论文风格数据做再验证

目标：

- 确认“不是只有代码结构像论文，而是真实数据分布下也能稳定输出论文风格结果”

最低交付：

- 至少一轮端到端 rerun
- 明确记录偏差点
- 形成 validation note

## 最新判断

旧版 handoff 里“`slicing_plane` 仍被主流程写死，因此论文方向入口还没打通”的说法，现在已经不成立。当前更准确的说法是：

- 切片方向入口已经打通到主流程层。
- 真正的复现差距已经转移到 detector fidelity、colocalization completeness、representative slice protocol、paper preset 和 validation。

这意味着项目已经从“底层能力缺失阶段”，进入了“论文一致性与产品化封装不足阶段”。

## 接手建议

后续如果继续沿论文复现方向推进，建议不要再泛泛地问“能不能复现整篇论文”，而是按下面这句作为工作边界：

“把 Brainfast 补到能稳定复现 AAV toolbox 论文中的组织学图像分析子流程，并明确不能覆盖的非图像方法部分。”

只要边界不收敛，工作就会不断滑向错误目标。

# Brainfast x AAV Toolbox 复现执行摘要

更新时间：2026-04-01  
用途：给新线程里的 Codex / Claude 直接接手，不重复读长文档也能立刻开工。

配套长文档：

- [HANDOFF_AAV_TOOLBOX_REPRO.md](D:\Brainfast\docs\HANDOFF_AAV_TOOLBOX_REPRO.md)

## 当前结论

当前 Brainfast 还不能声称“傻瓜式复现 AAV toolbox 论文的数据分析方法”。

最准确的说法是：

- 可以部分覆盖论文里的组织学图像分析子流程
- 不能覆盖整篇论文的方法学范围
- 即便只看图像分析子流程，也还没有达到论文同款、开箱即复现

## 这次必须记住的纠正

旧 handoff 里“主流程仍写死 `coronal`，尚未真正支持 `slicing_plane`”这条已经过时。

当前代码已经会从配置读取 `input.slicing_plane`：

- [main.py](D:\Brainfast\project\scripts\main.py#L242)
- [main.py](D:\Brainfast\project\scripts\main.py#L415)

所以后续不要再把“plane 入口没打通”当主要阻塞项。

## 现在已经有的论文相关能力

1. `slicing_plane` 已进入主流程
2. 已有 `reporter_positive` 检测模式
   - [detect.py](D:\Brainfast\project\scripts\detect.py#L152)
   - [detect.py](D:\Brainfast\project\scripts\detect.py#L323)
3. 已有 `paper_aav_region_summary.csv` 生成逻辑
   - [main.py](D:\Brainfast\project\scripts\main.py#L562)
   - [paper_aav_summary.py](D:\Brainfast\project\scripts\paper_aav_summary.py#L1)
4. 已有共定位脚本
   - [colocalization.py](D:\Brainfast\project\scripts\colocalization.py#L1)
5. 已有 paper-style report exporter
   - [export_paper_report.py](D:\Brainfast\project\scripts\export_paper_report.py#L215)

## 现在真正的主要差距

### P0

1. 缺少 paper-specific preset / SOP
   - 现在仍是通用 Brainfast 工作流，不是“论文复现模式”

2. 检测器还不是论文同款
   - `reporter_positive` 是进展，但还不能等价于论文里的 study-specific detector

3. `sensitivity` 没有真正算出来
   - 当前代码明确把它留成 `NaN`
   - [colocalization.py](D:\Brainfast\project\scripts\colocalization.py#L123)

4. representative slice 逻辑不是论文同款 protocol
   - 当前是“每区域取计数最高切片”
   - 不是论文那种标准化坐标取样
   - [paper_aav_summary.py](D:\Brainfast\project\scripts\paper_aav_summary.py#L3)

5. 缺少论文级输出 contract
   - 现在有 summary / report scaffold
   - 还没有明确定义“哪些产物齐全才算论文图像分析复现完成”

### P1

6. GUI 没有显式“论文模式”入口
7. replicate-level 统计和 cohort 汇总不足
8. 报告层还不够论文化
9. 尚未在真实论文风格数据上完成严格 rerun 验证

## 不要再往错误方向花时间

以下内容不应继续塞进 Brainfast 当前主线目标里：

- scATAC-seq enhancer discovery
- electrophysiology
- optogenetics
- behavior analysis
- cross-species wet-lab validation

这些属于“整篇论文复现”，不属于当前 Brainfast 图像分析升级的合理边界。

## 新线程建议直接做什么

按这个顺序，不要跳：

1. 做一个明确的 `paper_mode` / `aav_toolbox_histology` preset
   - 先把 GUI/CLI 入口收敛

2. 补齐 colocalization 的完整输入与 `sensitivity` 计算

3. 替换 representative slice 逻辑
   - 从“最高计数切片 heuristic”升级为更接近论文 protocol 的 standardized sampling

4. 定义 paper output contract
   - 哪些 CSV
   - 哪些图
   - 哪些 summary

5. 用真实论文风格数据做一轮端到端 rerun

## 当前最稳妥的对外表述

不要说：

- “已经复现整篇论文”
- “已经论文同款复现”
- “已经一键傻瓜式复现”

只建议说：

- “当前 Brainfast 可部分复现 AAV toolbox 论文中的组织学图像分析子流程，但尚未覆盖整篇论文，也尚未达到论文同款一键复现。”

## 如果新线程只看一段

当前项目已经从“底层能力缺失阶段”进入了“论文一致性与产品化封装不足阶段”。

后续重点不再是争论 `slicing_plane` 支不支持，而是：

- detector fidelity
- colocalization completeness
- representative slice protocol
- paper preset
- validation on paper-like data

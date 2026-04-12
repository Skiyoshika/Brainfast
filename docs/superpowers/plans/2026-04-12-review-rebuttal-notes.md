# Brainfast 当前代码库反驳稿

## 核心结论

这轮改动不是没有进步，但“代码局部变好”不等于“仓库已经稳定”。  
如果要技术上反驳“这一轮已经很稳、可以放心”的说法，最有力的结论不是去翻旧账，而是指出：

1. 有些旧 bug 确实已经修掉了，不能继续当作负面证据使用。
2. 但仓库级的可验证性仍然没有闭环。
3. 现在缺的不是又一个局部 patch，而是默认路径、环境检查、CI 覆盖和完整工作流的一致性证明。

---

## 先承认已经修掉的点

下面这条已经不适合再拿来批评当前代码：

- `project/scripts/registration_3d_ants.py`
  transform persistence 已经改成 best-effort；复制失败时会 warning 并回退到原始 transform 路径，而不是在注册成功后再崩溃。

这点必须承认。  
如果继续拿已经修掉的问题当主要论据，反而会削弱整体审查的可信度。

---

## 真正还能驳倒“已经很稳”的技术点

### 1. 环境检查会产生假阳性，说明“通过自检”不等于“项目可运行”

证据：

- `project/scripts/check_env.py` 当前只用 `importlib.util.find_spec()` 判断模块是否存在。
- 这只能说明“包名可见”，不能说明“包真的能 import 成功”。
- 当前机器上：
  - `python project/scripts/check_env.py --config project/configs/run_config.template.json` 返回通过
  - 但 `python -m pytest project/tests/unit -q` 在测试收集阶段直接因为 `NumPy 2.4.3` 与 `SciPy/skimage` ABI 不兼容而崩溃

结论：

- 现在的环境检查不具备发布级可信度。
- 只要这一条还存在，就不能说默认环境“已经稳”。

应该怎么改：

- 对 `scipy`、`skimage`、`cellpose` 做真实 import smoke test，而不是只看 `find_spec()`
- 对关键数值栈增加版本边界检查
- 把 ABI/版本冲突升级为 `FAIL`

---

### 2. CI 绿色不代表默认 shipped path 真的可启动

证据：

- `.github/workflows/test.yml` 的 unit job 只安装 `.[dev]`
- 但模板配置 `project/configs/run_config.template.json` 默认走：
  - `registration.scope = whole`
  - `whole_brain_backend = miki_3d`
  - `detection.primary_model = cpsam`

结论：

- 当前 CI 最多证明“最小开发依赖下，部分代码能过静态检查和单测”
- 它没有证明“默认 whole-brain + ANTs + Cellpose-SAM 路径能在干净环境里启动”

换句话说：

- 现在的 CI 是绿的，但默认产品路径没有被 CI 证明

应该怎么改：

- 增加一个 job 安装 `.[wholebrain,advanced,dev]`
- 在这个 job 里跑：
  - `python project/scripts/check_env.py --config project/configs/run_config.template.json`
  - 一个最小 whole-brain smoke test

---

### 3. Cellpose-SAM 已经接上，但默认 v4 路径还没有完全收紧

证据：

- `project/scripts/detect.py` 已经支持：
  - `cpsam`
  - `CellposeModel(pretrained_model=...)`
  - v4 3-value `eval()` 返回
- `project/tests/unit/test_detect.py` 也补了对应单测

但还存在一个技术风险：

- `detect.py` 里关于 `channels` 的条件仍然过宽
- 在 mixed-version 布局下，如果同时暴露了 `CellposeModel` 和 `Cellpose`，默认 `cpsam` 路径仍可能带上 legacy `channels` 参数

结论：

- 这是“已经接好”但“还没有完全版本收口”的典型例子
- 可以说功能通了，不能说默认路径已经充分稳定

应该怎么改：

- 只有在明确走 legacy `Cellpose` 分支时才传 `channels`
- 给 `cpsam` 默认 kwargs 增加断言级测试

---

### 4. 默认 `ml_flip=false` 和整脑主路径自己的方向说明存在张力

证据：

- 模板配置默认把 `ml_flip` 设成 `false`
- 但 `project/scripts/whole_brain_3d.py` 的代码注释写得很明确：
  显微图像通常需要做 ML flip，才能和 Allen 左半球约定对齐

结论：

- 这是一个“默认值改变缺少充分证据”的问题
- 如果没有真实 sample 的 A/B 结果支撑，这类改动不能被表述成“已经很稳”

更直接地说：

- 这可能正是你一直看到整脑配准 overlay 很差的来源之一

应该怎么改：

- 用真实样本比较 `ml_flip=true/false`
- 以 overlay 质量和 region mapping 一致性来决定默认值
- 在没有证据前，不要把这个说成稳定结论

---

### 5. 当前仓库还没有“结果可信”的完整证据链

证据：

- `ruff` 现在已经是绿的，这很好
- 但单测还因为数值栈问题在收集阶段崩溃
- 同时模板里的 `fail_score_threshold` 现在很低

这说明什么：

- 静态质量门恢复了一部分
- 但运行时质量门和结果可信度门还没有闭环

结论：

- 现在最多能说“工程原型在进步”
- 不能说“结果已经足够可信，可放心外推到更多 sample”

应该怎么改：

- 先恢复测试环境可运行
- 再把 registration quality gate 提升成硬门槛
- 明确规定：明显差的 overlay 不能进入 region mapping 结论

---

### 6. 最致命的问题不是某一段代码，而是用户仍然走不通一次完整工作流

证据：

- UI 里有：
  - manual landmark correction
  - liquify
  - `Save Calibration + Learn`
- 但这条链当前本质上是 2D atlas preview / overlay 学习链
- 它并不是 3D whole-brain 最终 truth 的完整人工闭环
- 它也不是 Cellpose-SAM 的人工校准闭环

因此：

- 现在用户仍然没法完整体验一次稳定的人机协同流程
- 这比任何单个 bug 都更能说明“系统还没稳”

结论：

- 只要用户还不能稳定完成：
  `预览 -> 自动配准 -> 手工修正 -> 保存校准 -> 检测预览 -> 导出`
  就不应该把“自然测试偶尔能跑通”当成成熟度证明

应该怎么改：

- 先做一个单 sample 的完整 interactive happy path
- 让用户真正走通一次
- 这条路打通之前，不要继续把大规模自然测试当主要质量论据

---

## 最适合作为反驳的总论述

可以直接这样说：

> 这轮改动确实修掉了一部分历史 bug，但目前修复的是“局部代码正确性”，不是“仓库级稳定性证明”。  
> 现在最大的缺口不在某一行逻辑，而在于：
> 1. 环境检查会假阳性  
> 2. CI 没有覆盖默认 shipped path  
> 3. Cellpose-SAM 默认 v4 路径还没完全收口  
> 4. 整脑默认方向设置缺少样本级证据  
> 5. 用户仍然无法稳定体验一次完整人工工作流  
> 
> 所以，这轮不能叫“已经很稳”，最多只能叫“比上一轮更接近可验证的工程原型”。

---

## 一句话压轴版

真正的问题不是“有没有修 bug”，而是“有没有形成可以被复现、被验证、被用户体验到的完整证据链”。  
在这件事上，当前代码库还没有闭环。

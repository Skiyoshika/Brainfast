# 人工实测指南 — Sample 35 首次端到端验证

> **什么时候看这份文档：** 当 Claude 告诉你"ready for manual run"时，按这份文档操作。
>
> **终端说明：** 所有命令在 **PowerShell** 里运行（VS Code 终端默认就是 PowerShell）。

---

## 前置检查（2 分钟）

```powershell
cd D:\Brainfast\project

# 1. 确认 atlas 文件存在（约 28MB）
Get-Item annotation_25.nii.gz | Select-Object Name, Length

# 2. 确认 demo 切片存在
(Get-ChildItem data\35_C0_demo\*.tif).Count
# 应显示 111（或类似数字，>0 即可）

# 3. 确认 Python 环境
python -c "import nibabel, tifffile, scipy; print('OK')"
```

如果步骤 1 或 2 失败（文件不存在），先解决再继续。

---

## 第一步：运行 pipeline（当前配置：Cellpose）

```powershell
cd D:\Brainfast\.worktrees\plan1-first-run\project

python scripts/main.py --config configs/run_config_35.json --run-real-input data/35_C0_demo
```

> **注意：** 配置文件现在已经是 `cellpose_cyto3`，所以这次直接跑 Cellpose。

**预期耗时：** 20-60 分钟（Cellpose CPU 模式，每张切片约 15-30 秒）

**观察终端输出：**
- 应该看到每个切片的进度信息
- 最后一行应显示：`Real-input end-to-end complete: detected=XXX, dedup=YYY`
- 如果看到 `RuntimeError: N slice(s) failed registration score threshold`，记下 N，继续即可

**运行完后立即记录：**
```
detected = _____ 个细胞
dedup    = _____ 个（去重后）
```

---

## 第二步：运行验证脚本

```powershell
cd D:\Brainfast\.worktrees\plan1-first-run
python project/scripts/validate_outputs.py
```

**预期输出（PASS）：**
```
Validating outputs in: ...\project\outputs

  overlay PNGs: XX found
  cell_counts_leaf.csv: XX regions
  registration QC: XX/XX slices passed (XX%)

PASS — all expected outputs present and non-empty
```

如果看到 `FAIL`，把完整输出复制给 Claude。

---

## 第三步：查看配准分数分布

```powershell
cd D:\Brainfast\.worktrees\plan1-first-run\project

python -c "
import pandas as pd
df = pd.read_csv('outputs/slice_registration_qc.csv')
print(df[['slice_id','best_z','best_score','registration_ok']].to_string())
print()
print('--- 摘要 ---')
print(f'通过率: {df.registration_ok.mean():.0%}')
print(f'分数均值: {df.best_score.mean():.3f}')
print(f'分数最小: {df.best_score.min():.3f}')
print(f'分数最大: {df.best_score.max():.3f}')
"
```

**记录结果：**
```
通过率  = _____%
均值分数 = _____
最低分数 = _____（对应 slice_id = _____）
```

---

## 第四步：人工检查 Overlay 质量（最重要）

用文件管理器（Win+E）打开：
```
D:\Brainfast\project\outputs\registered_slices\
```

或在 PowerShell 直接打开：
```powershell
explorer D:\Brainfast\project\outputs\registered_slices
```

找到 3-5 张 `*_overlay.png` 文件，双击用图片查看器打开。

**判断标准：**

| 看到的情况 | 结论 |
|---|---|
| 彩色脑区轮廓大致覆盖在组织上，形状吻合 | ✅ 配准良好 |
| 彩色区域整体偏移，但形状相似 | ⚠️ 平移偏差 |
| 彩色区域完全不在组织上，或大小严重不匹配 | ❌ 配准失败 |

**选几个不同位置的切片：**
- 前脑区域（slice_id 较小，AP~300 以上）
- 中脑区域（slice_id 居中）
- 后脑区域（slice_id 较大，AP~200 以下）

**记录你的判断：**
```
整体配准质量：良好 / 部分偏差 / 大多数失败
典型偏差描述：___________________________
```

---

## 第五步：查看使用的检测器

```powershell
python -c "
import pandas as pd
df = pd.read_csv('outputs/cells_detected.csv')
print(f'细胞总数: {len(df)}')
print(f'检测器: {df.detector.unique()}')
print(df.detector.value_counts().to_string())
"
```

**预期：** `detector` 列应包含 `cellpose_cyto3`（而非 `fallback_log`）

---

## 完成后告诉 Claude

把以下内容发给 Claude：

```
实测完成。结果如下：
- validate_outputs: PASS / FAIL（如果 FAIL，附上输出）
- 配准通过率: _____%，均值分数: _____
- 配准质量目测：良好 / 部分偏差 / 大多数失败
- 检测器: ___________，细胞数: _____
- 任何异常或疑问：___________________________
```

---

## 常见问题

**Q: `RuntimeError: N slice(s) failed registration score threshold` 是什么意思？**
A: 有 N 张切片配准分数低于阈值，被拒绝。记录 N 的数字，发给 Claude。

**Q: 想先用少量切片快速测试怎么办？**
```powershell
# 复制前 5 张切片到临时目录
New-Item -ItemType Directory -Force -Path D:\tmp\demo_5
Get-ChildItem D:\Brainfast\project\data\35_C0_demo\*.tif |
  Select-Object -First 5 |
  Copy-Item -Destination D:\tmp\demo_5\

cd D:\Brainfast\.worktrees\plan1-first-run\project
python scripts/main.py --config configs/run_config_35.json --run-real-input D:\tmp\demo_5
```

**Q: Cellpose 报错怎么办？**
A: 把完整错误信息发给 Claude。

**Q: 终端里多行命令怎么输入？**
A: PowerShell 支持用反引号 `` ` `` 换行，但本指南里的 `python -c "..."` 命令已经写成单行，直接整段复制粘贴即可。

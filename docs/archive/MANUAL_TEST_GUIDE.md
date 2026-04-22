# Brainfast 人工实测指南

更新：2026-04-01 | 适用分支：`main`

> **目标：** 一次测试走完"输入 TIFF → 配准 → 检测 → 论文级统计输出"全流程，并通过 GUI 验证结果展示。
>
> **终端：** 所有命令在 **PowerShell** 里运行（VS Code 终端默认就是）。

---

## 准备步骤：从原始 Z-stack 提取切片

> **如果你要用 `data\35_C0_demo\` 里已有的 demo 切片，跳过这一步直接去第 0 步。**
>
> 如果你要用原始样本文件 `Sample\35_High_..._C0.tif` 测试，先在这里把它拆成单张 TIFF。

### 文件信息

| 项目 | 值 |
|------|-----|
| 文件 | `35_High_1000ms_560nm_640nm_150W_z5um_Bothlaser - Pos 3 4 [1] 3DMontage_XY1763150824_Z000_T0_C0.tif` |
| 页数 | 646 张 z-slice |
| 单张尺寸 | 1636 × 1359 像素，uint16 |
| 像素大小 | 5.0 µm/pixel |
| z 步进 | 5 µm |

### 路径 A：快速测试（推荐先用，~100 张，与 demo 等效）

提取每隔 5 张（有效 z 间距 = 25 µm），覆盖脑组织集中的 z50–z545 区间：

```powershell
cd D:\Brainfast\project

python scripts\extract_zstack.py `
  --input "D:\Brainfast\Sample\35_High_1000ms_560nm_640nm_150W_z5um_Bothlaser - Pos 3 4 [1] 3DMontage_XY1763150824_Z000_T0_C0.tif" `
  --out_dir data\35_C0_full `
  --every_n 5 `
  --z_min 50 `
  --z_max 545
```

**期望输出：** `Extracted 100 slices (Z 50–545, every 5) → data\35_C0_full`

提取完直接用现有配置跑：

```powershell
python scripts\main.py --config configs\run_config_35.json --run-real-input data\35_C0_full
```

---

### 路径 B：全量提取（646 张，耗时更长）

提取全部切片，z 步进 5 µm，不跳帧，覆盖完整 z 轴：

```powershell
cd D:\Brainfast\project

python scripts\extract_zstack.py `
  --input "D:\Brainfast\Sample\35_High_1000ms_560nm_640nm_150W_z5um_Bothlaser - Pos 3 4 [1] 3DMontage_XY1763150824_Z000_T0_C0.tif" `
  --out_dir data\35_C0_full646
```

**期望输出：** `Extracted 646 slices (Z 0–645, every 1) → data\35_C0_full646`

用同一份配置跑（`atlas_z_z_scale=0.2` 自动适配，AP 范围 150–279）：

```powershell
python scripts\main.py --config configs\run_config_35.json --run-real-input data\35_C0_full646
```

> 646 张切片耗时约为路径 A 的 5–6 倍，建议路径 A 验证通过后再跑路径 B。

---

## 第 0 步：前置检查（2 分钟）

```powershell
cd D:\Brainfast\project

# 1. atlas 文件
Get-Item annotation_25.nii.gz | Select-Object Name, @{n='MB';e={[int]($_.Length/1MB)}}
```
**期望：** 文件存在，大小约 28 MB。没有的话先运行：
```powershell
python download_atlas.py --ensure
```

```powershell
# 2. 确认切片文件夹存在（用哪个就检查哪个）
(Get-ChildItem data\35_C0_demo\*.tif).Count      # 已有 demo
(Get-ChildItem data\35_C0_full\*.tif).Count       # 路径 A 提取后
(Get-ChildItem data\35_C0_full646\*.tif).Count    # 路径 B 提取后
```
**期望：** 至少一个输出数字 ≥ 10。

```powershell
# 3. Python 依赖
python -c "import nibabel, tifffile, scipy, skimage, pandas, matplotlib; print('依赖 OK')"
```
**期望：** `依赖 OK`。缺包就 `pip install <包名>`。

---

## 第 1 步：跑完整 Pipeline（CLI）

```powershell
cd D:\Brainfast\project

python scripts\main.py --config configs\run_config_35.json --run-real-input data\35_C0_full
```

结果会写到 `outputs\35_C0_full\`（默认与输入文件夹同名）。

如果想自定义文件夹名：
```powershell
python scripts\main.py --config configs\run_config_35.json --run-real-input data\35_C0_full --output-name my_run_01
```

> 换用其他切片文件夹就把路径替换掉：`data\35_C0_demo` 或 `data\35_C0_full646`，输出目录会自动对应。

**预计耗时：** 10–30 分钟（100 张切片）。终端会逐切片输出进度，最后一行应该是：
```
Real-input end-to-end complete: detected=XXXX, dedup=YYYY -> outputs/cell_counts_leaf.csv + QC
```

**立即记下：**
```
detected = _____
dedup    = _____
```

> 如果中途报错，把完整错误信息复制出来，不要继续后面的步骤。

---

## 第 2 步：确认输出文件齐全

```powershell
cd D:\Brainfast\project

# 把 RUN 换成你的实际输出文件夹名（默认与输入文件夹同名，如 35_C0_full）
$RUN = "35_C0_full"

# 核心输出（6 个）
Get-Item outputs\$RUN\cells_detected.csv,
         outputs\$RUN\cells_dedup.csv,
         outputs\$RUN\cells_mapped.csv,
         outputs\$RUN\cell_counts_leaf.csv,
         outputs\$RUN\cell_counts_hierarchy.csv,
         outputs\$RUN\slice_registration_qc.csv |
  Select-Object Name, @{n='KB';e={[int]($_.Length/1KB)}}
```
**期望：** 6 个文件全部存在，大小 > 0。

```powershell
# 论文级新输出（Phase 3/5）
Get-Item outputs\$RUN\region_areas.csv,
         outputs\$RUN\paper_aav_region_summary.csv,
         outputs\$RUN\paper_report\region_count_density.png,
         outputs\$RUN\paper_report\paper_report_summary.txt 2>$null |
  Select-Object Name
```
**期望：** 4 个文件全部存在。
- 缺 `region_areas.csv`：label TIF 路径问题，记录下来
- 缺 `paper_report/` 里的文件：运行 `pip install matplotlib Pillow`，然后单独生成：
  ```powershell
  python scripts\export_paper_report.py --outputs-dir outputs\$RUN
  ```

```powershell
# overlay 数量
(Get-ChildItem outputs\$RUN\registered_slices\*_overlay.png).Count
```
**期望：** 与切片数一致（约 100+）。

---

## 第 3 步：配准质量数据

```powershell
cd D:\Brainfast\project
$RUN = "35_C0_full"   # ← 改成你的实际输出文件夹名

python -c "
import pandas as pd
df = pd.read_csv(r'outputs/$RUN/slice_registration_qc.csv')
worst = df.loc[df.best_score.idxmin(), 'slice_id']
print('=== 配准摘要 ===')
print(f'切片总数  : {len(df)}')
print(f'通过率    : {df.registration_ok.mean():.0%}')
print(f'均值分数  : {df.best_score.mean():.3f}')
print(f'最低分数  : {df.best_score.min():.3f}  (slice ' + str(worst) + ')')
failed = df[~df.registration_ok]
if len(failed):
    print(f'失败切片 ({len(failed)} 张):')
    print(failed[['slice_id','best_z','best_score']].to_string(index=False))
else:
    print('全部切片通过')
"
```

**记录：**
```
通过率   = _____%
均值分数 = _____
最低分数 = _____（slice _____）
失败数量 = _____
```

---

## 第 4 步：人工目视 Overlay（最重要）

```powershell
$RUN = "35_C0_full"   # ← 改成你的实际输出文件夹名
explorer "D:\Brainfast\project\outputs\$RUN\registered_slices"
```

打开几张 `*_overlay.png`，**各选一张：**
- 前部（z0050–z0150 附近）
- 中部（z0250–z0350 附近）
- 后部（z0450–z0545 附近）

**判断标准：**

| 看到的情况 | 结论 |
|---|---|
| 彩色 atlas 轮廓大致覆盖组织，形状吻合 | ✅ 良好 |
| 轮廓整体偏移，但形状基本对 | ⚠️ 部分偏差 |
| 轮廓完全不在组织上，或大小严重错 | ❌ 配准失败 |

**记录：**
```
整体质量 = 良好 / 部分偏差 / 配准失败
描述     = ______________________________
```

---

## 第 5 步：检测结果

```powershell
cd D:\Brainfast\project
$RUN = "35_C0_full"   # ← 改成你的实际输出文件夹名

python -c "
import pandas as pd
det = pd.read_csv(r'outputs/$RUN/cells_detected.csv')
print(f'总细胞数  : {len(det)}')
print(f'检测器    :')
print(det['detector'].value_counts().to_string())
mapped = pd.read_csv(r'outputs/$RUN/cells_mapped.csv')
ok = (mapped['mapping_status'] == 'ok').sum()
print(f'成功映射率: {ok/max(len(mapped),1):.0%} ({ok}/{len(mapped)})')
"
```

**记录：**
```
检测器     = _____
总细胞数   = _____
成功映射率 = _____%
```

> 当前 `run_config_35.json` 用的是 `fallback_log`（LoG 检测器），这是正常的。

---

## 第 6 步：论文级统计输出（Phase 3/5）

```powershell
cd D:\Brainfast\project
$RUN = "35_C0_full"   # ← 改成你的实际输出文件夹名

python -c "
import pandas as pd
df = pd.read_csv(r'outputs/$RUN/paper_aav_region_summary.csv')
cols = [c for c in ['acronym','region_name','representative_slice_id','count','area_mm2','density_cells_per_mm2'] if c in df.columns]
print(f'脑区总数: {len(df)}')
print()
print('Top 10 regions (by count):')
print(df.head(10)[cols].to_string(index=False))
"
```

**期望：** 能看到脑区名、代表切片 ID、count、density_cells_per_mm2（不应全为 NaN）。

```powershell
# 查看文字摘要
Get-Content outputs\$RUN\paper_report\paper_report_summary.txt

# 打开图表
start outputs\$RUN\paper_report\region_count_density.png
```

**期望：** 图表能打开，能看到双轴柱状图（count 柱 + density 折线）。

**记录：**
```
脑区总数          = _____
density 列有值    = 是 / 否（全 NaN）
图表能打开        = 是 / 否
```

---

## 第 7 步：GUI 实测

### 7a. 启动服务器

```powershell
cd D:\Brainfast\project
python frontend\server.py
```

浏览器打开：[http://127.0.0.1:8787](http://127.0.0.1:8787)

---

### 7b. 查看 CLI 已生成的结果（验展示）

第 1 步的 pipeline 已经跑完，直接在 GUI 里看结果：

1. 点 **Results** 标签 → 应看到脑区计数表
2. 点 **QC** 标签 → 应看到 overlay 缩略图图库，点击可放大

**记录：**
```
Results 计数表显示正常 = 是 / 否
QC overlay 图库正常   = 是 / 否
```

---

### 7c. 用 GUI 直接跑 Pipeline（论文配准全流程）

> **这次 GUI 支持选 config 文件了。** 填入 `run_config_35.json` 就能跑论文级配准，不用再手动改模板。

**操作步骤：**

1. 在 **Input TIFF Folder** 输入框填入（或点 Browse 选择）：
   ```
   D:\Brainfast\project\data\35_C0_full
   ```
   （也可以用 `data\35_C0_demo` 或 `data\35_C0_full646`，用哪个填哪个）

2. 在 **Run Config JSON** 输入框填入（或点 Browse 选择）：
   ```
   D:\Brainfast\project\configs\run_config_35.json
   ```

3. 点 **Run Pipeline**

4. 观察：
   - 进度条是否实时更新（切片数 X/Y）
   - 日志区域是否显示每张切片的进度

5. 跑完后点 **Results** 和 **QC** 标签确认结果刷新

**记录：**
```
GUI 能触发 pipeline  = 是 / 否
Config 字段生效      = 是 / 否（若不确定，看日志里是否有 atlas_hemisphere=right_flipped）
进度条/日志正常      = 是 / 否
结果页面正常刷新     = 是 / 否
```

---

### 7d. 已知 GUI 未展示的新功能（下一轮迭代）

以下文件 pipeline 会生成到磁盘，但 GUI 还没有对应展示入口：

| 文件 | 备注 |
|------|------|
| `region_areas.csv` | 第 6 步用命令行查看 |
| `paper_aav_region_summary.csv` | 第 6 步用命令行查看 |
| `outputs/paper_report/*.png` | 第 6 步直接打开图片 |
| `colocalization_summary.csv` | 需在 config 里设 `marker_channel` 才生成 |

---

检查完用 `Ctrl+C` 停掉服务器。

---

## 实测结果模板

测完把下面填好发过来：

```
=== Brainfast 实测结果 2026-04-01 ===

【第 0 步】前置检查
- atlas 文件: 存在 / 不存在
- demo 切片数: _____
- 依赖检查: OK / 报错（附错误）

【第 1 步】Pipeline 运行
- 结果: 正常完成 / 中途报错（附错误）
- detected = _____
- dedup    = _____

【第 2 步】输出文件
- 核心 6 个文件: 全部存在 / 缺少（_____）
- region_areas.csv: 存在 / 不存在
- paper_aav_region_summary.csv: 存在 / 不存在
- paper_report/ 图表: 存在 / 不存在
- overlay 数量: _____

【第 3 步】配准质量
- 通过率:   _____%
- 均值分数: _____
- 最低分数: _____（slice _____）
- 失败数量: _____

【第 4 步】Overlay 目视
- 整体质量: 良好 / 部分偏差 / 配准失败
- 描述: ___________________________________

【第 5 步】检测
- 检测器: _____
- 总细胞数: _____
- 成功映射率: _____%

【第 6 步】论文级统计
- 脑区总数: _____
- density 列有值: 是 / 否
- 图表能打开: 是 / 否

【第 7 步】GUI
- Results 计数表正常: 是 / 否
- QC overlay 图库正常: 是 / 否
- GUI 触发 pipeline 正常: 是 / 否
- 进度条/日志正常: 是 / 否
- Config 字段生效: 是 / 否

【其他问题】
- _______________________________________
```

---

## 常见问题

**Q: pipeline 卡在某张切片超过 10 分钟不动？**  
`Ctrl+C` 中断，把最后几行输出发过来。

**Q: `RuntimeError: N slice(s) failed registration score threshold`**  
有 N 张配准失败。当前阈值 `fail_score_threshold=0.1` 已很宽松，若仍失败说明有严重问题，记录 N 和报错内容。

**Q: `density_cells_per_mm2` 全是 NaN？**  
`region_areas.csv` 和 `cells_mapped.csv` 的 `region_id` 没对上。把两个文件的前 5 行发过来。

**Q: `paper_report/region_count_density.png` 没生成？**  
```powershell
pip install matplotlib Pillow
python scripts\export_paper_report.py --outputs-dir outputs
```

**Q: GUI 里填了 config 路径但不生效？**  
检查路径是否用的绝对路径（`D:\...`），不能用相对路径。

**Q: 只想先快速跑 5 张切片 smoke test？**  
```powershell
New-Item -ItemType Directory -Force -Path D:\tmp\smoke5
Get-ChildItem data\35_C0_full\*.tif | Select-Object -First 5 | Copy-Item -Destination D:\tmp\smoke5

python scripts\main.py --config configs\run_config_35.json --run-real-input D:\tmp\smoke5 --output-name smoke5
# 结果写到 outputs\smoke5\
```

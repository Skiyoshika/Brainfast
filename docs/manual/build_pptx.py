"""Build the Brainfast Operations Manual PPT (16:9) from captured screenshots.

Slide structure (17 slides):
    01. Cover — Brainfast 操作说明书
    02. 概览 — 全流程 5 步
    03. 启动与界面
    04. New Sample 向导 (intro)
    05. Step 1: Inspect 真实 TIFF
    06. Step 2: 配置参数
    07. Step 3: Launch → 进度 + ETA
    08. Results — 脑区柱状图
    09. ★ 3D Liquify 是什么？(何时用、为什么)
    10. ★ Liquify 操作步骤 (1: 加载 job)
    11. ★ Liquify 操作步骤 (2: 加 landmark pair)
    12. ★ Liquify 操作步骤 (3: Apply 3D warp)
    13. ★ Liquify 操作步骤 (4: Finalize → 重新 aggregate)
    14. ★ Liquify 真实案例 (real-35-full 60 landmarks → 5249 cells 迁移)
    15. ★ Class Prior 闭环积累 (累计 → warm-start)
    16. 真实样本跑时统计 + ETA 自校准
    17. 常见问题 / 最佳实践 / 资源

Output: D:/Brainfast/docs/manual/Brainfast_Operations_Manual.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt, Emu

HERE = Path(__file__).parent
SHOTS = HERE / "screenshots"
OUT_PATH = HERE / "Brainfast_Operations_Manual.pptx"

# --- 16:9 slide geometry ---
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# --- colors ---
BG = RGBColor(0x14, 0x18, 0x25)          # deep navy
FG = RGBColor(0xE5, 0xE7, 0xEB)          # light gray
ACCENT = RGBColor(0x60, 0xA5, 0xFA)      # blue
ACCENT_2 = RGBColor(0xFB, 0xBF, 0x24)    # amber (highlight)
MUTED = RGBColor(0x94, 0xA3, 0xB8)       # slate
SUCCESS = RGBColor(0x34, 0xD3, 0x99)     # teal


def add_slide(prs: Presentation, bg: RGBColor = BG):
    layout = prs.slide_layouts[6]  # blank
    slide = prs.slides.add_slide(layout)
    # Background fill
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = bg
    return slide


def textbox(slide, x: float, y: float, w: float, h: float,
            text: str, *, size: int = 18, bold: bool = False,
            color: RGBColor = FG, align=PP_ALIGN.LEFT, font_name: str = "Microsoft YaHei"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.05)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font_name
    return tb


def bullet_list(slide, x: float, y: float, w: float, h: float,
                bullets: list[str], *, size: int = 14, color: RGBColor = FG,
                line_spacing: float = 1.25, font_name: str = "Microsoft YaHei"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = "• " + bullet if not bullet.startswith(("  ", "\t")) else bullet
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = font_name
    return tb


def add_image(slide, name: str, x: float, y: float, w: float | None = None,
              h: float | None = None, border: bool = True) -> None:
    path = SHOTS / name
    if not path.exists():
        textbox(slide, x, y, w or 5, h or 2, f"[missing: {name}]", color=RGBColor(0xE5, 0x7F, 0x7F))
        return
    kwargs: dict = {}
    if w is not None:
        kwargs["width"] = Inches(w)
    if h is not None:
        kwargs["height"] = Inches(h)
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kwargs)
    if border:
        line = pic.line
        line.color.rgb = ACCENT
        line.width = Emu(12700)  # ~1pt
    return pic


def header(slide, title: str, subtitle: str = "", slide_number: int | None = None):
    # Accent bar (left)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.18), SLIDE_H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    # Title
    textbox(slide, 0.45, 0.25, 10.5, 0.6, title, size=26, bold=True, color=FG)
    if subtitle:
        textbox(slide, 0.45, 0.90, 10.5, 0.4, subtitle, size=14, color=MUTED)
    if slide_number is not None:
        textbox(slide, 12.5, 7.05, 0.7, 0.35, f"{slide_number:02d}", size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def caption(slide, x: float, y: float, w: float, text: str, *, size: int = 11):
    textbox(slide, x, y, w, 0.35, text, size=size, color=MUTED)


# =====================================================================
# SLIDE BUILDERS
# =====================================================================

def build_slide_01_cover(prs):
    s = add_slide(prs, BG)
    # Big accent rectangle on the right
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(9.0), Inches(0), Inches(4.5), SLIDE_H)
    box.fill.solid()
    box.fill.fore_color.rgb = ACCENT
    box.line.fill.background()
    # Title lines
    textbox(s, 0.7, 2.2, 8.2, 1.0, "Brainfast", size=60, bold=True, color=FG)
    textbox(s, 0.7, 3.2, 8.2, 0.7, "全自动脑图谱配准 + 液化补偿", size=26, color=ACCENT)
    textbox(s, 0.7, 3.9, 8.2, 0.5, "操作说明书", size=22, color=FG)
    textbox(s, 0.7, 6.5, 8.2, 0.4, "v1.0.0-rc1  ·  真实样本演示 (ChATe27 / 35_C0)", size=12, color=MUTED)
    # White keyword on accent box
    textbox(s, 9.3, 2.8, 3.8, 0.6, "3D Liquify", size=28, bold=True, color=BG)
    textbox(s, 9.3, 3.4, 3.8, 1.0, "人工补偿 + 类先验闭环", size=14, color=BG)


def build_slide_02_overview(prs):
    s = add_slide(prs)
    header(s, "全流程概览", "从一个 TIFF 到带置信区间的脑区柱状图", slide_number=2)
    # 5-step flow
    steps = [
        ("① 指向 TIFF", "New Sample 向导"),
        ("② 启动 Pipeline", "一键配准 + 细胞检测"),
        ("③ 查看 Results", "脑区分布 + 置信区间"),
        ("④ 3D Liquify", "★ 手动补偿区域错位"),
        ("⑤ Class Prior", "跨样本积累 → 自动 warm-start"),
    ]
    x0, y0, boxw, gap = 0.55, 1.9, 2.4, 0.15
    for i, (label, desc) in enumerate(steps):
        x = x0 + i * (boxw + gap)
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y0), Inches(boxw), Inches(2.3))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0x1F, 0x29, 0x42) if i != 3 else ACCENT
        box.line.color.rgb = ACCENT
        box.line.width = Emu(12700)
        fg = FG if i != 3 else BG
        textbox(s, x, y0 + 0.35, boxw, 0.6, label, size=18, bold=True, color=fg, align=PP_ALIGN.CENTER)
        textbox(s, x + 0.1, y0 + 1.15, boxw - 0.2, 0.8, desc, size=12, color=fg, align=PP_ALIGN.CENTER)
    # Bottom call-out on liquify
    textbox(s, 0.55, 4.7, 12.2, 0.45, "★ 本册重点：3D Liquify 闭环 — 为什么、什么时候用、怎么用", size=20, bold=True, color=ACCENT_2)
    bullets = [
        "自动配准通常有 3–15 voxel 残余偏移 — 对大部分 region 影响极小，但核团边界上能改变上千个 cell 归属",
        "Liquify 通过少量人工 landmark 对（atlas-where-it-is, real-where-it-should-be）驱动 3D Laplacian 位移场",
        "Finalize 重新导出 truth slices + 重映射 cells → 输出 cell_counts_hierarchy_liquify3d.csv",
        "累积 ≥3 个同类样本的 landmark 后，下一个样本自动 warm-start（跳过手动修正）",
    ]
    bullet_list(s, 0.65, 5.2, 12.1, 2.1, bullets, size=13)


def build_slide_03_launch(prs):
    s = add_slide(prs)
    header(s, "启动 & 主界面", "双击 StartIdleBrainTrial.bat 或 python frontend/server.py → 浏览器打开 http://127.0.0.1:8787", slide_number=3)
    add_image(s, "01_home.png", 0.45, 1.45, w=9.5)
    add_image(s, "11_sidebar.png", 10.3, 1.45, h=5.6)
    caption(s, 0.45, 7.0, 9.5, "Registration Workflow 默认页 — 可直接配置 Source TIFF（手动模式）")
    caption(s, 10.3, 7.0, 2.5, "左侧主导航：8 个功能标签")


def build_slide_04_wizard_intro(prs):
    s = add_slide(prs)
    header(s, "New Sample 向导", "新手从此处开始 — 不用编辑 config JSON", slide_number=4)
    add_image(s, "02_wizard_empty.png", 0.45, 1.45, w=8.5)
    # Right column text
    textbox(s, 9.5, 1.5, 3.5, 0.5, "3 步走完一切", size=20, bold=True, color=ACCENT)
    bullets = [
        "① 指向源文件（目录 or 多页 TIFF）",
        "② 点 Inspect 自动读元数据",
        "③ 点 Launch 启动 pipeline",
        "",
        "全程不用编辑任何 JSON/YAML",
        "自动建议 pixel size + z 间距",
        "自动选 atlas_hemisphere + channel",
    ]
    bullet_list(s, 9.5, 2.15, 3.7, 5.0, bullets, size=13)
    caption(s, 0.45, 7.0, 8.5, "左侧 New Sample 标签进入 — 清爽 3 步表单")


def build_slide_05_wizard_inspect(prs):
    s = add_slide(prs)
    header(s, "Step 1 — Inspect 真实 TIFF", "指向 2.87GB 多页 TIFF，Brainfast 读 ImageJ 元数据并建议参数", slide_number=5)
    add_image(s, "03_wizard_inspected.png", 0.45, 1.45, w=12.4)
    # Call-out box on the detected metadata
    callout = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(6.0), Inches(12.2), Inches(1.25))
    callout.fill.solid()
    callout.fill.fore_color.rgb = RGBColor(0x1F, 0x29, 0x42)
    callout.line.color.rgb = ACCENT_2
    textbox(s, 0.7, 6.1, 12.0, 0.4, "自动检测结果：pages=646, shape=[1636, 1359], dtype=uint16, px=5.0µm, z=24.8µm", size=14, bold=True, color=ACCENT_2)
    textbox(s, 0.7, 6.55, 12.0, 0.6,
            "警告提示「多页 TIFF 需先 extract → 单页切片目录」— 用户一看就明白接下来该做什么",
            size=12, color=FG)


def build_slide_06_wizard_configure(prs):
    s = add_slide(prs)
    header(s, "Step 2 — 配置参数（自动预填）", "Inspect 后 Sample ID / pixel / z / hemisphere / channel 全自动填好，用户只需要确认", slide_number=6)
    add_image(s, "04_wizard_ready.png", 0.45, 1.45, w=12.4)
    callout = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(6.0), Inches(12.2), Inches(1.25))
    callout.fill.solid()
    callout.fill.fore_color.rgb = RGBColor(0x1F, 0x29, 0x42)
    callout.line.color.rgb = SUCCESS
    textbox(s, 0.7, 6.1, 12.0, 0.4, "小贴士：Sample ID 越短越好（会作为 outputs/jobs/<id>/ 的目录名）",
            size=14, bold=True, color=SUCCESS)
    textbox(s, 0.7, 6.55, 12.0, 0.6,
            "Atlas hemisphere 默认 right_flipped（清脑样本朝向）；其他方向在 dropdown 中可选。",
            size=12, color=FG)


def build_slide_07_progress_eta(prs):
    s = add_slide(prs)
    header(s, "Step 3 — Launch & 进度 + ETA", "新版 ETA 模块自校准：ANTs 阶段也能看到剩余时间", slide_number=7)
    add_image(s, "05_progress_eta.png", 0.45, 1.45, w=9.5)
    textbox(s, 10.2, 1.5, 3.0, 0.5, "ETA 时间模块", size=18, bold=True, color=ACCENT)
    bullets = [
        "6 个阶段独立估算",
        "  · ANTs ~27–40 min",
        "  · Truth Export ~7 s / slice",
        "  · Detection ~15 s / slice",
        "",
        "自校准：真实 / 预期",
        "  比值 clamp 0.5..2.0×",
        "",
        "每次完成自动 append",
        "到 eta_history.jsonl",
        "→ 3 次后收敛到本机",
    ]
    bullet_list(s, 10.2, 2.15, 3.0, 5.0, bullets, size=11)
    caption(s, 0.45, 7.0, 9.5, "完整 run 约 4 小时 40 分（真实 646 slice 样本）— 阶段时间见 slide 16")


def build_slide_08_results(prs):
    s = add_slide(prs)
    header(s, "Results — 脑区细胞分布", "real-35-full 跑完：5,932,161 cells across 551 regions", slide_number=8)
    add_image(s, "06_results_chart.png", 0.45, 1.45, h=5.8)
    textbox(s, 6.5, 1.5, 6.6, 0.5, "关键输出文件", size=20, bold=True, color=ACCENT)
    bullets = [
        "cell_counts_hierarchy.csv — 层级聚合（467–551 rows）",
        "cell_counts_leaf.csv — 叶节点最细粒度",
        "cells_mapped.csv — 每个 cell 带 region_id + (x,y,z)_µm",
        "slice_registration_qc.csv — 每张切片 Dice / NCC",
        "",
        "支持导出为 Excel / 中文标签 / 按脑区分组",
        "",
        "★ 下一步：发现某核团 cell 数跟预期差很多？",
        "    用 3D Liquify 手动修正 → 闭环重映射",
    ]
    bullet_list(s, 6.5, 2.15, 6.6, 5.0, bullets, size=13)


def build_slide_09_liquify_intro(prs):
    s = add_slide(prs)
    header(s, "★ 3D Liquify — 液化补偿", "自动配准的最后一公里 · 人工驱动 3D Laplacian 位移场", slide_number=9)
    # Two-column: "What it is" + "When to use"
    textbox(s, 0.45, 1.5, 6.0, 0.5, "是什么？", size=22, bold=True, color=ACCENT)
    bullets_a = [
        "在交互式 UI 里点几对「atlas where it IS」↔「real where it SHOULD BE」",
        "后端解 3D Laplacian 生成位移场 → 微调 annotation volume",
        "Finalize 重新导出 truth slices + 重新映射所有 cells",
        "同一 run 产出 cell_counts_hierarchy_liquify3d.csv",
    ]
    bullet_list(s, 0.55, 2.1, 6.0, 2.7, bullets_a, size=13)
    textbox(s, 6.7, 1.5, 6.0, 0.5, "什么时候用？", size=22, bold=True, color=ACCENT_2)
    bullets_b = [
        "Results 看到某核团（如 SNr / PVT）cell 数异常",
        "切片 QC 发现 Dice < 0.7 但 ANTs 已收敛",
        "ANTs 配准整体正确但局部错位 3–15 voxels",
        "要处理同类样本时希望建立 class prior",
    ]
    bullet_list(s, 6.8, 2.1, 6.0, 2.7, bullets_b, size=13)
    # Bottom — the 4 operations
    textbox(s, 0.45, 5.0, 12.5, 0.45, "4 步走完闭环", size=18, bold=True, color=FG)
    ops = [
        ("① 加载 job", "填 Job ID → Reload"),
        ("② 加 Landmarks", "点 atlas / real 对"),
        ("③ Apply", "Laplacian 解位移场"),
        ("④ Finalize", "重映射 cells"),
    ]
    bw = 2.95
    for i, (a, b) in enumerate(ops):
        x = 0.6 + i * (bw + 0.15)
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(5.6), Inches(bw), Inches(1.5))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0x1F, 0x29, 0x42)
        box.line.color.rgb = ACCENT
        textbox(s, x, 5.75, bw, 0.45, a, size=15, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
        textbox(s, x + 0.1, 6.3, bw - 0.2, 0.8, b, size=12, color=FG, align=PP_ALIGN.CENTER)


def build_slide_10_liquify_load(prs):
    s = add_slide(prs)
    header(s, "Liquify Step 1 — 加载 Job", "填 Job ID → Reload slice list → 看到 atlas 覆盖真实切片", slide_number=10)
    add_image(s, "07_liquify3d_loaded.png", 0.45, 1.45, w=10.2)
    textbox(s, 10.9, 1.5, 2.4, 0.5, "要点", size=20, bold=True, color=ACCENT)
    bullets = [
        "填 Job ID",
        "「real-35-full」",
        "",
        "点 Reload",
        "应见「646 slice",
        "overlays available」",
        "",
        "拖 z 滑块浏览",
        "0-645 切片",
        "",
        "彩色半透明区域",
        "就是 atlas 归属",
        "（region_id）",
    ]
    bullet_list(s, 10.9, 2.15, 2.4, 5.2, bullets, size=11)


def build_slide_11_liquify_pairs(prs):
    s = add_slide(prs)
    header(s, "Liquify Step 2 — 加 Landmark 对", "点 atlas (where the region IS) → 点 real (where it SHOULD BE)", slide_number=11)
    add_image(s, "09_liquify3d_full.png", 0.45, 1.45, w=7.5)
    textbox(s, 8.2, 1.5, 4.9, 0.5, "交互流程", size=22, bold=True, color=ACCENT)
    bullets = [
        "选 Next click = Atlas，点下图上某个 region 的当前位置",
        "再选 Real，点下图上同一解剖结构应该去的位置",
        "两次点击组成一对 landmark，自动入表",
        "表格最右可随时 × 删除某一对",
        "Ctrl+Z = 撤销最近一次",
        "",
        "密度建议",
        "  · 小样本 (~100 z) → 5-10 pairs 够",
        "  · 大样本 (>500 z) → 30-60 pairs 更稳",
        "  · 分布在 z 方向上 (比如每 50 层打一圈)",
        "",
        "本演示：real-35-full 用了 60 dense pairs",
        "max 位移 = 8.08 voxels",
    ]
    bullet_list(s, 8.3, 2.1, 4.8, 5.0, bullets, size=12)


def build_slide_12_liquify_apply(prs):
    s = add_slide(prs)
    header(s, "Liquify Step 3 — Apply 3D warp (Laplacian)", "点 Apply → 后端解 3D Laplacian → 生成 annotation_refined_liquify3d.nii.gz", slide_number=12)
    add_image(s, "08_liquify3d_pairs_table.png", 0.45, 1.45, w=7.2)
    textbox(s, 7.9, 1.5, 5.1, 0.5, "幕后算法", size=22, bold=True, color=ACCENT)
    bullets = [
        "vendored from D:/UCI-XuLab-RegTools",
        "",
        "Laplacian PDE with Dirichlet BC:",
        "  · landmark 是 anchor constraint",
        "  · 其他 voxel 是 free variable",
        "  · Conjugate Gradient + Jacobi 预处理",
        "  · 线性时间 O(N) 收敛",
        "",
        "输出 (3, D, H, W) 位移场",
        "  · nearest-neighbor sampling 保持 label 离散性",
        "  · 产出 annotation_refined_liquify3d.nii.gz",
        "",
        "real-full (57.6M voxels, 60 landmarks):",
        "  · 求解 ~3 分钟",
        "  · max 位移 8.08 voxels",
    ]
    bullet_list(s, 8.0, 2.1, 5.0, 5.1, bullets, size=11)


def build_slide_13_liquify_finalize(prs):
    s = add_slide(prs)
    header(s, "Liquify Step 4 — Finalize & 重新聚合", "用 refined annotation 重新导出 truth + 重映射 cells → hierarchy 自动更新", slide_number=13)
    # Content: numeric comparison table
    textbox(s, 0.45, 1.4, 12.5, 0.5, "real-35-full 真实闭环效果（60 dense landmarks）", size=18, bold=True, color=ACCENT_2)
    # Table as positioned text blocks
    rows = [
        ("指标", "Before Liquify", "After Liquify", "Δ"),
        ("root（全脑）cells", "5,656,618", "≈ 5,656,618", "≈ 0"),
        ("细胞迁移到不同 region", "—", "5,249", "+5,249 cells"),
        ("voxel-level region 改变", "—", "72,153 (0.125%)", "在 landmark 附近"),
        ("每 landmark 平均迁移 cells", "—", "87.5", "vs demo 960/pair"),
        ("hierarchy row 数（活跃 regions）", "551", "～551", "≈ 0"),
    ]
    cx = [0.55, 4.8, 7.6, 10.6]
    cw = [4.0, 2.6, 2.9, 2.4]
    row_h = 0.55
    for r, row in enumerate(rows):
        y = 2.05 + r * row_h
        if r == 0:
            bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(y), Inches(12.45), Inches(row_h))
            bg.fill.solid()
            bg.fill.fore_color.rgb = ACCENT
            bg.line.fill.background()
        for i, val in enumerate(row):
            color = BG if r == 0 else FG
            size = 14 if r == 0 else 12
            bold = r == 0
            textbox(s, cx[i], y + 0.05, cw[i], row_h - 0.1, val, size=size, bold=bold, color=color)
    # Insight box
    ins = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(5.6), Inches(12.5), Inches(1.6))
    ins.fill.solid()
    ins.fill.fore_color.rgb = RGBColor(0x1F, 0x29, 0x42)
    ins.line.color.rgb = ACCENT_2
    textbox(s, 0.6, 5.7, 12.2, 0.5, "关键洞见（本 session 新增）", size=16, bold=True, color=ACCENT_2)
    textbox(s, 0.6, 6.15, 12.2, 1.0,
            "Landmark 数量需要随体积 scale：demo 5 landmarks 动 4,800 cells（960/landmark），"
            "real-full 同样 5 landmarks 只动 1 cell；改 60 dense pairs 后动 5,249 cells。"
            "建议 ≥ 1 landmark per 10^7 voxels，并尽量覆盖 z 方向。",
            size=13, color=FG)


def build_slide_14_liquify_ui_detail(prs):
    s = add_slide(prs)
    header(s, "UI 细节 — landmark 表 + 批量操作", "60 pairs 全部在 UI 里可见，任意删除/撤销/清空", slide_number=14)
    add_image(s, "08_liquify3d_pairs_table.png", 0.45, 1.45, w=12.4)
    caption(s, 0.45, 7.0, 12.4, "Landmark pairs 表：# / z / ATLAS (y,x) / REAL (y,x) / 删除按钮。z=50/100/150... 均匀分布。")


def build_slide_15_classprior(prs):
    s = add_slide(prs)
    header(s, "Class Prior — 跨样本闭环积累", "同类样本越多，下一个越省事 — 3 次后自动 warm-start", slide_number=15)
    add_image(s, "10_classprior.png", 0.45, 1.45, w=8.5)
    textbox(s, 9.2, 1.5, 4.0, 0.5, "工作原理", size=22, bold=True, color=ACCENT)
    bullets = [
        "每完成 Finalize 后",
        "点「Save job → class prior」",
        "",
        "后端按 class name (如 ChATe27)",
        "做 running-mean 合并",
        "",
        "n ≥ 3 时:",
        "  - 新样本点「Warm-start",
        "    from prior」",
        "  - landmark 自动填入",
        "  - 再手动精调即可",
        "",
        "储存：",
        "  train_data_set/class_priors/",
        "    ChATe27/",
        "      landmark_prior.csv",
        "      sample_log.jsonl",
    ]
    bullet_list(s, 9.2, 2.15, 4.0, 5.2, bullets, size=10)


def build_slide_16_timing(prs):
    s = add_slide(prs)
    header(s, "真实样本耗时统计", "ChATe27 / 35_C0 全脑 (646 slices × 1636×1359) · ETA 自校准展示", slide_number=16)
    rows = [
        ("阶段", "耗时 (real-35-full)", "基线 (default)", "校准后 (2 samples)"),
        ("Volume Build", "~80 s", "0.6 s/slice (388 s)", "0.36 s/slice (232 s)"),
        ("Template Prep", "<1 min", "60 s", "60 s"),
        ("ANTS Registration", "27.8 min", "2000 s (33.3 min)", "2035 s (33.9 min)"),
        ("Laplacian Refinement", "<1 min", "30 s", "30 s"),
        ("Truth Export", "74 min (6.87 s/slice)", "7.0 s/slice", "7.22 s/slice"),
        ("Quantification", "178 min (16.5 s/slice)", "16.5 s/slice", "14.75 s/slice"),
        ("Total", "4h 40min", "~4.4 h (预估)", "~4.1 h (校准后)"),
    ]
    cx = [0.55, 4.0, 7.3, 10.3]
    cw = [3.2, 3.1, 2.7, 2.6]
    row_h = 0.52
    for r, row in enumerate(rows):
        y = 1.5 + r * row_h
        if r == 0:
            bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(y), Inches(12.45), Inches(row_h))
            bg.fill.solid()
            bg.fill.fore_color.rgb = ACCENT
            bg.line.fill.background()
        elif r == len(rows) - 1:
            bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(y), Inches(12.45), Inches(row_h))
            bg.fill.solid()
            bg.fill.fore_color.rgb = RGBColor(0x1F, 0x29, 0x42)
            bg.line.fill.background()
        for i, val in enumerate(row):
            color = BG if r == 0 else (ACCENT_2 if r == len(rows) - 1 else FG)
            size = 13 if r == 0 else 12
            bold = r in (0, len(rows) - 1)
            textbox(s, cx[i], y + 0.08, cw[i], row_h - 0.1, val, size=size, bold=bold, color=color)
    # Bottom note
    textbox(s, 0.45, 6.1, 12.5, 1.3,
            "ETA 模块在每次完成后自动 append 到 outputs/eta_history.jsonl；前端 /api/status 返回 eta.etr_total_s，"
            "UI 在所有阶段都能展示剩余时间。换台机器跑 ~3 次后基线自动收敛。",
            size=13, color=MUTED)


def build_slide_17_best_practices(prs):
    s = add_slide(prs)
    header(s, "最佳实践 & 资源", "常见问题 + 后续拓展方向", slide_number=17)
    textbox(s, 0.45, 1.5, 6.0, 0.5, "常见问题", size=22, bold=True, color=ACCENT)
    bullets_a = [
        "cpsam OOM？ wizard 默认 allow_fallback=true (LoG)",
        "路径有空格？ 全路径用正斜杠 / 或者加引号",
        "Results 空？ 看 slice_registration_qc.csv → Dice < 0.5 需要 liquify",
        "liquify 表空？ Job ID 要先填再点 Reload",
        "warm-start 提示 n<3？ 要先 save ≥3 个同类 sample",
    ]
    bullet_list(s, 0.55, 2.1, 6.1, 3.5, bullets_a, size=13)
    textbox(s, 6.8, 1.5, 6.0, 0.5, "相关文档", size=22, bold=True, color=ACCENT_2)
    bullets_b = [
        "CLAUDE.md — 开发者文档 + 架构",
        "docs/superpowers/plans/*.md — 各 phase 规划",
        "project/tests/unit/ — 401 个 unit tests",
        "GitHub: github.com/HiuraMika/Brainfast",
        "API 文档：/api/info 返回 version + endpoints",
    ]
    bullet_list(s, 6.9, 2.1, 6.0, 3.5, bullets_b, size=13)
    # Bottom call-out
    cta = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(5.6), Inches(12.5), Inches(1.55))
    cta.fill.solid()
    cta.fill.fore_color.rgb = ACCENT
    cta.line.fill.background()
    textbox(s, 0.7, 5.75, 12.0, 0.5, "从零到柱状图 · 新用户 typical flow", size=18, bold=True, color=BG)
    textbox(s, 0.7, 6.3, 12.0, 0.85,
            "① 双击启动 .bat  →  ② New Sample 指向 TIFF  →  ③ 4 小时后看 Results  "
            "→  ④ 对异常区域开 3D Liquify 补 10-50 landmarks  →  ⑤ Finalize → 保存到 Class Prior",
            size=13, color=BG)


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    builders = [
        build_slide_01_cover,
        build_slide_02_overview,
        build_slide_03_launch,
        build_slide_04_wizard_intro,
        build_slide_05_wizard_inspect,
        build_slide_06_wizard_configure,
        build_slide_07_progress_eta,
        build_slide_08_results,
        build_slide_09_liquify_intro,
        build_slide_10_liquify_load,
        build_slide_11_liquify_pairs,
        build_slide_12_liquify_apply,
        build_slide_13_liquify_finalize,
        build_slide_14_liquify_ui_detail,
        build_slide_15_classprior,
        build_slide_16_timing,
        build_slide_17_best_practices,
    ]

    for fn in builders:
        fn(prs)

    prs.save(str(OUT_PATH))
    print(f"[ok] Saved {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.0f} KB, {len(builders)} slides)")


if __name__ == "__main__":
    main()

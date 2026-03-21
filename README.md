[中文文档 →](README.zh-CN.md)

---

# Brainfast

**A reproducible, GUI-driven whole-brain analysis pipeline for enhancer AAV lightsheet data.**

> **For thesis committee reviewers:** Brainfast is not a new algorithm — it is a *workflow and analysis infrastructure* contribution. Its purpose is to consolidate a fragmented, multi-tool, manually-intensive lab process into a single, reproducible pipeline with structured biological output. The scientific value is in reproducibility, step reduction, and quantitative region-level output — not in outcompeting individual algorithms.

---

## Motivation — Why Brainfast?

### The problem with the current lab workflow

Whole-brain lightsheet fluorescence microscopy followed by enhancer AAV expression quantification involves a sequence of disconnected steps using multiple tools, each requiring expert configuration. In its current form, the workflow in our lab looks like this:

| Step | Tool | Pain point |
|------|------|------------|
| Image stitching / Z-stack export | Fiji / custom scripts | Manual parameter tuning per dataset |
| Atlas slice matching (AP position) | brainreg / manual lookup | Requires per-sample judgment on orientation and hemisphere |
| Section-to-atlas registration | brainreg / elastix / manual | No visual feedback during alignment; different tools produce different outputs |
| Cell / signal detection | Cellpose / manual threshold | Model choice and threshold tuned per imaging session |
| Region mapping | cellfinder / custom code | Separate tool, output format incompatible with registration result |
| Aggregation and statistics | Excel / R / Python scripts | Manual join of files from 3–4 different tools |
| Visualization and QC | Fiji / napari | Separate step, no link back to quantitative result |
| Report / export | Manual | Methods written from memory, no traceability to parameters |

**Consequence:** a single brain sample requires switching between 5–7 tools, 10–15 manual parameter decisions, and produces outputs in incompatible formats. Results vary between operators. New lab members spend weeks learning the stack. Reproducing a previous analysis requires documentation that rarely exists.

### What Brainfast does instead

Brainfast is **not** a replacement for brainreg, cellfinder, or ClearMap individually. Each of those tools does its job well in isolation. What Brainfast provides is the **connective tissue** missing between them: a single GUI-driven pipeline that takes a raw Z-stack as input and produces structured, atlas-mapped, per-region quantitative output — with full parameter traceability and QC at each step.

Specifically, Brainfast:

1. **Reduces 7+ fragmented steps to a single 4-step GUI workflow** (configure → register → review → detect+export)
2. **Replaces 5–7 disconnected tools** with one local desktop application
3. **Produces structured biological output directly** — per-region cell counts, 95% CI, morphology, AP-density distribution — without manual post-processing
4. **Encodes lab-specific defaults** into config templates (hemisphere orientation, voxel size, channel mapping for our enhancer AAV lightsheet setup)
5. **Enforces reproducibility** via full parameter logging, atlas SHA-256 fingerprinting, and auto-generated Methods text

This is the gap Brainfast fills: **turning a multi-tool, expert-dependent, inconsistently documented process into a reproducible, single-entry-point analysis pipeline.**

---

## Contribution Statement

This project's contribution is best described as a **pipeline innovation / analysis infrastructure contribution**, not an algorithmic one:

> *Brainfast establishes a reproducible, GUI-driven whole-brain analysis pipeline tailored to enhancer AAV lightsheet data, reducing a 7-step, 5-tool fragmented workflow to a 4-step single-application process, and producing structured per-region biological output suitable for multi-sample comparative analysis.*

The specific technical contributions are:

- **Unified registration–detection–mapping pipeline** with a single config file and one GUI entry point
- **Lab-specific registration heuristics** for cleared half-brain samples with specific hemisphere orientations (right_flipped mode, TPS nonlinear warp tuned to tissue-boundary conformance)
- **Calibration learning system** (17 curated training pairs) that adapts registration quality from operator feedback
- **Structured quantitative output** (region-wise counts, 95% Garwood–Poisson CI, morphology columns, AP-density distribution, cross-sample pivot table)
- **Built-in QC pipeline** (edge-SSIM per slice, Z-continuity outlier detection, preflight validation)
- **Batch processing and project management** enabling scalable multi-brain analysis

---

## Traditional Workflow vs Brainfast

| Metric | Traditional (lab current) | Brainfast |
|--------|--------------------------|-----------|
| Tools required | 5–7 (Fiji, brainreg, cellfinder, Cellpose, Excel, R/Python, napari) | 1 (Brainfast) |
| Manual parameter decisions per brain | 10–15 | 3–5 (voxel size, hemisphere, channel) |
| Steps requiring expert judgment | 4–5 | 1–2 (registration review) |
| Reproducibility of parameters | Low (undocumented) | Full (config JSON + atlas SHA-256 + auto Methods) |
| Time from raw Z-stack to regional counts | 4–8 h (experienced user) | 1–2 h (any lab member) |
| Output format | Inconsistent (varies per person) | Standardized CSV + Excel per run |
| New member onboarding | 2–4 weeks | < 1 day (GUI + guided tour) |
| Batch processing | Not supported | FIFO batch queue, N brains unattended |

*Runtime estimates based on a 111-slice coronal half-brain at 5 µm/pixel. Exact timings vary by GPU availability and image quality.*

---

## Pipeline Workflow

```
Input: TIFF Z-stack  (e.g., 111 coronal sections, 5 µm/pixel)
              │
              ▼
  ┌─────────────────────────┐
  │  Step 1 · Configuration │  ← single run_config.json (voxel size, hemisphere,
  │                         │    atlas path, channel map, detection params)
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Step 2 · Registration  │  AP slice matching (Z-filename → atlas coordinate)
  │                         │  Affine placement → TPS nonlinear warp
  │                         │  Output: registered_slices/, slice_registration_qc.csv
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Step 3 · Review / QC   │  ← browser GUI: per-slice overlay, SSIM scores,
  │                         │    manual liquify correction if needed,
  │                         │    Z-continuity AP-axis chart
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Step 4 · Detect+Export │  LoG or Cellpose detection → 3D KD-tree dedup
  │                         │  → CCFv3 region mapping → hierarchical aggregation
  │                         │  → Garwood 95% CI → Excel + CSV + Methods text
  └────────────┬────────────┘
               │
               ▼
Output: cell_counts_leaf.csv · cell_counts_hierarchy.csv
        cells_dedup.csv · brainfast_results.xlsx
        detection_summary.json (atlas_sha256, params, totals)
        Auto-generated Methods paragraph (EN + ZH)
```

---

## Biological Output

Brainfast is designed to answer the question:

> *"Where in the brain is my enhancer AAV expressed, how many cells express it per region, and how does that compare across samples?"*

Every pipeline run produces:

| Output | Biological meaning |
|--------|--------------------|
| `cell_counts_leaf.csv` | Per-region (leaf-level) cell count with 95% Poisson CI, mean elongation, mean area, mean intensity |
| `cell_counts_hierarchy.csv` | Counts aggregated through full Allen ontology (leaf → area → division → major region) |
| `cells_dedup.csv` | Individual cell centroids with (x, y, z_µm, score, region_id) — suitable for downstream spatial analysis |
| `AP-density chart` | Cells per anterior-posterior coordinate — visualises where along the AP axis expression is concentrated |
| `Cross-sample comparison` | Pivot table: rows = brain regions, columns = samples/animals — enables N > 1 group comparisons |
| `Methods paragraph` | Auto-generated reproducibility text for publication, citing atlas version, parameters, and SHA-256 |

Morphology columns (`mean_area_px`, `mean_elongation`, `mean_mean_intensity`) allow distinguishing between, e.g., dense compact nuclei vs diffuse axonal labelling within the same region.

---

## Minimal Validation

> **Note:** Brainfast has been validated on a single pilot brain (Sample 35, 111 coronal sections, 5 µm/pixel, right hemisphere, cleared tissue). This is a *minimal proof-of-concept validation*, not a systematic benchmark.

**Registration quality:** Edge-SSIM scores per slice are reported in `slice_registration_qc.csv`. Visual inspection of registered overlays showed correct hemisphere orientation and anatomical landmark alignment for all 111 sections.

**Detection:** LoG detector output was spot-checked against manual inspection in 3 representative ROIs. Cell centroid positions were qualitatively consistent with DAPI/signal-positive cells visible in the raw image.

**Limitations of current validation:** No formal manual-count vs automatic-count comparison has been performed. Accuracy across different signal intensities, imaging conditions, or enhancer constructs has not been systematically tested. See [Limitations](#limitations) below.

---

## Quick Start

### Requirements

- Python 3.10 or 3.11
- Windows 10/11 (primary) · Linux via Docker
- `annotation_25.nii.gz` — Allen CCFv3 25 µm annotation (place in `project/`)
- NVIDIA GPU recommended for Cellpose; LoG detector works on CPU

### Install

```powershell
git clone https://github.com/Skiyoshika/Brainfast.git
cd Brainfast
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[advanced,dev]"
```

### Run

```powershell
# Windows:
.\Start_Brainfast.bat

# Or directly:
python project\frontend\server.py
```

Open **http://127.0.0.1:8787** in your browser.

### Config template (lightsheet half-brain, our lab setup)

```json
{
  "project": { "name": "Sample_01", "pixel_size_um": 5.0 },
  "input": { "atlas_hemisphere": "right_flipped", "atlas_z_z_scale": 0.2 },
  "detection": { "detector": "log", "channels": ["red"], "min_score": 0.0 },
  "outputs": { "output_dir": "project/outputs/jobs/sample_01" }
}
```

All parameters are documented in [docs/user_guide.md](docs/user_guide.md) and validated against [project/configs/run_config.schema.json](project/configs/run_config.schema.json).

### Docker (Linux server / headless)

```bash
docker compose up -d
# Open http://localhost:8787
```

---

## Features

- **4-tab browser GUI** — Workflow / QC / Results / Projects — no install beyond Python
- **Bilingual EN/ZH** — full interface toggle, all labels and hints translated
- **Config-driven** — single JSON file controls all parameters; schema-validated on load
- **Batch queue** — SQLite-backed FIFO worker; multiple brains run unattended overnight
- **Projects & sample management** — group samples, track runs, re-run with different params
- **Cross-sample comparison** — merge leaf CSVs from N runs into a pivot table (N > 1 analysis)
- **Multi-channel co-expression** — per-region red/green channel counts side-by-side
- **Garwood 95% CI** — Wilson-Hilferty Poisson confidence intervals on every region count
- **Atlas fingerprint** — SHA-256 of annotation volume written to output for reproducibility
- **Auto Methods text** — publication-ready paragraph generated from run parameters (EN + ZH)
- **Z-continuity QC** — AP-axis smoothness chart with outlier flagging
- **3D volume pipeline** — full volumetric registration via ANTs/Elastix with HTML reports
- **Light/dark theme** — localStorage-persisted theme toggle
- **Guided tour** — 5-step onboarding for new lab members
- **Docker-ready** — `Dockerfile` + `docker-compose.yml` for headless Linux server deployment
- **97 unit tests**, CI on GitHub Actions (Windows + Ubuntu)

---

## Output Files

| File | Contents |
|------|----------|
| `cell_counts_leaf.csv` | Per-region leaf counts with `ci_low`, `ci_high`, morphology columns |
| `cell_counts_hierarchy.csv` | Counts rolled up through the full Allen ontology tree |
| `cells_dedup.csv` | Deduplicated cell centroids (x, y, z_µm, score, region_id) |
| `detection_summary.json` | Detector choice, sampling mode, totals, `atlas_sha256` |
| `slice_registration_qc.csv` | Edge-SSIM per slice |
| `z_smoothness_report.json` | AP-axis continuity analysis (outlier flags) |
| `brainfast_results.xlsx` | 3-sheet Excel: Hierarchy / Leaf / Run parameters |

All outputs are written to `project/outputs/jobs/<job_id>/` — fully isolated per run.

---

## Architecture

```
project/
├── frontend/
│   ├── server.py              Flask entry point (70-line orchestration layer)
│   ├── server_context.py      Shared run state, job isolation, GC
│   ├── blueprints/            12 API blueprints (fully documented at /api/docs)
│   │   ├── api_pipeline.py    run / cancel / poll / preflight / methods-text
│   │   ├── api_outputs.py     CSV / Excel / Z-continuity / AP-density / coexpression
│   │   ├── api_projects.py    project + sample CRUD (SQLite)
│   │   ├── api_batch.py       FIFO batch queue
│   │   ├── api_compare.py     cross-sample region comparison
│   │   └── …
│   ├── index.html             Single-page UI (4 tabs, bilingual)
│   ├── app.js                 Frontend JS (~3500 lines, full i18n)
│   └── styles.css             Dark/light theme CSS variables
├── scripts/
│   ├── main.py                2D pipeline entry point
│   ├── detect.py              LoG + Cellpose detection
│   ├── map_and_aggregate.py   Region mapping + hierarchical counts + Garwood CI
│   ├── z_smoothness.py        AP-axis continuity analysis
│   ├── config_validation.py   JSON Schema validation + runtime checks
│   └── …
└── tests/
    ├── unit/                  97 tests, no atlas file required
    └── integration/           Requires annotation_25.nii.gz
```

**Security invariants (v0.5+):**
- Config paths are containment-checked — no arbitrary filesystem access
- `running = True` set inside lock before thread start — no race condition
- `_job_states` capped at 200 entries with LRU eviction — no unbounded memory growth

---

## Limitations

Brainfast is a research-stage pipeline. Current known limitations:

1. **Single-brain validation only.** All registration heuristics and detection parameters have been tuned on Sample 35 (one right-hemisphere cleared brain). Performance on different enhancers, imaging conditions, brain regions, or hemisphere orientations has not been systematically evaluated.

2. **No formal accuracy benchmark.** Cell counting accuracy relative to manual ground truth has not been quantified. A systematic ROI-level manual-count vs automatic-count comparison is needed before drawing quantitative conclusions.

3. **LoG detector limitations.** The built-in LoG (Laplacian of Gaussian) detector works well for isolated nuclei but may over- or under-count in dense regions, axonal labelling, or low-SNR images. Cellpose is available but requires GPU and per-dataset model selection.

4. **Registration quality is data-dependent.** The TPS nonlinear warp is tuned for half-brain coronal sections at 5 µm/pixel. Full-brain samples, sagittal sections, or significantly different imaging resolutions require re-tuning of registration parameters.

5. **No inter-animal normalization.** Cell counts are raw counts per region. Brain volume normalization, hemisphere symmetry correction, and group-level statistical testing are not yet implemented and must be done downstream.

6. **Windows-primary.** The GUI is tested on Windows 10/11. Linux support is via Docker (headless); native Linux GUI is not supported.

---

## Future Work

- [ ] Systematic manual-count vs automatic-count validation across 3+ brains and 2+ enhancers
- [ ] Inter-animal normalization (counts per mm³) and group-level statistical output
- [ ] Sagittal and horizontal section support
- [ ] Cellpose model fine-tuning on lab-specific lightsheet data
- [ ] Integration with Allen Brain Cell Atlas for cell-type annotation
- [ ] Automated report generation (HTML/PDF) from multi-sample runs

---

## Science Methods

Algorithmic detail — registration stages, Garwood CI derivation, Z-continuity detection, atlas fingerprinting — is documented in [docs/science_methods.md](docs/science_methods.md).

**Auto-generated Methods paragraph** (from UI, with run parameters filled in):

> Brain atlas registration was performed using Brainfast v0.5. Microscopy images were acquired at N µm/pixel. Section registration was carried out against the Allen Mouse Brain Atlas (CCFv3, annotation_25.nii.gz, 25 µm voxel spacing, sha256: …) using nonlinear (thin-plate spline) transformation. Alignment quality was evaluated by edge-SSIM per slice. Cell counting used LoG detection on native single slices, followed by 3D KD-tree deduplication and hierarchical atlas aggregation. 95% Poisson confidence intervals were computed using the Garwood–Wilson-Hilferty method. Channels: red.

---

## Repository Layout

| Path | Purpose |
|------|---------|
| `project/configs/` | Run configs, Allen metadata, JSON Schema, sample presets |
| `project/frontend/` | Flask app, UI assets, desktop launcher |
| `project/scripts/` | Registration, detection, mapping, aggregation, utility scripts |
| `project/tests/` | Unit (97 tests) and integration test suites |
| `project/train_data_set/` | Manual calibration pairs (17 samples) |
| `docs/` | [User guide](docs/user_guide.md) · [API reference](docs/api_reference.md) · [Science methods](docs/science_methods.md) |

---

## Tests

```powershell
# Unit tests (no atlas file needed)
python -m pytest project/tests/unit -v

# With coverage report (unit-testable modules; atlas-dependent scripts excluded)
python -m pytest project/tests/unit --cov=project/scripts --cov-report=term-missing

# Lint
ruff check project/scripts/ project/frontend/blueprints/
```

CI: [GitHub Actions](.github/workflows/test.yml) — unit tests on Windows, ruff lint on every push/PR to `main`.

---

## License

See [LICENSE](LICENSE).

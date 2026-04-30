/* ============================================================
   Brainfast UI - app.js
   Bilingual (EN default / 中文 toggle)
   ============================================================ */

// ================================================================
// TRANSLATIONS
// ================================================================
// ================================================================
// LANGUAGE / i18n
// ================================================================
let currentLang = localStorage.getItem('brainfast.lang') || 'en';

const LANGS = {
  en: {
    'nav.workflow': 'Registration Workflow',
    'nav.manualTiff': 'Manual TIFF Check',
    'nav.qc': 'Batch QC Review',
    'nav.results': 'Results',
    'nav.projects': 'Projects',
    'projects.title': 'My Projects',
    'projects.create': '+ Create Project',
    'projects.empty': 'No projects yet. Create one above.',
    'projects.namePh': 'Project name…',
    'projects.descPh': 'Description (optional)',
    'projects.samples': 'Samples',
    'projects.delete': 'Delete',
    'sample.run': 'Load & Run',
    'sample.status.done': 'Done',
    'sample.status.running': 'Running',
    'sample.status.queued': 'Queued',
    'sample.status.pending': 'Pending',
    'sample.status.error': 'Error',
    'sample.addBtn': '+ Add Sample',
    'sample.configPh': 'Config path…',
    'sample.inputPh': 'Input directory…',
    'sample.namePh': 'Sample name…',
    'batch.title': 'Batch Queue',
    'batch.hint': 'Samples queued here are processed one by one automatically.',
    'batch.empty': 'Queue is empty.',
    'batch.cancel': 'Cancel',
    'batch.enqueue': 'Enqueue',
    'status.idle': 'Idle',
    'status.running': 'Running...',
    'status.error': 'Error',
    'errorPanel.title': 'Errors',
    'errorPanel.empty': 'No errors recorded.',
    'preflight.title': 'Preflight Check',
    'preflight.desc.warn': 'Review the structured issues below before starting the pipeline.',
    'preflight.desc.error': 'Fix the blocking issues below before starting the pipeline.',
    'preflight.back': 'Back',
    'preflight.continue': 'Continue Anyway',
    'progress.phase.queued': 'Queued',
    'progress.phase.ap_selection': 'AP Selection',
    'progress.phase.registration': 'Registration',
    'progress.phase.detection': 'Detection',
    'progress.phase.dedup': 'Deduplication',
    'progress.phase.mapping': 'Mapping',
    'progress.phase.done': 'Done',
    'progress.phase.error': 'Error',
    'progress.phase.cancelled': 'Cancelled',
    'btn.guide': '<i data-lucide="book-open" class="btn-icon"></i> Guide',
    'btn.run': '<i data-lucide="play" class="btn-icon"></i> Run Pipeline',
    'btn.cancel': '<i data-lucide="x" class="btn-icon"></i> Cancel',
    'btn.openOutputs': '<i data-lucide="folder" class="btn-icon"></i> Open Output Folder',
    'btn.copy': '<i data-lucide="clipboard-copy" class="btn-icon"></i> Copy to Clipboard',
    'btn.close': 'Close',
    'btn.refreshResults': '<i data-lucide="refresh-cw" class="btn-icon"></i> Refresh',
    'btn.exportCsv': '<i data-lucide="download" class="btn-icon"></i> Export CSV',
    'btn.exportExcel': '<i data-lucide="download" class="btn-icon"></i> Export Excel',
    'btn.exportMethods': '<i data-lucide="file-text" class="btn-icon"></i> Export Methods Text',
    'tour.btnTitle': 'Start guided tour',
    'tour.skip': 'Skip tour',
    'tour.next': 'Next →',
    'tour.done': 'Done',
    'tour.step1.title': '① Input paths',
    'tour.step1.body': 'Set your input image folder (TIFF Z-stack) and output folder here. The atlas file is auto-filled if found.',
    'tour.step2.title': '② Atlas selection',
    'tour.step2.body': 'Brainfast auto-selects the best Allen CCFv3 coronal plane for each slice. Choose hemisphere and pixel size to match your sample.',
    'tour.step3.title': '③ Registration mode',
    'tour.step3.body': 'Affine is fast and robust. Nonlinear (TPS) handles curved or deformed tissue. Adjust the confidence threshold to filter detections.',
    'tour.step4.title': '④ Run & monitor',
    'tour.step4.body': 'Click Run Pipeline. The log and slice progress bar update in real-time. Cancel at any time.',
    'tour.step5.title': '⑤ Results',
    'tour.step5.body': 'Switch to the Results tab after the run. Export to CSV or Excel, view the Garwood CI per region, and copy the Methods paragraph.',
    'coexpr.title': 'Co-expression by Region',
    'coexpr.hint': 'Cell counts per atlas region for each fluorescence channel — only shown when per-channel leaf CSVs exist.',
    'coexpr.th.region': 'Region',
    'coexpr.th.red': 'Red (count)',
    'coexpr.th.green': 'Green (count)',
    'btn.browse': 'Browse',
    'btn.savePreset': '<i data-lucide="save" class="btn-icon"></i> Save Config',
    'btn.loadPreset': '<i data-lucide="folder-open" class="btn-icon"></i> Load Config',
    'btn.autoPick': '<i data-lucide="crosshair" class="btn-icon"></i> Auto-pick Atlas Slice',
    'btn.refreshPreview': '<i data-lucide="image" class="btn-icon"></i> Refresh Preview',
    'btn.extractSlice': '✓ Confirm Layer & Continue',
    'btn.aiAlign': '<i data-lucide="bot" class="btn-icon"></i> AI Landmark Registration',
    'btn.landmarkView': '<i data-lucide="map" class="btn-icon"></i> View Landmark Map',
    'btn.startManual': '<i data-lucide="pin" class="btn-icon"></i> Enter Manual Mode',
    'btn.applyManual': '<i data-lucide="check" class="btn-icon"></i> Apply Manual Landmarks',
    'btn.clearManual': '<i data-lucide="trash-2" class="btn-icon"></i> Clear Manual Points',
    'btn.undo': '<i data-lucide="undo-2" class="btn-icon"></i> Undo',
    'btn.scalebar': '<i data-lucide="ruler" class="btn-icon"></i> Scale',
    'btn.clearAnnotations': '<i data-lucide="trash-2" class="btn-icon"></i> Clear All',
    'btn.exportFigure': '<i data-lucide="download" class="btn-icon"></i> Export Figure',
    'btn.detectPreview': '<i data-lucide="scan" class="btn-icon"></i> Detect Cells',
    'btn.detectParams': '<i data-lucide="settings" class="btn-icon"></i> Detection Parameters',
    'detect.model': 'Model',
    'detect.diameter': 'Diameter (µm)',
    'detect.flowThreshold': 'Flow Threshold',
    'detect.cellprobThreshold': 'Cell Probability',
    'detect.minSize': 'Min Size (px)',
    'detect.gpu': 'GPU',
    'detect.running': 'Running cell detection…',
    'detect.done': '{count} cells detected ({detector})',
    'detect.error': 'Detection failed: {err}',
    'detect.noSlice': 'Select a real slice first',
    'detect.noRuntime': 'Cellpose not available — install cellpose to enable detection',
    'btn.refreshQc': '<i data-lucide="refresh-cw" class="btn-icon"></i> Refresh',
    'btn.regenDemo': '<i data-lucide="settings" class="btn-icon"></i> Regen Demo',
    'btn.oneClickStart': 'Start One-Click Workflow',
    'hint.oneClickFlow': 'Flow: auto-pick atlas → auto registration → manual review / liquify → export.',
    'ch.red': '<span class="channel-dot channel-dot-red"></span> Red',
    'ch.green': '<span class="channel-dot channel-dot-green"></span> Green',
    'ch.farred': '<span class="channel-dot channel-dot-farred"></span> Far-Red',
    'ch.all': '<i data-lucide="layers" class="btn-icon"></i> All Channels',
    'chname.red': 'Red',
    'chname.green': 'Green',
    'chname.farred': 'Far-Red',
    'step1.title': 'Configure File Paths',
    'step1.desc': 'Specify your slice image folder, atlas annotation file, and output directory',
    'step2.title': 'Atlas Preview & Adjustment',
    'step2.desc': 'Verify atlas orientation and slice level before running registration',
    'step3.title': 'AI Registration',
    'step3.desc': 'Automatically detect landmarks and warp the atlas to match the tissue slice',
    'step4.title': 'Run Pipeline',
    'step4.desc': 'Detect cells, deduplicate, map to brain regions, and export results',
    'label.inputDir': 'Input TIFF Folder',
    'label.outputDir': 'Output Folder',
    'label.atlasPath': 'Atlas Annotation File (annotation_25.nii.gz)',
    'label.structPath': 'Brain Region Mapping File (CSV/JSON)',
    'label.workflowMode': 'Workflow Mode',
    'label.sourceFile': 'Source TIFF',
    'label.regScope': 'Registration Scope',
    'opt.modeOneClick': 'One-Click Mode (Recommended)',
    'opt.modePro': 'Professional Mode',
    'opt.scopeSingle': 'Single-layer Registration (single-slice)',
    'opt.scopeWhole': 'Whole-brain Registration (full workflow)',
    'hint.scopeSingle': 'Single-layer mode will include Z-layer selection for 3D TIFF.',
    'hint.scopeWhole': '3D volume will be processed slice-by-slice with auto AP localization.',
    'label.realSlicePath': 'Single Real Slice (preview / registration)',
    'label.atlasLabelPath': 'Single Atlas Slice (auto-generated by Auto-pick)',
    'label.pixelSizeUm': 'Pixel Size (µm/pixel)',
    'label.slicingPlane': 'Slicing Plane',
    'label.rotateAtlas': 'Rotate Atlas (°)',
    'label.flipAtlas': 'Flip Atlas',
    'label.alphaRange': 'Overlay Alpha',
    'label.overlayMode': 'Overlay Mode',
    'label.fitMode': 'Fit Mode',
    'label.majorTopK': 'Major Contour Count (Top-K)',
    'label.minMeanThreshold': 'Min Mean Threshold',
    'label.alignMode': 'Registration Mode',
    'label.channels': 'Fluorescence Channel',
    'label.maxPoints': 'Max Landmark Points',
    'label.minDistance': 'Min Point Distance',
    'label.ransacResidual': 'RANSAC Residual Threshold',
    'hint.pixelSizeUm': 'Fluorescence microscopy: typically ~0.65 µm',
    'hint.rotateAtlas': 'Try 180° if atlas appears upside down',
    'hint.maxPoints': 'Fewer points → faster but less accurate',
    'hint.minDistance': 'Pixels between detected landmarks',
    'hint.ransacResidual': 'Higher → more permissive matching',
    'hint.scopeSingle': 'Single-layer mode will include Z-layer selection for 3D TIFF.',
    'hint.scopeWhole': 'Whole-brain mode registers all Z-slices to the Allen Atlas automatically.',
    'label.hemisphere': 'Hemisphere',
    'opt.hemiAuto': 'Auto-detect (recommended)',
    'opt.hemiFull': 'Full brain (both hemispheres)',
    'opt.hemiLeft': 'Left hemisphere',
    'opt.hemiRightFlipped': 'Right hemisphere (flipped, lateral=left)',
    'hint.hemiAuto': 'Auto-detect tries all orientations and picks the best match.',
    'hint.hemiFull': 'Register as a complete coronal section with both hemispheres.',
    'hint.hemiLeft': 'Left hemisphere only — medial face on right side of image.',
    'hint.hemiRightFlipped': 'Right hemisphere, flipped so lateral cortex is on the left side of the image.',
    'label.atlasVersion': 'Atlas Version',
    'opt.atlasCcfv3': 'CCFv3 (Allen 2017)',
    'opt.atlasCcfv3bbp': 'CCFv3-BBP (Extended)',
    'hint.atlasCcfv3': 'Standard Allen Mouse Brain CCFv3, 25\u00b5m resolution',
    'hint.atlasCcfv3bbp': 'Blue Brain Project extended atlas with olfactory bulb, cerebellum, medulla improvements and averaged Nissl template',
    'label.regMode': 'Registration Mode',
    'opt.regCrossModal': 'Cross-Modal (Default)',
    'opt.regNissl': 'Nissl Template (Single-Modal)',
    'hint.regCrossModal': 'Matches fluorescence edges against atlas annotation edges',
    'hint.regNissl': 'Matches fluorescence against averaged Nissl template for higher quality (requires CCFv3-BBP atlas)',
    'label.targetRegion': 'Target Brain Region',
    'hint.targetRegion': 'Select target region(s) to restrict AP search range (e.g., STN + CP for ChAT)',
    'hint.targetRegionSelected': 'AP search restricted to slices {start}–{end} ({startMm} to {endMm} mm)',
    'hint.targetRegionNone': 'No region selected — full atlas AP range will be searched',
    'hint.channelGuide': 'For multi-channel data: load the reporter channel (C0) for registration. After registration, use Step 4 to process each channel separately.',
    'label.confidenceThreshold': 'Confidence Threshold',
    'hint.confidenceThreshold': 'Filter cell detections by minimum score (0 = keep all, 1 = strictest).',
    'opt.coronal': 'Coronal (default)',
    'opt.sagittal': 'Sagittal',
    'opt.horizontal.plane': 'Horizontal (Axial)',
    'opt.noflip': 'No flip',
    'opt.flipH': 'Horizontal (L-R mirror)',
    'opt.flipV': 'Vertical (U-D mirror)',
    'opt.fill': 'Fill (colored regions)',
    'opt.contour': 'Contour (all boundaries)',
    'opt.contourMajor': 'Contour (major boundaries)',
    'opt.contain': 'Contain (default, preserve aspect ratio)',
    'opt.cover': 'Cover (crop risk)',
    'opt.widthLock': 'Width-lock',
    'opt.heightLock': 'Height-lock',
    'opt.affine': 'Affine (fast, small deformation)',
    'opt.nonlinear': 'Nonlinear (slow, large deformation)',
    'adv.options': 'Advanced Options',
    'adv.params': 'Advanced Parameters',
    'required': 'Required',
    'progress.slicesLabel': 'Slices registered',
    'progress.waiting': 'Waiting to start...',
    'progress.queued': 'Queued...',
    'progress.running': 'Running: {ch}',
    'progress.slices': 'Processing slice {cur} / {total}',
    'progress.eta': 'ETA {eta}',
    'progress.slicesEta': 'Processing slice {cur} / {total} · ETA {eta}',
    'progress.done': 'Done.',
    'progress.cancelled': 'Cancelled.',
    'progress.startFailed': 'Failed to start.',
    'progress.starting': 'Starting...',
    'progress.submitting': 'Submitting...',
    'progress.processing': 'Processing...',
    'toast.folderNotFile': 'Please select a .tif file, not a folder.',
    'toast.pixelSizeMismatch': 'Warning: filename suggests pixel size ~{hint}\u00b5m but current value is {current}\u00b5m. Please verify.',
    'progress.autopickFailed': 'Auto-pick failed',
    'progress.extractingZ': 'Extracting selected Z slice...',
    'progress.usingSlice': 'Using extracted slice: {path}',
    'log.title': 'Live Logs ▶',
    'log.ready': '[ready] Frontend initialized',
    'quality.title': 'Registration Quality',
    'quality.before': 'Before',
    'quality.after': 'After',
    'quality.excellent': 'Excellent',
    'quality.good': 'Good',
    'quality.fair': 'Fair',
    'quality.poor': 'Poor',
    'quality.tip.excellent': 'Registration quality is excellent. Ready for cell counting.',
    'quality.tip.good': 'Good registration. Minor manual correction may help.',
    'quality.tip.fair': 'Fair registration. Manual landmark correction recommended.',
    'quality.tip.poor': 'Poor registration. Please add manual landmarks to improve.',
    'quality.noImprove': 'Alignment did not improve. Try a different mode.',
    'results.title': 'Cell Counts by Brain Region',
    'results.expandDepth': 'Expand to depth:',
    'results.expandAll': 'All',
    'results.total': '{n} regions total',
    'results.filtered': 'Showing {found} of {total} regions',
    'results.expandHint': 'Expand to browse and search',
    'results.tableHint': 'This tree is for browsing hierarchy totals. Use the summary above for interpretation.',
    'compare.title': 'Channel Comparison (Total Cell Count)',
    'compare.multi.title': 'Cross-Sample Region Comparison',
    'compare.multi.hint': 'Enter output directories from multiple runs to compare cell counts across samples.',
    'compare.multi.addDir': '+ Add Directory',
    'compare.multi.run': 'Compare',
    'compare.multi.label': 'Label',
    'compare.multi.dirPlaceholder': 'Output directory path...',
    'compare.multi.empty': 'Enter at least 2 output directories and click Compare.',
    'compare.multi.noData': 'No matching regions found. Check that hierarchy CSV files exist in the selected directories.',
    'history.title': 'Run History',
    'th.region': 'Region Name',
    'th.count': 'Cell Count',
    'th.confidence': 'Confidence',
    'th.pct': '%',
    'th.elongation': 'Elongation',
    'th.area': 'Area (px)',
    'th.intensity': 'Intensity',
    'th.ci': '95% CI',
    'results.morphToggle': 'Show morphology',
    'th.bar': 'Distribution',
    'chart.title': 'Distribution — Analysis Regions',
    'chart.imgTitle': 'Cell Count Summary',
    'chart.apDensityTitle': 'AP-Axis Cell Density Profile',
    'chart.apDensityHint': 'Cell count per atlas AP position — shows injection spread along the anterior-posterior axis.',
    'summary.title': 'Result Snapshot',
    'summary.hint': 'Check scope and mapping coverage before reading regional biology.',
    'summary.sample': 'Sample',
    'summary.scope': 'Scope',
    'summary.mode': 'Counting Mode',
    'summary.detectors': 'Detector',
    'summary.detected': 'Detected Cells',
    'summary.mapped': 'Mapped to Atlas',
    'summary.outside': 'Outside Atlas',
    'summary.regions': 'Mapped Regions',
    'summary.topRegion': 'Top Region',
    'summary.none': 'No summary available yet.',
    'cellconf.title': 'Cell Count Confidence Samples',
    'cellconf.hint': 'Three representative raw slices with the final counted-cell markers overlaid.',
    'cellconf.empty': 'No counted-cell sample images yet.',
    'cellconf.detector': 'Detector',
    'cellconf.cells': 'cells',
    'th.channel': 'Channel',
    'th.total': 'Total Count',
    'reg3d.title': '3D Registration Reports',
    'reg3d.hint': 'Check the final overview first. Open the summary or metadata only when something looks suspicious.',
    'reg3d.empty': 'No 3D registration runs found yet.',
    'reg3d.pipeline': 'Pipeline',
    'reg3d.updated': 'Updated',
    'reg3d.hemisphere': 'Hemisphere',
    'reg3d.target': 'Target',
    'reg3d.staining': 'Staining Rate',
    'reg3d.coverage': 'Atlas Coverage',
    'reg3d.positiveAtlas': 'Positive / Atlas',
    'reg3d.before': 'Before',
    'reg3d.after': 'Final',
    'reg3d.noBefore': 'No pre-refinement overview',
    'reg3d.openSummary': 'Open Summary',
    'reg3d.openMetadata': 'Open Metadata',
    'reg3d.openReport': 'Open HTML Report',
    'reg3d.summaryTitle': '3D Run Summary',
    'reg3d.summaryDesc': 'This is the plain-text summary for the selected 3D registration run.',
    'reg3d.metadataTitle': '3D Run Metadata',
    'reg3d.metadataDesc': 'This JSON contains the paths, metrics, backend parameters, and staining stats for the selected 3D registration run.',
    'reg3d.menu': 'More actions',
    'reg3d.detailInfo': 'Detailed Info',
    'reg3d.deleteBad': 'Delete Bad Report',
    'reg3d.pinReport': 'Pin This Report',
    'reg3d.pinned': 'Pinned',
    'reg3d.pinDone': 'Report pinned to top.',
    'reg3d.deleteDone': 'Report removed from active list.',
    'reg3d.deleteConfirm': 'Move this report out of the active list?',
    'outputs.title': 'Output Files',
    'outputs.hint': 'Click a PNG to preview · Click CSV/JSON to view content',
    'outputs.empty': 'No output files yet',
    'wb3d.status.title': '3D Registration Status',
    'wb3d.status.notice': 'Whole-brain automatic truth comes from the 3D pipeline. The 2D tools elsewhere remain preview and manual-correction helpers only.',
    'wb3d.status.idle': 'Waiting for a whole-brain 3D run',
    'wb3d.status.stage': 'Stage {current}/{total}',
    'wb3d.status.running': 'Running',
    'wb3d.status.done': 'Done',
    'wb3d.status.pending': 'Pending',
    'wb3d.status.failed': 'Failed',
    'wb3d.qc.title': '3D QC Summary',
    'wb3d.qc.loading': 'Loading volume registration QC...',
    'wb3d.qc.empty': 'Volume QC will appear after the 3D pipeline writes volume_registration_qc.csv.',
    'wb3d.slice.title': 'Slice Inspector',
    'wb3d.slice.hint': 'These overlays are exported from the final 3D truth volume. The 2D tools below remain auxiliary.',
    'wb3d.slice.empty': 'No exported 3D slice overlays yet. Run the whole-brain 3D pipeline first.',
    'qc.hint': 'Click any image to enlarge. Generated after running the pipeline.',
    'qc.empty': 'No QC images yet. Please run the pipeline in the Registration Workflow tab first.',
    'qc.annotatedSliceTitle': 'Atlas Registration — Annotated Brain Regions',
    'qc.annotatedSliceHint': 'Lightsheet image with Allen CCFv3 region boundaries and labels. Click to view full size.',
    'qc.bestSliceTitle': 'Registration Slice vs Atlas Registration',
    'qc.bestSliceHint': 'Registered-slice comparison — click to view full resolution',
    'qc.zContinuityTitle': 'AP-Axis Z Continuity',
    'qc.zContinuityHint': 'Atlas AP index per slice — blue=raw, green=smoothed, red=outlier. Outliers may indicate registration errors.',
    'qc.zContinuityOk': 'AP series monotone — no outliers detected',
    'qc.zContinuityWarn': '{n} AP outlier(s) detected — review registration for flagged slices',
    'qc.panelTitle': 'Whole-Brain Registration Overview',
    'qc.panelHint': 'Multi-slice atlas registration panel — click to view full size',
    'tab.manualTiff.title': 'Manual TIFF Check',
    'tab.qc.title': 'Batch QC Review',
    'tab.results.title': 'Results',
    'ph.outputDir': '(default: outputs/)',
    'ph.atlasLabelPath': '(auto-filled by Auto-pick)',
    'ph.regionSearch': 'Search region name...',
    'preview.placeholder': 'Preview will appear here after clicking "Refresh Preview"',
    'align.placeholder': 'Alignment comparison will appear here after running AI registration',
    'manual.title': '<i data-lucide="pen-tool" class="icon-inline"></i> Manual Landmark Correction',
    'manual.desc': 'Click corresponding points on real and atlas slices to add correction landmarks',
    'manual.realSide': 'Real Slice → click to mark point',
    'manual.atlasSide': 'Atlas Slice → click corresponding point',
    'manual.th.real': 'Real (x, y)',
    'manual.th.atlas': 'Atlas (x, y)',
    'manual.pendingReal': 'Real point marked at ({x}, {y}) — now click the matching atlas point',
    'manual.needReal': 'Click on the Real slice first to set a point',
    'manual.pairAdded': '{n} landmark pair(s) set',
    'manual.needImages': 'Please set Real Slice path and Atlas Label path first',
    'manual.enterMode': 'Manual landmark mode: click corresponding points on both images',
    'manual.exitMode': 'Manual landmark mode exited',
    'manual.applyFail': 'Apply failed: {err}',
    'manual.applyOk': '{n} landmark pairs applied. Re-aligning...',
    'toolbar.tools': 'Tools',
    'toolbar.color': 'Color',
    'toolbar.lineWidth': 'Width',
    'lightbox.overlay': 'Overlay Preview',
    'lightbox.compare': 'Before / After Comparison',
    'lightbox.landmark': 'Landmark Map ({n} points)',
    'guide.title': '<i data-lucide="book-open" class="icon-inline"></i> Getting Started Guide',
    'guide.step1': '<strong>Step 1 → Configure Paths:</strong> Click "Browse" to select your TIFF folder, Atlas annotation file, and brain region CSV. Fields marked "Required" must be filled in.',
    'guide.step2': '<strong>Step 2 → Preview Atlas:</strong> Set the slicing plane (usually Coronal), pixel size (default 0.65 µm), click "Auto-pick Atlas Slice", then "Refresh Preview" to check the initial overlay.',
    'guide.step3': '<strong>Step 3 → AI Registration:</strong> Choose the mode (Affine for small deformation; Nonlinear for tears/large warping), then click "AI Landmark Registration". Check the quality panel to confirm SSIM improved.',
    'guide.step4': '<strong>Step 4 → Run Pipeline:</strong> Select the fluorescence channel, then click "Run Pipeline". After completion, switch to the "Results" tab to view per-region cell counts.',
    'guide.step5': '<strong>Export:</strong> In the Results tab, click "Export CSV" for data. Click "Export Methods Text" to get a pre-written Methods paragraph you can paste directly into your paper.',
    'guide.tip': '<i data-lucide="lightbulb" class="icon-inline"></i> Tip: All run parameters are automatically saved to outputs/run_params_YYYYMMDD_HHMMSS.json for reproducibility.',
    'guide.ok': 'Got it → Start Using',
    'guide.workflowHtml': '<ol><li><strong>Step 1 → Configure Paths:</strong> Click "Browse" to select your TIFF folder, Atlas annotation file, and brain region CSV.</li><li><strong>Step 2 → Preview Atlas:</strong> Set slicing plane, pixel size, click "Auto-pick Atlas Slice", then "Refresh Preview".</li><li><strong>Step 3 → AI Registration:</strong> Choose Affine or Nonlinear mode, click "AI Landmark Registration".</li><li><strong>Step 4 → Run Pipeline:</strong> Select fluorescence channel, click "Run Pipeline".</li><li><strong>Export:</strong> In Results tab, "Export CSV" for data, "Export Methods Text" for paper.</li></ol><p class="guide-tip">Tip: All parameters auto-saved to outputs/ for reproducibility.</p>',
    'guide.manualCheckHtml': '<ol><li><strong>Load TIFF:</strong> Browse to your source TIFF file to view it slice by slice.</li><li><strong>Navigate Z:</strong> Use the mouse wheel or slider to scroll through Z layers.</li><li><strong>Manual Count:</strong> Left-click to add count points, right-click to remove. Points are tracked per Z layer.</li><li><strong>Export:</strong> Click "Export" to save your manual count data as CSV.</li></ol><p class="guide-tip">Tip: Use this tab to validate cell detection results against manual counts.</p>',
    'guide.batchQcHtml': '<ol><li><strong>Overview Panel:</strong> Shows the 12-slice demo panel — a quick visual check of atlas registration quality across slices.</li><li><strong>Stats Bar:</strong> Displays registration success rate and mean quality score. Green = good, Yellow = marginal, Red = poor.</li><li><strong>Slice Gallery:</strong> Click any slice to see a detailed raw vs atlas comparison in the lightbox.</li><li><strong>Regenerate:</strong> Click "Regenerate" to refresh demo visuals after re-running registration.</li></ol>',
    'guide.resultsHtml': '<ol><li><strong>Cell Count Table:</strong> Shows per-region cell counts, percentages, and hierarchical structure. Use depth filter to expand/collapse.</li><li><strong>Charts:</strong> Bar + pie chart showing distribution of cells across brain regions.</li><li><strong>Export CSV:</strong> Downloads the full cell count hierarchy table.</li><li><strong>Methods Text:</strong> Generates a paragraph describing your analysis pipeline for paper methods sections.</li></ol><p class="guide-tip">Tip: Toggle "Show hierarchy" for parent region rollup counts.</p>',
    'methods.title': '<i data-lucide="file-text" class="icon-inline"></i> Methods Paragraph',
    'methods.desc': 'The following text is auto-generated from your most recent run parameters. Edit as needed before pasting into your Methods section.',
    'methods.loading': 'Loading...',
    'text.dialog': 'Enter annotation text:',
    'scalebar.dialog': 'Enter scale bar length in µm:',
    'scalebar.invalid': 'Please enter a valid number in µm',
    'toast.browseFail': 'File browser failed to open',
    'toast.validateFail': 'Path validation failed: {issues}',
    'toast.validateOk': '✓ Paths validated',
    'toast.presetSaved': 'Configuration saved',
    'toast.noPreset': 'No saved configuration found',
    'toast.presetLoaded': 'Configuration loaded',
    'toast.previewNeedPath': 'Please set Real Slice path first',
    'toast.previewFailed': 'Preview generation failed',
    'toast.fillModeFallback': 'Fill mode failed — switched to Contour mode',
    'toast.previewUpdated': 'Preview updated',
    'toast.autoPickNeedPath': 'Please set both Real Slice and Atlas paths first',
    'toast.autoPickWaiting': 'Auto-picking best atlas slice...',
    'toast.autoPickFailed': 'Auto-pick failed. See the progress dialog for details.',
    'toast.autoPickSuccess': 'Auto-picked: plane={plane}, Z={z}, score={score}',
    'toast.aspectWarning': 'Aspect ratio mismatch: real={ra}, atlas={aa}. Check pixel size or flip settings.',
    'toast.zDetected': '3D stack detected: {z} slices, {h}×{w} px',
    'toast.zExtracted': 'Extracted Z={z} → {path}',
    'toast.zExtractFail': 'Z extraction failed: {err}',
    'toast.autoLoadPreset': 'Last configuration auto-loaded',
    'toast.methodsFailed': 'Failed to generate Methods text',
    'toast.copyOk': 'Copied to clipboard',
    'toast.copyFailed': 'Copy failed — please copy manually',
    'toast.landmarkNeedPath': 'Please set Real Slice and Atlas Label paths first',
    'toast.landmarkExtracting': 'Extracting landmarks...',
    'toast.landmarkApplyFailed': 'Landmark alignment failed: {err}',
    'toast.landmarkSuccess': '{n} landmark pairs applied',
    'toast.alignUnexpected': 'Unexpected alignment error',
    'toast.alignFailedManualHint': 'Auto-alignment could not find enough landmarks. Please add manual correction points below.',
    'toast.runFailed': 'Pipeline failed: {err}',
    'toast.runStarted': 'Pipeline started: {channels}',
    'toast.runComplete': 'Pipeline completed',
    'toast.cancelOk': 'Pipeline cancelled',
    'toast.cancelNone': 'No pipeline is running',
    'toast.outputsPath': 'Output folder: {path}',
    'toast.qcLoadFailed': 'Failed to load QC images',
    'toast.chooseSourceFirst': 'Please choose source TIFF first.',
    'toast.autoPickPreviewFailed': 'Auto-pick failed, cannot generate preview.',
    'toast.oneClickDone': 'Registration done! Review the result, then click "Run Pipeline" to start cell counting.',
    'toast.3dDetected': '3D detected. The Z selector is now shown below the Start button.',
    'toast.regenStarted': 'Demo visuals regeneration started. Refreshing in 15s...',
    'toast.regenFailed': 'Regen failed: {err}',
    'toast.regenError': 'Regen error: {err}',
    'toast.pixelSizeDetected': 'Pixel size auto-detected from TIFF: {size} µm/px',
    'warn.pixelSizeNotDetected': 'Pixel size not detected from image metadata. Registration quality depends on this value \u2014 please verify or enter manually.',
    'toast.atlasPathNotSet': 'Atlas annotation path not set. Check /api/info defaults.',
    'toast.setRealSliceFirst': 'Please set Real Slice path first.',
    'toast.autoLearnDone': 'Auto-learning finished. Tuned params updated.',
    'toast.autoLearnStarted': 'Auto-learning started in background.',
    'toast.devError': 'Dev Error: {msg}',
    'toast.uncaughtError': 'Uncaught: {msg}',
    'toast.promiseReject': 'Promise Reject: {msg}',
    'hint.zChoose': 'Choose a Z layer below. Click "Extract This Slice" to continue immediately, or click Start again to use the selected Z.',

    'manualCount.title': 'Manual TIFF Check',
    'manualCount.desc': 'Open a source TIFF directly for visual inspection and manual counting. Use the mouse wheel to move through Z slices.',
    'manualCount.source': 'Manual Count Source TIFF',
    'manualCount.zoom': 'Zoom',
    'manualCount.palette': 'Display',
    'manualCount.palette.gray': 'Gray',
    'manualCount.palette.green': 'Green',
    'manualCount.palette.magenta': 'Magenta',
    'manualCount.palette.amber': 'Amber',
    'manualCount.palette.turbo': 'Turbo',
    'manualCount.load': 'Load TIFF',
    'manualCount.export': 'Export Count CSV',
    'manualCount.undo': 'Undo Last Point',
    'manualCount.clearSlice': 'Clear Current Z',
    'manualCount.clearAll': 'Clear All Points',
    'manualCount.z': 'Z',
    'manualCount.sliceCount': 'Current Z',
    'manualCount.totalCount': 'Total',
    'manualCount.placeholder': 'Load a TIFF to start manual counting.',
    'manualCount.help': 'Left click adds a count. Right click removes the nearest count. Mouse wheel moves through Z slices.',
    'manualCount.ready': 'Loaded {name}. Mouse wheel changes Z. Left click adds a point.',
    'manualCount.needPath': 'Please choose a TIFF source first',
    'manualCount.loadFail': 'Failed to load TIFF: {err}',
    'manualCount.noPoints': 'No manual count points to export',
    'manualCount.exported': 'Manual count CSV exported',
    'manualCount.pathChanged': 'Switched TIFF source. Previous manual points were cleared.',
    'manualCount.sliceCleared': 'Cleared points on current Z',
    'manualCount.allCleared': 'Cleared all manual count points',
    'btn.editMasks': 'Edit Masks',
    'maskEditor.title': 'Mask Editor',
    'maskEditor.brushSize': 'Brush Size',
    'maskEditor.opacity': 'Mask Opacity',
    'maskEditor.cells': 'Cells',
    'maskEditor.save': 'Save to Training Set',
    'nav.training': 'Model Training',
    'training.title': 'Cellpose Model Training',
    'training.datasetTitle': 'Training Dataset',
    'training.datasetDesc': 'Annotated images saved from the Mask Editor',
    'training.images': 'Images',
    'training.cells': 'Cells',
    'training.avgPerImage': 'Avg/Image',
    'training.readiness': 'Status',
    'training.refresh': 'Refresh',
    'training.configTitle': 'Training Configuration',
    'training.configDesc': 'Select base model and start training',
    'training.baseModel': 'Base Model',
    'training.modelName': 'Model Name',
    'training.epochs': 'Epochs',
    'training.gpu': 'Use GPU',
    'training.start': 'Start Training',
    'training.cancel': 'Cancel Training',
    'training.progressTitle': 'Training Progress',
    'training.resultTitle': 'Training Complete',
    'training.apply': 'Apply Model',
    'toast.runDetailsFailed': 'Failed to open run details.',
    'outputs.previewDesc': 'Text preview for the selected output file.',
    // ----- Pre-existing i18n gaps (index.html refs without LANGS entries) -----
    'label.pixelSizeQuick': 'Pixel size not auto-detected:',
    'hint.pixelSizeQuick': '(Cleared tissue: ~5 µm, thin sections: ~0.65 µm)',
    'label.configPath': 'Run Config JSON (optional, defaults to template)',
    'label.runName': 'Output Run Name (defaults to input folder name)',
    'btn.exportExcel': '📊 Export Excel',
    // ----- New Sample wizard -----
    'nav.newsample': 'New Sample',
    'newsample.title': 'New Sample — Onboarding Wizard',
    'newsample.hint': 'Point at a folder of TIFF slices (or a multi-page TIFF), let Brainfast read the metadata, then launch the registration pipeline without editing any config files.',
    'newsample.sourcePath': 'Step 1. Source path (directory of slice TIFFs, or a multi-page TIFF)',
    'newsample.inspect': 'Inspect',
    'newsample.step2': 'Step 2.',
    'newsample.step2hint': 'Configure (defaults auto-filled from inspection)',
    'newsample.sampleId': 'Sample ID',
    'newsample.pixelUm': 'XY pixel (µm)',
    'newsample.zUm': 'Z spacing (µm)',
    'newsample.hemi': 'What did you scan?',
    'newsample.hemiRight': 'Right hemisphere · coronal',
    'newsample.hemiLeft': 'Left hemisphere · coronal',
    'newsample.hemiWhole': 'Whole brain · coronal',
    'newsample.browseDir': 'Browse dir',
    'newsample.browseFile': 'Browse file',
    'newsample.extractTitle': 'Multi-page TIFF detected — extract slices first',
    'newsample.extractHint': 'The pipeline runs on a directory of single-page TIFFs. Extract this multi-page file into a slice directory; it will auto-Inspect when done.',
    'newsample.extractOutDir': 'Output dir',
    'newsample.extract': 'Extract slices',
    'newsample.channel': 'Channel',
    'newsample.launch': 'Step 3. Generate config & start pipeline',
    'newsample.addSecondChannel': 'Add second channel',
    'newsample.secondChannelHint': 'Runs only detection (~15 min) on the 2nd channel — reuses the first channel\'s registration. Both channels become overlay-able in 3D Liquify.',
    'newsample.secondSource': '2nd source path (directory or multi-page TIFF)',
    'newsample.secondChannel': '2nd channel',
    'newsample.advanced': 'Advanced (Xu Lab parity)',
    'newsample.antsTransform': 'ANTs transform',
    'newsample.fixedMaxDim': 'Fixed max dim (SyN memory)',
    'newsample.fixedMaxDimHint': 'Downsample fixed side so longest axis \u2264 this before SyN. Empty = full-res.',
    'newsample.axisAlign': 'Pre-align axes via midline fissure',
    'newsample.cellToCcf': 'Xu Lab cell\u2192CCF mapping',
    'ng.title': 'Neuroglancer 3D Viewer',
    'ng.hint': 'Launch a self-hosted Neuroglancer instance to browse the registered volume and CCF annotation overlay in 3D. Requires the neuroglancer extras.',
    'ng.volumePath': 'Volume NIfTI',
    'ng.segPath': 'Segmentation NIfTI',
    'ng.launch': 'Launch Neuroglancer',
    'ng.urlReady': 'Viewer ready:',
    'stitch.title': 'Stitching (TissueCyte)',
    'stitch.hint': 'Stitch raw TissueCyte mosaic tiles into full sections before running the pipeline. Requires the stitching extras.',
    'stitch.inputDir': 'Tile input directory',
    'stitch.outputDir': 'Output directory',
    'stitch.bezierPath': 'Bezier calibration file',
    'stitch.start': 'Start stitching',
    'stitch.jobId': 'Job:',
    'stitch.jobStatus': 'Status:',
    // ----- 3D Liquify -----
    'nav.liquify3d': '3D Liquify',
    'liquify3d.title': '3D Landmark Liquify',
    'liquify3d.hint': 'Click on the overlay: first click marks where the atlas region currently sits, second click marks where the real anatomy actually is.',
    'liquify3d.jobLabel': 'Job ID',
    'liquify3d.reload': 'Reload slice list',
    'liquify3d.noSlices': 'No slices loaded',
    'liquify3d.z': 'z =',
    'liquify3d.modeLabel': 'Next click:',
    'liquify3d.modeAtlas': 'Atlas (where region currently is)',
    'liquify3d.modeReal': 'Real (where it should be)',
    'liquify3d.apply': 'Apply 3D warp (Laplacian)',
    'liquify3d.undo': 'Undo last (Ctrl+Z)',
    'liquify3d.clear': 'Clear all pairs',
    'liquify3d.pairsTitle': 'Landmark pairs',
    'liquify3d.atlasHdr': 'Atlas (y, x)',
    'liquify3d.realHdr': 'Real (y, x)',
    'liquify3d.finalize': 'Finalize & re-export cell counts',
    'liquify3d.qcDone': 'Mark QC done',
    'liquify3d.qcNote': 'QC note',
    'liquify3d.classLabel': 'Class',
    'liquify3d.priorStatus': 'Prior status',
    'liquify3d.saveToPrior': 'Save job → class prior',
    'liquify3d.loadFromPrior': 'Warm-start from prior',
    'liquify3d.priorHeat': 'Coverage heatmap',
    'liquify3d.priorHeatHint': 'Class-prior landmark density across z (taller bar = more contributing samples at that z):',
    'liquify3d.overlay2ndChannel': 'Overlay 2nd channel',
    'liquify3d.overlayChannel': 'Channel:',
    'liquify3d.overlayColor': 'Tint:',
    'liquify3d.overlayOpacity': 'Opacity:',
    'atlas.banner.title': 'Atlas file missing',
    'atlas.banner.bodyAnnotation': 'Brainfast needs the Allen CCFv3 atlas annotation_25.nii.gz to register brain slices. Run the command below or double-click Start_Brainfast.bat to fetch it (~50 MB). Reload once the download finishes.',
    'atlas.banner.bodyStructureOnly': 'Brainfast needs the Allen structure graph to map cells to brain regions. Run the command below or double-click Start_Brainfast.bat to fetch it. Reload once the download finishes.',
    'atlas.banner.retry': 'Recheck',
    'atlas.banner.dismiss': 'Dismiss',
    'atlas.banner.structureAlsoMissing': 'The structure graph is also missing.',
  },
  zh: {
    'nav.workflow': '配准工作流',
    'nav.manualTiff': '手动TIFF检查',
    'nav.qc': '批量QC审查',
    'nav.results': '统计结果',
    'nav.projects': '项目管理',
    'projects.title': '我的项目',
    'projects.create': '+ 新建项目',
    'projects.empty': '暂无项目，请在上方创建。',
    'projects.namePh': '项目名称…',
    'projects.descPh': '描述（可选）',
    'projects.samples': '样本',
    'projects.delete': '删除',
    'sample.run': '加载并运行',
    'sample.status.done': '完成',
    'sample.status.running': '运行中',
    'sample.status.queued': '排队中',
    'sample.status.pending': '待处理',
    'sample.status.error': '错误',
    'sample.addBtn': '+ 添加样本',
    'sample.configPh': '配置文件路径…',
    'sample.inputPh': '输入目录…',
    'sample.namePh': '样本名称…',
    'batch.title': '批处理队列',
    'batch.hint': '队列中的样本会自动逐个处理。',
    'batch.empty': '队列为空。',
    'batch.cancel': '取消',
    'batch.enqueue': '加入队列',
    'status.idle': '空闲',
    'status.running': '运行中...',
    'status.error': '错误',
    'btn.guide': '<i data-lucide="book-open" class="btn-icon"></i> 使用指南',
    'btn.run': '<i data-lucide="play" class="btn-icon"></i> 运行流水线',
    'btn.cancel': '<i data-lucide="x" class="btn-icon"></i> 取消',
    'btn.openOutputs': '<i data-lucide="folder" class="btn-icon"></i> 打开输出目录',
    'btn.copy': '<i data-lucide="clipboard-copy" class="btn-icon"></i> 复制到剪贴板',
    'btn.close': '关闭',
    'btn.refreshResults': '<i data-lucide="refresh-cw" class="btn-icon"></i> 刷新',
    'btn.exportCsv': '<i data-lucide="download" class="btn-icon"></i> 导出CSV',
    'btn.exportMethods': '<i data-lucide="file-text" class="btn-icon"></i> 导出方法段落',
    'errorPanel.title': '错误面板',
    'errorPanel.empty': '当前没有记录到错误。',
    'preflight.title': '运行前检查',
    'preflight.desc.warn': '运行前请先检查下面这些结构化提示。',
    'preflight.desc.error': '下面这些阻断问题需要先修复，才能开始运行。',
    'preflight.back': '返回修改',
    'preflight.continue': '仍然继续',
    'progress.phase.queued': '已排队',
    'progress.phase.ap_selection': 'AP选层',
    'progress.phase.registration': '配准',
    'progress.phase.detection': '检测',
    'progress.phase.dedup': '去重',
    'progress.phase.mapping': '映射',
    'progress.phase.done': '完成',
    'progress.phase.error': '错误',
    'progress.phase.cancelled': '已取消',
    'tour.btnTitle': '开始引导游览',
    'tour.skip': '跳过',
    'tour.next': '下一步 →',
    'tour.done': '完成',
    'tour.step1.title': '① 输入路径',
    'tour.step1.body': '在这里设置输入图像文件夹（TIFF Z-stack）和输出文件夹。若检测到图谱文件则自动填充。',
    'tour.step2.title': '② 图谱选层',
    'tour.step2.body': 'Brainfast 自动为每张切片匹配最佳 Allen CCFv3 冠状面。请根据样本设置半球和像素大小。',
    'tour.step3.title': '③ 配准模式',
    'tour.step3.body': '仿射变换速度快且鲁棒；非线性（TPS）适用于弯曲或变形组织。调整置信度阈值可过滤检测结果。',
    'tour.step4.title': '④ 运行与监控',
    'tour.step4.body': '点击运行流程，日志和切片进度条实时更新。可随时取消。',
    'tour.step5.title': '⑤ 查看结果',
    'tour.step5.body': '运行完成后切换到结果 Tab，可按脑区导出 CSV/Excel，查看 Garwood CI，并复制方法段落用于论文。',
    'coexpr.title': '各通道区域共表达',
    'coexpr.hint': '每个荧光通道在各脑图谱区域的细胞数——仅当存在分通道叶区CSV时显示。',
    'coexpr.th.region': '区域',
    'coexpr.th.red': '红色通道（数量）',
    'coexpr.th.green': '绿色通道（数量）',
    'btn.browse': '浏览',
    'btn.savePreset': '<i data-lucide="save" class="btn-icon"></i> 保存配置',
    'btn.loadPreset': '<i data-lucide="folder-open" class="btn-icon"></i> 加载配置',
    'btn.autoPick': '<i data-lucide="crosshair" class="btn-icon"></i> 自动选取图谱层',
    'btn.refreshPreview': '<i data-lucide="image" class="btn-icon"></i> 刷新预览',
    'btn.extractSlice': '✓ 确认选层并继续',
    'btn.aiAlign': '<i data-lucide="bot" class="btn-icon"></i> AI地标配准',
    'btn.landmarkView': '<i data-lucide="map" class="btn-icon"></i> 查看地标图',
    'btn.startManual': '<i data-lucide="pin" class="btn-icon"></i> 进入手动模式',
    'btn.applyManual': '<i data-lucide="check" class="btn-icon"></i> 应用手动地标',
    'btn.clearManual': '<i data-lucide="trash-2" class="btn-icon"></i> 清除手动点',
    'btn.undo': '<i data-lucide="undo-2" class="btn-icon"></i> 撤销',
    'btn.scalebar': '<i data-lucide="ruler" class="btn-icon"></i> 比例尺',
    'btn.clearAnnotations': '<i data-lucide="trash-2" class="btn-icon"></i> 清除全部',
    'btn.exportFigure': '<i data-lucide="download" class="btn-icon"></i> 导出图片',
    'btn.detectPreview': '<i data-lucide="scan" class="btn-icon"></i> 检测细胞',
    'btn.detectParams': '<i data-lucide="settings" class="btn-icon"></i> 检测参数',
    'detect.model': '模型',
    'detect.diameter': '直径 (µm)',
    'detect.flowThreshold': '流量阈值',
    'detect.cellprobThreshold': '细胞概率',
    'detect.minSize': '最小面积 (px)',
    'detect.gpu': 'GPU',
    'detect.running': '正在检测细胞…',
    'detect.done': '检测到 {count} 个细胞（{detector}）',
    'detect.error': '检测失败：{err}',
    'detect.noSlice': '请先选择真实切片',
    'detect.noRuntime': 'Cellpose 未安装——请安装 cellpose 以启用检测',
    'btn.refreshQc': '<i data-lucide="refresh-cw" class="btn-icon"></i> 刷新',
    'btn.regenDemo': '<i data-lucide="settings" class="btn-icon"></i> 重新生成演示图',
    'btn.oneClickStart': '启动一键工作流',
    'hint.oneClickFlow': '流程：自动选取图谱 → 自动配准 → 手动审查 / 液化校正 → 导出。',
    'ch.red': '<span class="channel-dot channel-dot-red"></span> 红通道',
    'ch.green': '<span class="channel-dot channel-dot-green"></span> 绿通道',
    'ch.farred': '<span class="channel-dot channel-dot-farred"></span> 远红通道',
    'ch.all': '<i data-lucide="layers" class="btn-icon"></i> 全部通道',
    'chname.red': '红通道',
    'chname.green': '绿通道',
    'chname.farred': '远红通道',
    'step1.title': '配置文件路径',
    'step1.desc': '指定切片图像文件夹、图谱标注文件和输出目录',
    'step2.title': '图谱预览与调整',
    'step2.desc': '在配准前验证图谱方向和层面',
    'step3.title': 'AI配准',
    'step3.desc': '自动检测地标并将图谱变形以匹配组织切片',
    'step4.title': '运行流水线',
    'step4.desc': '检测细胞、去重、映射到脑区并导出结果',
    'label.inputDir': '输入TIFF文件夹',
    'label.outputDir': '输出文件夹',
    'label.atlasPath': '图谱标注文件 (annotation_25.nii.gz)',
    'label.structPath': '脑区映射文件 (CSV/JSON)',
    'label.workflowMode': '流程模式',
    'label.sourceFile': '源 TIFF 文件',
    'label.regScope': '配准范围',
    'opt.modeOneClick': '一键模式（推荐）',
    'opt.modePro': '专业模式',
    'opt.scopeSingle': '单层配准（单切片）',
    'opt.scopeWhole': '全脑配准（完整流程）',
    'hint.scopeSingle': '单层模式下，3D TIFF 将提供 Z 层选择。',
    'hint.scopeWhole': '3D 体数据将逐层配准并自动定位 AP 位置。',
    'label.realSlicePath': '真实切片（预览/配准）',
    'label.atlasLabelPath': '图谱标签层（由自动选取生成）',
    'label.pixelSizeUm': '像素尺寸 (µm/像素)',
    'label.slicingPlane': '切片方向',
    'label.rotateAtlas': '旋转图谱 (°)',
    'label.flipAtlas': '翻转图谱',
    'label.alphaRange': '叠加透明度',
    'label.overlayMode': '叠加模式',
    'label.fitMode': '适配模式',
    'label.majorTopK': '主要轮廓数量 (Top-K)',
    'label.minMeanThreshold': '最低均值阈値',
    'label.alignMode': '配准模式',
    'label.channels': '荧光通道',
    'label.maxPoints': '最大地标点数',
    'label.minDistance': '地标最小间距',
    'label.ransacResidual': 'RANSAC残差阈値',
    'hint.pixelSizeUm': '荧光显微镜：通常约 0.65 µm',
    'hint.rotateAtlas': '若图谱上下颠倒，尝试旋转180°',
    'hint.maxPoints': '点数越少 → 越快但精度较低',
    'hint.minDistance': '检测地标之间的像素距离',
    'hint.ransacResidual': '值越大 → 匹配容忍度越高',
    'hint.scopeSingle': '单层模式将为3D TIFF提供Z层选择。',
    'hint.scopeWhole': '全脑模式自动将所有Z切片配准到Allen Atlas。',
    'label.hemisphere': '半球方向',
    'opt.hemiAuto': '自动检测（推荐）',
    'opt.hemiFull': '全脑（双半球）',
    'opt.hemiLeft': '左半球',
    'opt.hemiRightFlipped': '右半球（翻转，外侧在左）',
    'hint.hemiAuto': '自动检测会尝试所有朝向并选择最佳匹配。',
    'hint.hemiFull': '作为完整冠状切面配准，包含双侧半球。',
    'hint.hemiLeft': '仅左半球——内侧面在图像右侧。',
    'hint.hemiRightFlipped': '右半球，翻转使外侧皮质在图像左侧。',
    'label.atlasVersion': '图谱版本',
    'opt.atlasCcfv3': 'CCFv3 (Allen 2017)',
    'opt.atlasCcfv3bbp': 'CCFv3-BBP (扩展版)',
    'hint.atlasCcfv3': '标准Allen小鼠脑图谱CCFv3，25µm分辨率',
    'hint.atlasCcfv3bbp': 'Blue Brain Project扩展图谱：补全嗅球、小脑、延髓注释，含734脑平均Nissl模板',
    'label.regMode': '配准模式',
    'opt.regCrossModal': '跨模态配准（默认）',
    'opt.regNissl': 'Nissl模板配准（单模态）',
    'hint.regCrossModal': '荧光边缘 vs 图谱注释边缘匹配',
    'hint.regNissl': '荧光图像 vs 平均Nissl模板匹配，配准质量更高（需要CCFv3-BBP图谱）',
    'label.targetRegion': '目标脑区',
    'hint.targetRegion': '选择目标脑区以限制AP搜索范围（如ChAT看STN + CP）',
    'hint.targetRegionSelected': 'AP搜索范围限制为切片 {start}–{end}（{startMm} 至 {endMm} mm）',
    'hint.targetRegionNone': '未选择脑区——将搜索全部AP范围',
    'hint.channelGuide': '多通道数据：请加载reporter通道（C0）用于配准。配准完成后，在步骤4中分别处理各通道。',
    'label.confidenceThreshold': '置信度阈值',
    'hint.confidenceThreshold': '按最低置信度过滤检测结果（0 = 保留全部，1 = 最严格）。',
    'opt.coronal': '冠状面（默认）',
    'opt.sagittal': '射状面',
    'opt.horizontal.plane': '水平面（轴位）',
    'opt.noflip': '不翻转',
    'opt.flipH': '水平翻转（左右镜像）',
    'opt.flipV': '垂直翻转（上下镜像）',
    'opt.fill': '填充（区域着色）',
    'opt.contour': '轮廓（所有边界）',
    'opt.contourMajor': '轮廓（主要边界）',
    'opt.contain': '适配（默认，保持宽高比）',
    'opt.cover': '覆盖（有裁剪风险）',
    'opt.widthLock': '锁定宽度',
    'opt.heightLock': '锁定高度',
    'opt.affine': '仿射变换（快速，适合小变形）',
    'opt.nonlinear': '非线性（较慢，适合大变形）',
    'adv.options': '高级选项',
    'adv.params': '高级参数',
    'required': '必填',
    'progress.slicesLabel': '已配准切片',
    'progress.waiting': '等待开始...',
    'progress.queued': '已排队...',
    'progress.running': '运行中：{ch}',
    'progress.slices': '处理切片 {cur} / {total}',
    'progress.eta': '预计剩余 {eta}',
    'progress.slicesEta': '处理切片 {cur} / {total} · 预计剩余 {eta}',
    'progress.done': '完成。',
    'progress.cancelled': '已取消。',
    'progress.startFailed': '启动失败。',
    'progress.starting': '正在启动...',
    'progress.submitting': '正在提交...',
    'progress.processing': '处理中...',
    'toast.folderNotFile': '请选择 .tif 文件，而非文件夹。',
    'toast.pixelSizeMismatch': '警告：文件名提示像素尺寸约为 {hint}\u00b5m，但当前值为 {current}\u00b5m，请确认。',
    'progress.autopickFailed': '自动选取失败',
    'progress.extractingZ': '正在提取选定的Z层...',
    'progress.usingSlice': '正在使用提取的切片：{path}',
    'log.title': '实时日志 ▶',
    'log.ready': '[就绪] 前端已初始化',
    'quality.title': '配准质量',
    'quality.before': '配准前',
    'quality.after': '配准后',
    'quality.excellent': '优秀',
    'quality.good': '良好',
    'quality.fair': '一般',
    'quality.poor': '较差',
    'quality.tip.excellent': '配准质量优秀，可以直接进行细胞计数。',
    'quality.tip.good': '配准良好，少量手动校正可能有帮助。',
    'quality.tip.fair': '配准一般，建议添加手动地标进行校正。',
    'quality.tip.poor': '配准较差，请添加手动地标改善配准。',
    'quality.noImprove': '对齐未改善，请尝试其他模式。',
    'results.title': '脑区细胞计数',
    'results.expandDepth': '展开到层级：',
    'results.expandAll': '全部',
    'results.total': '共 {n} 个脑区',
    'results.filtered': '显示 {found} / {total} 个脑区',
    'results.expandHint': '展开后查看和搜索',
    'results.tableHint': '这张树表用于浏览层级累计值；真正用于解释分布的请以上方摘要和图表为准。',
    'compare.title': '通道比较（细胞总数）',
    'compare.multi.title': '跨样本脑区细胞数对比',
    'compare.multi.hint': '输入多个运行结果目录，对比各脑区的细胞计数。',
    'compare.multi.addDir': '+ 添加目录',
    'compare.multi.run': '开始对比',
    'compare.multi.label': '标签',
    'compare.multi.dirPlaceholder': '输出目录路径...',
    'compare.multi.empty': '请输入至少2个输出目录后点击"开始对比"。',
    'compare.multi.noData': '未找到匹配脑区。请检查所选目录中是否存在层次CSV文件。',
    'history.title': '运行历史',
    'th.region': '脑区名称',
    'th.count': '细胞计数',
    'th.pct': '占比',
    'th.bar': '分布',
    'th.elongation': '细胞延伸度',
    'th.area': '面积(px)',
    'th.intensity': '荧光强度',
    'th.ci': '95% 置信区间',
    'results.morphToggle': '显示形态特征',
    'chart.title': '细胞分布 — 分析脑区',
    'chart.imgTitle': '细胞计数汇总',
    'chart.apDensityTitle': 'AP轴细胞密度分布',
    'chart.apDensityHint': '每个图谱AP坐标的细胞数 — 反映注射点沿前后轴的扩散范围。',
    'summary.title': '结果摘要',
    'summary.hint': '先确认范围和图谱映射覆盖，再解读脑区分布。',
    'summary.sample': '样本',
    'summary.scope': '范围',
    'summary.mode': '计数模式',
    'summary.detectors': '检测器',
    'summary.detected': '检测到的细胞',
    'summary.mapped': '成功映射到图谱',
    'summary.outside': '落在图谱外',
    'summary.regions': '映射到的脑区数',
    'summary.topRegion': '最高脑区',
    'summary.none': '当前还没有可用摘要。',
    'cellconf.title': '细胞计数置信样本',
    'cellconf.hint': '展示 3 张代表性真实切片，并叠加最终计入统计的细胞标记点。',
    'cellconf.empty': '暂无细胞计数样本图。',
    'cellconf.detector': '检测器',
    'cellconf.cells': '个细胞',
    'th.confidence': '置信度',
    'th.channel': '通道',
    'th.total': '总计数',
    'reg3d.title': '3D配准报告',
    'reg3d.hint': '先看最终总览图；只有当结果可疑时，再打开摘要或元数据。',
    'reg3d.empty': '还没有发现3D配准结果。',
    'reg3d.pipeline': '流程',
    'reg3d.updated': '更新时间',
    'reg3d.hemisphere': '半脑',
    'reg3d.target': '目标分辨率',
    'reg3d.staining': '染色率',
    'reg3d.coverage': '图谱覆盖率',
    'reg3d.positiveAtlas': '阳性/图谱',
    'reg3d.before': '细化前',
    'reg3d.after': '最终结果',
    'reg3d.noBefore': '没有细化前总览图',
    'reg3d.openSummary': '打开摘要',
    'reg3d.openMetadata': '打开元数据',
    'reg3d.openReport': '打开HTML报告',
    'reg3d.summaryTitle': '3D运行摘要',
    'reg3d.summaryDesc': '这是当前3D配准结果的纯文本摘要。',
    'reg3d.metadataTitle': '3D运行元数据',
    'reg3d.metadataDesc': '这个JSON包含当前3D配准结果的路径、指标、后端参数和染色率。',
    'reg3d.menu': '更多操作',
    'reg3d.detailInfo': '详细信息',
    'reg3d.deleteBad': '删除不良报告',
    'reg3d.pinReport': '置顶该报告',
    'reg3d.pinned': '已置顶',
    'reg3d.pinDone': '报告已置顶。',
    'reg3d.deleteDone': '报告已从当前列表移除。',
    'reg3d.deleteConfirm': '确认将该报告移出当前列表吗？',
    'outputs.title': '输出文件',
    'outputs.hint': '点击PNG预览 · 点击CSV/JSON查看内容',
    'outputs.empty': '暂无输出文件',
    'wb3d.status.title': '3D配准状态',
    'wb3d.status.notice': '全脑自动真值来自3D流水线。其他2D工具仅作为预览和手动修正辅助。',
    'wb3d.status.idle': '等待全脑3D运行',
    'wb3d.status.stage': '阶段 {current}/{total}',
    'wb3d.status.running': '运行中',
    'wb3d.status.done': '完成',
    'wb3d.status.pending': '等待中',
    'wb3d.status.failed': '失败',
    'wb3d.qc.title': '3D QC摘要',
    'wb3d.qc.loading': '正在加载体素配准QC...',
    'wb3d.qc.empty': '当3D流水线写入 volume_registration_qc.csv 后，这里会显示体素级QC摘要。',
    'wb3d.slice.title': '切片检查器',
    'wb3d.slice.hint': '这些叠加图来自最终的3D真值体。下面的2D工具仅作为辅助。',
    'wb3d.slice.empty': '尚无导出的3D切片叠加图，请先运行全脑3D流水线。',
    'qc.hint': '点击图片可放大。运行流水线后生成。',
    'qc.empty': '暂无QC图片，请先在”配准工作流”标签页运行流水线。',
    'qc.annotatedSliceTitle': '图谱配准 — 脑区标注示例',
    'qc.annotatedSliceHint': '光片图像叠加 Allen CCFv3 脑区边界与标签。点击查看大图。',
    'qc.bestSliceTitle': '配准切片 vs 图谱配准',
    'qc.bestSliceHint': '配准后切片左右对比图 — 点击查看原始分辨率',
    'qc.zContinuityTitle': 'AP轴Z连续性检测',
    'qc.zContinuityHint': '每切片图谱AP坐标 — 蓝=原始，绿=平滑，红=异常。异常点可能表示配准错误。',
    'qc.zContinuityOk': 'AP序列单调 — 未检测到异常',
    'qc.zContinuityWarn': '检测到 {n} 个AP异常 — 请检查标红切片的配准结果',
    'qc.panelTitle': '全脑配准总览',
    'qc.panelHint': '多切片图谱配准面板 — 点击查看大图',
    'tab.manualTiff.title': '手动TIFF检查',
    'tab.qc.title': '批量QC审查',
    'tab.results.title': '统计结果',
    'ph.outputDir': '（默认：outputs/）',
    'ph.atlasLabelPath': '（由自动选取自动填充）',
    'ph.regionSearch': '搜索脑区名称...',
    'preview.placeholder': '点击“刷新预览”后，预览将显示在此处',
    'align.placeholder': '运行AI配准后，对比图将显示在此处',
    'manual.title': '<i data-lucide="pen-tool" class="icon-inline"></i> 手动地标校正',
    'manual.desc': '在真实切片和图谱切片上点击对应位置，添加校正地标',
    'manual.realSide': '真实切片 → 点击标记位置',
    'manual.atlasSide': '图谱切片 → 点击对应位置',
    'manual.th.real': '真实 (x, y)',
    'manual.th.atlas': '图谱 (x, y)',
    'manual.pendingReal': '已标记真实点 ({x}, {y})，请在图谱上点击对应位置',
    'manual.needReal': '请先在真实切片上点击标记一个位置',
    'manual.pairAdded': '已添加 {n} 对地标',
    'manual.needImages': '请先设置真实切片路径和图谱标签路径',
    'manual.enterMode': '手动地标模式：在两张图上点击对应位置',
    'manual.exitMode': '已退出手动地标模式',
    'manual.applyFail': '应用失败：{err}',
    'manual.applyOk': '已应用 {n} 对地标，正在重新对齐...',
    'toolbar.tools': '工具',
    'toolbar.color': '颜色',
    'toolbar.lineWidth': '线宽',
    'lightbox.overlay': '叠加预览',
    'lightbox.compare': '配准前后对比',
    'lightbox.landmark': '地标图（{n} 个点）',
    'guide.title': '<i data-lucide="book-open" class="icon-inline"></i> 使用指南',
    'guide.step1': '<strong>第1步 → 配置路径：</strong>点击“浏览”选择TIFF文件夹、图谱标注文件和脑区CSV。',
    'guide.step2': '<strong>第2步 → 预览图谱：</strong>设置切片方向（通常选冠状面）、像素尺寸，点击“自动选取图谱层”，再点击“刷新预览”。',
    'guide.step3': '<strong>第3步 → AI配准：</strong>选择配准模式（仿射或非线性），点击“AI地标配准”，查看质量面板SSIM是否改善。',
    'guide.step4': '<strong>第4步 → 运行流水线：</strong>选择荧光通道，点击“运行流水线”，完成后切换到“统计结果”标签查看脑区细胞计数。',
    'guide.step5': '<strong>导出：</strong>在结果标签点击“导出CSV”获取数据，点击“导出方法段落”获得可直接粘贴到论文的方法描述。',
    'guide.tip': '<i data-lucide="lightbulb" class="icon-inline"></i> 提示：所有运行参数自动保存到 outputs/run_params_YYYYMMDD_HHMMSS.json，便于复现。',
    'guide.ok': '明白了 → 开始使用',
    'guide.workflowHtml': '<ol><li><strong>步骤1 → 配置路径：</strong>点击"浏览"选择TIFF文件夹、Atlas注释文件和脑区CSV。</li><li><strong>步骤2 → 预览Atlas：</strong>设置切面方向、像素大小，点击"自动选择Atlas切片"，然后"刷新预览"。</li><li><strong>步骤3 → AI配准：</strong>选择仿射或非线性模式，点击"AI地标配准"。</li><li><strong>步骤4 → 运行流程：</strong>选择荧光通道，点击"运行流程"。</li><li><strong>导出：</strong>在结果标签中，"导出CSV"获取数据，"导出方法文本"用于论文。</li></ol><p class="guide-tip">提示：所有参数自动保存到outputs/用于复现。</p>',
    'guide.manualCheckHtml': '<ol><li><strong>加载TIFF：</strong>浏览并选择源TIFF文件，逐层查看。</li><li><strong>导航Z层：</strong>使用鼠标滚轮或滑块在Z层间滚动。</li><li><strong>手动计数：</strong>左键添加计数点，右键删除。每个Z层分别跟踪。</li><li><strong>导出：</strong>点击"导出"将手动计数数据保存为CSV。</li></ol><p class="guide-tip">提示：使用此标签验证细胞检测结果与手动计数的一致性。</p>',
    'guide.batchQcHtml': '<ol><li><strong>概览面板：</strong>显示12切片演示面板——快速检查全脑配准质量。</li><li><strong>统计栏：</strong>显示配准成功率和平均质量分数。绿色=好，黄色=一般，红色=差。</li><li><strong>切片图库：</strong>点击任意切片可在灯箱中查看详细的原图vs Atlas对比。</li><li><strong>重新生成：</strong>重新运行配准后，点击"重新生成"刷新演示图。</li></ol>',
    'guide.resultsHtml': '<ol><li><strong>细胞计数表：</strong>显示各脑区的细胞计数、百分比和层级结构。使用深度过滤展开/折叠。</li><li><strong>图表：</strong>柱状图+饼图显示细胞在各脑区的分布。</li><li><strong>导出CSV：</strong>下载完整的分层细胞计数表。</li><li><strong>方法文本：</strong>生成描述分析流程的段落，用于论文方法部分。</li></ol><p class="guide-tip">提示：切换"显示层级"查看父区域汇总计数。</p>',
    'methods.title': '<i data-lucide="file-text" class="icon-inline"></i> 方法段落',
    'methods.desc': '以下文本根据最近一次运行参数自动生成，粘贴到方法部分前请自行修改。',
    'methods.loading': '加载中...',
    'text.dialog': '请输入标注文字：',
    'scalebar.dialog': '请输入比例尺长度（µm）：',
    'scalebar.invalid': '请输入有效的µm数値',
    'toast.browseFail': '文件浏览器打开失败',
    'toast.validateFail': '路径验证失败：{issues}',
    'toast.validateOk': '✓ 路径验证通过',
    'toast.presetSaved': '配置已保存',
    'toast.noPreset': '未找到已保存的配置',
    'toast.presetLoaded': '配置已加载',
    'toast.previewNeedPath': '请先设置真实切片路径',
    'toast.previewFailed': '预览生成失败',
    'toast.fillModeFallback': '填充模式失败，已切换到轮廓模式',
    'toast.previewUpdated': '预览已更新',
    'toast.autoPickNeedPath': '请先设置真实切片路径和图谱路径',
    'toast.autoPickWaiting': '正在自动选取最佳图谱层...',
    'toast.autoPickFailed': '自动选取失败。请查看进度对话框了解详情。',
    'toast.autoPickSuccess': '自动选取完成：平面={plane}，Z={z}，评分={score}',
    'toast.aspectWarning': '宽高比不匹配：真实={ra}，图谱={aa}。请检查像素尺寸或翻转设置。',
    'toast.zDetected': '检测到3D数据：{z}层，{h}×{w} px',
    'toast.zExtracted': '已提取 Z={z} → {path}',
    'toast.zExtractFail': 'Z层提取失败：{err}',
    'toast.autoLoadPreset': '已自动加载上次配置',
    'toast.methodsFailed': '方法段落生成失败',
    'toast.copyOk': '已复制到剪贴板',
    'toast.copyFailed': '复制失败，请手动复制',
    'toast.landmarkNeedPath': '请先设置真实切片路径和图谱标签路径',
    'toast.landmarkExtracting': '正在提取地标...',
    'toast.landmarkApplyFailed': '地标对齐失败：{err}',
    'toast.landmarkSuccess': '已应用 {n} 对地标',
    'toast.alignUnexpected': '对齐时发生意外错误',
    'toast.alignFailedManualHint': '\u81ea\u52a8\u914d\u51c6\u672a\u80fd\u627e\u5230\u8db3\u591f\u7684\u7279\u5f81\u70b9\u3002\u8bf7\u5728\u4e0b\u65b9\u624b\u52a8\u6dfb\u52a0\u6821\u6b63\u70b9\u3002',
    'toast.runFailed': '流水线失败：{err}',
    'toast.runStarted': '流水线已启动：{channels}',
    'toast.runComplete': '流水线已完成',
    'toast.cancelOk': '流水线已取消',
    'toast.cancelNone': '当前没有运行中的流水线',
    'toast.outputsPath': '输出目录：{path}',
    'toast.qcLoadFailed': 'QC图片加载失败',
    'toast.chooseSourceFirst': '请先选择源TIFF文件。',
    'toast.autoPickPreviewFailed': '自动选取失败，无法生成预览。',
    'toast.oneClickDone': '配准完成！检查结果后，点击"运行流水线"开始细胞计数。',
    'toast.3dDetected': '检测到3D数据。Z层选择器已显示在开始按钮下方。',
    'toast.regenStarted': '正在重新生成演示图，15秒后刷新...',
    'toast.regenFailed': '重新生成失败：{err}',
    'toast.regenError': '重新生成出错：{err}',
    'toast.pixelSizeDetected': '已从TIFF自动检测像素尺寸：{size} µm/px',
    'warn.pixelSizeNotDetected': '未从图像元数据中检测到像素尺寸。配准质量取决于此值——请核实或手动输入。',
    'toast.atlasPathNotSet': '未设置Atlas标注路径。请检查 /api/info 默认值。',
    'toast.setRealSliceFirst': '请先设置切片图像路径。',
    'toast.autoLearnDone': '自动学习完成，参数已更新。',
    'toast.autoLearnStarted': '自动学习已在后台启动。',
    'toast.devError': '开发错误：{msg}',
    'toast.uncaughtError': '未捕获异常：{msg}',
    'toast.promiseReject': 'Promise拒绝：{msg}',
    'hint.zChoose': '请选择Z层。点击"提取此切片"立即继续，或再次点击开始使用所选Z层。',

    'manualCount.title': '手动TIFF检查',
    'manualCount.desc': '直接打开源TIFF进行人眼检查和手动计数，可用鼠标滚轮切换Z层。',
    'manualCount.source': '手动检查源TIFF',
    'manualCount.zoom': '缩放',
    'manualCount.palette': '显示',
    'manualCount.palette.gray': '灰度',
    'manualCount.palette.green': '绿色',
    'manualCount.palette.magenta': '洋红',
    'manualCount.palette.amber': '琥珀',
    'manualCount.palette.turbo': 'Turbo 伪彩',
    'manualCount.load': '加载TIFF',
    'manualCount.export': '导出计数CSV',
    'manualCount.undo': '撤销上一个点',
    'manualCount.clearSlice': '清空当前Z层',
    'manualCount.clearAll': '清空全部点',
    'manualCount.z': 'Z',
    'manualCount.sliceCount': '当前Z层',
    'manualCount.totalCount': '总数',
    'manualCount.placeholder': '加载TIFF后即可开始手动检查。',
    'manualCount.help': '左键添加计数点，右键删除最近的点，鼠标滚轮切换Z层。',
    'manualCount.ready': '已加载 {name}。可滚轮切换Z层，左键添加计数点。',
    'manualCount.needPath': '请先选择一个TIFF源文件',
    'manualCount.loadFail': '加载TIFF失败：{err}',
    'manualCount.noPoints': '当前没有可导出的手动计数点',
    'manualCount.exported': '手动计数CSV已导出',
    'manualCount.pathChanged': '已切换TIFF源文件，之前的手动计数点已清空。',
    'manualCount.sliceCleared': '已清空当前Z层的计数点',
    'manualCount.allCleared': '已清空全部手动计数点',
    'btn.editMasks': '编辑掩码',
    'maskEditor.title': '掩码编辑器',
    'maskEditor.brushSize': '画笔大小',
    'maskEditor.opacity': '掩码透明度',
    'maskEditor.cells': '细胞',
    'maskEditor.save': '保存到训练集',
    'nav.training': '模型训练',
    'training.title': 'Cellpose 模型训练',
    'training.datasetTitle': '训练数据集',
    'training.datasetDesc': '从掩码编辑器保存的标注图像',
    'training.images': '图像',
    'training.cells': '细胞',
    'training.avgPerImage': '平均/图',
    'training.readiness': '状态',
    'training.refresh': '刷新',
    'training.configTitle': '训练配置',
    'training.configDesc': '选择基础模型并开始训练',
    'training.baseModel': '基础模型',
    'training.modelName': '模型名称',
    'training.epochs': '训练轮数',
    'training.gpu': '使用GPU',
    'training.start': '开始训练',
    'training.cancel': '取消训练',
    'training.progressTitle': '训练进度',
    'training.resultTitle': '训练完成',
    'training.apply': '应用模型',
    'toast.runDetailsFailed': '打开运行详情失败。',
    'outputs.previewDesc': '所选输出文件的文本预览。',
    // ----- 补充老版 UI 缺失的 i18n 键 -----
    'label.pixelSizeQuick': '像素尺寸未自动检测：',
    'hint.pixelSizeQuick': '（清脑组织：约 5 µm；薄切片：约 0.65 µm）',
    'label.configPath': '运行配置 JSON（可选，默认使用模板）',
    'label.runName': '输出运行名称（默认使用输入文件夹名称）',
    'btn.exportExcel': '📊 导出 Excel',
    // ----- 新样本向导 -----
    'nav.newsample': '新样本',
    'newsample.title': '新样本 — 引导向导',
    'newsample.hint': '指向一个 TIFF 切片文件夹（或多页 TIFF），让 Brainfast 读取元数据，无需编辑配置文件即可启动配准管线。',
    'newsample.sourcePath': '步骤 1。源路径（切片 TIFF 文件夹，或多页 TIFF）',
    'newsample.inspect': '检查',
    'newsample.step2': '步骤 2。',
    'newsample.step2hint': '配置（默认值已根据检查结果自动填充）',
    'newsample.sampleId': '样本 ID',
    'newsample.pixelUm': 'XY 像素 (µm)',
    'newsample.zUm': 'Z 间距 (µm)',
    'newsample.hemi': '你扫的是哪边？',
    'newsample.hemiRight': '右半球 · 冠状切片',
    'newsample.hemiLeft': '左半球 · 冠状切片',
    'newsample.hemiWhole': '完整脑 · 冠状切片',
    'newsample.browseDir': '浏览文件夹',
    'newsample.browseFile': '浏览文件',
    'newsample.extractTitle': '检测到多页 TIFF —— 需要先拆成切片',
    'newsample.extractHint': '后续管线只接受单页 TIFF 切片目录。点 Extract 把多页文件拆成切片目录，拆完会自动重新 Inspect。',
    'newsample.extractOutDir': '输出目录',
    'newsample.extract': '拆切片',
    'newsample.channel': '通道',
    'newsample.launch': '步骤 3。生成配置并启动管线',
    'newsample.addSecondChannel': '添加第二通道',
    'newsample.secondChannelHint': '仅对第二通道跑检测（约 15 分钟）— 复用第一通道的配准结果。两个通道都能在 3D Liquify 里叠加查看。',
    'newsample.secondSource': '第二通道源路径（目录或多页 TIFF）',
    'newsample.secondChannel': '第二通道',
    'newsample.advanced': '高级（与 Xu Lab 对齐）',
    'newsample.antsTransform': 'ANTs 变换类型',
    'newsample.fixedMaxDim': '固定侧最大尺寸（SyN 省内存）',
    'newsample.fixedMaxDimHint': '将固定侧最长轴降采样到该值再跑 SyN。留空则原分辨率。',
    'newsample.axisAlign': '按中线裂预对齐主轴',
    'newsample.cellToCcf': 'Xu Lab 细胞\u2192CCF 映射',
    'ng.title': 'Neuroglancer 三维查看器',
    'ng.hint': '启动本地 Neuroglancer 实例，在浏览器里 3D 浏览配准体积 + CCF 注释叠加。需要装 neuroglancer 附加依赖。',
    'ng.volumePath': '体积 NIfTI',
    'ng.segPath': '分割 NIfTI',
    'ng.launch': '启动 Neuroglancer',
    'ng.urlReady': '查看器就绪：',
    'stitch.title': '拼接（TissueCyte）',
    'stitch.hint': '把 TissueCyte 原始瓦片拼成完整切片，再跑后续管线。需要装 stitching 附加依赖。',
    'stitch.inputDir': '瓦片输入目录',
    'stitch.outputDir': '输出目录',
    'stitch.bezierPath': 'Bezier 标定文件',
    'stitch.start': '开始拼接',
    'stitch.jobId': '任务 ID：',
    'stitch.jobStatus': '状态：',
    // ----- 3D 液化 -----
    'nav.liquify3d': '3D 液化',
    'liquify3d.title': '3D 地标液化',
    'liquify3d.hint': '在叠图上点击：第一次标记图谱区域当前位置，第二次标记实际解剖位置。',
    'liquify3d.jobLabel': '作业 ID',
    'liquify3d.reload': '重新加载切片列表',
    'liquify3d.noSlices': '尚未加载切片',
    'liquify3d.z': 'z =',
    'liquify3d.modeLabel': '下一次点击：',
    'liquify3d.modeAtlas': '图谱（区域当前所在）',
    'liquify3d.modeReal': '实际（应该在的位置）',
    'liquify3d.apply': '应用 3D 形变（Laplacian）',
    'liquify3d.undo': '撤销最近一对（Ctrl+Z）',
    'liquify3d.clear': '清除所有点对',
    'liquify3d.pairsTitle': '地标点对',
    'liquify3d.atlasHdr': '图谱 (y, x)',
    'liquify3d.realHdr': '实际 (y, x)',
    'liquify3d.finalize': '完成并重新导出细胞计数',
    'liquify3d.qcDone': '标记 QC 完成',
    'liquify3d.qcNote': 'QC 备注',
    'liquify3d.classLabel': '类别',
    'liquify3d.priorStatus': '先验状态',
    'liquify3d.saveToPrior': '把作业存入类别先验',
    'liquify3d.loadFromPrior': '用先验热启动',
    'liquify3d.priorHeat': '覆盖热力图',
    'liquify3d.priorHeatHint': '类先验地标在 z 方向的密度分布（柱越高 = 该 z 位置贡献样本越多）：',
    'liquify3d.overlay2ndChannel': '叠加第二通道',
    'liquify3d.overlayChannel': '通道:',
    'liquify3d.overlayColor': '颜色:',
    'liquify3d.overlayOpacity': '透明度:',
    'atlas.banner.title': '缺少图谱文件',
    'atlas.banner.bodyAnnotation': 'Brainfast 需要 Allen CCFv3 图谱 annotation_25.nii.gz 才能完成脑片配准。请运行下方命令，或双击 Start_Brainfast.bat 自动下载（约 50 MB）。下载完成后请刷新页面。',
    'atlas.banner.bodyStructureOnly': 'Brainfast 需要 Allen 脑区结构图谱才能完成细胞-脑区映射。请运行下方命令，或双击 Start_Brainfast.bat 下载。下载完成后请刷新页面。',
    'atlas.banner.retry': '重新检查',
    'atlas.banner.dismiss': '关闭',
    'atlas.banner.structureAlsoMissing': '同时缺少脑区结构文件。',
  },
};

function t(key, vars) {
  const dict = currentLang === 'zh' ? LANGS.zh : LANGS.en;
  let str = dict[key];
  if (str === undefined) str = LANGS.en[key];
  if (str === undefined) str = key;
  if (vars) {
    Object.entries(vars).forEach(function(kv) {
      str = str.split('{' + kv[0] + '}').join(String(kv[1]));
    });
  }
  return str;
}

function applyLang(lang) {
  currentLang = lang || 'en';
  localStorage.setItem('brainfast.lang', currentLang);
  document.querySelectorAll('[data-i18n]').forEach(function(el) {
    const key = el.dataset.i18n;
    const val = t(key);
    if (val === key) return;
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
      // skip — inputs use data-i18n-ph for placeholder
    } else if (el.tagName === 'OPTION') {
      el.textContent = val;
    } else {
      el.innerHTML = val;
    }
  });
  document.querySelectorAll('[data-i18n-ph]').forEach(function(el) {
    const val = t(el.dataset.i18nPh);
    if (val !== el.dataset.i18nPh) el.placeholder = val;
  });
  document.querySelectorAll('.lang-btn[data-lang]').forEach(function(btn) {
    btn.classList.toggle('active', btn.dataset.lang === currentLang);
  });
  // Re-initialize Lucide icons after i18n innerHTML updates
  if (typeof lucide !== 'undefined') lucide.createIcons();
}

// Language toggle buttons
document.querySelectorAll('.lang-btn[data-lang]').forEach(function(btn) {
  btn.onclick = function() { applyLang(btn.dataset.lang); };
});

// ================================================================
// THEME TOGGLE (light / dark)
// ================================================================
(function initTheme() {
  const saved = localStorage.getItem('idlebrain.theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved === 'light' ? 'light' : 'dark');
  const btn = document.getElementById('themeToggleBtn');
  if (btn) btn.textContent = saved === 'light' ? '🌙' : '☀️';
})();

document.getElementById('themeToggleBtn')?.addEventListener('click', function() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('idlebrain.theme', next);
  this.textContent = next === 'light' ? '🌙' : '☀️';
});



// DOM refs
const logBox        = document.getElementById('logBox');
const barFill       = document.getElementById('barFill');
const stepText      = document.getElementById('stepText');
const progressPct   = document.getElementById('progressPct');
const statusBadge   = document.getElementById('statusBadge');
const resultRows    = document.getElementById('resultRows');
const compareRows   = document.getElementById('compareRows');
const historyList   = document.getElementById('historyList');
const versionText   = document.getElementById('versionText');
const sliceProgress = document.getElementById('sliceProgress');
const wholeBrainStageList = document.getElementById('wholeBrainStageList');
const wholeBrainStageMeta = document.getElementById('wholeBrainStageMeta');
const volumeQcSummaryEl = document.getElementById('volumeQcSummary');
const volumeQcSourceEl = document.getElementById('volumeQcSource');
const sliceInspectorGrid = document.getElementById('sliceInspectorGrid');
const sliceInspectorEmpty = document.getElementById('sliceInspectorEmpty');
const sliceInspectorCount = document.getElementById('sliceInspectorCount');
const qcAllCount = document.getElementById('qcAllCount');
const validateStatus = document.getElementById('validateStatus');
const workflowModeEl = document.getElementById('workflowMode');
const oneClickSourcePathEl = document.getElementById('oneClickSourcePath');
const oneClickScopeEl = document.getElementById('oneClickScope');
const oneClickStartBtn = document.getElementById('oneClickStartBtn');
const quickExportBtn = document.getElementById('quickExportBtn');
const quickExportFormatEl = document.getElementById('quickExportFormat');
const methodsModalTitleEl = document.getElementById('methodsModalTitle');
const methodsModalDescEl = document.getElementById('methodsModalDesc');
const errorPanel = document.getElementById('errorPanel');
const errorPanelToggle = document.getElementById('errorPanelToggle');
const errorPanelBody = document.getElementById('errorPanelBody');
const errorPanelList = document.getElementById('errorPanelList');
const errorPanelEmpty = document.getElementById('errorPanelEmpty');
const errorBadge = document.getElementById('errorBadge');
const preflightModal = document.getElementById('preflightModal');
const preflightModalTitle = document.getElementById('preflightModalTitle');
const preflightModalDesc = document.getElementById('preflightModalDesc');
const preflightIssuesEl = document.getElementById('preflightIssues');
const preflightContinueBtn = document.getElementById('preflightContinueBtn');
const preflightCancelBtn = document.getElementById('preflightCancelBtn');

const state = {
  running: false,
  channel: 'red',
  runAll: false,
  allResults: [],
  cellSummary: null,
  useHierarchy: false,
  startEpoch: null,
  backendErrors: [],
  frontendErrors: [],
  activeJobId: localStorage.getItem('idlebrain.activeJobId') || '',
};

const overlayJobState = {
  jobId: localStorage.getItem('brainfast.overlayJobId') || '',
};

renderErrorPanel();

if (errorPanelToggle) {
  errorPanelToggle.onclick = () => {
    const willOpen = errorPanelBody?.classList.contains('hidden');
    errorPanelBody?.classList.toggle('hidden');
    errorPanelToggle.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
  };
}

function buildOverlayJobId() {
  if (window.crypto && typeof window.crypto.randomUUID === 'function') {
    return `job-${window.crypto.randomUUID()}`;
  }
  return `job-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function getOverlayJobId() {
  if (!overlayJobState.jobId) {
    overlayJobState.jobId = buildOverlayJobId();
    localStorage.setItem('brainfast.overlayJobId', overlayJobState.jobId);
  }
  return overlayJobState.jobId;
}

function setActiveJobId(jobId) {
  state.activeJobId = String(jobId || '').trim();
  if (state.activeJobId) localStorage.setItem('idlebrain.activeJobId', state.activeJobId);
  else localStorage.removeItem('idlebrain.activeJobId');
}

function withActiveJobQuery(path, extra = {}) {
  const url = new URL(path, window.location.origin);
  if (state.activeJobId) url.searchParams.set('job', state.activeJobId);
  Object.entries(extra || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v));
  });
  return `${url.pathname}${url.search}`;
}

function syncOverlayJobId(resp) {
  const jobId = String(resp?.jobId || '').trim();
  if (!jobId) return;
  overlayJobState.jobId = jobId;
  localStorage.setItem('brainfast.overlayJobId', jobId);
}

function withOverlayJobQuery(path, extra = {}) {
  const url = new URL(path, window.location.origin);
  url.searchParams.set('jobId', getOverlayJobId());
  Object.entries(extra || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v));
  });
  return `${url.pathname}${url.search}`;
}

// ================================================================
// TOAST
// ================================================================
const FIELD_ERROR_MAP = {
  inputDir: 'inputDirError',
  atlasPath: 'atlasPathError',
  structPath: 'structPathError',
};
const FIELD_INPUT_MAP = {
  inputDir: 'inputDir',
  atlasPath: 'atlasPath',
  structPath: 'structPath',
};
function normalizeFieldKey(field) {
  const raw = String(field || '').trim();
  if (!raw) return '';
  const map = {
    'input.slice_dir': 'inputDir',
    'inputDir': 'inputDir',
    'atlasPath': 'atlasPath',
    'structPath': 'structPath',
  };
  return map[raw] || raw;
}

function renderFieldIssues(issues = []) {
  const byField = new Map();
  (issues || []).forEach(issue => {
    const field = normalizeFieldKey(issue?.field);
    if (!field || byField.has(field)) return;
    byField.set(field, String(issue?.message || '').trim());
  });
  Object.entries(FIELD_ERROR_MAP).forEach(([field, errorId]) => {
    const errorEl = document.getElementById(errorId);
    const inputEl = document.getElementById(FIELD_INPUT_MAP[field] || field);
    const msg = byField.get(field) || '';
    if (errorEl) {
      errorEl.textContent = msg;
      errorEl.classList.toggle('hidden', !msg);
    }
    if (inputEl) inputEl.classList.toggle('field-input-error', !!msg);
  });
}

function renderErrorPanel() {
  const merged = [...(state.backendErrors || []), ...(state.frontendErrors || [])]
    .filter(item => item && item.message)
    .sort((a, b) => String(b.timestamp || '').localeCompare(String(a.timestamp || '')));
  if (errorPanelList) {
    errorPanelList.innerHTML = merged
      .map(item => {
        const step = escapeHtml(item.step || 'general');
        const ts = escapeHtml(item.timestamp || '');
        const message = escapeHtml(item.message || '');
        const source = escapeHtml(item.source || 'backend');
        const recoverable = item.recoverable === false ? 'blocking' : 'recoverable';
        return `
          <div class="error-item">
            <div class="error-item-header">
              <span class="error-item-step">${step}</span>
              <span class="error-item-time">${ts}</span>
            </div>
            <div class="error-item-message">${message}</div>
            <div class="error-item-source">${source} · ${recoverable}</div>
          </div>
        `;
      })
      .join('');
  }
  if (errorPanelEmpty) errorPanelEmpty.classList.toggle('hidden', merged.length > 0);
  if (errorBadge) {
    errorBadge.textContent = String(merged.length);
    errorBadge.classList.toggle('hidden', merged.length <= 0);
  }
  errorPanel?.classList.toggle('has-errors', merged.length > 0);
}

function pushPersistentError(message, opts = {}) {
  const item = {
    timestamp: new Date().toISOString(),
    message: String(message || '').trim(),
    step: String(opts.step || 'ui'),
    recoverable: opts.recoverable !== false,
    source: String(opts.source || 'frontend'),
  };
  if (!item.message) return;
  state.frontendErrors = [...(state.frontendErrors || []), item].slice(-50);
  renderErrorPanel();
  if (errorPanelBody) {
    errorPanelBody.classList.remove('hidden');
    errorPanelToggle?.setAttribute('aria-expanded', 'true');
  }
  if (!state.running && statusBadge) {
    statusBadge.textContent = t('status.error');
    statusBadge.className = 'status-badge error';
  }
}

async function refreshErrorLog() {
  try {
    const res = await fetch(withActiveJobQuery('/api/error-log')).then(r => r.json());
    state.backendErrors = Array.isArray(res?.errors) ? res.errors : [];
    renderErrorPanel();
  } catch {}
}

function showToast(msg, type = 'info', duration = 4500) {
  if (type === 'error') {
    pushPersistentError(msg, { step: 'ui', source: 'frontend', recoverable: true });
    return;
  }
  const container = document.getElementById('toastContainer');
  // Deduplicate: skip if an identical message is already showing
  const existing = Array.from(container.children);
  if (existing.some(el => el.querySelector('.toast-msg')?.textContent === msg)) return;
  // Cap at 3 visible toasts — remove oldest non-error first, then oldest error
  while (container.children.length >= 3) {
    const kids = Array.from(container.children);
    const victim = kids.find(el => !el.classList.contains('toast-error')) || kids[0];
    victim.remove();
  }
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  const icons = { success: 'OK', warning: 'WARN', error: 'ERR', info: 'i' };
  const escapedMsg = msg.replace(/</g, '&lt;').replace(/>/g, '&gt;');
  toast.innerHTML = `<span class="toast-icon">${icons[type] || 'i'}</span><span class="toast-msg">${escapedMsg}</span>`;
  const cb = document.createElement('button');
  cb.className = 'toast-close';
  cb.innerHTML = '&times;';
  cb.onclick = () => toast.remove();
  toast.appendChild(cb);
  container.appendChild(toast);
  // All toasts auto-dismiss — errors get extra time
  const effectiveDuration = type === 'error' ? Math.max(duration, 10000) : duration;
  setTimeout(() => { toast.classList.add('fade-out'); setTimeout(() => toast.remove(), 380); }, effectiveDuration);
}

// ================================================================
// STEP CARD COLLAPSE / EXPAND
// ================================================================
function updateWorkflowStepIndicator(activeStep) {
  const indicator = document.getElementById('workflowStepIndicator');
  if (!indicator) return;
  indicator.classList.remove('hidden');
  indicator.querySelectorAll('.wsi-step').forEach(el => {
    const step = Number(el.dataset.step);
    el.classList.toggle('active', step === activeStep);
    el.classList.toggle('done', step < activeStep);
  });
}

function collapseAllStepsExcept(keepId) {
  const apply = () => {
    document.querySelectorAll('.step-card').forEach(card => {
      if (card.id === keepId) {
        card.classList.remove('collapsed');
      } else {
        card.classList.add('collapsed');
      }
    });
  };
  // Apply immediately and again after a frame to survive any pending DOM updates.
  apply();
  requestAnimationFrame(apply);
}
document.querySelectorAll('.step-card .step-header').forEach(header => {
  header.addEventListener('click', () => {
    header.closest('.step-card').classList.toggle('collapsed');
  });
});

// ================================================================
// LIGHTBOX
// ================================================================
function openLightbox(src, caption = '') {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightboxCaption').textContent = caption;
  document.getElementById('lightbox').classList.remove('hidden');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.add('hidden');
  document.getElementById('lightboxImg').src = '';
}
document.getElementById('lightboxClose').onclick = closeLightbox;
document.getElementById('lightboxBg').onclick    = closeLightbox;
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ================================================================
// GUIDE MODAL
// ================================================================
document.getElementById('guideBtn').onclick = () => {
  const activeTab = document.querySelector('.sidebar .active')?.textContent?.trim();
  const guideContent = document.getElementById('guideContent');
  if (guideContent) {
    const guideMap = {
      [t('nav.workflow')]: 'guide.workflowHtml',
      [t('nav.manualTiff')]: 'guide.manualCheckHtml',
      [t('nav.qc')]: 'guide.batchQcHtml',
      [t('nav.results')]: 'guide.resultsHtml',
    };
    const key = guideMap[activeTab] || 'guide.workflowHtml';
    guideContent.innerHTML = t(key);
  }
  document.getElementById('guideModal').classList.remove('hidden');
};
document.getElementById('guideModalClose').onclick = () => document.getElementById('guideModal').classList.add('hidden');
document.getElementById('guideModalOk').onclick    = () => document.getElementById('guideModal').classList.add('hidden');

// ================================================================
// TAB SWITCHING
// ================================================================
document.querySelectorAll('.nav-btn[data-tab]').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    if (btn.dataset.tab === 'results') refreshOutputsAndFiles();
    if (btn.dataset.tab === 'qc')      refreshQcAll();
    if (btn.dataset.tab === 'projects') { loadProjects(); refreshBatchQueue(); }
  };
});

// ================================================================
// BROWSE (backend tkinter dialog)
// ================================================================
async function browseFor(targetId, type, filetypes = '') {
  try {
    const endpoint = type === 'folder' ? '/api/browse/folder' : '/api/browse/file';
    const body = type === 'file' ? { filetypes } : {};
    const res = await fetch(endpoint, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json());
    if (res.ok && res.path) {
      document.getElementById(targetId).value = res.path;
      document.getElementById(targetId).dispatchEvent(new Event('change'));
    }
  } catch (err) {
    console.error('Browse failed:', err);
    showToast(t('toast.browseFail'), 'warning');
  }
}
document.querySelectorAll('.btn-browse').forEach(btn => {
  btn.onclick = () => browseFor(btn.dataset.target, btn.dataset.type, btn.dataset.filetypes || '');
});

function applyWorkflowMode(mode) {
  const m = (mode || 'oneclick').toLowerCase();
  document.body.classList.toggle('mode-oneclick', m === 'oneclick');
  // In One-Click mode, collapse Steps 2-4 so the user sees only Step 1 initially.
  // Steps expand automatically as the workflow progresses.
  if (m === 'oneclick') {
    collapseAllStepsExcept('step1');
  } else {
    // Manual mode: expand all steps
    document.querySelectorAll('.step-card').forEach(c => c.classList.remove('collapsed'));
  }
}

if (workflowModeEl) {
  workflowModeEl.onchange = () => {
    applyWorkflowMode(workflowModeEl.value);
    // Sync source path between One-Click and Pro modes (#2)
    const oneClick = document.getElementById('oneClickSourcePath');
    const proInput = document.getElementById('inputDir');
    if (oneClick && proInput) {
      if (workflowModeEl.value === 'oneclick' && proInput.value) {
        // Switching TO oneclick: bring Pro path into OneClick
        oneClick.value = proInput.value;
        oneClick.dispatchEvent(new Event('change', { bubbles: true }));
      } else if (workflowModeEl.value === 'pro' && oneClick.value) {
        // Switching TO pro: bring OneClick path into Pro
        proInput.value = oneClick.value;
      }
    }
  };
  // Apply on page load
  applyWorkflowMode(workflowModeEl.value);
}

// ================================================================
// PROGRESS / STATUS
// ================================================================
function setProgress(p, text) {
  barFill.style.width = `${p}%`;
  stepText.textContent = text;
  progressPct.textContent = `${Math.round(p)}%`;
}
function formatEtaSeconds(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  if (total < 60) return `${total}s`;
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  if (mins < 60) return secs ? `${mins}m ${secs}s` : `${mins}m`;
  const hours = Math.floor(mins / 60);
  const remMins = mins % 60;
  return remMins ? `${hours}h ${remMins}m` : `${hours}h`;
}
function getRunEtaSeconds(status) {
  // Prefer backend ETA (per-stage baselines + self-correction). It works
  // throughout the entire pipeline including ANTs where slicesDone is 0.
  const backendEta = status?.eta?.etr_total_s;
  if (Number.isFinite(backendEta) && backendEta > 0) {
    return Math.round(backendEta);
  }
  // Fallback: naive (total - done) × per-slice elapsed. Only useful during
  // slice-iterating phases (Truth Export, Detection) where slicesDone > 0.
  const done = Number(status?.slicesDone || 0);
  const total = Number(status?.slicesTotal || 0);
  const startEpoch = Number(status?.startEpoch || 0);
  if (!(status?.running) || done <= 0 || total <= done || startEpoch <= 0) return null;
  const elapsed = Math.max(1, Math.floor(Date.now() / 1000 - startEpoch));
  const perSlice = elapsed / done;
  return Math.max(1, Math.round((total - done) * perSlice));
}
function setRunning(r) {
  state.running = r;
  statusBadge.textContent = r ? t('status.running') : t('status.idle');
  statusBadge.className = 'status-badge' + (r ? ' running' : '');
  document.getElementById('runBtn').disabled = r;
}

// Allow clicking the log title to toggle it (if hidden via CSS)
document.addEventListener('click', e => {
  if (e.target && e.target.textContent && (e.target.textContent.includes('实时日志') || e.target.textContent.includes('Live Logs'))) {
    const box = document.getElementById('logBox');
    if (box) box.classList.toggle('hidden');
  }
});

function log(msg) {
  const box = logBox || document.getElementById('logBox');
  if (!box) return;
  const ts = new Date().toLocaleTimeString();
  box.textContent += `\n[${ts}] ${msg}`;
  box.scrollTop = box.scrollHeight;
}

// ================================================================
// FRONTEND ERROR & DEV LOG TRACKING (嵌入式开发者日志)
// ================================================================
(function setupDevLogger() {
  const origConsoleError = console.error;
  const origConsoleWarn = console.warn;

  function safeStr(arg) {
    if (typeof arg === 'string') return arg;
    if (arg instanceof Error) return arg.stack || arg.message;
    try { return JSON.stringify(arg); } catch { return String(arg); }
  }

  console.error = function(...args) {
    origConsoleError.apply(console, args);
    const msg = args.map(safeStr).join(' ');
    log(`❌ [FE ERROR] ${msg}`);
    showToast(t('toast.devError', { msg: msg.substring(0, 70) }), 'error');
  };
  console.warn = function(...args) {
    origConsoleWarn.apply(console, args);
    log(`⚠️ [FE WARN] ${args.map(safeStr).join(' ')}`);
  };

  window.addEventListener('error', e => {
    const msg = `${e.message} at ${e.filename}:${e.lineno}`;
    log(`❌ [UNCAUGHT] ${msg}`);
    showToast(t('toast.uncaughtError', { msg: e.message.substring(0, 50) }), 'error');
  });
  window.addEventListener('unhandledrejection', e => {
    const msg = safeStr(e.reason);
    log(`❌ [PROMISE REJECT] ${msg}`);
    showToast(t('toast.promiseReject', { msg: msg.substring(0, 60) }), 'error');
  });
})();

// ================================================================
// PATH VALIDATION (inline, no alert)
// ================================================================
async function validatePaths(showMsg = true) {
  const q = new URLSearchParams({
    inputDir:   document.getElementById('inputDir').value,
    atlasPath:  document.getElementById('atlasPath').value,
    structPath: document.getElementById('structPath').value,
  });
  try {
    const res = await fetch(`/api/validate?${q}`).then(r => r.json());
    renderFieldIssues(res.fieldIssues || []);
    if (!res.ok) {
      const issues = res.issues.join('; ');
      validateStatus.textContent = t('toast.validateFail', { issues });
      validateStatus.className = 'validate-status err';
      statusBadge.textContent = t('status.error');
      statusBadge.className = 'status-badge error';
      if (showMsg) showToast(t('toast.validateFail', { issues }), 'error');
    } else {
      validateStatus.textContent = t('toast.validateOk');
      validateStatus.className = 'validate-status';
       renderFieldIssues([]);
      if (!state.running) { statusBadge.textContent = t('status.idle'); statusBadge.className = 'status-badge'; }
    }
    return res.ok;
  } catch { return false; }
}
['inputDir', 'atlasPath', 'structPath', 'realSlicePath', 'atlasLabelPath'].forEach(id => {
  const el = document.getElementById(id);
  // In one-click mode, skip auto-validation (inputDir is not required upfront)
  if (el) el.addEventListener('change', () => {
    if (workflowModeEl?.value === 'oneclick') return;
    validatePaths(false);
  });
});

// ================================================================
// PRESET SAVE / LOAD
// ================================================================
const PRESET_KEYS = ['inputDir','outputDir','atlasPath','structPath','realSlicePath','atlasLabelPath',
  'pixelSizeUm','rotateAtlas','flipAtlas','slicingPlane','majorTopK','fitMode',
  'overlayMode','alignMode','maxPoints','minDistance','ransacResidual',
  'oneClickHemisphere','oneClickAtlasVersion','oneClickRegMode','oneClickScope'];

function savePreset() {
  const preset = {};
  PRESET_KEYS.forEach(k => { const el = document.getElementById(k); if (el) preset[k] = el.value; });
  preset.channel = state.channel;
  preset.runAll  = state.runAll;
  localStorage.setItem('brainfast.preset', JSON.stringify(preset));
  showToast(t('toast.presetSaved'), 'success', 2500);
}
function loadPreset(silent = false) {
  const raw = localStorage.getItem('brainfast.preset');
  if (!raw) { if (!silent) showToast(t('toast.noPreset'), 'warning'); return false; }
  const p = JSON.parse(raw);
  PRESET_KEYS.forEach(k => { const el = document.getElementById(k); if (el && p[k] !== undefined) el.value = p[k]; });
  if (p.channel) { const pill = document.querySelector(`.pill[data-channel="${p.channel}"]`); if (pill) pill.click(); }
  state.runAll = !!p.runAll;
  document.getElementById('batchAll').classList.toggle('active', state.runAll);
  if (!silent) showToast(t('toast.presetLoaded'), 'success', 2500);
  return true;
}
document.getElementById('savePreset').onclick = savePreset;
document.getElementById('loadPreset').onclick = () => loadPreset(false);

// ================================================================
// CHANNEL SELECTION
// ================================================================
document.querySelectorAll('.pill[data-channel]').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.pill[data-channel]').forEach(x => x.classList.remove('active'));
    btn.classList.add('active');
    state.channel = btn.dataset.channel;
    state.runAll  = false;
    document.getElementById('batchAll').classList.remove('active');
  };
});
document.getElementById('batchAll').onclick = () => {
  state.runAll = !state.runAll;
  document.getElementById('batchAll').classList.toggle('active', state.runAll);
};

// ================================================================
// ALPHA SLIDER
// ================================================================
const alphaRange = document.getElementById('alphaRange');
const alphaValue = document.getElementById('alphaValue');
alphaRange.oninput = () => { alphaValue.textContent = `${alphaRange.value}%`; refreshOverlayPreview(); };
document.getElementById('overlayMode').onchange = refreshOverlayPreview;

// ================================================================
// OVERLAY PREVIEW
// ================================================================
async function refreshOverlayPreview() {
  const realPath = document.getElementById('realSlicePath').value;
  if (!realPath) { showToast(t('toast.previewNeedPath'), 'warning'); return; }
  const alpha   = Number(alphaRange.value) / 100;
  const modeEl  = document.getElementById('overlayMode');
  let mode      = modeEl.value;
  const fitMode = document.getElementById('fitMode')?.value || 'cover';
  const payload = {
    jobId: getOverlayJobId(),
    realPath,
    realZIndex:      getSelectedRealZIndex(),
    labelPath:        document.getElementById('atlasLabelPath').value || '../outputs/test_label.tif',
    structureCsv:     document.getElementById('structPath').value || '',
    minMeanThreshold: Number(document.getElementById('minMeanThreshold').value || 8),
    pixelSizeUm:      Number(document.getElementById('pixelSizeUm').value || 0.65),
    rotateAtlas:      Number(document.getElementById('rotateAtlas').value || 0),
    flipAtlas:        document.getElementById('flipAtlas').value || 'none',
    majorTopK:        Number(document.getElementById('majorTopK').value || 20),
    fitMode, alpha, mode, edgeSmoothIter: mode === 'fill' ? 2 : 1,
  };
  
  let respJson = await _runWithProgress('/api/overlay/preview', '/api/overlay/preview/status', payload, 'Generating Preview...');
  if (!respJson) {
    if (mode !== 'contour') {
      mode = 'contour'; modeEl.value = 'contour';
      respJson = await _runWithProgress('/api/overlay/preview', '/api/overlay/preview/status', { ...payload, mode: 'contour' }, 'Generating Preview (Fallback)...');
      if (respJson && respJson.ok) { showToast(t('toast.fillModeFallback'), 'warning'); }
      else { showToast(t('toast.previewFailed'), 'error'); return; }
    } else { showToast(t('toast.previewFailed'), 'error'); return; }
  }
  syncOverlayJobId(respJson);
  try {
    const dg = respJson?.diagnostic;
    if (dg) {
      const ra = Number(dg.real_aspect || 0), aa = Number((dg.atlas_aspect ?? dg.atlas_aspect_before) || 0);
      if (ra > 0 && aa > 0 && Math.abs(ra / aa - 1) > 0.35)
        showToast(t('toast.aspectWarning', { ra: ra.toFixed(2), aa: aa.toFixed(2) }), 'warning', 7000);
    }
  } catch (e) { console.error(e); }
  const img = document.getElementById('previewImg');
  img.src = withOverlayJobQuery('/api/outputs/overlay-preview', { ts: Date.now() });
  img.classList.remove('hidden');
  document.getElementById('previewPlaceholder').classList.add('hidden');
  img.onclick = () => openLightbox(img.src, t('lightbox.overlay'));
  showToast(t('toast.previewUpdated'), 'success', 2000);
}
document.getElementById('refreshPreviewBtn').onclick = refreshOverlayPreview;

// ================================================================
// DETECT PREVIEW (single-slice cell detection)
// ================================================================
let _detectOverlayVisible = true;

async function runDetectPreview() {
  const slicePath = document.getElementById('realSlicePath').value;
  if (!slicePath) { showToast(t('detect.noSlice'), 'warning'); return; }

  const resultDiv = document.getElementById('detectPreviewResult');
  const summary   = document.getElementById('detectResultSummary');
  const details   = document.getElementById('detectResultDetails');
  const btn       = document.getElementById('detectPreviewBtn');

  // Show running state
  resultDiv.classList.remove('hidden');
  details.classList.add('hidden');
  summary.textContent = t('detect.running');
  btn.disabled = true;

  try {
    const res = await fetch('/api/detect/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        slicePath,
        jobId: getOverlayJobId(),
        params: {
          model: document.getElementById('detectModelSelect').value,
          diameter_um: parseFloat(document.getElementById('detectDiameterUm').value) || 12.0,
          flow_threshold: parseFloat(document.getElementById('detectFlowThreshold').value),
          cellprob_threshold: parseFloat(document.getElementById('detectCellprobThreshold').value),
          min_size_px: parseInt(document.getElementById('detectMinSizePx').value, 10) || 8,
          gpu: document.getElementById('detectGpuToggle').checked,
        },
      }),
    });
    const data = await res.json();

    if (!res.ok || !data.ok) {
      const errMsg = data.runtimeAvailable === false
        ? t('detect.noRuntime')
        : t('detect.error', { err: data.error || 'unknown' });
      summary.textContent = errMsg;
      showToast(errMsg, 'error', 5000);
      return;
    }

    // Success — show results
    summary.textContent = t('detect.done', { count: data.cellCount, detector: data.detector });
    showToast(t('detect.done', { count: data.cellCount, detector: data.detector }), 'success', 3000);

    // Show overlay image and CSV link
    if (data.overlayUrl) {
      const overlayImg = document.getElementById('detectOverlayImg');
      overlayImg.src = data.overlayUrl + '&ts=' + Date.now();
      overlayImg.onclick = () => openLightbox(overlayImg.src, t('detect.done', { count: data.cellCount, detector: data.detector }));
      _detectOverlayVisible = true;
    }
    if (data.csvUrl) {
      document.getElementById('detectCsvLink').href = data.csvUrl;
    }
    details.classList.remove('hidden');

  } catch (err) {
    summary.textContent = t('detect.error', { err: err.message });
    showToast(t('detect.error', { err: err.message }), 'error');
  } finally {
    btn.disabled = false;
  }
}

// --- Detection parameter panel logic ---
async function loadCellposeModels() {
  try {
    const res = await fetch('/api/cellpose/models');
    const data = await res.json();
    if (!data.ok) return;

    const select = document.getElementById('detectModelSelect');
    select.innerHTML = '';
    for (const m of data.models) {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.type === 'custom' ? `${m.name} (custom)` : m.name;
      select.appendChild(opt);
    }
  } catch (err) {
    console.warn('Failed to load Cellpose models:', err);
  }
}

// Slider value displays
document.getElementById('detectFlowThreshold').oninput = function() {
  document.getElementById('detectFlowThresholdVal').textContent = this.value;
};
document.getElementById('detectCellprobThreshold').oninput = function() {
  document.getElementById('detectCellprobThresholdVal').textContent = this.value;
};

// Load models when panel is first opened
document.getElementById('detectParamsPanel').addEventListener('toggle', function() {
  if (this.open) loadCellposeModels();
});

document.getElementById('detectPreviewBtn').onclick = runDetectPreview;
document.getElementById('detectResultClose').onclick = () => {
  document.getElementById('detectPreviewResult').classList.add('hidden');
};
document.getElementById('detectToggleOverlay').onclick = () => {
  const img = document.getElementById('detectOverlayImg');
  _detectOverlayVisible = !_detectOverlayVisible;
  img.style.display = _detectOverlayVisible ? '' : 'none';
};

document.getElementById('editMasksBtn').onclick = async function() {
  var slicePath = document.getElementById('realSlicePath').value;
  if (!slicePath) { showToast('No slice loaded', 'warning'); return; }

  var btn = this;
  btn.disabled = true;
  btn.textContent = 'Loading...';

  try {
    var res = await fetch('/api/detect/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        slicePath: slicePath,
        jobId: getOverlayJobId(),
        returnMasks: true,
        params: {
          model: document.getElementById('detectModelSelect').value,
          diameter_um: parseFloat(document.getElementById('detectDiameterUm').value) || 12.0,
          flow_threshold: parseFloat(document.getElementById('detectFlowThreshold').value),
          cellprob_threshold: parseFloat(document.getElementById('detectCellprobThreshold').value),
          min_size_px: parseInt(document.getElementById('detectMinSizePx').value, 10) || 8,
          gpu: document.getElementById('detectGpuToggle').checked,
        },
      }),
    });
    var data = await res.json();
    if (!data.ok) {
      showToast('Detection failed: ' + (data.error || 'unknown'), 'error');
      return;
    }
    MaskEditor.open(slicePath, getOverlayJobId());
  } catch (err) {
    showToast('Failed: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Edit Masks';
  }
};

// ================================================================
// AUTOPICK ASYNC PROGRESS HELPERS
// ================================================================
let _autopickAbortFlag = false;
let _modalCloseTimer = null;
let _autopickToken = '';

function _showAutopickModal() {
  const modal = document.getElementById('autopickModal');
  if (!modal) return;
  if (_modalCloseTimer) {
    clearTimeout(_modalCloseTimer);
    _modalCloseTimer = null;
  }
  modal.classList.remove('hidden');
  document.getElementById('autopickBarFill').style.width = '0%';
  document.getElementById('autopickBarFill').style.background = 'var(--accent,#4c72f5)';
  document.getElementById('autopickPct').textContent = '0%';
  document.getElementById('autopickProgressMsg').textContent = t('progress.starting');
  document.getElementById('autopickStepText').textContent = '';
  document.getElementById('autopickErrorDetail').classList.add('hidden');
  document.getElementById('autopickModalActions').style.display = 'flex';
  document.getElementById('autopickModalFooter').classList.add('hidden');
  document.getElementById('autopickModalFooter').style.display = 'none';
  _autopickAbortFlag = false;
  _autopickToken = '';
  const closeBtn = document.getElementById('autopickModalClose');
  if (closeBtn) closeBtn.onclick = () => {
    _autopickAbortFlag = true;
    document.getElementById('autopickModal').classList.add('hidden');
  };
}

function _closeAutopickModal() {
  const modal = document.getElementById('autopickModal');
  if (modal) modal.classList.add('hidden');
}

function _scheduleCloseModal(delay = 800) {
  if (_modalCloseTimer) clearTimeout(_modalCloseTimer);
  _modalCloseTimer = setTimeout(() => {
    _closeAutopickModal();
    _modalCloseTimer = null;
  }, delay);
}

function _updateAutopickProgress(progress, msg, stepText) {
  const pct = Math.min(100, Math.max(0, progress || 0));
  const fillEl = document.getElementById('autopickBarFill');
  if (fillEl) fillEl.style.width = pct + '%';
  const pctEl = document.getElementById('autopickPct');
  if (pctEl) pctEl.textContent = pct + '%';
  if (msg) {
    const msgEl = document.getElementById('autopickProgressMsg');
    if (msgEl) msgEl.textContent = msg;
  }
  if (stepText !== undefined) {
    const stepEl = document.getElementById('autopickStepText');
    if (stepEl) stepEl.textContent = stepText;
  }
}

function _showAutopickError(errMsg) {
  const fillEl = document.getElementById('autopickBarFill');
  if (fillEl) fillEl.style.background = '#f87171';
  const msgEl = document.getElementById('autopickProgressMsg');
  if (msgEl) msgEl.textContent = t('progress.autopickFailed');
  const detailEl = document.getElementById('autopickErrorDetail');
  if (detailEl) {
    detailEl.textContent = errMsg || 'Unknown error';
    detailEl.classList.remove('hidden');
  }
  const actions = document.getElementById('autopickModalActions');
  if (actions) actions.style.display = 'none';
  const footer = document.getElementById('autopickModalFooter');
  if (footer) { footer.classList.remove('hidden'); footer.style.display = 'flex'; }
  // Auto-close after 5 seconds so the user isn't stuck
  _scheduleCloseModal(5000);
}

document.getElementById('autopickModalCancel').onclick = async () => {
  _autopickAbortFlag = true;
  const msgEl = document.getElementById('autopickProgressMsg');
  if (msgEl) msgEl.textContent = 'Cancelling...';
  if (_autopickToken) {
    try {
      await fetch('/api/atlas/autopick/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: _autopickToken }),
      });
    } catch {}
  }
  _scheduleCloseModal(300);
};

async function _runAutopickAsync(payload) {
  _showAutopickModal();
  _updateAutopickProgress(3, 'Submitting request...', '');

  let startRes;
  try {
    startRes = await fetch('/api/atlas/autopick-z', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(r => r.json());
  } catch (e) {
    _showAutopickError('Network error: ' + String(e));
    return null;
  }

  if (!startRes.ok) {
    _showAutopickError(startRes.error || 'Request failed');
    return null;
  }

  // Legacy synchronous response (has label_slice_tif directly)
  if (startRes.label_slice_tif !== undefined) {
    _updateAutopickProgress(100, 'Done!', '');
    _scheduleCloseModal(800);
    return startRes;
  }

  const token = startRes.token;
  if (!token) {
    _showAutopickError('No progress token returned from server');
    return null;
  }
  _autopickToken = token;

  // Poll for progress
  while (!_autopickAbortFlag) {
    await new Promise(r => setTimeout(r, 600));
    let status;
    try {
      status = await fetch('/api/atlas/autopick-z/status?token=' + token).then(r => r.json());
    } catch (err) { 
      console.warn('Polling error (autopick):', err);
      continue; 
    }

    if (status.progress !== undefined) {
      _updateAutopickProgress(status.progress, status.message, '');
    }

    if (status.status === 'done') {
      _updateAutopickProgress(100, 'Done!', '');
      if (status.jobId) syncOverlayJobId({ jobId: status.jobId });
      _scheduleCloseModal(900);
      return { ok: true, jobId: status.jobId, ...status.result };
    }
    if (status.status === 'cancelled') {
      showToast('Auto-pick cancelled.', 'warning', 2500);
      _scheduleCloseModal(300);
      return null;
    }
    if (status.status === 'error') {
      _showAutopickError(status.error || 'Unknown error during autopick');
      return null;
    }
  }
  return null;
}

async function _runWithProgress(postUrl, statusUrl, payload, modalTitle) {
  _showAutopickModal();
  document.getElementById('autopickModalActions').style.display = 'none';
  // Update modal title
  const h2 = document.querySelector('#autopickModal h2') || document.querySelector('#autopickModal .modal-title');
  if (h2) h2.textContent = modalTitle || `\uD83E\uDDE0 ${t('progress.processing')}`;
  _updateAutopickProgress(3, t('progress.submitting'), '');

  let startRes;
  try {
    startRes = await fetch(postUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(r => r.json());
  } catch (e) {
    _showAutopickError('Network error: ' + String(e));
    return null;
  }
  if (!startRes.ok) {
    _showAutopickError(startRes.error || 'Request failed');
    return null;
  }
  // Legacy sync response
  if (!startRes.token) {
    _updateAutopickProgress(100, t('progress.done'), '');
    _scheduleCloseModal(600);
    return startRes;
  }
  const token = startRes.token;
  _autopickToken = token;
  while (!_autopickAbortFlag) {
    await new Promise(r => setTimeout(r, 700));
    let status;
    try {
      status = await fetch(statusUrl + '?token=' + token).then(r => r.json());
    } catch (err) { 
      console.warn('Polling error:', err);
      continue; 
    }
    if (status.progress !== undefined) _updateAutopickProgress(status.progress, status.message, '');
    if (status.status === 'done') {
      _updateAutopickProgress(100, 'Done!', '');
      if (status.jobId) syncOverlayJobId({ jobId: status.jobId });
      _scheduleCloseModal(700);
      return { ok: true, ...status };
    }
    if (status.status === 'cancelled') {
      showToast('Task cancelled.', 'warning', 2500);
      _scheduleCloseModal(300);
      return null;
    }
    if (status.status === 'error') {
      _showAutopickError(status.error || 'Unknown error');
      return null;
    }
  }
  return null;
}

// ================================================================
// AUTO-PICK ATLAS SLICE
// ================================================================
document.getElementById('autoPickBtn').onclick = async () => {
  const realPath = document.getElementById('realSlicePath').value;
  const annotationPath = document.getElementById('atlasPath').value;
  if (!realPath || !annotationPath) { showToast(t('toast.autoPickNeedPath'), 'warning'); return; }
  const r = await _runAutopickAsync({
    jobId: getOverlayJobId(),
    realPath, annotationPath,
    realZIndex: getSelectedRealZIndex(),
    zStep: 2,
    pixelSizeUm: Number(document.getElementById('pixelSizeUm').value || 0.65),
    slicingPlane: document.getElementById('slicingPlane').value || 'coronal',
  });
  if (!r) return;
  document.getElementById('atlasLabelPath').value = r.label_slice_tif;
  lastAutoPickedLabelPath = r.label_slice_tif;
  autoPickCacheKey = buildAutoPickKey(
    realPath,
    annotationPath,
    document.getElementById('slicingPlane').value || 'coronal',
    Number(document.getElementById('pixelSizeUm').value || 0.65),
  );
  showToast(t('toast.autoPickSuccess', { plane: r.slicing_plane || 'coronal', z: r.best_z, score: Number(r.best_score).toFixed(4) }), 'success', 5000);

  // Auto-jump to next step by refreshing the preview
  await refreshOverlayPreviewWithCanvas();
};

// ================================================================
// ALIGNMENT QUALITY PANEL
// ================================================================
function getQualityLevel(score) {
  if (score >= 0.70) return 'excellent';
  if (score >= 0.50) return 'good';
  if (score >= 0.30) return 'fair';
  return 'poor';
}
function showAlignQuality(beforeEdge, afterEdge, improved) {
  const panel = document.getElementById('alignQualityPanel');
  panel.classList.remove('hidden');
  const b = Number(beforeEdge), a = Number(afterEdge);
  document.getElementById('scoreBeforeNum').textContent = b.toFixed(4);
  document.getElementById('scoreAfterNum').textContent  = a.toFixed(4);
  document.getElementById('scoreBefore').style.width = `${Math.min(100, b * 100).toFixed(1)}%`;
  const afterBar = document.getElementById('scoreAfter');
  afterBar.style.width = `${Math.min(100, a * 100).toFixed(1)}%`;
  const level = getQualityLevel(a);
  afterBar.className = `quality-bar quality-${level}`;
    const impPct  = b > 0 ? ((a - b) / b * 100).toFixed(0) : '--';
  const impSign = Number(impPct) >= 0 ? '+' : '';
  const impCls  = Number(impPct) < 0 ? 'negative' : '';
  const verdictEl = document.getElementById('qualityVerdict');
  verdictEl.innerHTML = `
    <span class="verdict-badge verdict-${level}">${t(`quality.${level}`)}</span>
    <span class="verdict-text">${improved ? t(`quality.tip.${level}`) : t('quality.noImprove')}</span>
    <span class="verdict-improve ${impCls}">${impSign}${impPct}%</span>
  `;
  const toastType = improved ? (level === 'poor' || level === 'fair' ? 'warning' : 'success') : 'error';
  showToast(`SSIM ${b.toFixed(4)} \u2192 ${a.toFixed(4)} (${impSign}${impPct}%) \u2014 ${t(`quality.${level}`)}`, toastType, 6000);
}

// ================================================================
// AI REGISTRATION
// ================================================================
document.getElementById('aiAlignBtn').onclick = async () => {
  const realPath  = document.getElementById('realSlicePath').value;
  const atlasPath = document.getElementById('atlasLabelPath').value;
  if (!realPath || !atlasPath) { showToast(t('toast.landmarkNeedPath'), 'warning'); return; }
  const alignMode = document.getElementById('alignMode').value;
  const atlasVersion = document.getElementById('oneClickAtlasVersion')?.value || 'ccfv3';
  const registrationMode = document.getElementById('oneClickRegMode')?.value || 'cross_modal';
  showToast(t('toast.landmarkExtracting'), 'info', 15000);
  try {
    const lm = await fetch('/api/align/landmarks', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jobId: getOverlayJobId(),
        realPath, atlasPath,
        maxPoints:      Number(document.getElementById('maxPoints').value || 30),
        minDistance:    Number(document.getElementById('minDistance').value || 12),
        ransacResidual: Number(document.getElementById('ransacResidual').value || 8),
        atlasVersion,
        registrationMode,
      }),
    }).then(r => r.json());
    if (!lm.ok) { showToast(t('toast.landmarkApplyFailed', { err: lm.error || '?' }), 'error'); return; }
    syncOverlayJobId(lm);
    showToast(t('toast.landmarkSuccess', { n: lm.landmark_pairs }), 'info', 10000);
    const ep = alignMode === 'nonlinear' ? '/api/align/nonlinear' : '/api/align/apply';
    const ap = await fetch(ep, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ jobId: getOverlayJobId(), realPath, atlasLabelPath: atlasPath, atlasVersion, registrationMode, hemisphere: document.getElementById('oneClickHemisphere')?.value || 'auto' }),
    }).then(r => r.json());
    if (!ap.ok) { showToast(t('toast.landmarkApplyFailed', { err: ap.error || '?' }), 'error'); return; }
    syncOverlayJobId(ap);
    showAlignQuality(ap.beforeEdgeScore, ap.afterEdgeScore, !ap.scoreWarning);
    const compareImg = document.getElementById('alignPreviewImg');
    compareImg.src = alignMode === 'nonlinear'
      ? withOverlayJobQuery('/api/outputs/overlay-compare-nonlinear', { ts: Date.now() })
      : withOverlayJobQuery('/api/outputs/overlay-compare', { ts: Date.now() });
    compareImg.classList.remove('hidden');
    document.getElementById('alignPreviewPlaceholder').classList.add('hidden');
    compareImg.onclick = () => openLightbox(compareImg.src, t('lightbox.compare'));
    log(`AI ${alignMode} | pairs=${lm.landmark_pairs} | SSIM ${Number(ap.beforeEdgeScore).toFixed(4)} \u2192 ${Number(ap.afterEdgeScore).toFixed(4)}`);
  } catch { showToast(t('toast.alignUnexpected'), 'error'); }
};

// ================================================================
// VIEW LANDMARKS
// ================================================================
document.getElementById('landmarkViewBtn').onclick = async () => {
  const realPath  = document.getElementById('realSlicePath').value;
  const atlasPath = document.getElementById('atlasLabelPath').value;
  if (!realPath || !atlasPath) { showToast(t('toast.landmarkNeedPath'), 'warning'); return; }
  try {
    const p = await fetch('/api/align/landmark-preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ jobId: getOverlayJobId(), realPath, atlasPath }),
    }).then(r => r.json());
    if (!p.ok) { showToast(t('toast.landmarkApplyFailed', { err: p.error || '?' }), 'error'); return; }
    syncOverlayJobId(p);
    openLightbox(withOverlayJobQuery('/api/outputs/landmark-preview', { ts: Date.now() }), t('lightbox.landmark', { n: p.points }));
  } catch { showToast(t('toast.landmarkApplyFailed', { err: '?' }), 'error'); }
};

// ================================================================
// RUN PAYLOAD / PREFLIGHT
// ================================================================
function buildRunPayload() {
  const channels = state.runAll ? ['red', 'green', 'farred'] : [state.channel];
  return {
    configPath: '../configs/run_config.template.json',
    inputDir: document.getElementById('inputDir').value,
    outputDir: document.getElementById('outputDir').value,
    atlasPath: document.getElementById('atlasPath').value,
    structPath: document.getElementById('structPath').value,
    channels,
    params: {
      inputDir: document.getElementById('inputDir').value,
      outputDir: document.getElementById('outputDir').value,
      atlasPath: document.getElementById('atlasPath').value,
      structPath: document.getElementById('structPath').value,
      realSlicePath: document.getElementById('realSlicePath').value,
      pixelSizeUm: document.getElementById('pixelSizeUm').value,
      slicingPlane: document.getElementById('slicingPlane').value,
      rotateAtlas: document.getElementById('rotateAtlas').value,
      flipAtlas: document.getElementById('flipAtlas').value,
      alignMode: document.getElementById('alignMode').value,
      maxPoints: document.getElementById('maxPoints').value,
      minDistance: document.getElementById('minDistance').value,
      ransacResidual: document.getElementById('ransacResidual').value,
      confidenceThreshold: parseFloat(document.getElementById('confidenceThreshold')?.value || '0') || 0,
      version: versionText.textContent,
    },
  };
}

function phaseLabel(phase) {
  const key = `progress.phase.${String(phase || '').trim()}`;
  const translated = t(key);
  return translated === key ? String(phase || 'running') : translated;
}

function computeRunProgress(status) {
  const progress = status?.progress || {};
  const stepCurrent = Math.max(0, Number(progress.stepCurrent || 0));
  const stepTotal = Math.max(0, Number(progress.stepTotal || 0));
  const slicesDone = Math.max(0, Number(status?.slicesDone || 0));
  const slicesTotal = Math.max(0, Number(status?.slicesTotal || 0));
  if (stepCurrent <= 0 || stepTotal <= 0) {
    if (slicesTotal > 0) return Math.min(96, 20 + Math.round((slicesDone / slicesTotal) * 70));
    return Math.min(94, 20 + Math.floor((status?.logCount || 0) * 0.6));
  }
  const completedSteps = Math.max(0, stepCurrent - 1);
  const sliceFraction = slicesTotal > 0 && progress.phase === 'registration'
    ? (slicesDone / slicesTotal)
    : 0;
  const pct = ((completedSteps + sliceFraction) / stepTotal) * 100;
  return Math.max(5, Math.min(progress.phase === 'done' ? 100 : 98, Math.round(pct)));
}

function showPreflightModal(issues = []) {
  return new Promise(resolve => {
    const hasBlocking = (issues || []).some(item => item?.severity === 'error');
    preflightModalTitle.textContent = t('preflight.title');
    preflightModalDesc.textContent = hasBlocking ? t('preflight.desc.error') : t('preflight.desc.warn');
    preflightIssuesEl.innerHTML = (issues || [])
      .map(item => {
        const severity = String(item?.severity || 'warning').toLowerCase();
        const field = escapeHtml(item?.field || 'general');
        const message = escapeHtml(item?.message || '');
        return `
          <div class="preflight-issue ${escapeHtml(severity)}">
            <div class="preflight-issue-header">
              <span class="preflight-issue-badge">${escapeHtml(severity)}</span>
              <span class="preflight-issue-field">${field}</span>
            </div>
            <div class="preflight-issue-message">${message}</div>
          </div>
        `;
      })
      .join('');
    preflightContinueBtn.classList.toggle('hidden', hasBlocking);
    preflightModal.classList.remove('hidden');

    const close = result => {
      preflightModal.classList.add('hidden');
      preflightContinueBtn.onclick = null;
      preflightCancelBtn.onclick = null;
      document.getElementById('preflightModalClose').onclick = null;
      resolve(result);
    };

    preflightContinueBtn.onclick = () => close(true);
    preflightCancelBtn.onclick = () => close(false);
    document.getElementById('preflightModalClose').onclick = () => close(false);
  });
}

async function runPreflightGate(payload) {
  try {
    const res = await fetch('/api/pipeline/preflight', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(r => r.json());
    const issues = Array.isArray(res?.issues) ? res.issues : [];
    if (!issues.length) return true;
    return await showPreflightModal(issues);
  } catch (err) {
    showToast(`Preflight failed: ${err?.message || err || '?'}`, 'error');
    return false;
  }
}

// ================================================================
// RUN PIPELINE
// ================================================================
document.getElementById('runBtn').onclick = async () => {
  if (state.running) return;
  if (!(await validatePaths(true))) return;
  const payload = buildRunPayload();
  if (!(await runPreflightGate(payload))) return;
  setRunning(true);
  setProgress(5, t('progress.queued'));
  const res = await fetch('/api/run', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(r => r.json());
  if (!res.ok) {
    showToast(t('toast.runFailed', { err: res.error || '?' }), 'error');
    setRunning(false); setProgress(0, t('progress.startFailed')); return;
  }
  setActiveJobId(res.jobId || '');
  state.startEpoch = Math.floor(Date.now() / 1000);
  setProgress(20, t('progress.running', { ch: payload.channels.join(' + ') }));
  showToast(t('toast.runStarted', { channels: payload.channels.join(' + ') }), 'info', 5000);
  await pollLogsUntilDone();
  setProgress(100, t('progress.done'));
  await refreshOutputs();
  setRunning(false);
  showToast(t('toast.runComplete'), 'success', 6000);
};

// ================================================================
// UNIFIED POLL (/api/poll — replaces pollLogsUntilDone + _pollSliceProgress + refreshErrorLog)
// ================================================================
let _uniPollTimer = null;

function _applyPollResponse(p) {
  // Structured errors
  state.backendErrors = Array.isArray(p?.errors) ? p.errors : [];
  renderErrorPanel();

  // Log tail
  if (Array.isArray(p?.logTail) && logBox) {
    const feLogs = logBox.textContent.split('\n').filter(l => l.includes('❌') || l.includes('⚠️') || l.includes('[ready]'));
    logBox.textContent = p.logTail.join('\n') + (feLogs.length ? '\n\n--- Frontend Dev Logs ---\n' + feLogs.join('\n') : '');
    logBox.scrollTop = logBox.scrollHeight;
  }

  // Slice progress bar
  state.startEpoch = Number(p.startEpoch || state.startEpoch || 0) || null;
  state.lastBackendEta = p?.eta || null;
  _updateSliceProgressBar(p.slicesDone || 0, p.slicesTotal || 0);

  // Running state divergence detection
  if (Boolean(p.running) !== state.running) setRunning(Boolean(p.running));

  if (p.running) {
    const cur = Number(p.slicesDone || 0);
    const total = Number(p.slicesTotal || 0);
    const etaSeconds = getRunEtaSeconds(p);
    const phase = phaseLabel(p.progress?.phase || 'running');
    const detail = String(p.progress?.message || '').trim() || t('progress.running', { ch: p.currentChannel || '' });
    if (total > 0) {
      sliceProgress.classList.remove('hidden');
      sliceProgress.textContent = etaSeconds != null
        ? t('progress.slicesEta', { cur, total, eta: formatEtaSeconds(etaSeconds) })
        : t('progress.slices', { cur, total });
    } else {
      sliceProgress.classList.add('hidden');
    }
    setProgress(computeRunProgress(p), `${phase} · ${detail}`);
  }
}

async function _runUnifiedPoll() {
  try {
    const p = await fetch(withActiveJobQuery('/api/poll')).then(r => r.json());
    if (!p.ok) return;
    _applyPollResponse(p);
  } catch {}
}

function _startUnifiedPoll(activeMode = false) {
  _stopUnifiedPoll();
  _runUnifiedPoll();  // immediate first tick
  const interval = activeMode ? 500 : 30000;
  _uniPollTimer = setInterval(_runUnifiedPoll, interval);
}

function _stopUnifiedPoll() {
  if (_uniPollTimer != null) { clearInterval(_uniPollTimer); _uniPollTimer = null; }
}

async function pollLogsUntilDone() {
  _startUnifiedPoll(true);
  while (true) {
    await new Promise(r => setTimeout(r, 600));
    try {
      const p = await fetch(withActiveJobQuery('/api/poll')).then(r => r.json());
      if (!p.ok) continue;
      _applyPollResponse(p);
      if (!p.running) {
        state.startEpoch = null;
        sliceProgress.classList.add('hidden');
        if (p.error && p.error !== 'cancelled by user') showToast(p.error, 'error');
        break;
      }
    } catch { continue; }
  }
  _startUnifiedPoll(false);  // switch to idle rate after done
}

// ================================================================
// CANCEL
// ================================================================
document.getElementById('cancelBtn').onclick = async () => {
  const res = await fetch('/api/cancel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ jobId: state.activeJobId || '' }),
  }).then(r => r.json());
  if (res.ok) { state.startEpoch = null; setRunning(false); setProgress(0, t('progress.cancelled')); sliceProgress.classList.add('hidden'); showToast(t('toast.cancelOk'), 'warning', 3000); }
  else showToast(t('toast.cancelNone'), 'info');
};

// ================================================================
// OPEN OUTPUT FOLDER
// ================================================================
document.getElementById('openOutputsBtn').onclick = async () => {
  const [info, status] = await Promise.all([
    fetch('/api/info').then(r => r.json()),
    fetch(withActiveJobQuery('/api/status')).then(r => r.json()).catch(() => ({})),
  ]);
  showToast(t('toast.outputsPath', { path: status.outputsDir || info.outputs }), 'info', 8000);
};

// ================================================================
// 3D WHOLE-BRAIN QC SURFACES
// ================================================================
const WHOLE_BRAIN_STAGE_NAMES = [
  'Volume Build',
  'Template Prep',
  'ANTS Registration',
  'Laplacian Refinement',
  'Truth Export',
  'Quantification',
];

let latestWholeBrainStage = null;

function renderWholeBrain3dStage(stage) {
  latestWholeBrainStage = stage || null;
  if (!wholeBrainStageList) return;

  const activeName = String(stage?.stageName || '').trim();
  const activeIndex = Number(stage?.stageIndex || 0);
  const stageCount = Number(stage?.stageCount || WHOLE_BRAIN_STAGE_NAMES.length);
  const activePercent = Number(stage?.percent || 0);
  const hasActiveStage = !!activeName;

  if (wholeBrainStageMeta) {
    if (!hasActiveStage) {
      wholeBrainStageMeta.textContent = t('wb3d.status.idle');
    } else {
      wholeBrainStageMeta.textContent = t('wb3d.status.stage', {
        current: activeIndex || 1,
        total: stageCount || WHOLE_BRAIN_STAGE_NAMES.length,
      });
    }
  }

  wholeBrainStageList.innerHTML = '';
  WHOLE_BRAIN_STAGE_NAMES.forEach((name, idx) => {
    const isActive = activeName === name;
    const isDone = hasActiveStage && (idx + 1 < activeIndex || (isActive && activePercent >= 100));
    const isFuture = hasActiveStage && !isActive && !isDone && idx + 1 > activeIndex;
    const pct = isDone ? 100 : (isActive ? Math.max(0, Math.min(100, activePercent)) : 0);
    const row = document.createElement('div');
    row.className = `stage-row${isActive ? ' active' : ''}${isDone ? ' done' : ''}${isFuture ? ' future' : ''}${stage?.error && isActive ? ' error' : ''}`;
    row.innerHTML = `
      <div class="stage-head">
        <div class="stage-title">
          <span class="stage-index">${idx + 1}</span>
          <div>
            <div class="stage-name"></div>
            <div class="stage-sub"></div>
          </div>
        </div>
        <span class="stage-status"></span>
      </div>
      <div class="stage-bar"><div></div></div>
    `;
    row.querySelector('.stage-name').textContent = name;
    row.querySelector('.stage-status').textContent = isActive
      ? (stage?.error ? t('wb3d.status.failed') : (pct >= 100 ? t('wb3d.status.done') : t('wb3d.status.running')))
      : (isDone ? t('wb3d.status.done') : t('wb3d.status.pending'));
    const stageSub = row.querySelector('.stage-sub');
    const artifactText = stage?.artifacts && typeof stage.artifacts === 'object'
      ? Object.entries(stage.artifacts)
          .filter(([, value]) => value !== undefined && value !== null && String(value).trim() !== '')
          .map(([key, value]) => `${key}: ${value}`)
          .join(' · ')
      : '';
    if (isActive && (stage?.message || artifactText)) {
      stageSub.textContent = [stage.message, artifactText].filter(Boolean).join(' · ');
    } else if (isDone) {
      stageSub.textContent = t('wb3d.status.done');
    } else {
      stageSub.textContent = '';
    }
    row.querySelector('.stage-bar > div').style.width = `${pct}%`;
    wholeBrainStageList.appendChild(row);
  });
}

function _parseSimpleCsvRows(text) {
  const lines = String(text || '').trim().split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(part => part.trim());
  const rows = [];
  for (const line of lines.slice(1)) {
    const cols = line.split(',').map(part => part.trim());
    if (headers.length <= 1) {
      rows.push({ key: headers[0] || 'value', value: cols[0] || '' });
      continue;
    }
    const key = cols[0] || headers[0] || 'metric';
    const value = cols.slice(1).join(', ').trim() || cols[0] || '';
    rows.push({ key, value });
  }
  return rows;
}

function renderVolumeQcSummary(text) {
  if (!volumeQcSummaryEl) return;
  volumeQcSummaryEl.innerHTML = '';
  const rows = _parseSimpleCsvRows(text);
  if (volumeQcSourceEl) {
    volumeQcSourceEl.textContent = rows.length ? 'volume_registration_qc.csv' : '';
  }
  if (!rows.length) {
    const empty = document.createElement('div');
    empty.className = 'volume-qc-empty';
    empty.textContent = t('wb3d.qc.empty');
    volumeQcSummaryEl.appendChild(empty);
    return;
  }

  const list = document.createElement('div');
  list.className = 'volume-qc-list';
  rows.slice(0, 12).forEach(row => {
    const item = document.createElement('div');
    item.className = 'volume-qc-item';
    const key = document.createElement('span');
    key.className = 'volume-qc-key';
    key.textContent = row.key;
    const value = document.createElement('strong');
    value.className = 'volume-qc-val';
    value.textContent = row.value;
    item.appendChild(key);
    item.appendChild(value);
    list.appendChild(item);
  });
  volumeQcSummaryEl.appendChild(list);
}

let _volumeRegStatsAvailable = null; // null = unknown, true/false = cached
async function refreshVolumeQcSummary() {
  if (!volumeQcSummaryEl) return;
  // Skip if we already know the endpoint doesn't exist (avoids 404 spam)
  if (_volumeRegStatsAvailable === false) { renderVolumeQcSummary(''); return; }
  if (volumeQcSourceEl) {
    volumeQcSourceEl.textContent = t('wb3d.qc.loading');
  }
  try {
    const res = await fetch('/api/outputs/volume-reg-stats');
    if (!res.ok) {
      if (res.status === 404) _volumeRegStatsAvailable = false;
      renderVolumeQcSummary('');
      return;
    }
    _volumeRegStatsAvailable = true;
    renderVolumeQcSummary(await res.text());
  } catch {
    renderVolumeQcSummary('');
  }
}

function _renderSliceInspectorGrid(payload) {
  if (!sliceInspectorGrid || !sliceInspectorEmpty || !sliceInspectorCount) return;
  const regList = payload?.regList || null;
  const items = regList?.ok && Array.isArray(regList.files) && regList.files.length > 0
    ? regList.files.map(name => ({
        name,
        source: '3d',
      }))
    : [];

  sliceInspectorGrid.innerHTML = '';
  if (!items.length) {
    sliceInspectorEmpty.classList.remove('hidden');
    sliceInspectorCount.textContent = '';
    if (qcAllCount) qcAllCount.textContent = '';
    return;
  }

  sliceInspectorEmpty.classList.add('hidden');
  sliceInspectorCount.textContent = `${items.length}`;
  if (qcAllCount) qcAllCount.textContent = `${items.length} slices`;
  items.forEach(entry => {
    const wrap = document.createElement('div');
    wrap.className = 'qc-thumb';
    const img = document.createElement('img');
    const is3d = entry.source === '3d';
    img.src = is3d ? `/api/outputs/reg-slice/${entry.name}?${Date.now()}` : `/api/outputs/qc-file/${entry.name}?${Date.now()}`;
    img.alt = entry.name;
    img.onerror = () => wrap.remove();
    const label = document.createElement('div');
    label.className = 'qc-thumb-label';
    label.textContent = is3d
      ? entry.name.replace('slice_', '').replace('_overlay.png', '')
      : entry.name.replace('overlay_', '').replace('.png', '');
    wrap.appendChild(img);
    wrap.appendChild(label);
    wrap.onclick = () => {
      openLightbox(img.src, entry.name);
    };
    sliceInspectorGrid.appendChild(wrap);
  });
}

async function refreshSliceInspector() {
  if (!sliceInspectorGrid || !sliceInspectorEmpty || !sliceInspectorCount) return;
  try {
    const regList = await fetch('/api/outputs/reg-slice-list').then(r => r.json());
    _renderSliceInspectorGrid({ regList });
  } catch {
    _renderSliceInspectorGrid({});
  }
}

// ================================================================
// BATCH QC ALL
// ================================================================
async function refreshQcAll() {
  renderWholeBrain3dStage(latestWholeBrainStage);
  await Promise.all([refreshVolumeQcSummary(), refreshSliceInspector()]);

  // Load annotated slice (region labels)
  try {
    const annSection = document.getElementById('annotatedSliceSection');
    const annImg = document.getElementById('annotatedSliceImg');
    const ra = await fetch(withActiveJobQuery('/api/outputs/demo-annotated-slice'), {method:'HEAD'});
    if (ra.ok) {
      annSection.style.display = '';
      annImg.src = withActiveJobQuery('/api/outputs/demo-annotated-slice', { ts: Date.now() });
    }
  } catch {}

  // Load best-slice comparison
  try {
    const bestSection = document.getElementById('bestSliceSection');
    const bestImg = document.getElementById('bestSliceImg');
    const r = await fetch(withActiveJobQuery('/api/outputs/demo-best-slice'), {method:'HEAD'});
    if (r.ok) {
      bestSection.style.display = '';
      bestImg.src = withActiveJobQuery('/api/outputs/demo-best-slice', { ts: Date.now() });
    }
  } catch {}

  // Load demo panel overview
  try {
    const panelSection = document.getElementById('demoPanelSection');
    const panelImg = document.getElementById('demoPanelImg');
    const statsBar = document.getElementById('regStatsBar');
    const panelHead = await fetch(withActiveJobQuery('/api/outputs/demo-panel'), { method: 'HEAD' });
    if (panelHead.ok) {
      panelSection.style.display = '';
      panelImg.src = withActiveJobQuery('/api/outputs/demo-panel', { ts: Date.now() });
      statsBar.innerHTML = '';
      try {
        const stats = await fetch(withActiveJobQuery('/api/outputs/reg-stats')).then(r => r.json());
        if (stats.ok) {
          if (stats.mode === 'registration_run') {
            statsBar.innerHTML = [
              `<span>Pipeline: <strong>${escapeHtml(stats.pipeline || '-')}</strong></span>`,
              `<span>NCC: <strong>${escapeHtml(formatFixed(stats.ncc, 4))}</strong></span>`,
              `<span>SSIM: <strong>${escapeHtml(formatFixed(stats.ssim, 4))}</strong></span>`,
              `<span>Dice: <strong>${escapeHtml(formatFixed(stats.dice, 4))}</strong></span>`,
              `<span>Staining: <strong>${escapeHtml(formatPercent(stats.staining_rate))}</strong></span>`,
              `<span>Coverage: <strong>${escapeHtml(formatPercent(stats.atlas_coverage))}</strong></span>`,
            ].join('<span style="color:#444">  |  </span>');
          } else {
            const scoreColor = stats.mean_score > 0.7 ? '#5c9' : stats.mean_score > 0.4 ? '#fc5' : '#f55';
            statsBar.innerHTML = [
              `<span>✅ <strong>${stats.ok_count}/${stats.total}</strong> slices registered</span>`,
              `<span>Score mean: <strong style="color:${scoreColor}">${stats.mean_score.toFixed(3)}</strong></span>`,
              `<span>Range: ${stats.min_score.toFixed(3)} – ${stats.max_score.toFixed(3)}</span>`,
            ].join('<span style="color:#444">  |  </span>');
          }
        }
      } catch {}
    } else {
      panelSection.style.display = 'none';
    }
  } catch {}

  // Load individual QC thumbnails (use registered slice gallery if available)
  try {
    // Prefer the vibrant registered slice overlays
    const regList = await fetch(withActiveJobQuery('/api/outputs/reg-slice-list')).then(r => r.json());
    if (regList.ok && regList.files.length > 0) {
      empty.classList.add('hidden');
      count.textContent = `${regList.count}`;
      grid.innerHTML = '';
      regList.files.forEach(fname => {
        const wrap = document.createElement('div');
        wrap.className = 'qc-thumb';
        const img = document.createElement('img');
        img.src = withActiveJobQuery(`/api/outputs/reg-slice/${fname}`, { ts: Date.now() });
        img.alt = fname; img.onerror = () => wrap.remove();
        const label = document.createElement('div');
        label.className = 'qc-thumb-label';
        const sliceIdx = parseInt(fname.replace('slice_','').replace('_overlay.png','')) || 0;
        label.textContent = fname.replace('slice_', '').replace('_overlay.png', '');
        wrap.appendChild(img); wrap.appendChild(label);
        // Click → open side-by-side comparison
        wrap.onclick = () => openLightbox(withActiveJobQuery(`/api/outputs/demo-comparison/${sliceIdx}`, { ts: Date.now() }), `Slice ${sliceIdx} — Raw vs Atlas`);
        grid.appendChild(wrap);
      });
      return;
    }
    // Fallback to qc_overlays
    const res = await fetch(withActiveJobQuery('/api/outputs/qc-list')).then(r => r.json());
    if (!res.ok || res.files.length === 0) { grid.innerHTML = ''; empty.classList.remove('hidden'); count.textContent = ''; return; }
    empty.classList.add('hidden');
    count.textContent = `${res.count}`;
    grid.innerHTML = '';
    res.files.forEach(fname => {
      const wrap = document.createElement('div');
      wrap.className = 'qc-thumb';
      const img = document.createElement('img');
      img.src = withActiveJobQuery(`/api/outputs/qc-file/${fname}`, { ts: Date.now() });
      img.alt = fname; img.onerror = () => wrap.remove();
      const label = document.createElement('div');
      label.className = 'qc-thumb-label';
      label.textContent = fname.replace('overlay_', '').replace('.png', '');
      wrap.appendChild(img); wrap.appendChild(label);
      wrap.onclick = () => openLightbox(img.src, fname);
      grid.appendChild(wrap);
    });
  } catch { showToast(t('toast.qcLoadFailed'), 'warning'); }

  // Load Z continuity chart
  refreshZContinuity();
}
document.getElementById('refreshQcAllBtn').onclick = refreshQcAll;

// ----------------------------------------------------------------
// NEUROGLANCER VIEWER (optional; requires neuroglancer extras)
// ----------------------------------------------------------------
async function probeNeuroglancerAvailability() {
  const section = document.getElementById('neuroglancerSection');
  const status = document.getElementById('ngStatus');
  const launchBtn = document.getElementById('ngLaunchBtn');
  if (!section) return;
  try {
    const resp = await fetch('/api/neuroglancer/available');
    const data = await resp.json();
    if (data.available) {
      section.style.display = '';
      launchBtn.disabled = false;
      if (status) status.textContent = '';
    } else {
      section.style.display = '';
      launchBtn.disabled = true;
      if (status) {
        status.textContent = `Missing: ${data.missing || 'neuroglancer'} — run ${data.install}`;
      }
    }
  } catch (err) {
    section.style.display = 'none';
  }
}

async function launchNeuroglancerViewer() {
  const volume = document.getElementById('ngVolumePath')?.value?.trim();
  const seg = document.getElementById('ngSegPath')?.value?.trim();
  const launchBtn = document.getElementById('ngLaunchBtn');
  const urlBlock = document.getElementById('ngUrlBlock');
  const urlLink = document.getElementById('ngUrlLink');
  const status = document.getElementById('ngStatus');
  if (!volume) {
    if (status) status.textContent = 'Provide a Volume NIfTI path first.';
    return;
  }
  launchBtn.disabled = true;
  if (status) status.textContent = 'Starting viewer…';
  try {
    const payload = {
      imageInputs: [{ path: volume, type: 'nii', name: 'Registered Volume' }],
    };
    if (seg) payload.segmentationPath = seg;
    const resp = await fetch('/api/neuroglancer/launch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!data.ok) {
      if (status) status.textContent = 'Launch failed: ' + (data.error || resp.status);
      launchBtn.disabled = false;
      return;
    }
    if (urlLink) {
      urlLink.href = data.url;
      urlLink.textContent = data.url;
    }
    if (urlBlock) urlBlock.style.display = '';
    if (status) status.textContent = 'Viewer ready — click the URL to open it.';
    window.open(data.url, '_blank', 'noopener');
  } catch (err) {
    if (status) status.textContent = 'Launch failed: ' + err.message;
  } finally {
    launchBtn.disabled = false;
  }
}

document.getElementById('ngLaunchBtn')?.addEventListener('click', launchNeuroglancerViewer);
probeNeuroglancerAvailability();

// ----------------------------------------------------------------
// STITCHING (TissueCyte) (optional; requires stitching extras)
// ----------------------------------------------------------------
async function probeStitchingAvailability() {
  const section = document.getElementById('stitchingSection');
  const status = document.getElementById('stitchStatus');
  const startBtn = document.getElementById('stitchStartBtn');
  const bezierInput = document.getElementById('stitchBezierPath');
  if (!section) return;
  try {
    const resp = await fetch('/api/stitching/available');
    const data = await resp.json();
    section.style.display = '';
    if (bezierInput && data.defaultBezierPath) {
      bezierInput.placeholder = data.requiresBezierPath
        ? `Required: ${data.defaultBezierPath}`
        : `Default: ${data.defaultBezierPath}`;
    }
    if (data.available) {
      startBtn.disabled = false;
      if (status) {
        status.textContent = data.requiresBezierPath
          ? 'Bezier calibration required; provide a bezierPath before starting.'
          : '';
      }
    } else {
      startBtn.disabled = true;
      if (status) status.textContent = `Missing: ${data.missing || 'cv2'} — run ${data.install}`;
    }
  } catch (err) {
    section.style.display = 'none';
  }
}

let _stitchPollTimer = null;
async function pollStitchingStatus(jobId) {
  const statusSpan = document.getElementById('stitchJobStatus');
  try {
    const resp = await fetch(`/api/stitching/status?jobId=${encodeURIComponent(jobId)}`);
    const data = await resp.json();
    if (!data.ok) {
      if (statusSpan) statusSpan.textContent = 'unknown';
      clearInterval(_stitchPollTimer);
      return;
    }
    if (statusSpan) statusSpan.textContent = data.status;
    if (data.status === 'done' || data.status === 'error') {
      clearInterval(_stitchPollTimer);
      const status = document.getElementById('stitchStatus');
      if (status) {
        status.textContent = data.status === 'done'
          ? 'Stitching finished.'
          : 'Stitching failed: ' + (data.error || 'unknown error');
      }
    }
  } catch (err) {
    clearInterval(_stitchPollTimer);
  }
}

async function startStitching() {
  const inputDir = document.getElementById('stitchInputDir')?.value?.trim();
  const outputDir = document.getElementById('stitchOutputDir')?.value?.trim();
  const bezierPath = document.getElementById('stitchBezierPath')?.value?.trim();
  const startBtn = document.getElementById('stitchStartBtn');
  const status = document.getElementById('stitchStatus');
  const jobBlock = document.getElementById('stitchJobBlock');
  const jobIdSpan = document.getElementById('stitchJobId');
  const statusSpan = document.getElementById('stitchJobStatus');
  if (!inputDir || !outputDir) {
    if (status) status.textContent = 'Provide both input + output directories.';
    return;
  }
  startBtn.disabled = true;
  if (status) status.textContent = 'Starting stitch job…';
  try {
    const resp = await fetch('/api/stitching/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        inputDir,
        outputDir,
        ...(bezierPath ? { bezierPath } : {}),
      }),
    });
    const data = await resp.json();
    if (!data.ok) {
      if (status) status.textContent = 'Start failed: ' + (data.error || resp.status);
      startBtn.disabled = false;
      return;
    }
    if (jobIdSpan) jobIdSpan.textContent = data.jobId;
    if (statusSpan) statusSpan.textContent = data.status || 'queued';
    if (jobBlock) jobBlock.style.display = '';
    if (status) status.textContent = 'Job started; polling status every 5s.';
    if (_stitchPollTimer) clearInterval(_stitchPollTimer);
    _stitchPollTimer = setInterval(() => pollStitchingStatus(data.jobId), 5000);
  } catch (err) {
    if (status) status.textContent = 'Start failed: ' + err.message;
    startBtn.disabled = false;
  }
}

document.getElementById('stitchStartBtn')?.addEventListener('click', startStitching);
probeStitchingAvailability();

// ----------------------------------------------------------------
// Z-CONTINUITY SVG CHART
// ----------------------------------------------------------------
async function refreshZContinuity() {
  const section = document.getElementById('zContinuitySection');
  const chartDiv = document.getElementById('zContinuityChart');
  const summaryDiv = document.getElementById('zContinuitySummary');
  if (!section || !chartDiv || !summaryDiv) return;
  try {
    const r = await fetch(withActiveJobQuery('/api/outputs/z-continuity'));
    if (!r.ok) { section.style.display = 'none'; return; }
    const j = await r.json();
    if (!j.ok || !Array.isArray(j.slice_ids) || j.slice_ids.length === 0) {
      section.style.display = 'none'; return;
    }
    section.style.display = '';
    const slices = j.slice_ids;
    const orig = j.original_z;
    const smooth = j.smoothed_z;
    const outliers = j.is_outlier || slices.map(() => false);
    const outlierCount = j.outlier_count || 0;

    // SVG dimensions
    const W = Math.max(600, slices.length * 6);
    const H = 180;
    const PAD = { t: 16, r: 20, b: 32, l: 48 };
    const cW = W - PAD.l - PAD.r;
    const cH = H - PAD.t - PAD.b;

    const allZ = [...orig, ...(smooth || [])].filter(v => v != null);
    const zMin = Math.min(...allZ);
    const zMax = Math.max(...allZ);
    const zRange = zMax - zMin || 1;
    const xMin = Math.min(...slices);
    const xMax = Math.max(...slices);
    const xRange = xMax - xMin || 1;

    const px = s => PAD.l + ((s - xMin) / xRange) * cW;
    const py = z => PAD.t + (1 - (z - zMin) / zRange) * cH;

    const pts = (arr) => arr.map((z, i) => `${px(slices[i]).toFixed(1)},${py(z).toFixed(1)}`).join(' ');

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block;max-width:100%">`;

    // Y-axis label
    svg += `<text x="10" y="${PAD.t + cH/2}" text-anchor="middle" transform="rotate(-90,10,${PAD.t + cH/2})" fill="#888" font-size="11">AP index</text>`;

    // Axes
    svg += `<line x1="${PAD.l}" y1="${PAD.t}" x2="${PAD.l}" y2="${PAD.t+cH}" stroke="#444" stroke-width="1"/>`;
    svg += `<line x1="${PAD.l}" y1="${PAD.t+cH}" x2="${PAD.l+cW}" y2="${PAD.t+cH}" stroke="#444" stroke-width="1"/>`;

    // Y tick
    [zMin, Math.round((zMin+zMax)/2), zMax].forEach(v => {
      const y = py(v);
      svg += `<line x1="${PAD.l-4}" y1="${y}" x2="${PAD.l}" y2="${y}" stroke="#555"/>`;
      svg += `<text x="${PAD.l-6}" y="${y+4}" text-anchor="end" fill="#888" font-size="10">${v}</text>`;
    });

    // X ticks (every ~10 slices)
    const step = Math.max(1, Math.round(slices.length / 10));
    slices.filter((_, i) => i % step === 0).forEach(s => {
      const x = px(s);
      svg += `<line x1="${x}" y1="${PAD.t+cH}" x2="${x}" y2="${PAD.t+cH+4}" stroke="#555"/>`;
      svg += `<text x="${x}" y="${PAD.t+cH+15}" text-anchor="middle" fill="#888" font-size="10">${s}</text>`;
    });
    svg += `<text x="${PAD.l + cW/2}" y="${H-2}" text-anchor="middle" fill="#888" font-size="11">slice</text>`;

    // Smoothed line (green)
    if (smooth && smooth.length === slices.length) {
      svg += `<polyline points="${pts(smooth)}" fill="none" stroke="#4caf50" stroke-width="1.5" opacity="0.85"/>`;
    }
    // Original line (blue, thinner)
    svg += `<polyline points="${pts(orig)}" fill="none" stroke="#5b9bd5" stroke-width="1.5" stroke-dasharray="4 2" opacity="0.7"/>`;

    // Outlier markers (red circles)
    outliers.forEach((isOut, i) => {
      if (!isOut) return;
      const x = px(slices[i]);
      const y = py(orig[i]);
      svg += `<circle cx="${x}" cy="${y}" r="4" fill="#e53935" opacity="0.9"/>`;
    });

    // Legend
    const lx = PAD.l + cW - 130;
    const ly = PAD.t + 4;
    svg += `<line x1="${lx}" y1="${ly+6}" x2="${lx+18}" y2="${ly+6}" stroke="#5b9bd5" stroke-width="1.5" stroke-dasharray="4 2"/>`;
    svg += `<text x="${lx+22}" y="${ly+10}" fill="#aaa" font-size="10">original</text>`;
    svg += `<line x1="${lx}" y1="${ly+20}" x2="${lx+18}" y2="${ly+20}" stroke="#4caf50" stroke-width="1.5"/>`;
    svg += `<text x="${lx+22}" y="${ly+24}" fill="#aaa" font-size="10">smoothed</text>`;
    svg += `<circle cx="${lx+9}" cy="${ly+34}" r="4" fill="#e53935"/>`;
    svg += `<text x="${lx+22}" y="${ly+38}" fill="#aaa" font-size="10">outlier</text>`;

    svg += '</svg>';
    chartDiv.innerHTML = svg;

    // Summary badge
    if (outlierCount > 0) {
      summaryDiv.innerHTML = `<span style="background:#b71c1c;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.85em">${t('qc.zContinuityWarn', { n: outlierCount })}</span>`;
    } else {
      summaryDiv.innerHTML = `<span style="background:#1b5e20;color:#c8e6c9;padding:3px 10px;border-radius:12px;font-size:0.85em">${t('qc.zContinuityOk')}</span>`;
    }
  } catch { section.style.display = 'none'; }
}

async function regenDemoVisuals() {
  const btn = document.getElementById('regenDemoBtn');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Regenerating...'; }
  try {
    const r = await fetch(withActiveJobQuery('/api/outputs/refresh-demo'), { method: 'POST' });
    const j = await r.json();
    if (j.ok) {
      showToast(t('toast.regenStarted'), 'info');
      setTimeout(() => { refreshQcAll(); if (btn) { btn.disabled = false; btn.textContent = t('btn.regenDemo'); } }, 15000);
    } else {
      showToast(t('toast.regenFailed', { err: j.error || 'unknown' }), 'warning');
      if (btn) { btn.disabled = false; btn.textContent = t('btn.regenDemo'); }
    }
  } catch (e) {
    showToast(t('toast.regenError', { err: String(e) }), 'warning');
    if (btn) { btn.disabled = false; btn.textContent = t('btn.regenDemo'); }
  }
}

// ================================================================
// RESULTS
// ================================================================
function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const head  = lines.shift().split(',');
  return lines.map(line => { const cols = line.split(','); const obj = {}; head.forEach((h, i) => (obj[h] = cols[i] || '')); return obj; });
}

async function refreshOutputs() {
  try {
    // Prefer hierarchy CSV for a richer tree view; fall back to leaf
    let data = null;
    let useHierarchy = false;
    try {
      const hierText = await fetch(withActiveJobQuery('/api/outputs/hierarchy')).then(r => r.ok ? r.text() : null);
      if (hierText) { data = parseCsv(hierText); useHierarchy = true; }
    } catch {}
    if (!data) {
      const leaf = await fetch(withActiveJobQuery('/api/outputs/leaf')).then(r => r.text());
      data = parseCsv(leaf);
    }
    state.allResults = data;
    state.useHierarchy = useHierarchy;
    state.cellSummary = null;
    await refreshCellSummary();
    renderResultsTable(state.allResults);

    // Load static cell count chart
    try {
      const chartRes = await fetch(withActiveJobQuery('/api/outputs/cell-chart'), { method: 'HEAD' });
      const chartSection = document.getElementById('cellChartSection');
      const chartImg = document.getElementById('cellChartImg');
      if (chartRes.ok && chartSection && chartImg) {
        chartSection.style.display = '';
        chartImg.src = withActiveJobQuery('/api/outputs/cell-chart', { ts: Date.now() });
      }
    } catch {}
    refreshApDensity();
    refreshCoexpression();
    refreshZContinuity();
    await refreshDetectionConfidenceSamples();
    compareRows.innerHTML = '';
    for (const ch of ['red', 'green', 'farred']) {
      try {
        const txt = await fetch(withActiveJobQuery(`/api/outputs/leaf/${ch}`)).then(r => (r.ok ? r.text() : ''));
        if (!txt) continue;
        const arr = parseCsv(txt);
        const total = arr.reduce((s, x) => s + Number(x.count || 0), 0);
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${t(`chname.${ch}`)}</td><td>${total.toLocaleString()}</td>`;
        compareRows.appendChild(tr);
      } catch {}
    }
    await refreshRegistrationRuns();
    await refreshHistory();
  } catch (err) {
    console.error('Refresh outputs failed:', err);
  }
}

async function refreshCellSummary() {
  const section = document.getElementById('resultSummarySection');
  const cards = document.getElementById('resultSummaryCards');
  const warnings = document.getElementById('resultSummaryWarnings');
  const lead = document.getElementById('resultSummaryLead');
  if (!section || !cards || !warnings || !lead) return;

  try {
    const res = await fetch(withActiveJobQuery('/api/outputs/cell-summary')).then(r => r.json());
    const summary = res?.ok ? res.summary : null;
    state.cellSummary = summary || null;
    if (!summary) {
      section.style.display = 'none';
      cards.innerHTML = '';
      warnings.innerHTML = '';
      lead.textContent = t('summary.none');
      return;
    }

    section.style.display = '';
    lead.textContent = `${summary.slice_summary || '-'} · ${summary.mode_note || ''}`.trim();
    warnings.innerHTML = '';
    (summary.warnings || []).forEach((message) => {
      const item = document.createElement('div');
      item.className = 'results-warning';
      item.textContent = String(message || '');
      warnings.appendChild(item);
    });

    const topRegion = summary.top_region
      ? `${summary.top_region.label} · ${Number(summary.top_region.count || 0).toLocaleString()}`
      : '-';
    const mappedText = `${Number(summary.mapped_count || 0).toLocaleString()} (${formatPercent(summary.mapped_pct || 0)})`;
    const outsideText = `${Number(summary.outside_count || 0).toLocaleString()} (${formatPercent(summary.outside_pct || 0)})`;
    cards.innerHTML = [
      renderSummaryCard(t('summary.sample'), summary.sample_name || '-', summary.slice_summary || ''),
      renderSummaryCard(t('summary.scope'), summary.scope_label || '-', summary.scope_kind || ''),
      renderSummaryCard(t('summary.mode'), summary.counting_mode || '-', summary.mode_note || ''),
      renderSummaryCard(t('summary.detectors'), summary.detectors || '-', ''),
      renderSummaryCard(t('summary.detected'), Number(summary.total_detected || 0).toLocaleString(), ''),
      renderSummaryCard(t('summary.mapped'), mappedText, ''),
      renderSummaryCard(t('summary.outside'), outsideText, ''),
      renderSummaryCard(t('summary.regions'), Number(summary.regions_mapped || 0).toLocaleString(), ''),
      renderSummaryCard(t('summary.topRegion'), topRegion, ''),
    ].join('');
  } catch (err) {
    console.error('Refresh cell summary failed:', err);
    section.style.display = 'none';
    cards.innerHTML = '';
    warnings.innerHTML = '';
  }
}

async function refreshDetectionConfidenceSamples() {
  const section = document.getElementById('cellConfidenceSection');
  const grid = document.getElementById('cellConfidenceGrid');
  const empty = document.getElementById('cellConfidenceEmpty');
  if (!section || !grid || !empty) return;

  try {
    const res = await fetch(withActiveJobQuery('/api/outputs/detection-samples')).then(r => r.json());
    if (!res.ok || !Array.isArray(res.samples) || !res.samples.length) {
      section.style.display = 'none';
      grid.innerHTML = '';
      empty.style.display = '';
      return;
    }

    section.style.display = '';
    empty.style.display = 'none';
    grid.innerHTML = '';
    res.samples.forEach(sample => {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'cell-confidence-card';
      const title = sample.source_name || `slice ${sample.slice_id ?? ''}`;
      const subtitle = `${Number(sample.count || 0).toLocaleString()} ${t('cellconf.cells')}`;
      const detectorText = sample.detectors ? `${t('cellconf.detector')}: ${sample.detectors}` : '';
      const sampleUrl = withActiveJobQuery(sample.url || '', { ts: Date.now() });
      card.innerHTML = `
        <img src="${sampleUrl}" alt="${escapeHtml(title)}" />
        <div class="cell-confidence-meta">
          <div class="cell-confidence-title">${escapeHtml(title)}</div>
          <div class="cell-confidence-subtitle">${escapeHtml(subtitle)}</div>
          <div class="cell-confidence-detector">${escapeHtml(detectorText)}</div>
        </div>
      `;
      card.onclick = () => openLightbox(sampleUrl, title);
      grid.appendChild(card);
    });
  } catch (err) {
    console.error('Refresh detection samples failed:', err);
    section.style.display = 'none';
  }
}

function renderSummaryCard(label, value, note = '') {
  return `
    <article class="result-summary-card">
      <div class="result-summary-label">${escapeHtml(label)}</div>
      <div class="result-summary-value">${escapeHtml(value)}</div>
      <div class="result-summary-note">${escapeHtml(note || '')}</div>
    </article>
  `;
}

function safeDepth(value) {
  const parsed = Number.parseInt(value ?? 0, 10);
  if (!Number.isFinite(parsed) || parsed < 0) return 0;
  return Math.min(parsed, 12);
}

function renderRegionChart(summary) {
  const section = document.getElementById('chartSection');
  const container = document.getElementById('regionChart');
  const regions = Array.isArray(summary?.major_regions) ? summary.major_regions : [];
  if (!regions.length) { section.style.display = 'none'; return; }
  const maxCount = regions[0] ? Number(regions[0].count) : 1;
  section.style.display = '';
  container.innerHTML = '';
  regions.forEach((r) => {
    const count = Number(r.count || 0);
    const pct = Number(r.pct || 0) * 100;
    const barPct = (count / maxCount * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'region-chart-row';
    row.innerHTML = `
      <div class="region-chart-label" title="${escapeHtml(r.label || '')}">${escapeHtml(r.label || '-')}</div>
      <div class="region-chart-bar">
        <div class="region-chart-fill" style="width:${barPct}%;background:${escapeHtml(r.color || '#78909C')}"></div>
      </div>
      <div class="region-chart-value">${count.toLocaleString()} <span>(${pct.toFixed(1)}%)</span></div>`;
    container.appendChild(row);
  });
}

// Tree collapse state: tracks which region_ids are collapsed
const _treeCollapsed = new Set();
let _treeMaxDepth = 2; // default expand depth

function _sortTreeDFS(data) {
  // Build children map keyed by parent_structure_id
  const childrenOf = new Map();
  data.forEach(d => {
    const pid = String(d.parent_structure_id || '0');
    if (!childrenOf.has(pid)) childrenOf.set(pid, []);
    childrenOf.get(pid).push(d);
  });
  // Sort children by graph_order or region_id
  for (const [, kids] of childrenOf) {
    kids.sort((a, b) => (Number(a.graph_order || 0) - Number(b.graph_order || 0)) || (Number(a.region_id) - Number(b.region_id)));
  }
  // DFS from roots (entries whose parent is not in the dataset)
  const allIds = new Set(data.map(d => String(d.region_id)));
  const ordered = [];
  const visited = new Set();
  function dfs(id) {
    const kids = childrenOf.get(id) || [];
    kids.forEach(k => {
      const rid = String(k.region_id);
      if (visited.has(rid)) return;
      visited.add(rid);
      ordered.push(k);
      dfs(rid);
    });
  }
  // Find roots
  const roots = data.filter(d => !allIds.has(String(d.parent_structure_id || '0')));
  roots.sort((a, b) => (Number(a.graph_order || 0) - Number(b.graph_order || 0)) || (Number(a.region_id) - Number(b.region_id)));
  roots.forEach(r => { const rid = String(r.region_id); if (!visited.has(rid)) { visited.add(rid); ordered.push(r); dfs(rid); } });
  // Append any orphans not reached by DFS
  data.forEach(d => { if (!visited.has(String(d.region_id))) ordered.push(d); });
  return ordered;
}

function _hasChildren(regionId, data, idx) {
  const depth = parseInt(data[idx].depth || 0);
  for (let i = idx + 1; i < data.length; i++) {
    const d = parseInt(data[i].depth || 0);
    if (d <= depth) break;
    if (String(data[i].parent_structure_id) === String(regionId)) return true;
  }
  return false;
}


function renderResultsTable(data) {
  resultRows.innerHTML = '';
  const keyword = (document.getElementById('regionSearch')?.value || '').toLowerCase();
  const filtered = keyword ? data.filter(d => (d.region_name || d.region || '').toLowerCase().includes(keyword) || (d.acronym || '').toLowerCase().includes(keyword)) : data;
  const rootCount = data.find(d => safeDepth(d.depth) === 0);
  const total = rootCount ? Number(rootCount.count||0) : Math.max(...data.map(d=>Number(d.count||0)));

  // Sort into tree order for hierarchy mode
  const treeMode = state.useHierarchy && !keyword;
  const ordered = treeMode ? _sortTreeDFS(filtered) : filtered;

  // Show/hide depth controls
  const depthControls = document.getElementById('treeDepthControls');
  if (depthControls) depthControls.classList.toggle('hidden', !treeMode);

  // Morphology: check if CSV has these columns
  const hasMorph = data.length > 0 && (data[0].mean_elongation !== undefined || data[0].mean_area_px !== undefined);
  const morphToggleLabel = document.getElementById('morphToggleLabel');
  const morphToggle = document.getElementById('morphToggle');
  if (morphToggleLabel) morphToggleLabel.style.display = hasMorph ? 'flex' : 'none';
  const showMorph = hasMorph && morphToggle && morphToggle.checked;
  document.querySelectorAll('.morph-col').forEach(el => el.style.display = showMorph ? '' : 'none');

  // Build ancestor-collapsed lookup for click-based collapse
  const collapsedAncestor = new Set();
  ordered.forEach((d, i) => {
    const depth = Math.max(0, parseInt(d.depth || 0));
    const rid = String(d.region_id || i);
    const name = escapeHtml(d.region_name || d.region || '-');
    const acronym = d.acronym ? `<span style="color:#888;font-size:0.85em"> (${escapeHtml(d.acronym)})</span>` : '';
    const count = Number(d.count || 0);
    const countStr = count > 0 ? count.toLocaleString() : '<span style="color:#555">\u2014</span>';
    const pct = total > 0 && count > 0 ? (count/total*100).toFixed(1)+'%' : '';
    const barWidth = total > 0 && count > 0 ? Math.max(2, count/total*120).toFixed(0) : 0;
    const barColor = depth <= 2 ? '#5c9' : depth === 3 ? '#59c' : '#888';
    const bar = (depth >= 2 && depth <= 5 && count > 0)
      ? `<div style="display:inline-block;width:${barWidth}px;height:8px;background:${barColor};border-radius:2px;vertical-align:middle;opacity:0.75"></div>`
      : '';

    const hasKids = treeMode && _hasChildren(rid, ordered, i);
    const isCollapsed = _treeCollapsed.has(rid);
    const indent = treeMode ? '\u00A0'.repeat(Math.min(depth, 10) * 3) : '';
    const toggle = treeMode && hasKids
      ? `<span class="tree-toggle${isCollapsed ? ' collapsed' : ''}" data-rid="${rid}">\u25BC</span>`
      : (treeMode ? '<span style="display:inline-block;width:16px"></span>' : '');
    const ciLow = d.ci_low != null ? Number(d.ci_low).toFixed(0) : null;
    const ciHigh = d.ci_high != null ? Number(d.ci_high).toFixed(0) : null;
    const ciStr = ciLow != null && ciHigh != null
      ? `<span style="font-size:0.78em;color:#666">[${ciLow}–${ciHigh}]</span>` : '';
    const morphCols = showMorph
      ? `<td class="morph-col" style="text-align:right;color:#888;font-size:0.85em">${d.mean_elongation != null ? Number(d.mean_elongation).toFixed(2) : '—'}</td><td class="morph-col" style="text-align:right;color:#888;font-size:0.85em">${d.mean_area_px != null ? Number(d.mean_area_px).toFixed(0) : '—'}</td><td class="morph-col" style="text-align:right;color:#888;font-size:0.85em">${d.mean_mean_intensity != null ? Number(d.mean_mean_intensity).toFixed(0) : '—'}</td>`
      : '<td class="morph-col" style="display:none"></td><td class="morph-col" style="display:none"></td><td class="morph-col" style="display:none"></td>';
    const tr = document.createElement('tr');
    tr.dataset.depth = depth;
    tr.dataset.rid = rid;
    if (treeMode && depth <= 2) tr.style.fontWeight = 'bold';
    if (depth === 0) tr.style.cssText = 'font-weight:bold;background:rgba(255,255,255,0.04)';

    // Determine visibility: hidden if depth exceeds max OR any ancestor is collapsed
    let hidden = false;
    if (treeMode && depth > 0) {
      // Check if parent is in collapsedAncestor set
      const pid = String(d.parent_structure_id || '');
      if (collapsedAncestor.has(pid) || depth > _treeMaxDepth) hidden = true;
    }
    if (hidden) tr.classList.add('tree-hidden');
    // Track collapsed ancestors for descendants
    if (isCollapsed || hidden) collapsedAncestor.add(rid);

    tr.innerHTML = `<td>${indent}${toggle}${name}${acronym}</td><td style="text-align:right">${countStr}</td><td class="ci-col" style="text-align:right">${ciStr}</td><td style="color:#888;font-size:0.85em">${pct}</td><td>${bar}</td>${morphCols}`;
    resultRows.appendChild(tr);
  });

  // Attach click handlers for tree toggles
  if (treeMode) {
    resultRows.querySelectorAll('.tree-toggle').forEach(el => {
      el.onclick = (e) => {
        e.stopPropagation();
        const rid = el.dataset.rid;
        if (_treeCollapsed.has(rid)) _treeCollapsed.delete(rid);
        else _treeCollapsed.add(rid);
        renderResultsTable(data);
      };
    });
  }

  const meta = document.getElementById('resultsMeta');
  const baseMeta = keyword
    ? t('results.filtered', { found: filtered.length, total: data.length })
    : t('results.total', { n: data.length });
  meta.textContent = `${baseMeta} · ${t('results.tableHint')}`;
  if (!keyword) renderRegionChart(state.cellSummary);
}

document.getElementById('regionSearch').addEventListener('input', () => renderResultsTable(state.allResults));

// Depth-level buttons for tree expand
document.querySelectorAll('.depth-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    _treeMaxDepth = parseInt(btn.dataset.depth || 99);
    _treeCollapsed.clear();
    document.querySelectorAll('.depth-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderResultsTable(state.allResults);
  });
});
document.getElementById('morphToggle')?.addEventListener('change', () => renderResultsTable(state.allResults));
document.getElementById('refreshBtn').onclick = refreshOutputs;
document.getElementById('exportBtn').onclick      = () => window.open(withActiveJobQuery('/api/outputs/leaf'), '_blank');
document.getElementById('exportExcelBtn').onclick  = () => window.open(withActiveJobQuery('/api/outputs/excel'), '_blank');

// ================================================================
// METHODS TEXT EXPORT
// ================================================================
function openTextModal(title, description, text) {
  if (methodsModalTitleEl) methodsModalTitleEl.textContent = title;
  if (methodsModalDescEl) methodsModalDescEl.textContent = description;
  document.getElementById('methodsText').textContent = text;
  document.getElementById('methodsModal').classList.remove('hidden');
}

document.getElementById('exportMethodsBtn').onclick = async () => {
  try {
    const res = await fetch(withActiveJobQuery('/api/export/methods-text')).then(r => r.json());
    if (!res.ok) { showToast(t('toast.methodsFailed'), 'error'); return; }
    openTextModal(t('methods.title'), t('methods.desc'), res.text);
  } catch { showToast(t('toast.methodsFailed'), 'error'); }
};
document.getElementById('methodsModalClose').onclick  = () => document.getElementById('methodsModal').classList.add('hidden');
document.getElementById('methodsModalClose2').onclick = () => document.getElementById('methodsModal').classList.add('hidden');
document.getElementById('methodsCopyBtn').onclick = async () => {
  const text = document.getElementById('methodsText').textContent;
  try { await navigator.clipboard.writeText(text); showToast(t('toast.copyOk'), 'success', 2500); }
  catch { showToast(t('toast.copyFailed'), 'warning'); }
};

// ================================================================
// RUN HISTORY
// ================================================================
async function refreshHistory() {
  try {
    const h = await fetch(withActiveJobQuery('/api/history')).then(r => r.json());
    historyList.innerHTML = '';
    (h.history || []).slice().reverse().forEach(item => {
      const li = document.createElement('li');
      li.className = item.ok ? 'ok' : 'err';
            const ts    = item.timestamp || '--';
      const chStr = (item.channels || []).map(c => t(`chname.${c}`)).join(' + ');
            const status = item.ok ? 'OK' : `ERR ${item.error || '?'}`;
      li.textContent = `[${ts}]  ${status}  ${chStr}  (${item.logCount || 0} lines)`;
      historyList.appendChild(li);
    });
  } catch {}
}

// ================================================================
// 3D REGISTRATION REPORTS
// ================================================================
function formatRunTimestamp(value) {
  if (!value) return '-';
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString();
}

function formatFixed(value, digits = 4) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : '-';
}

function formatPercent(value, digits = 1) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : '-';
}

function metricDeltaInfo(finalValue, beforeValue, lowerIsBetter = false) {
  const after = Number(finalValue);
  const before = Number(beforeValue);
  if (!Number.isFinite(after) || !Number.isFinite(before)) {
    return { text: 'final only', cls: 'neutral' };
  }
  const delta = after - before;
  if (Math.abs(delta) < 1e-6) {
    return { text: 'no change', cls: 'neutral' };
  }
  const direction = lowerIsBetter ? -delta : delta;
  return {
    text: `${delta > 0 ? '+' : ''}${delta.toFixed(4)}`,
    cls: direction > 0 ? 'good' : 'bad',
  };
}

function renderRegistrationMetric(label, finalValue, beforeValue, opts = {}) {
  const { lowerIsBetter = false, formatter = (v) => formatFixed(v, 4) } = opts;
  const delta = metricDeltaInfo(finalValue, beforeValue, lowerIsBetter);
  return `
    <div class="registration-metric">
      <div class="registration-metric-label">${escapeHtml(label)}</div>
      <div class="registration-metric-value">${escapeHtml(formatter(finalValue))}</div>
      <div class="registration-metric-delta ${delta.cls}">${escapeHtml(delta.text)}</div>
    </div>
  `;
}

function renderRegistrationPreview(url, caption) {
  if (!url) {
    return `
      <div class="registration-preview">
        <div class="registration-preview-empty">${escapeHtml(caption)}</div>
      </div>
    `;
  }
  const previewUrl = withActiveJobQuery(url, { ts: Date.now() });
  return `
    <div class="registration-preview">
      <img src="${escapeHtml(previewUrl)}" alt="${escapeHtml(caption)}" data-lightbox-src="${escapeHtml(previewUrl)}" data-lightbox-caption="${escapeHtml(caption)}" />
      <div class="registration-preview-caption">${escapeHtml(caption)}</div>
    </div>
  `;
}

function renderRegistrationMenu(run) {
  const detailUrl = run?.artifacts?.report || run?.artifacts?.summary || run?.artifacts?.metadata || '';
  const detailType = run?.artifacts?.report ? 'link' : 'text';
  const detailTitle = run?.artifacts?.report ? '' : (run?.artifacts?.summary ? t('reg3d.summaryTitle') : t('reg3d.metadataTitle'));
  const detailDesc = run?.artifacts?.report ? '' : (run?.artifacts?.summary ? t('reg3d.summaryDesc') : t('reg3d.metadataDesc'));
  const pinLabel = t('reg3d.pinReport');
  return `
    <div class="registration-menu-wrap" data-registration-menu="${escapeHtml(run.name || '')}">
      <button
        class="registration-menu-trigger"
        type="button"
        aria-label="${escapeHtml(t('reg3d.menu'))}"
        data-registration-menu-btn="${escapeHtml(run.name || '')}"
      >⋯</button>
      <div class="registration-menu-dropdown">
        <button
          type="button"
          class="registration-menu-item"
          data-registration-detail="${escapeHtml(run.name || '')}"
          data-detail-url="${escapeHtml(detailUrl)}"
          data-detail-type="${escapeHtml(detailType)}"
          data-detail-title="${escapeHtml(detailTitle)}"
          data-detail-desc="${escapeHtml(detailDesc)}"
        >${escapeHtml(t('reg3d.detailInfo'))}</button>
        <button
          type="button"
          class="registration-menu-item"
          data-registration-delete="${escapeHtml(run.name || '')}"
        >${escapeHtml(t('reg3d.deleteBad'))}</button>
        <button
          type="button"
          class="registration-menu-item"
          data-registration-pin="${escapeHtml(run.name || '')}"
        >${escapeHtml(pinLabel)}</button>
      </div>
    </div>
  `;
}

function renderRegistrationRunCard(run) {
  const metrics = run.metrics || {};
  const pre = run.pre_metrics || {};
  const staining = run.staining_stats || {};
  const targetText = run.target_um === null || run.target_um === undefined
    ? 'native'
    : `${formatFixed(run.target_um, 1)} um`;
  return `
    <article class="registration-card">
      <div class="registration-card-header">
        <div class="registration-card-header-main">
          <div class="registration-card-title">${escapeHtml(run.input_name || run.name || 'Unnamed run')}</div>
          <div class="registration-card-subtitle">${escapeHtml(run.verdict_body || '')}</div>
        </div>
        <div class="registration-card-header-side">
          ${run.pinned ? `<span class="registration-pill registration-pill-pinned">${escapeHtml(t('reg3d.pinned'))}</span>` : ''}
          <span class="registration-badge ${escapeHtml(run.verdict_tone || 'neutral')}">${escapeHtml(run.verdict_title || '')}</span>
          ${renderRegistrationMenu(run)}
        </div>
      </div>

      <div class="registration-meta">
        <div class="registration-meta-label">${escapeHtml(t('reg3d.pipeline'))}</div>
        <div class="registration-meta-value">${escapeHtml(run.pipeline_label || '-')}</div>
        <div class="registration-meta-label">${escapeHtml(t('reg3d.hemisphere'))}</div>
        <div class="registration-meta-value">${escapeHtml(run.hemisphere || '-')}</div>
        <div class="registration-meta-label">${escapeHtml(t('reg3d.target'))}</div>
        <div class="registration-meta-value">${escapeHtml(targetText)}</div>
        <div class="registration-meta-label">${escapeHtml(t('reg3d.updated'))}</div>
        <div class="registration-meta-value">${escapeHtml(formatRunTimestamp(run.updated_at))}</div>
      </div>

      <div class="registration-preview-grid">
        ${renderRegistrationPreview(run?.artifacts?.overview_before, run?.artifacts?.overview_before ? t('reg3d.before') : t('reg3d.noBefore'))}
        ${renderRegistrationPreview(run?.artifacts?.overview, t('reg3d.after'))}
      </div>

      <div class="registration-metrics-grid">
        ${renderRegistrationMetric('NCC', metrics.NCC, pre.NCC)}
        ${renderRegistrationMetric('SSIM', metrics.SSIM, pre.SSIM)}
        ${renderRegistrationMetric('Dice', metrics.Dice, pre.Dice)}
        ${renderRegistrationMetric('MSE', metrics.MSE, pre.MSE, { lowerIsBetter: true })}
        ${renderRegistrationMetric(t('reg3d.staining'), staining.staining_rate, undefined, { formatter: (v) => formatPercent(v) })}
        ${renderRegistrationMetric(t('reg3d.coverage'), staining.atlas_coverage, undefined, { formatter: (v) => formatPercent(v) })}
      </div>
    </article>
  `;
}

async function openRegistrationText(url, title, description) {
  try {
    const resp = await fetch(withActiveJobQuery(url));
    if (!resp.ok) throw new Error('request failed');
    const text = await resp.text();
    const capped = text.length > 20000 ? `${text.slice(0, 20000)}\n...(truncated)` : text;
    openTextModal(title, description, capped);
  } catch (err) {
    console.error('Open registration text failed:', err);
    showToast(t('toast.runDetailsFailed'), 'warning');
  }
}

function closeRegistrationMenus() {
  document.querySelectorAll('.registration-menu-wrap.open').forEach((el) => el.classList.remove('open'));
}

function toggleRegistrationMenu(runName) {
  const target = Array.from(document.querySelectorAll('.registration-menu-wrap')).find(
    (el) => el.dataset.registrationMenu === runName
  );
  if (!target) return;
  const nextState = !target.classList.contains('open');
  closeRegistrationMenus();
  if (nextState) target.classList.add('open');
}

async function pinRegistrationRun(runName) {
  try {
    const resp = await fetch(withActiveJobQuery(`/api/outputs/registration-run/${encodeURIComponent(runName)}/pin`), {
      method: 'POST',
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data?.error || 'pin failed');
    }
    showToast(t('reg3d.pinDone'), 'success', 2500);
    await refreshRegistrationRuns();
  } catch (err) {
    console.error('Pin registration run failed:', err);
    showToast(`${t('toast.runDetailsFailed')} ${err?.message || ''}`.trim(), 'warning');
  }
}

async function deleteBadRegistrationRun(runName) {
  if (!window.confirm(t('reg3d.deleteConfirm'))) return;
  try {
    const resp = await fetch(withActiveJobQuery(`/api/outputs/registration-run/${encodeURIComponent(runName)}/delete-bad`), {
      method: 'POST',
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data?.error || 'delete failed');
    }
    showToast(t('reg3d.deleteDone'), 'success', 2500);
    await refreshRegistrationRuns();
  } catch (err) {
    console.error('Delete registration run failed:', err);
    showToast(`${t('toast.runDetailsFailed')} ${err?.message || ''}`.trim(), 'warning');
  }
}

async function refreshRegistrationRuns() {
  const grid = document.getElementById('registrationRunsGrid');
  const empty = document.getElementById('registrationRunsEmpty');
  if (!grid || !empty) return;
  try {
    const res = await fetch(withActiveJobQuery('/api/outputs/registration-runs')).then(r => r.json());
    const runs = Array.isArray(res?.runs) ? res.runs : [];
    if (!res.ok || runs.length === 0) {
      grid.innerHTML = '';
      empty.classList.remove('hidden');
      return;
    }

    empty.classList.add('hidden');
    grid.innerHTML = runs.map(renderRegistrationRunCard).join('');

    grid.querySelectorAll('[data-lightbox-src]').forEach((img) => {
      img.onclick = () => openLightbox(img.dataset.lightboxSrc, img.dataset.lightboxCaption || '');
    });
    grid.querySelectorAll('[data-registration-menu-btn]').forEach((btn) => {
      btn.onclick = (event) => {
        event.stopPropagation();
        toggleRegistrationMenu(btn.dataset.registrationMenuBtn || '');
      };
    });
    grid.querySelectorAll('[data-registration-detail]').forEach((btn) => {
      btn.onclick = () => {
        closeRegistrationMenus();
        const detailType = btn.dataset.detailType || 'link';
        const detailUrl = btn.dataset.detailUrl || '';
        if (!detailUrl) return;
        if (detailType === 'text') {
          openRegistrationText(detailUrl, btn.dataset.detailTitle || '', btn.dataset.detailDesc || '');
          return;
        }
        window.open(withActiveJobQuery(detailUrl), '_blank');
      };
    });
    grid.querySelectorAll('[data-registration-delete]').forEach((btn) => {
      btn.onclick = () => {
        closeRegistrationMenus();
        deleteBadRegistrationRun(btn.dataset.registrationDelete || '');
      };
    });
    grid.querySelectorAll('[data-registration-pin]').forEach((btn) => {
      btn.onclick = () => {
        closeRegistrationMenus();
        pinRegistrationRun(btn.dataset.registrationPin || '');
      };
    });
  } catch (err) {
    console.error('Refresh 3D registration runs failed:', err);
    grid.innerHTML = '';
    empty.classList.remove('hidden');
  }
}

document.getElementById('refreshRegistrationRunsBtn').onclick = refreshRegistrationRuns;
document.addEventListener('click', () => closeRegistrationMenus());

// ================================================================
// INIT
// ── Slice progress bar (sidebar, always visible) ─────────────────────────────
function _updateSliceProgressBar(done, total) {
  const wrap = document.getElementById('sliceProgressWrap');
  const bar  = document.getElementById('sliceProgressBar');
  const txt  = document.getElementById('sliceProgressText');
  if (!wrap) return;
  if (done === 0 && total === 0) { wrap.style.display = 'none'; return; }
  wrap.style.display = '';
  // Prefer the backend ETA captured by the latest /api/status poll —
  // works during ANTs etc. when slicesDone is still 0.
  const etaSeconds = getRunEtaSeconds({
    running: state.running,
    slicesDone: done,
    slicesTotal: total,
    startEpoch: state.startEpoch,
    eta: state.lastBackendEta,
  });
  txt.textContent = etaSeconds != null
    ? `${done} / ${total || '?'} · ${t('progress.eta', { eta: formatEtaSeconds(etaSeconds) })}`
    : `${done} / ${total || '?'}`;
  const pct = total > 0 ? Math.round(done / total * 100) : 0;
  bar.style.width = pct + '%';
  // Change color when complete
  bar.style.background = done >= total && total > 0
    ? 'linear-gradient(90deg,#2196F3,#64B5F6)'   // blue = done
    : 'linear-gradient(90deg,#4CAF50,#81C784)';  // green = in progress
}

// Start unified background poll at idle rate (30s); switches to 500ms during active run
_startUnifiedPoll(false);

// Sync quick pixel-size input (Step 1) ↔ main pixel-size input (Step 2)
const _oneClickPixelSize = document.getElementById('oneClickPixelSize');
const _mainPixelSize = document.getElementById('pixelSizeUm');
if (_oneClickPixelSize && _mainPixelSize) {
  _oneClickPixelSize.oninput = () => { _mainPixelSize.value = _oneClickPixelSize.value; };
  _mainPixelSize.oninput = () => { _oneClickPixelSize.value = _mainPixelSize.value; };
}

// ================================================================
// FORM PERSISTENCE — save key fields to localStorage on change
// ================================================================
const _PERSIST_FIELDS = [
  'oneClickSourcePath', 'oneClickPixelSize', 'oneClickScope', 'oneClickHemisphere',
  'pixelSizeUm', 'alignMode', 'slicingPlane',
];
function _saveFormField(id) {
  const el = document.getElementById(id);
  if (el) localStorage.setItem(`brainfast.field.${id}`, el.value);
}
function _restoreFormFields() {
  for (const id of _PERSIST_FIELDS) {
    const saved = localStorage.getItem(`brainfast.field.${id}`);
    if (saved !== null) {
      const el = document.getElementById(id);
      if (el) { el.value = saved; }
    }
  }
}
// Attach change/input listeners for auto-save
for (const id of _PERSIST_FIELDS) {
  const el = document.getElementById(id);
  if (el) {
    el.addEventListener('change', () => _saveFormField(id));
    el.addEventListener('input', () => _saveFormField(id));
  }
}

// ================================================================
async function checkAtlasStatus() {
  const banner = document.getElementById('atlasMissingBanner');
  if (!banner) return;
  if (localStorage.getItem('brainfast.atlasBanner.dismissed') === '1') return;
  try {
    const data = await fetch('/api/atlas/status').then(r => r.json());
    if (!data || !data.ok) return;
    if (data.allRequiredReady) {
      banner.classList.add('hidden');
      return;
    }
    const bodyEl = banner.querySelector('.amb-body');
    if (bodyEl) {
      let body;
      if (!data.annotationReady) {
        body = t('atlas.banner.bodyAnnotation');
        if (!data.structureReady) body += ' ' + t('atlas.banner.structureAlsoMissing');
      } else {
        body = t('atlas.banner.bodyStructureOnly');
      }
      bodyEl.textContent = body;
    }
    banner.classList.remove('hidden');
  } catch (e) {
    console.warn('atlas-status check failed:', e);
  }
}

function wireAtlasBanner() {
  const retry = document.getElementById('atlasMissingRetry');
  const dismiss = document.getElementById('atlasMissingDismiss');
  if (retry) retry.onclick = () => { checkAtlasStatus(); };
  if (dismiss) dismiss.onclick = () => {
    const banner = document.getElementById('atlasMissingBanner');
    if (banner) banner.classList.add('hidden');
    localStorage.setItem('brainfast.atlasBanner.dismissed', '1');
  };
}

async function init() {
  // Restore persisted form values before applying defaults
  _restoreFormFields();

  // Apply saved or default language
  applyLang(currentLang);
  applyWorkflowMode(workflowModeEl?.value || 'oneclick');

  wireAtlasBanner();
  checkAtlasStatus();

  try {
    const info = await fetch('/api/info').then(r => r.json());
    versionText.textContent = `v${info.version || '0.0.0'}`;
    if (!document.getElementById('outputDir').value) document.getElementById('outputDir').value = info.outputs || '';
    if (!document.getElementById('atlasPath').value && info?.defaults?.atlasPath) {
      document.getElementById('atlasPath').value = info.defaults.atlasPath;
    }
    if (!document.getElementById('structPath').value && info?.defaults?.structPath) {
      document.getElementById('structPath').value = info.defaults.structPath;
    }
  } catch {}

  // Auto-load last preset silently
  if (localStorage.getItem('brainfast.preset')) {
    if (loadPreset(true)) showToast(t('toast.autoLoadPreset'), 'info', 3000);
  }
}

init();

// ================================================================
// 3D Z-SLICER
// ================================================================
let zSlicerPath = '';
const zSliderEl    = document.getElementById('zSlider');
const zNumInputEl  = document.getElementById('zNumInput');
const zValDisplay  = document.getElementById('zValDisplay');
const zMaxDisplay  = document.getElementById('zMaxDisplay');
const zSlicerBox   = document.getElementById('zSlicerBox');
const zExtractBtn  = document.getElementById('zExtractBtn');
const zExtractStatus = document.getElementById('zExtractStatus');
const oneClickSourcePathEl2 = document.getElementById('oneClickSourcePath');

function revealZSlicer() {
  if (!zSlicerBox) return;
  zSlicerBox.classList.remove('hidden');
  zSlicerBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

async function checkSliceIs3D(path) {
  if (!path) return;
  try {
    const res = await fetch(`/api/slice/info?path=${encodeURIComponent(path)}`).then(r => r.json());
    if (res.ok && res.is3d) {
      zSlicerPath = path;
      const zMax = res.z_count - 1;
      zSliderEl.max   = zMax;
      zNumInputEl.max = zMax;
      const midZ = Math.round(zMax / 2);
      zSliderEl.value = midZ;
      zNumInputEl.value = midZ;
      zValDisplay.textContent = midZ;
      zMaxDisplay.textContent = zMax;
      zExtractStatus.textContent = '3D TIFF detected. Choose a Z layer, then click "Extract This Slice" or click Start again.';
      showToast(t('toast.zDetected', { z: res.z_count, h: res.shape[1] || '?', w: res.shape[2] || '?' }), 'info', 5000);
      // Auto-switch to whole-brain mode for 3D stacks
      const scopeEl = document.getElementById('oneClickScope');
      if (scopeEl && scopeEl.value !== 'whole') {
        scopeEl.value = 'whole';
        scopeEl.dispatchEvent(new Event('change'));
      }
      // Only show Z-slicer for single-slice mode; whole-brain processes all slices automatically
      if (scopeEl?.value === 'single') {
        revealZSlicer();
      } else {
        zSlicerBox.classList.add('hidden');
      }
    } else {
      zSlicerBox.classList.add('hidden');
      zExtractStatus.textContent = '';
      // Auto-switch to single mode for 2D images
      const scopeEl = document.getElementById('oneClickScope');
      if (scopeEl && scopeEl.value !== 'single') {
        scopeEl.value = 'single';
        scopeEl.dispatchEvent(new Event('change'));
      }
    }
    
    const pixelWarnEl = document.getElementById('pixelSizeWarning');
    if (res.pixel_size_um) {
      const psEl = document.getElementById('pixelSizeUm');
      if (psEl && !psEl.dataset.userModified) {
        psEl.value = res.pixel_size_um;
        psEl.dataset.autoDetected = '1';
        showToast(t('toast.pixelSizeDetected', { size: res.pixel_size_um }), 'info', 4000);
      }
      if (pixelWarnEl) pixelWarnEl.classList.add('hidden');
    } else {
      // Pixel size not detected — show prominent warning + quick input in Step 1
      if (pixelWarnEl) pixelWarnEl.classList.remove('hidden');
      const quickRow = document.getElementById('oneClickPixelSizeRow');
      if (quickRow) quickRow.classList.remove('hidden');
      showToast(t('warn.pixelSizeNotDetected'), 'warning', 8000);
    }

    // Show thumbnail preview
    const previewBox = document.getElementById('oneClickPreviewBox');
    const previewImg = document.getElementById('oneClickPreviewImg');
    const previewInfo = document.getElementById('oneClickPreviewInfo');
    if (previewBox && previewImg) {
      try {
        const z = res.is3d ? Math.floor((res.z_count || 0) / 2) : 0;
        previewImg.src = `/api/slice/thumbnail?path=${encodeURIComponent(path)}&z=${z}&size=360`;
        previewImg.onload = () => { previewBox.classList.remove('hidden'); };
        previewImg.onerror = () => { previewBox.classList.add('hidden'); };
        if (previewInfo) {
          const dims = res.is3d ? `${res.z_count} slices, ${res.shape[1] || '?'}×${res.shape[2] || '?'} px` : `${res.shape[0] || '?'}×${res.shape[1] || '?'} px`;
          previewInfo.textContent = dims + (res.pixel_size_um ? `, ${res.pixel_size_um} µm/px` : '');
        }
      } catch { /* thumbnail is optional */ }
    }
  } catch (err) {
    console.error('checkSliceIs3D failed:', err);
  }
}

document.getElementById('realSlicePath').addEventListener('change', e => {
  const path = String(e.target.value || '').trim();
  seedManualCountSourceFromWorkflow(path);
  checkSliceIs3D(path);
});

if (oneClickSourcePathEl2) {
  // Restore saved source path on load
  const savedSource = localStorage.getItem('brainfast.sourcePath');
  if (savedSource && !oneClickSourcePathEl2.value) {
    oneClickSourcePathEl2.value = savedSource;
    document.getElementById('realSlicePath').value = savedSource;
    checkSliceIs3D(savedSource);
  }
  oneClickSourcePathEl2.addEventListener('change', e => {
    const path = String(e.target.value || '').trim();
    localStorage.setItem('brainfast.sourcePath', path);
    document.getElementById('realSlicePath').value = path;
    seedManualCountSourceFromWorkflow(path);
    if (oneClickStartBtn) oneClickStartBtn.dataset.zConfirmed = '0';
    if (path) checkSliceIs3D(path);
  });
}

function syncZ(val) {
  const z = Math.max(0, Math.min(Number(val), Number(zSliderEl.max)));
  zSliderEl.value   = z;
  zNumInputEl.value = z;
  zValDisplay.textContent = z;
  // Update thumbnail preview when Z changes
  const previewImg = document.getElementById('oneClickPreviewImg');
  if (previewImg && zSlicerPath) {
    previewImg.src = `/api/slice/thumbnail?path=${encodeURIComponent(zSlicerPath)}&z=${z}&size=360`;
  }
}
zSliderEl.oninput   = () => syncZ(zSliderEl.value);
zNumInputEl.oninput = () => syncZ(zNumInputEl.value);

zExtractBtn.onclick = async () => {
  const z = Number(zSliderEl.value);
  zExtractStatus.textContent = t('progress.extractingZ');
  try {
    const res = await fetch('/api/slice/extract-z', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: zSlicerPath, z }),
    }).then(r => r.json());
    if (!res.ok) { showToast(t('toast.zExtractFail', { err: res.error }), 'error'); return; }
    document.getElementById('realSlicePath').value = res.path;
    zExtractStatus.textContent = t('progress.usingSlice', { path: res.path });
    showToast(t('toast.zExtracted', { z, path: res.path }), 'success', 4000);

    // Hide Z-slicer immediately so subsequent operations treat the new file as a standard 2D slice
    if (zSlicerBox) zSlicerBox.classList.add('hidden');
    zSlicerPath = '';

    // Auto-continue to next step
    const mode = workflowModeEl?.value || 'oneclick';
    if (mode === 'oneclick' && oneClickStartBtn) {
      oneClickStartBtn.dataset.zConfirmed = '1';
      oneClickStartBtn.click();
    } else {
      await refreshOverlayPreviewWithCanvas();
    }
  } catch (err) { 
    console.error('Extract Z failed:', err);
    showToast(t('toast.zExtractFail', { err: '?' }), 'error'); 
  }
};


// ================================================================
// MANUAL COUNT VIEWER
// ================================================================
const manualCountSourcePathEl = document.getElementById('manualCountSourcePath');
const manualCountLoadBtn = document.getElementById('manualCountLoadBtn');
const manualCountExportBtn = document.getElementById('manualCountExportBtn');
const manualCountUndoBtn = document.getElementById('manualCountUndoBtn');
const manualCountClearSliceBtn = document.getElementById('manualCountClearSliceBtn');
const manualCountClearAllBtn = document.getElementById('manualCountClearAllBtn');
const manualCountViewport = document.getElementById('manualCountViewport');
const manualCountStage = document.getElementById('manualCountStage');
const manualCountPlaceholder = document.getElementById('manualCountPlaceholder');
const manualCountImg = document.getElementById('manualCountImg');
const manualCountCanvas = document.getElementById('manualCountCanvas');
const manualCountCtx = manualCountCanvas?.getContext('2d');
const manualCountZoomEl = document.getElementById('manualCountZoom');
const manualCountZoomValueEl = document.getElementById('manualCountZoomValue');
const manualCountPaletteEl = document.getElementById('manualCountPalette');
const manualCountZTextEl = document.getElementById('manualCountZText');
const manualCountSliceCountEl = document.getElementById('manualCountSliceCount');
const manualCountTotalCountEl = document.getElementById('manualCountTotalCount');
const manualCountStatusEl = document.getElementById('manualCountStatus');

const manualCountState = {
  path: '',
  points: [],
  z: 0,
  zMax: 0,
  is3d: false,
  loaded: false,
  loading: false,
  naturalWidth: 0,
  naturalHeight: 0,
  zoom: Number(manualCountZoomEl?.value || 1),
  palette: String(manualCountPaletteEl?.value || 'green'),
  renderSeq: 0,
  nextPointId: 1,
};

function manualCountBaseName(path) {
  const parts = String(path || '').split(/[\/\\]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : String(path || '');
}

function manualCountCurrentPoints() {
  return manualCountState.points
    .filter(point => point.z === manualCountState.z)
    .sort((a, b) => a.id - b.id);
}

function updateManualCountSummary() {
  if (manualCountZTextEl) {
    manualCountZTextEl.textContent = `${manualCountState.z} / ${manualCountState.zMax}`;
  }
  if (manualCountSliceCountEl) {
    manualCountSliceCountEl.textContent = String(manualCountCurrentPoints().length);
  }
  if (manualCountTotalCountEl) {
    manualCountTotalCountEl.textContent = String(manualCountState.points.length);
  }
}

function setManualCountStatus(message) {
  if (manualCountStatusEl) {
    manualCountStatusEl.textContent = message || t('manualCount.placeholder');
  }
}

function applyManualCountZoom(zoomValue) {
  const zoom = Math.min(2.5, Math.max(0.25, Number(zoomValue) || 1));
  manualCountState.zoom = zoom;
  if (manualCountZoomEl) manualCountZoomEl.value = zoom.toFixed(2);
  if (manualCountZoomValueEl) manualCountZoomValueEl.textContent = `${zoom.toFixed(2)}?`;
  if (!manualCountState.loaded) return;
  const width = Math.max(1, Math.round(manualCountState.naturalWidth * zoom));
  const height = Math.max(1, Math.round(manualCountState.naturalHeight * zoom));
  manualCountStage.style.width = `${width}px`;
  manualCountStage.style.height = `${height}px`;
  manualCountImg.style.width = `${width}px`;
  manualCountImg.style.height = `${height}px`;
  manualCountCanvas.style.width = `${width}px`;
  manualCountCanvas.style.height = `${height}px`;
  redrawManualCountCanvas();
}

function manualCountCanvasCoords(evt) {
  const rect = manualCountCanvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const x = ((evt.clientX - rect.left) / rect.width) * manualCountState.naturalWidth;
  const y = ((evt.clientY - rect.top) / rect.height) * manualCountState.naturalHeight;
  return {
    x: Math.max(0, Math.min(manualCountState.naturalWidth, x)),
    y: Math.max(0, Math.min(manualCountState.naturalHeight, y)),
  };
}

function redrawManualCountCanvas() {
  if (!manualCountCtx || !manualCountState.loaded) return;
  manualCountCtx.clearRect(0, 0, manualCountCanvas.width, manualCountCanvas.height);
  const points = manualCountCurrentPoints();
  if (!points.length) return;
  manualCountCtx.save();
  manualCountCtx.lineWidth = 3;
  manualCountCtx.font = '24px "Segoe UI", sans-serif';
  manualCountCtx.textBaseline = 'middle';
  manualCountCtx.textAlign = 'left';
  points.forEach((point, index) => {
    manualCountCtx.beginPath();
    manualCountCtx.fillStyle = 'rgba(255, 91, 91, 0.92)';
    manualCountCtx.strokeStyle = 'rgba(255, 255, 255, 0.98)';
    manualCountCtx.arc(point.x, point.y, 12, 0, Math.PI * 2);
    manualCountCtx.fill();
    manualCountCtx.stroke();

    const label = String(index + 1);
    manualCountCtx.lineWidth = 5;
    manualCountCtx.strokeStyle = 'rgba(0, 0, 0, 0.75)';
    manualCountCtx.strokeText(label, point.x + 18, point.y);
    manualCountCtx.fillStyle = 'rgba(255, 255, 255, 0.98)';
    manualCountCtx.fillText(label, point.x + 18, point.y);
    manualCountCtx.lineWidth = 3;
  });
  manualCountCtx.restore();
}

function csvEscape(value) {
  const raw = String(value ?? '');
  if (!/[",\n]/.test(raw)) return raw;
  return `"${raw.replace(/"/g, '""')}"`;
}

function seedManualCountSourceFromWorkflow(path) {
  const nextPath = String(path || '').trim();
  if (!nextPath || !manualCountSourcePathEl) return;
  const currentValue = String(manualCountSourcePathEl.value || '').trim();
  if (!currentValue || currentValue === manualCountState.path) {
    manualCountSourcePathEl.value = nextPath;
  }
}

async function renderManualCountSlice() {
  if (!manualCountState.path) return;
  const seq = ++manualCountState.renderSeq;
  const params = new URLSearchParams({
    path: manualCountState.path,
    kind: 'real',
    z: String(manualCountState.z),
    palette: manualCountState.palette,
  });
  const previewUrl = `/api/align/manual-image?${params.toString()}`;
  manualCountState.loading = true;
  try {
    const probe = new Image();
    await new Promise((resolve, reject) => {
      probe.onload = resolve;
      probe.onerror = () => reject(new Error('image decode failed'));
      probe.src = previewUrl;
    });
    if (seq !== manualCountState.renderSeq) return;
    manualCountImg.src = probe.src;
    manualCountState.naturalWidth = probe.naturalWidth || probe.width || 1;
    manualCountState.naturalHeight = probe.naturalHeight || probe.height || 1;
    manualCountCanvas.width = manualCountState.naturalWidth;
    manualCountCanvas.height = manualCountState.naturalHeight;
    manualCountPlaceholder.classList.add('hidden');
    manualCountStage.classList.remove('hidden');
    applyManualCountZoom(manualCountState.zoom);
    updateManualCountSummary();
    redrawManualCountCanvas();
    setManualCountStatus(t('manualCount.ready', { name: manualCountBaseName(manualCountState.path) }));
  } finally {
    if (seq === manualCountState.renderSeq) {
      manualCountState.loading = false;
    }
  }
}

async function loadManualCountStack({ preserveZ = false } = {}) {
  const path = String(manualCountSourcePathEl?.value || '').trim();
  if (!path) {
    setManualCountStatus(t('manualCount.needPath'));
    showToast(t('manualCount.needPath'), 'warning');
    return;
  }

  manualCountLoadBtn.disabled = true;
  try {
    const info = await fetch(`/api/slice/info?path=${encodeURIComponent(path)}`).then(r => r.json());
    if (!info.ok) throw new Error(info.error || 'unknown error');

    const pathChanged = manualCountState.path && manualCountState.path !== path;
    manualCountState.path = path;
    manualCountState.is3d = Boolean(info.is3d);
    manualCountState.zMax = Math.max(0, Number(info.z_count || 1) - 1);
    manualCountState.z = preserveZ ? Math.min(manualCountState.z, manualCountState.zMax) : 0;
    manualCountState.loaded = true;

    if (pathChanged) {
      manualCountState.points = [];
      manualCountState.nextPointId = 1;
      showToast(t('manualCount.pathChanged'), 'info', 3500);
    }

    updateManualCountSummary();
    await renderManualCountSlice();
    manualCountViewport.focus();
  } catch (err) {
    console.error('loadManualCountStack failed:', err);
    const msg = t('manualCount.loadFail', { err: err?.message || '?' });
    setManualCountStatus(msg);
    showToast(msg, 'error', 5000);
  } finally {
    manualCountLoadBtn.disabled = false;
  }
}

manualCountLoadBtn.onclick = () => loadManualCountStack();

if (manualCountSourcePathEl) {
  manualCountSourcePathEl.addEventListener('change', () => {
    const nextPath = String(manualCountSourcePathEl.value || '').trim();
    if (!nextPath) return;
    if (manualCountState.loaded && manualCountState.path && manualCountState.path !== nextPath) {
      setManualCountStatus(t('manualCount.pathChanged'));
    }
  });
}

if (manualCountZoomEl) {
  manualCountZoomEl.addEventListener('input', () => applyManualCountZoom(manualCountZoomEl.value));
}

if (manualCountPaletteEl) {
  manualCountPaletteEl.addEventListener('change', async () => {
    manualCountState.palette = String(manualCountPaletteEl.value || 'green');
    if (!manualCountState.loaded) return;
    try {
      await renderManualCountSlice();
    } catch (err) {
      const msg = t('manualCount.loadFail', { err: err?.message || '?' });
      setManualCountStatus(msg);
      showToast(msg, 'error', 5000);
    }
  });
}

manualCountViewport.addEventListener('wheel', async evt => {
  if (!manualCountState.loaded || manualCountState.zMax <= 0) return;
  evt.preventDefault();
  if (manualCountState.loading) return;
  const direction = evt.deltaY > 0 ? 1 : -1;
  const nextZ = Math.max(0, Math.min(manualCountState.zMax, manualCountState.z + direction));
  if (nextZ === manualCountState.z) return;
  manualCountState.z = nextZ;
  updateManualCountSummary();
  try {
    await renderManualCountSlice();
  } catch (err) {
    const msg = t('manualCount.loadFail', { err: err?.message || '?' });
    setManualCountStatus(msg);
    showToast(msg, 'error', 5000);
  }
}, { passive: false });

manualCountCanvas.addEventListener('click', evt => {
  if (!manualCountState.loaded || evt.button !== 0) return;
  const coords = manualCountCanvasCoords(evt);
  if (!coords) return;
  manualCountState.points.push({
    id: manualCountState.nextPointId++,
    z: manualCountState.z,
    x: Number(coords.x.toFixed(2)),
    y: Number(coords.y.toFixed(2)),
  });
  updateManualCountSummary();
  redrawManualCountCanvas();
});

manualCountCanvas.addEventListener('contextmenu', evt => {
  evt.preventDefault();
  if (!manualCountState.loaded) return;
  const coords = manualCountCanvasCoords(evt);
  if (!coords) return;
  const currentPoints = manualCountCurrentPoints();
  if (!currentPoints.length) return;
  const nearest = currentPoints.reduce((best, point) => {
    const dist2 = ((point.x - coords.x) ** 2) + ((point.y - coords.y) ** 2);
    return !best || dist2 < best.dist2 ? { point, dist2 } : best;
  }, null);
  const removeThreshold = Math.max(12, 48 / Math.max(manualCountState.zoom, 0.25));
  if (!nearest || nearest.dist2 > removeThreshold ** 2) return;
  manualCountState.points = manualCountState.points.filter(point => point.id !== nearest.point.id);
  updateManualCountSummary();
  redrawManualCountCanvas();
});

manualCountUndoBtn.onclick = () => {
  if (!manualCountState.points.length) return;
  manualCountState.points.pop();
  updateManualCountSummary();
  redrawManualCountCanvas();
};

manualCountClearSliceBtn.onclick = () => {
  if (!manualCountState.points.length) return;
  const before = manualCountState.points.length;
  manualCountState.points = manualCountState.points.filter(point => point.z !== manualCountState.z);
  if (manualCountState.points.length !== before) {
    updateManualCountSummary();
    redrawManualCountCanvas();
    showToast(t('manualCount.sliceCleared'), 'info', 2500);
    setManualCountStatus(t('manualCount.sliceCleared'));
  }
};

manualCountClearAllBtn.onclick = () => {
  if (!manualCountState.points.length) return;
  manualCountState.points = [];
  manualCountState.nextPointId = 1;
  updateManualCountSummary();
  redrawManualCountCanvas();
  showToast(t('manualCount.allCleared'), 'info', 2500);
  setManualCountStatus(t('manualCount.allCleared'));
};

manualCountExportBtn.onclick = () => {
  if (!manualCountState.points.length) {
    showToast(t('manualCount.noPoints'), 'warning');
    return;
  }
  const ordered = manualCountState.points.slice().sort((a, b) => a.z - b.z || a.id - b.id);
  const sliceCounters = new Map();
  const lines = ['source_path,z,x,y,point_index,slice_point_index'];
  ordered.forEach((point, index) => {
    const slicePointIndex = (sliceCounters.get(point.z) || 0) + 1;
    sliceCounters.set(point.z, slicePointIndex);
    lines.push([
      csvEscape(manualCountState.path),
      point.z,
      point.x.toFixed(2),
      point.y.toFixed(2),
      index + 1,
      slicePointIndex,
    ].join(','));
  });
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
  const stem = manualCountBaseName(manualCountState.path).replace(/\.[^.]+$/, '') || 'manual_count';
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${stem}_manual_counts.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  setManualCountStatus(t('manualCount.exported'));
  showToast(t('manualCount.exported'), 'success', 3000);
};

updateManualCountSummary();
applyManualCountZoom(manualCountState.zoom);
setManualCountStatus(t('manualCount.placeholder'));

// ================================================================
// CANVAS DRAWING EDITOR
// ================================================================
const drawCanvas    = document.getElementById('drawCanvas');
const drawCtx       = drawCanvas.getContext('2d');
const drawToolbar   = document.getElementById('drawToolbar');
const canvasWrap    = document.getElementById('canvasWrap');
const previewImgEl  = document.getElementById('previewImg');
const regionHoverTooltip = document.getElementById('regionHoverTooltip');
const liquifyRadiusEl = document.getElementById('liquifyRadius');
const liquifyStrengthEl = document.getElementById('liquifyStrength');
const saveCalibLearnBtn = document.getElementById('saveCalibLearnBtn');
const autoLearnToggle = document.getElementById('autoLearnToggle');

let hoverTimer = null;
let hoverReqSeq = 0;
let hoverLastPixelKey = '';
let autoPickCacheKey = '';
let lastAutoPickedLabelPath = '';

let currentTool     = 'select';
let isDrawing       = false;
let drawStartX      = 0;
let drawStartY      = 0;
let annotations     = [];   // stored vector annotations
let pendingTextPos  = null;
let liquifyBusy     = false;
let calibLearnPollTimer = null;

// Tool selection
document.querySelectorAll('.tool-btn[data-tool]').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.tool-btn[data-tool]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentTool = btn.dataset.tool;
    drawCanvas.style.cursor = currentTool === 'select' ? 'default' : 'crosshair';
    if (currentTool !== 'select') hideRegionTooltip();
  };
});

function getDrawColor()     { return document.getElementById('drawColor').value; }
function getDrawLineWidth() { return Number(document.getElementById('drawLineWidth').value) || 2; }
function getLiquifyRadius() {
  const v = Number(liquifyRadiusEl?.value ?? 80);
  return Math.max(8, Math.min(260, Number.isFinite(v) ? v : 80));
}
function getLiquifyStrength() {
  const v = Number(liquifyStrengthEl?.value ?? 0.72);
  return Math.max(0.05, Math.min(1.5, Number.isFinite(v) ? v : 0.72));
}

function buildOverlayRequestPayload(modeOverride = null) {
  const modeEl = document.getElementById('overlayMode');
  const mode = modeOverride || modeEl?.value || 'fill';
  const alpha = Number(alphaRange.value) / 100;
  return {
    realPath: document.getElementById('realSlicePath').value,
    realZIndex: getSelectedRealZIndex(),
    labelPath: document.getElementById('atlasLabelPath').value || '../outputs/test_label.tif',
    structureCsv: document.getElementById('structPath').value || '',
    minMeanThreshold: Number(document.getElementById('minMeanThreshold').value || 8),
    pixelSizeUm: Number(document.getElementById('pixelSizeUm').value || 0.65),
    rotateAtlas: Number(document.getElementById('rotateAtlas').value || 0),
    flipAtlas: document.getElementById('flipAtlas').value || 'none',
    majorTopK: Number(document.getElementById('majorTopK').value || 20),
    fitMode: document.getElementById('fitMode')?.value || 'cover',
    edgeSmoothIter: mode === 'fill' ? 2 : 1,
    warpParams: {},
    alpha,
    mode,
  };
}

function escapeHtml(txt) {
  return String(txt ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function hideRegionTooltip() {
  if (regionHoverTooltip) regionHoverTooltip.classList.add('hidden');
}

function moveRegionTooltip(clientX, clientY) {
  if (!regionHoverTooltip || regionHoverTooltip.classList.contains('hidden')) return;
  const rect = drawCanvas.getBoundingClientRect();
  const leftRaw = (clientX - rect.left) + 14;
  const topRaw = (clientY - rect.top) + 14;
  const maxLeft = Math.max(4, rect.width - regionHoverTooltip.offsetWidth - 6);
  const maxTop = Math.max(4, rect.height - regionHoverTooltip.offsetHeight - 6);
  regionHoverTooltip.style.left = `${Math.max(4, Math.min(maxLeft, leftRaw))}px`;
  regionHoverTooltip.style.top = `${Math.max(4, Math.min(maxTop, topRaw))}px`;
}

function renderRegionTooltip(region, clientX, clientY) {
  if (!regionHoverTooltip) return;
  const color = (region.color && String(region.color).length === 6) ? `#${region.color}` : '#8cc8ff';
  const name = escapeHtml(region.name || region.acronym || 'Unknown Region');
    const acronym = escapeHtml(region.acronym || '-');
    const parent = escapeHtml(region.parent || '-');
  const rid = Number(region.region_id || 0);
  regionHoverTooltip.innerHTML = `
    <div class="region-tip-title"><span class="region-tip-dot" style="background:${color};"></span>${name}</div>
    <div class="region-tip-row"><b>Acronym:</b> ${acronym}</div>
    <div class="region-tip-row"><b>Parent:</b> ${parent}</div>
        <div class="region-tip-row"><b>ID:</b> ${rid || '-'}</div>
  `;
  regionHoverTooltip.classList.remove('hidden');
  moveRegionTooltip(clientX, clientY);
}

async function fetchRegionAtPixel(x, y, clientX, clientY) {
  const seq = ++hoverReqSeq;
  try {
    const res = await fetch(withOverlayJobQuery('/api/overlay/region-at', { x, y })).then(r => r.json());
    if (seq !== hoverReqSeq) return;
    if (!res.ok || !res.inside || !res.region_id) {
      hideRegionTooltip();
      return;
    }
    renderRegionTooltip(res, clientX, clientY);
  } catch {
    if (seq !== hoverReqSeq) return;
    hideRegionTooltip();
  }
}

function buildAutoPickKey(realPath, annotationPath, slicingPlane, pixelSizeUm) {
  const z = getSelectedRealZIndex();
  return `${realPath}|${annotationPath}|${slicingPlane}|${pixelSizeUm}|${z ?? 'auto'}`;
}

function getSelectedRealZIndex() {
  if (!zSlicerBox || zSlicerBox.classList.contains('hidden')) return null;
  const z = Number(zSliderEl?.value);
  if (!Number.isFinite(z)) return null;
  return Math.max(0, Math.round(z));
}

async function ensureAutoPickedAtlasSlice(realPath) {
  const atlasLabelEl = document.getElementById('atlasLabelPath');
  const annotationPath = document.getElementById('atlasPath').value;
  const slicingPlane = document.getElementById('slicingPlane').value || 'coronal';
  const pixelSizeUm = Number(document.getElementById('pixelSizeUm').value || 0.65);
  if (!annotationPath) {
    if (!atlasLabelEl.value) showToast(t('toast.atlasPathNotSet'), 'error', 6000);
    return !!atlasLabelEl.value;
  }

  const canAutoReplace = !atlasLabelEl.value || atlasLabelEl.value === lastAutoPickedLabelPath;
  if (!canAutoReplace) return true;

  const k = buildAutoPickKey(realPath, annotationPath, slicingPlane, pixelSizeUm);
  if (autoPickCacheKey === k && atlasLabelEl.value) return true;

  const atlasVersion = document.getElementById('oneClickAtlasVersion')?.value || 'ccfv3';
  const registrationMode = document.getElementById('oneClickRegMode')?.value || 'cross_modal';
  const targetAp = getTargetApRange();
  const r = await _runAutopickAsync({
    jobId: getOverlayJobId(),
    realPath,
    realZIndex: getSelectedRealZIndex(),
    annotationPath,
    zStep: 2,
    pixelSizeUm,
    slicingPlane,
    roiMode: 'auto',
    atlasVersion,
    registrationMode,
    apRangeStart: targetAp?.start ?? null,
    apRangeEnd: targetAp?.end ?? null,
  });

  if (!r) {
    if (!atlasLabelEl.value) {
      showToast(t('toast.autoPickPreviewFailed'), 'error', 5000);
    }
    return !!atlasLabelEl.value;
  }

  atlasLabelEl.value = r.label_slice_tif;
  lastAutoPickedLabelPath = r.label_slice_tif;
  autoPickCacheKey = k;
  showToast(t('toast.autoPickSuccess', {
    plane: r.slicing_plane || slicingPlane,
    z: r.best_z,
    score: Number(r.best_score).toFixed(4),
  }), 'success', 2800);
  return true;
}

/** Load overlay preview from server into canvas */
async function loadPreviewIntoCanvas(ts) {
  return new Promise(resolve => {
    const src = withOverlayJobQuery('/api/outputs/overlay-preview', { ts: ts || Date.now() });
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (img.naturalWidth === 0) {
        resolve(false);
        return;
      }
      // Constrain canvas to max display size (480px height) to prevent
      // the overlay from blocking downstream UI elements (Step 4 button).
      const MAX_DISPLAY_H = 480;
      const scale = Math.min(1, MAX_DISPLAY_H / img.naturalHeight);
      drawCanvas.width  = Math.round(img.naturalWidth * scale);
      drawCanvas.height = Math.round(img.naturalHeight * scale);
      previewImgEl.src  = src;
      previewImgEl.classList.remove('hidden');
      previewImgEl.style.display = 'block';
      canvasWrap.classList.remove('hidden');
      document.getElementById('previewPlaceholder').classList.add('hidden');
      drawToolbar.classList.remove('hidden');
      redrawAnnotations();
      resolve(true);
    };
    img.onerror = () => {
      console.error('Failed to load preview image from ' + src);
      resolve(false);
    };
    img.src = src;
  });
}

function redrawAnnotations() {
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  annotations.forEach(drawAnnotation);
}

function drawAnnotation(ann) {
  const ctx = drawCtx;
  ctx.save();
  ctx.strokeStyle = ann.color;
  ctx.fillStyle   = ann.color;
  ctx.lineWidth   = ann.lw || 2;
  ctx.lineCap = 'round';

  if (ann.type === 'line') {
    ctx.beginPath(); ctx.moveTo(ann.x1, ann.y1); ctx.lineTo(ann.x2, ann.y2); ctx.stroke();

  } else if (ann.type === 'arrow') {
    ctx.beginPath(); ctx.moveTo(ann.x1, ann.y1); ctx.lineTo(ann.x2, ann.y2); ctx.stroke();
    const ang = Math.atan2(ann.y2 - ann.y1, ann.x2 - ann.x1);
    const sz  = Math.max(8, ann.lw * 4);
    ctx.beginPath();
    ctx.moveTo(ann.x2, ann.y2);
    ctx.lineTo(ann.x2 - sz * Math.cos(ang - 0.4), ann.y2 - sz * Math.sin(ang - 0.4));
    ctx.lineTo(ann.x2 - sz * Math.cos(ang + 0.4), ann.y2 - sz * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();

  } else if (ann.type === 'scalebar') {
    const px = ann.pixelLen;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(ann.x - 6, ann.y - 22, px + 12, 32);
    ctx.fillStyle   = ann.color;
    ctx.strokeStyle = ann.color;
    ctx.lineWidth   = ann.lw + 1;
    ctx.beginPath(); ctx.moveTo(ann.x, ann.y); ctx.lineTo(ann.x + px, ann.y);
    ctx.moveTo(ann.x, ann.y - 6); ctx.lineTo(ann.x, ann.y + 6);
    ctx.moveTo(ann.x + px, ann.y - 6); ctx.lineTo(ann.x + px, ann.y + 6);
    ctx.stroke();
    ctx.font = `bold ${Math.max(12, ann.lw * 5)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(`${ann.umLen} µm`, ann.x + px / 2, ann.y - 8);

  } else if (ann.type === 'text') {
    ctx.font = `${ann.size || 16}px sans-serif`;
    ctx.textAlign = 'left';
    ctx.shadowColor = 'rgba(0,0,0,0.8)'; ctx.shadowBlur = 3;
    ctx.fillText(ann.text, ann.x, ann.y);
  }
  ctx.restore();
}

// Live preview line/arrow while dragging
function drawPreviewStroke(x2, y2) {
  redrawAnnotations();
  const ctx = drawCtx;
  ctx.save();
  ctx.strokeStyle = getDrawColor();
  ctx.fillStyle   = getDrawColor();
  ctx.lineWidth   = getDrawLineWidth();
  ctx.lineCap = 'round';
  if (currentTool === 'line') {
    ctx.beginPath(); ctx.moveTo(drawStartX, drawStartY); ctx.lineTo(x2, y2); ctx.stroke();
  } else if (currentTool === 'arrow') {
    ctx.beginPath(); ctx.moveTo(drawStartX, drawStartY); ctx.lineTo(x2, y2); ctx.stroke();
    const ang = Math.atan2(y2 - drawStartY, x2 - drawStartX);
    const sz  = Math.max(8, getDrawLineWidth() * 4);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - sz * Math.cos(ang - 0.4), y2 - sz * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - sz * Math.cos(ang + 0.4), y2 - sz * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
  } else if (currentTool === 'liquify') {
    const r = getLiquifyRadius();
    ctx.lineWidth = Math.max(1, getDrawLineWidth());
    ctx.beginPath(); ctx.moveTo(drawStartX, drawStartY); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.arc(drawStartX, drawStartY, r, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(x2, y2, Math.max(5, r * 0.35), 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

async function applyLiquifyDrag(x1, y1, x2, y2) {
  if (liquifyBusy) return;
  const dist = Math.hypot(x2 - x1, y2 - y1);
  if (dist < 2.0) return;

  const payload = buildOverlayRequestPayload();
  if (!payload.realPath) {
    showToast(t('toast.setRealSliceFirst'), 'warning');
    return;
  }
  payload.x1 = Number(x1);
  payload.y1 = Number(y1);
  payload.x2 = Number(x2);
  payload.y2 = Number(y2);
  payload.radius = getLiquifyRadius();
  payload.strength = getLiquifyStrength();
  payload.jobId = getOverlayJobId();

  liquifyBusy = true;
  try {
    const res = await fetch('/api/overlay/liquify-drag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(r => r.json());
    if (!res.ok) {
      showToast(`Liquify failed: ${res.error || '?'}`, 'error');
      return;
    }
    syncOverlayJobId(res);
    if (res.correctedLabelPath) {
      document.getElementById('atlasLabelPath').value = res.correctedLabelPath;
    }
    await loadPreviewIntoCanvas();
    hoverLastPixelKey = '';
    hideRegionTooltip();
    showToast(`Liquify applied (${dist.toFixed(1)} px).`, 'success', 1600);
  } catch (e) {
    showToast(`Liquify failed: ${e?.message || '?'}`, 'error');
  } finally {
    liquifyBusy = false;
  }
}

async function pollCalibrationLearnStatus() {
  try {
    const st = await fetch('/api/calibration/learn-status').then(r => r.json());
    if (!st.ok || !st.state) return;
    const s = st.state;
    if (s.running) return;
    if (calibLearnPollTimer) {
      clearInterval(calibLearnPollTimer);
      calibLearnPollTimer = null;
    }
    if (s.ok === true) {
      showToast(t('toast.autoLearnDone'), 'success', 5000);
    } else {
      showToast(`Auto-learning failed: ${s.error || '?'}`, 'warning', 7000);
    }
  } catch {
    if (calibLearnPollTimer) {
      clearInterval(calibLearnPollTimer);
      calibLearnPollTimer = null;
    }
  }
}

function canvasCoords(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width  / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
}

drawCanvas.addEventListener('mousedown', e => {
  if (currentTool === 'select') return;
  const { x, y } = canvasCoords(e);
  if (currentTool === 'text') {
    const txt = prompt(t('text.dialog'));
    if (txt) {
      annotations.push({ type: 'text', x, y, text: txt, color: getDrawColor(), size: Math.max(12, getDrawLineWidth() * 6) });
      redrawAnnotations();
    }
    return;
  }
  if (currentTool === 'scalebar') {
    const umStr = prompt(t('scalebar.dialog'));
    const um = Number(umStr);
    if (!um || isNaN(um)) { showToast(t('scalebar.invalid'), 'warning'); return; }
    const pixelSizeUm = Number(document.getElementById('pixelSizeUm').value || 0.65);
    const pixelLen = Math.round(um / pixelSizeUm);
    annotations.push({ type: 'scalebar', x: Math.round(x), y: Math.round(y), umLen: um, pixelLen, color: getDrawColor(), lw: getDrawLineWidth() });
    redrawAnnotations();
    return;
  }
  isDrawing = true;
  drawStartX = x; drawStartY = y;
});

drawCanvas.addEventListener('mousemove', e => {
  const { x, y } = canvasCoords(e);
  const px = Math.round(x);
  const py = Math.round(y);

  if (isDrawing) {
    drawPreviewStroke(x, y);
    hideRegionTooltip();
    return;
  }

  if (currentTool !== 'select') {
    hideRegionTooltip();
    return;
  }

  const pixelKey = `${px},${py}`;
  if (pixelKey === hoverLastPixelKey) {
    moveRegionTooltip(e.clientX, e.clientY);
    return;
  }
  hoverLastPixelKey = pixelKey;
  if (hoverTimer) clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    fetchRegionAtPixel(px, py, e.clientX, e.clientY);
  }, 45);
});

drawCanvas.addEventListener('mouseleave', () => {
  hoverLastPixelKey = '';
  if (hoverTimer) clearTimeout(hoverTimer);
  hideRegionTooltip();
});

drawCanvas.addEventListener('mouseup', e => {
  if (!isDrawing) return;
  isDrawing = false;
  const { x, y } = canvasCoords(e);
  const dx = x - drawStartX, dy = y - drawStartY;
  if (Math.sqrt(dx*dx + dy*dy) < 3) return; // ignore tiny clicks
  if (currentTool === 'liquify') {
    applyLiquifyDrag(drawStartX, drawStartY, x, y);
    redrawAnnotations();
    return;
  }
  annotations.push({ type: currentTool, x1: drawStartX, y1: drawStartY, x2: x, y2: y, color: getDrawColor(), lw: getDrawLineWidth() });
  redrawAnnotations();
});

document.getElementById('undoAnnotationBtn').onclick = () => {
  annotations.pop(); redrawAnnotations();
};
document.getElementById('clearAnnotationsBtn').onclick = () => {
  annotations = []; redrawAnnotations();
};

document.getElementById('exportCanvasBtn').onclick = () => {
  // Composite: real overlay PNG + drawing canvas
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width  = drawCanvas.width;
  exportCanvas.height = drawCanvas.height;
  const ec = exportCanvas.getContext('2d');
  // Draw base overlay image
  const baseImg = previewImgEl;
  if (baseImg.src) ec.drawImage(baseImg, 0, 0);
  // Draw annotations
  ec.drawImage(drawCanvas, 0, 0);
  exportCanvas.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `brainfast_figure_${Date.now()}.png`;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
};

if (saveCalibLearnBtn) {
  saveCalibLearnBtn.onclick = async () => {
    const payload = buildOverlayRequestPayload();
    if (!payload.realPath) {
      showToast(t('toast.setRealSliceFirst'), 'warning');
      return;
    }
    payload.autoLearn = autoLearnToggle ? !!autoLearnToggle.checked : true;
    payload.note = 'manual_liquify_or_landmark_adjust';
    payload.jobId = getOverlayJobId();
    try {
      const res = await fetch('/api/overlay/calibration/finalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }).then(r => r.json());
      if (!res.ok) {
        showToast(`Finalize failed: ${res.error || '?'}`, 'error');
        return;
      }
      syncOverlayJobId(res);
      const sid = res?.sample?.sample_id;
      showToast(`Calibration saved as sample #${sid}.`, 'success', 3500);
      const pruned = Number(res?.sample?.prune?.pruned || 0);
      const kept = Number(res?.sample?.prune?.kept || 0);
      const maxN = Number(res?.sample?.sample_limit || 0);
      if (pruned > 0) {
        showToast(`Sample library pruned: removed ${pruned}, kept ${kept}/${maxN}.`, 'info', 5000);
      }
      if (res.learningStarted) {
        showToast(t('toast.autoLearnStarted'), 'info', 3000);
        if (calibLearnPollTimer) clearInterval(calibLearnPollTimer);
        calibLearnPollTimer = setInterval(pollCalibrationLearnStatus, 5000);
      }
    } catch (e) {
      showToast(`Finalize failed: ${e?.message || '?'}`, 'error');
    }
  };
}

// Override refreshOverlayPreview to load result into canvas
const _origRefreshOverlayPreview = refreshOverlayPreview;
// Patch: after server returns OK, also load into canvas
async function refreshOverlayPreviewWithCanvas() {
  try {
    const realPath = document.getElementById('realSlicePath').value;
    if (!realPath) { showToast(t('toast.previewNeedPath'), 'warning'); return; }
    if (!(await ensureAutoPickedAtlasSlice(realPath))) return;
    
    const alpha   = Number(alphaRange.value) / 100;
    const modeEl  = document.getElementById('overlayMode');
    let mode      = modeEl.value;
    const fitMode = document.getElementById('fitMode')?.value || 'cover';
    const payload = {
      jobId: getOverlayJobId(),
      realPath,
      realZIndex:      getSelectedRealZIndex(),
      labelPath:        document.getElementById('atlasLabelPath').value || '../outputs/test_label.tif',
      structureCsv:     document.getElementById('structPath').value || '',
      minMeanThreshold: Number(document.getElementById('minMeanThreshold').value || 8),
      pixelSizeUm:      Number(document.getElementById('pixelSizeUm').value || 0.65),
      rotateAtlas:      Number(document.getElementById('rotateAtlas').value || 0),
      flipAtlas:        document.getElementById('flipAtlas').value || 'none',
      majorTopK:        Number(document.getElementById('majorTopK').value || 20),
      fitMode, alpha, mode, edgeSmoothIter: mode === 'fill' ? 2 : 1,
    };
    
    let respJson = await _runWithProgress('/api/overlay/preview', '/api/overlay/preview/status', payload, 'Generating Preview...');
    if (!respJson) {
      if (mode !== 'contour') {
        mode = 'contour'; modeEl.value = 'contour';
        respJson = await _runWithProgress('/api/overlay/preview', '/api/overlay/preview/status', { ...payload, mode: 'contour' }, 'Generating Preview (Fallback)...');
        if (respJson && respJson.ok) { showToast(t('toast.fillModeFallback'), 'warning'); }
        else { showToast(t('toast.previewFailed'), 'error'); return; }
      } else { showToast(t('toast.previewFailed'), 'error'); return; }
    }
    syncOverlayJobId(respJson);
    const dg = respJson?.diagnostic;
    if (dg) {
      const ra = Number(dg.real_aspect || 0), aa = Number((dg.atlas_aspect ?? dg.atlas_aspect_before) || 0);
      if (ra > 0 && aa > 0 && Math.abs(ra / aa - 1) > 0.35)
        showToast(t('toast.aspectWarning', { ra: ra.toFixed(2), aa: aa.toFixed(2) }), 'warning', 7000);
    }
    
    // Load into canvas
    const loaded = await loadPreviewIntoCanvas();
    if (loaded) {
      hoverLastPixelKey = '';
      hideRegionTooltip();
      showToast(t('toast.previewUpdated'), 'success', 2000);
    } else {
      showToast(t('toast.previewFailed'), 'error');
    }
  } catch (err) {
    console.error('refreshOverlayPreviewWithCanvas failed:', err);
  }
}

// Replace the existing onclick handler
document.getElementById('refreshPreviewBtn').onclick = refreshOverlayPreviewWithCanvas;
// Also re-hook alpha/mode change
alphaRange.oninput = () => { alphaValue.textContent = `${alphaRange.value}%`; refreshOverlayPreviewWithCanvas(); };
document.getElementById('overlayMode').onchange = refreshOverlayPreviewWithCanvas;

async function runOneClickWorkflow() {
  const source = String(oneClickSourcePathEl?.value || '').trim();
  const scope = String(oneClickScopeEl?.value || 'single');
  if (!source) {
    showToast(t('toast.chooseSourceFirst'), 'warning');
    return;
  }

  // Validate: reject folder paths (no .tif extension)
  const srcLower = source.toLowerCase().replace(/\\/g, '/');
  if (srcLower.endsWith('/') || srcLower.endsWith('\\') ||
      (!srcLower.endsWith('.tif') && !srcLower.endsWith('.tiff') && !srcLower.endsWith('.nii.gz'))) {
    showToast(t('toast.folderNotFile'), 'warning');
    return;
  }

  // Auto-detect pixel size from filename (e.g. "z5um" → 5)
  const pxMatch = source.match(/[_\-]z?(\d+(?:\.\d+)?)um/i);
  if (pxMatch) {
    const hintPx = parseFloat(pxMatch[1]);
    const currentPxEl = document.getElementById('oneClickPixelSize');
    const currentPx = parseFloat(currentPxEl?.value || '0');
    if (currentPx > 0 && hintPx > 0 && (currentPx / hintPx > 2 || hintPx / currentPx > 2)) {
      showToast(t('toast.pixelSizeMismatch', { hint: hintPx, current: currentPx }), 'warning', 8000);
    }
  }

  document.getElementById('realSlicePath').value = source;
  document.getElementById('realSlicePath').dispatchEvent(new Event('change'));
  try { await checkSliceIs3D(source); } catch {}

  // Single-layer flow: force user to pass through Z selection step for 3D stacks.
  const zVisible = !!(zSlicerBox && !zSlicerBox.classList.contains('hidden'));
  if (scope === 'single' && zVisible && oneClickStartBtn?.dataset?.zConfirmed !== '1') {
    if (oneClickStartBtn) oneClickStartBtn.dataset.zConfirmed = '1';
    zExtractStatus.textContent = t('hint.zChoose');
    revealZSlicer();
    showToast(t('toast.3dDetected'), 'info', 6000);
    return;
  }
  if (oneClickStartBtn) oneClickStartBtn.dataset.zConfirmed = '0';

  // Show workflow step indicator — starting Step 1 (config)
  updateWorkflowStepIndicator(1);

  try {
    const info = await fetch('/api/info').then(r => r.json());
    const defs = info?.defaults || {};
    if (!document.getElementById('atlasPath').value && defs.atlasPath) {
      document.getElementById('atlasPath').value = defs.atlasPath;
    }
    if (!document.getElementById('structPath').value && defs.structPath) {
      document.getElementById('structPath').value = defs.structPath;
    }
    if (!document.getElementById('outputDir').value && info.outputs) {
      document.getElementById('outputDir').value = info.outputs;
    }
    if (!document.getElementById('inputDir').value) {
      const p = source.replaceAll('\\', '/');
      document.getElementById('inputDir').value = p.includes('/') ? p.substring(0, p.lastIndexOf('/')) : p;
    }
  } catch {}

  // Whole-brain uses stronger default align mode.
  if (scope === 'whole') {
    const alignEl = document.getElementById('alignMode');
    if (alignEl) alignEl.value = 'nonlinear';
  }

  // Apply hemisphere selection to flip atlas control.
  const hemiEl = document.getElementById('oneClickHemisphere');
  if (hemiEl && hemiEl.value !== 'auto') {
    const flipEl = document.getElementById('flipAtlas');
    if (flipEl) {
      const hemiMap = { 'full': 'none', 'left': 'none', 'right_flipped': 'h' };
      flipEl.value = hemiMap[hemiEl.value] || 'none';
    }
  }

  // Step 2: Auto-pick atlas slice
  updateWorkflowStepIndicator(2);
  const okAuto = await ensureAutoPickedAtlasSlice(source);
  if (!okAuto) {
    showToast(t('toast.autoPickFailed'), 'error', 5000);
    return;
  }
  try { await refreshOverlayPreviewWithCanvas(); } catch (e) { console.warn('Pre-align preview skipped:', e); }

  // Step 3: AI registration
  updateWorkflowStepIndicator(3);
  const aiAlignHandler = document.getElementById('aiAlignBtn')?.onclick;
  let aiAlignOk = true;
  if (typeof aiAlignHandler === 'function') {
    try {
      await aiAlignHandler();
    } catch {
      aiAlignOk = false;
    }
    // Check if the alignment produced a visible result image
    const compareImg = document.getElementById('alignPreviewImg');
    if (!compareImg?.src || compareImg.classList.contains('hidden')) {
      aiAlignOk = false;
    }
  }

  if (!aiAlignOk) {
    // Refresh overlay preview so user sees the current state even if alignment failed.
    try { await refreshOverlayPreviewWithCanvas(); } catch (e) { console.warn('Post-fail preview skipped:', e); }
    // Collapse all steps except Step 3 so manual correction is the focus.
    collapseAllStepsExcept('step3');
    // Scroll to manual landmark section and activate it
    const manualSection = document.getElementById('manualLandmarkSection');
    if (manualSection) {
      manualSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
      manualSection.style.outline = '2px solid #f59e0b';
      setTimeout(() => { manualSection.style.outline = ''; }, 5000);
    }
    if (manualModeBtn && !manualState.active) {
      manualModeBtn.click();
    }
    loadManualImages();
    showToast(t('toast.alignFailedManualHint'), 'warning', 8000);
    return;
  }

  // Refresh overlay preview with post-alignment result before entering manual review.
  try { await refreshOverlayPreviewWithCanvas(); } catch (e) { console.warn('Post-align preview skipped:', e); }

  // Mark Step 3 done and advance to Step 4
  updateWorkflowStepIndicator(4);

  // Expand both Step 3 (manual review available) and Step 4 (run pipeline)
  document.querySelectorAll('.step-card').forEach(card => {
    if (card.id === 'step3' || card.id === 'step4') {
      card.classList.remove('collapsed');
    } else {
      card.classList.add('collapsed');
    }
  });

  // Activate manual mode so user can review/correct if needed
  if (manualModeBtn && !manualState.active) {
    manualModeBtn.click();
  }
  loadManualImages();

  // Scroll to Step 4 (Run Pipeline) so user sees the action button
  const step4Card = document.getElementById('step4');
  if (step4Card) {
    step4Card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  showToast(t('toast.oneClickDone'), 'success', 5000);
}

if (oneClickStartBtn) {
  oneClickStartBtn.onclick = runOneClickWorkflow;
}

if (oneClickScopeEl) {
  oneClickScopeEl.onchange = () => {
    const hint = document.getElementById('oneClickScopeHint');
    if (hint) {
      hint.textContent = oneClickScopeEl.value === 'whole'
        ? t('hint.scopeWhole')
        : t('hint.scopeSingle');
    }
    // Toggle z-slicer visibility: show for single-slice when 3D stack is loaded
    if (zSlicerBox) {
      const has3D = zSlider && parseInt(zSlider.max, 10) > 0;
      if (oneClickScopeEl.value === 'single' && has3D) {
        revealZSlicer();
      } else {
        zSlicerBox.classList.add('hidden');
      }
    }
  };
}

// Auto-persist One-Click settings to localStorage on change
function _autoSaveOneClickSettings() {
  const settings = {};
  ['oneClickHemisphere', 'oneClickAtlasVersion', 'oneClickRegMode', 'oneClickScope'].forEach(id => {
    const el = document.getElementById(id);
    if (el) settings[id] = el.value;
  });
  // Also save target region selections (just IDs)
  settings._targetRegionIds = _targetRegionState.selected.map(r => r.id);
  localStorage.setItem('brainfast.oneclick', JSON.stringify(settings));
}

const oneClickHemiEl = document.getElementById('oneClickHemisphere');
if (oneClickHemiEl) {
  const hemiHintKeys = { 'auto': 'hint.hemiAuto', 'full': 'hint.hemiFull', 'left': 'hint.hemiLeft', 'right_flipped': 'hint.hemiRightFlipped' };
  oneClickHemiEl.onchange = () => {
    const hint = document.getElementById('oneClickHemiHint');
    if (hint) hint.textContent = t(hemiHintKeys[oneClickHemiEl.value] || 'hint.hemiAuto');
    _autoSaveOneClickSettings();
  };
}

const oneClickAtlasVersionEl = document.getElementById('oneClickAtlasVersion');
if (oneClickAtlasVersionEl) {
  const atlasHintKeys = { 'ccfv3': 'hint.atlasCcfv3', 'ccfv3bbp': 'hint.atlasCcfv3bbp' };
  oneClickAtlasVersionEl.onchange = () => {
    const hint = document.getElementById('atlasVersionHint');
    if (hint) hint.textContent = t(atlasHintKeys[oneClickAtlasVersionEl.value] || 'hint.atlasCcfv3');
    // Auto-enable Nissl mode when CCFv3-BBP is selected
    const regModeEl = document.getElementById('oneClickRegMode');
    if (oneClickAtlasVersionEl.value === 'ccfv3bbp' && regModeEl) {
      regModeEl.value = 'nissl_template';
      regModeEl.dispatchEvent(new Event('change'));
    }
    _autoSaveOneClickSettings();
  };
}

const oneClickRegModeEl = document.getElementById('oneClickRegMode');
if (oneClickRegModeEl) {
  const regHintKeys = { 'cross_modal': 'hint.regCrossModal', 'nissl_template': 'hint.regNissl' };
  oneClickRegModeEl.onchange = () => {
    const hint = document.getElementById('regModeHint');
    if (hint) hint.textContent = t(regHintKeys[oneClickRegModeEl.value] || 'hint.regCrossModal');
    // Warn if Nissl selected without CCFv3-BBP atlas
    const atlasVerEl = document.getElementById('oneClickAtlasVersion');
    if (oneClickRegModeEl.value === 'nissl_template' && atlasVerEl && atlasVerEl.value !== 'ccfv3bbp') {
      showToast(t('hint.regNissl'), 'warning', 5000);
    }
    _autoSaveOneClickSettings();
  };
}

// Also auto-save when scope changes
if (oneClickScopeEl) {
  const _origScopeChange = oneClickScopeEl.onchange;
  oneClickScopeEl.onchange = () => { if (_origScopeChange) _origScopeChange(); _autoSaveOneClickSettings(); };
}

// --- Target Brain Region Selector with multi-select tags ---
const _targetRegionState = { allRegions: [], selected: [], apRange: null };

(async function initTargetRegionSelector() {
  const searchEl = document.getElementById('targetRegionSearch');
  const selectEl = document.getElementById('targetRegionSelect');
  const clearBtn = document.getElementById('targetRegionClearBtn');
  const tagsEl = document.getElementById('targetRegionTags');
  const hintEl = document.getElementById('targetRegionHint');
  if (!searchEl || !selectEl) return;

  // Fetch region list
  try {
    const res = await fetch('/api/atlas/region-ap-ranges').then(r => r.json());
    if (res.ok) _targetRegionState.allRegions = res.regions;
  } catch (e) { console.warn('Failed to load region AP ranges:', e); }

  function renderTags() {
    tagsEl.innerHTML = '';
    _targetRegionState.selected.forEach(r => {
      const tag = document.createElement('span');
      tag.style.cssText = 'display:inline-flex;align-items:center;gap:3px;background:#2a4a6a;color:#cde;padding:2px 8px;border-radius:12px;font-size:0.82em;';
      tag.innerHTML = `${escapeHtml(r.acronym)} <span style="cursor:pointer;font-weight:bold;margin-left:2px;" title="Remove">&times;</span>`;
      tag.querySelector('span').onclick = () => { removeRegion(r.id); };
      tagsEl.appendChild(tag);
    });
    updateApRange();
  }

  function updateApRange() {
    const sel = _targetRegionState.selected;
    if (sel.length === 0) {
      _targetRegionState.apRange = null;
      if (hintEl) hintEl.textContent = t('hint.targetRegionNone');
      return;
    }
    const apStart = Math.min(...sel.map(r => r.ap_start));
    const apEnd = Math.max(...sel.map(r => r.ap_end));
    _targetRegionState.apRange = { start: apStart, end: apEnd };
    if (hintEl) hintEl.textContent = t('hint.targetRegionSelected', {
      start: apStart, end: apEnd,
      startMm: sel.reduce((m, r) => Math.min(m, r.ap_start_mm), 99).toFixed(1),
      endMm: sel.reduce((m, r) => Math.max(m, r.ap_end_mm), -99).toFixed(1),
    });
  }

  function addRegion(region) {
    if (_targetRegionState.selected.find(r => r.id === region.id)) return;
    _targetRegionState.selected.push(region);
    searchEl.value = '';
    selectEl.style.display = 'none';
    renderTags();
    _autoSaveOneClickSettings();
  }

  function removeRegion(id) {
    _targetRegionState.selected = _targetRegionState.selected.filter(r => r.id !== id);
    renderTags();
    _autoSaveOneClickSettings();
  }

  searchEl.oninput = () => {
    const q = searchEl.value.trim().toLowerCase();
    if (q.length < 1) { selectEl.style.display = 'none'; return; }
    const filtered = _targetRegionState.allRegions
      .filter(r => r.acronym.toLowerCase().includes(q) || r.name.toLowerCase().includes(q));
    // Sort: exact acronym > prefix acronym > prefix name > substring
    filtered.sort((a, b) => {
      const aAcr = a.acronym.toLowerCase(), bAcr = b.acronym.toLowerCase();
      const aExact = aAcr === q ? 0 : 1, bExact = bAcr === q ? 0 : 1;
      if (aExact !== bExact) return aExact - bExact;
      const aPrefix = aAcr.startsWith(q) ? 0 : 1, bPrefix = bAcr.startsWith(q) ? 0 : 1;
      if (aPrefix !== bPrefix) return aPrefix - bPrefix;
      const aNPrefix = a.name.toLowerCase().startsWith(q) ? 0 : 1;
      const bNPrefix = b.name.toLowerCase().startsWith(q) ? 0 : 1;
      if (aNPrefix !== bNPrefix) return aNPrefix - bNPrefix;
      return a.depth - b.depth;
    });
    const matches = filtered.slice(0, 30);
    selectEl.innerHTML = '';
    matches.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.id;
      const indent = '\u00A0'.repeat(Math.max(0, (r.depth - 2) * 2));
      opt.textContent = `${indent}${r.acronym} — ${r.name} (AP ${r.ap_start}–${r.ap_end})`;
      selectEl.appendChild(opt);
    });
    selectEl.style.display = matches.length > 0 ? '' : 'none';
  };

  selectEl.onchange = () => {
    const id = Number(selectEl.value);
    const region = _targetRegionState.allRegions.find(r => r.id === id);
    if (region) addRegion(region);
  };

  // Also handle Enter key to select first match
  searchEl.onkeydown = (e) => {
    if (e.key === 'Enter' && selectEl.options.length > 0) {
      e.preventDefault();
      const id = Number(selectEl.options[0].value);
      const region = _targetRegionState.allRegions.find(r => r.id === id);
      if (region) addRegion(region);
    }
  };

  if (clearBtn) clearBtn.onclick = () => {
    _targetRegionState.selected = [];
    searchEl.value = '';
    selectEl.style.display = 'none';
    renderTags();
    _autoSaveOneClickSettings();
  };

  // Restore saved One-Click settings (hemisphere, atlas version, reg mode, target regions)
  try {
    const saved = JSON.parse(localStorage.getItem('brainfast.oneclick') || 'null');
    if (saved) {
      ['oneClickHemisphere', 'oneClickAtlasVersion', 'oneClickRegMode', 'oneClickScope'].forEach(id => {
        const el = document.getElementById(id);
        if (el && saved[id]) { el.value = saved[id]; el.dispatchEvent(new Event('change')); }
      });
      // Restore target regions by ID
      if (saved._targetRegionIds && saved._targetRegionIds.length && _targetRegionState.allRegions.length) {
        saved._targetRegionIds.forEach(id => {
          const region = _targetRegionState.allRegions.find(r => r.id === id);
          if (region) addRegion(region);
        });
      }
    }
  } catch (e) { console.warn('Failed to restore One-Click settings:', e); }
})();

// Expose target region AP range for autopick
function getTargetApRange() { return _targetRegionState.apRange; }

if (quickExportBtn) {
  quickExportBtn.onclick = async () => {
    const fmt = String(quickExportFormatEl?.value || 'png');
    try {
      const res = await fetch('/api/overlay/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId: getOverlayJobId(), format: fmt }),
      }).then(r => r.json());
      if (!res.ok) {
        showToast(`Export failed: ${res.error || '?'}`, 'error');
        return;
      }
      showToast(`Exported: ${res.path}`, 'success', 5000);
    } catch (e) {
      showToast(`Export failed: ${e?.message || '?'}`, 'error');
    }
  };
}

// ================================================================
// MANUAL LANDMARK CORRECTION
// ================================================================
const manualState = {
  active: false,
  pendingReal: null,    // { x, y } in canvas coords
  pairs: [],
};

const manualModeBtn   = document.getElementById('manualModeBtn');
const applyManualBtn  = document.getElementById('applyManualBtn');
const clearManualBtn  = document.getElementById('clearManualBtn');
const manualCanvases  = document.getElementById('manualCanvases');
const manualStatus    = document.getElementById('manualStatus');
const manualPairsWrap = document.getElementById('manualPairsWrap');
const manualPairsBody = document.getElementById('manualPairsBody');
const manualRealCanvas  = document.getElementById('manualRealCanvas');
const manualAtlasCanvas = document.getElementById('manualAtlasCanvas');
const manualRealImg     = document.getElementById('manualRealImg');
const manualAtlasImg    = document.getElementById('manualAtlasImg');
const mrcCtx = manualRealCanvas.getContext('2d');
const macCtx = manualAtlasCanvas.getContext('2d');

async function loadManualImages() {
  const rPath = document.getElementById('realSlicePath').value;
  const aPath = document.getElementById('atlasLabelPath').value;
  if (!rPath || !aPath) return;
  // Ensure atlas-layer PNG exists (one-click workflow may not have generated it yet)
  const jobId = getOverlayJobId();
  const atlasUrl = withOverlayJobQuery('/api/outputs/atlas-layer', { ts: Date.now() });
  const checkRes = await fetch(atlasUrl, { method: 'HEAD' }).catch(() => null);
  if (!checkRes || !checkRes.ok) {
    // Trigger atlas-layer render on the fly
    try {
      await fetch('/api/overlay/atlas-layer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jobId,
          labelPath: aPath,
          realPath: rPath,
          structureCsv: document.getElementById('structPath')?.value || '',
          pixelSizeUm: parseFloat(document.getElementById('pixelSizeUm')?.value) || 5,
          fitMode: document.getElementById('fitMode')?.value || 'cover',
        }),
      });
    } catch (e) { console.warn('atlas-layer render failed:', e); }
  }
  // Real slice: combined overlay preview; Atlas: atlas-only label layer
  manualRealImg.src = withOverlayJobQuery('/api/outputs/overlay-preview', { ts: Date.now() });
  manualAtlasImg.src = withOverlayJobQuery('/api/outputs/atlas-layer', { ts: Date.now() });
  manualRealImg.onload = () => {
    manualRealCanvas.width  = manualRealImg.naturalWidth;
    manualRealCanvas.height = manualRealImg.naturalHeight;
    redrawManual();
  };
  manualAtlasImg.onload = () => {
    manualAtlasCanvas.width  = manualAtlasImg.naturalWidth;
    manualAtlasCanvas.height = manualAtlasImg.naturalHeight;
    redrawManual();
  };
}

function redrawManual() {
  mrcCtx.clearRect(0, 0, manualRealCanvas.width, manualRealCanvas.height);
  macCtx.clearRect(0, 0, manualAtlasCanvas.width, manualAtlasCanvas.height);
  const DOT = 6;
  manualState.pairs.forEach((p, i) => {
    // Real side
    mrcCtx.fillStyle = '#00ff88'; mrcCtx.strokeStyle = '#000';
    mrcCtx.beginPath(); mrcCtx.arc(p.real_x, p.real_y, DOT, 0, 2*Math.PI); mrcCtx.fill(); mrcCtx.stroke();
    mrcCtx.fillStyle = '#fff'; mrcCtx.font = 'bold 11px sans-serif'; mrcCtx.textAlign = 'center';
    mrcCtx.fillText(i+1, p.real_x, p.real_y - DOT - 2);
    // Atlas side
    macCtx.fillStyle = '#ffcc00'; macCtx.strokeStyle = '#000';
    macCtx.beginPath(); macCtx.arc(p.atlas_x, p.atlas_y, DOT, 0, 2*Math.PI); macCtx.fill(); macCtx.stroke();
    macCtx.fillStyle = '#fff'; macCtx.font = 'bold 11px sans-serif'; macCtx.textAlign = 'center';
    macCtx.fillText(i+1, p.atlas_x, p.atlas_y - DOT - 2);
  });
  if (manualState.pendingReal) {
    mrcCtx.fillStyle = '#ff4444'; mrcCtx.strokeStyle = '#000';
    mrcCtx.beginPath(); mrcCtx.arc(manualState.pendingReal.x, manualState.pendingReal.y, DOT, 0, 2*Math.PI); mrcCtx.fill(); mrcCtx.stroke();
  }
}

function updateManualPairsTable() {
  manualPairsBody.innerHTML = '';
  manualState.pairs.forEach((p, i) => {
    const tr = document.createElement('tr');
  tr.innerHTML = `<td>${i+1}</td><td>(${Math.round(p.real_x)}, ${Math.round(p.real_y)})</td><td>(${Math.round(p.atlas_x)}, ${Math.round(p.atlas_y)})</td><td><button type="button" aria-label="Remove landmark pair ${i+1}" onclick="removeManualPair(${i})" style="background:transparent;color:var(--danger);border:none;cursor:pointer;">&times;</button></td>`;
    manualPairsBody.appendChild(tr);
  });
  manualPairsWrap.classList.toggle('hidden', manualState.pairs.length === 0);
  applyManualBtn.classList.toggle('hidden', manualState.pairs.length === 0);
  clearManualBtn.classList.toggle('hidden', manualState.pairs.length === 0);
}
window.removeManualPair = (i) => {
  manualState.pairs.splice(i, 1);
  updateManualPairsTable(); redrawManual();
};

function manualCanvasCoords(canvas, e) {
  const rect = canvas.getBoundingClientRect();
  return { x: (e.clientX - rect.left) * (canvas.width / rect.width), y: (e.clientY - rect.top) * (canvas.height / rect.height) };
}

manualRealCanvas.addEventListener('click', e => {
  if (!manualState.active) return;
  const { x, y } = manualCanvasCoords(manualRealCanvas, e);
  manualState.pendingReal = { x, y };
  manualStatus.textContent = t('manual.pendingReal', { x: Math.round(x), y: Math.round(y) });
  manualStatus.classList.remove('hidden');
  redrawManual();
});

manualAtlasCanvas.addEventListener('click', e => {
  if (!manualState.active) return;
  if (!manualState.pendingReal) { manualStatus.textContent = t('manual.needReal'); return; }
  const { x, y } = manualCanvasCoords(manualAtlasCanvas, e);
  const pair = { real_x: manualState.pendingReal.x, real_y: manualState.pendingReal.y, atlas_x: x, atlas_y: y };
  manualState.pairs.push(pair);
  manualState.pendingReal = null;
  const n = manualState.pairs.length;
  manualStatus.textContent = t('manual.pairAdded', { n });
  updateManualPairsTable(); redrawManual();
});

manualModeBtn.onclick = () => {
  const rPath = document.getElementById('realSlicePath').value;
  const aPath = document.getElementById('atlasLabelPath').value;
  if (!rPath || !aPath) { showToast(t('manual.needImages'), 'warning'); return; }
  manualState.active = !manualState.active;
  if (manualState.active) {
    manualModeBtn.classList.add('active-mode');
    manualCanvases.classList.remove('hidden');
    manualStatus.classList.remove('hidden');
    manualStatus.textContent = t('manual.enterMode');
    loadManualImages();
    showToast(t('manual.enterMode'), 'info', 4000);
  } else {
    manualModeBtn.classList.remove('active-mode');
    manualCanvases.classList.add('hidden');
    manualStatus.classList.add('hidden');
    showToast(t('manual.exitMode'), 'info', 2000);
  }
};

clearManualBtn.onclick = () => {
  manualState.pairs = []; manualState.pendingReal = null;
  updateManualPairsTable(); redrawManual();
};

applyManualBtn.onclick = async () => {
  if (manualState.pairs.length === 0) return;
  try {
    const res = await fetch('/api/align/add-manual-landmarks', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ jobId: getOverlayJobId(), pairs: manualState.pairs }),
    }).then(r => r.json());
    if (!res.ok) { showToast(t('manual.applyFail', { err: res.error }), 'error'); return; }
    syncOverlayJobId(res);
    showToast(t('manual.applyOk', { n: res.total_pairs }), 'success', 5000);
    // Auto-trigger re-alignment
    document.getElementById('aiAlignBtn').click();
  } catch { showToast(t('manual.applyFail', { err: '?' }), 'error'); }
};

// ================================================================
// OUTPUT FILE BROWSER
// ================================================================
async function refreshFileList() {
  const grid  = document.getElementById('outputFileGrid');
  const empty = document.getElementById('outputFileEmpty');
  try {
  const res = await fetch(withActiveJobQuery('/api/outputs/file-list')).then(r => r.json());
    if (!res.ok || res.files.length === 0) { grid.innerHTML = ''; empty.classList.remove('hidden'); return; }
    empty.classList.add('hidden');
    grid.innerHTML = '';
    const ICONS = { '.png': 'image', '.tif': 'microscope', '.tiff': 'microscope', '.csv': 'table', '.json': 'braces', '.txt': 'file-text' };
    res.files.forEach(f => {
      const card = document.createElement('div');
      card.className = 'output-file-card';
      const iconName = ICONS[f.ext] || 'folder';
      const sizeStr = f.size > 1024*1024 ? `${(f.size/1024/1024).toFixed(1)} MB` : `${(f.size/1024).toFixed(0)} KB`;
      card.innerHTML = `<span class="file-icon"><i data-lucide="${iconName}"></i></span><span class="file-name" title="${f.name}">${f.name}</span><span class="file-size">${sizeStr}</span>`;
      card.onclick = () => handleOutputFileClick(f);
      grid.appendChild(card);
    });
    if (typeof lucide !== 'undefined') lucide.createIcons();
  } catch {}
}

async function handleOutputFileClick(f) {
  if (f.ext === '.png') {
        openLightbox(withActiveJobQuery(`/api/outputs/named/${f.name}`, { ts: Date.now() }), f.name);
  } else if (f.ext === '.csv' || f.ext === '.json' || f.ext === '.txt') {
    try {
        const text = await fetch(withActiveJobQuery(`/api/outputs/named/${f.name}`)).then(r => r.text());
      const capped = text.slice(0, 8000) + (text.length > 8000 ? '\n...(truncated)' : '');
      openTextModal(f.name, t('outputs.previewDesc'), capped);
    } catch {}
  }
}

document.getElementById('refreshFileListBtn').onclick = refreshFileList;

// ================================================================
// CROSS-SAMPLE COMPARISON
// ================================================================
(function initSampleCompare() {
  const dirList = document.getElementById('compareDirList');
  const resultTable = document.getElementById('compareResultTable');

  function addDirRow(dir = '', label = '') {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:8px;align-items:center';
    row.innerHTML = `
      <input class="compare-dir-input" type="text" value="${dir.replace(/"/g,'')}"
        placeholder="${t('compare.multi.dirPlaceholder')}"
        style="flex:3;padding:5px 8px;background:#1e1e1e;border:1px solid #333;border-radius:4px;color:#ddd;font-size:0.85em"/>
      <input class="compare-label-input" type="text" value="${label.replace(/"/g,'')}"
        placeholder="${t('compare.multi.label')}"
        style="flex:1;padding:5px 8px;background:#1e1e1e;border:1px solid #333;border-radius:4px;color:#ddd;font-size:0.85em"/>
      <button class="btn-ghost" style="padding:4px 8px;font-size:0.85em" onclick="this.closest('div').remove()">✕</button>`;
    dirList.appendChild(row);
  }

  // Seed with 2 rows
  addDirRow(); addDirRow();

  document.getElementById('compareAddDirBtn').onclick = () => addDirRow();

  document.getElementById('compareRunBtn').onclick = async () => {
    const dirs = [...dirList.querySelectorAll('.compare-dir-input')].map(el => el.value.trim()).filter(Boolean);
    const labels = [...dirList.querySelectorAll('.compare-label-input')].map(el => el.value.trim());
    if (dirs.length < 2) { showToast(t('compare.multi.empty'), 'warning'); return; }

    resultTable.innerHTML = '<div style="color:#888;padding:12px">Loading...</div>';
    try {
      const res = await fetch('/api/compare/regions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ output_dirs: dirs, labels }),
      }).then(r => r.json());

      if (!res.ok) { resultTable.innerHTML = `<div style="color:#e53935">${res.error || 'Error'}</div>`; return; }
      const sampleCols = res.sample_labels || dirs.map((_, i) => labels[i] || `Sample ${i + 1}`);
      const regions = res.regions || [];
      if (!regions.length) { resultTable.innerHTML = `<div class="empty-hint">${t('compare.multi.noData')}</div>`; return; }

      let html = '<table style="width:100%;border-collapse:collapse;font-size:0.85em"><thead><tr>';
      html += `<th style="text-align:left;padding:6px 8px;border-bottom:1px solid #333;color:#aaa">Region</th>`;
      sampleCols.forEach(s => { html += `<th style="text-align:right;padding:6px 8px;border-bottom:1px solid #333;color:#aaa">${s}</th>`; });
      html += '</tr></thead><tbody>';
      regions.forEach((row, ri) => {
        const bg = ri % 2 === 0 ? '' : 'background:rgba(255,255,255,0.02)';
        html += `<tr style="${bg}"><td style="padding:5px 8px;color:#ccc">${row.region_name || row.region || '-'}</td>`;
        sampleCols.forEach((_, si) => {
          const val = row[`count_${si}`] ?? row[sampleCols[si]] ?? '—';
          html += `<td style="text-align:right;padding:5px 8px;color:#ddd">${typeof val === 'number' ? val.toLocaleString() : val}</td>`;
        });
        html += '</tr>';
      });
      html += '</tbody></table>';
      resultTable.innerHTML = html;
    } catch (err) {
      resultTable.innerHTML = `<div style="color:#e53935">Request failed: ${err?.message || err}</div>`;
    }
  };
})();
// Auto-refresh when switching to results tab
const _origResultsRefresh = refreshOutputs;
async function refreshOutputsAndFiles() {
  await _origResultsRefresh();
  await refreshFileList();
}
document.getElementById('refreshBtn').onclick = refreshOutputsAndFiles;

// ----------------------------------------------------------------
// AP-AXIS DENSITY PROFILE CHART
// ----------------------------------------------------------------
async function refreshApDensity() {
  const section = document.getElementById('apDensitySection');
  const chartDiv = document.getElementById('apDensityChart');
  if (!section || !chartDiv) return;
  try {
    const r = await fetch(withActiveJobQuery('/api/outputs/ap-density'));
    if (!r.ok) { section.style.display = 'none'; return; }
    const j = await r.json();
    if (!j.ok || !Array.isArray(j.ap_slices) || !j.ap_slices.length) {
      section.style.display = 'none'; return;
    }
    section.style.display = '';
    const items = j.ap_slices;
    const maxCount = Math.max(...items.map(d => d.cell_count), 1);
    const W = Math.max(600, items.length * 8);
    const H = 160;
    const PAD = { t: 12, r: 20, b: 36, l: 52 };
    const cW = W - PAD.l - PAD.r;
    const cH = H - PAD.t - PAD.b;

    const barW = Math.max(2, cW / items.length - 1);
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block;max-width:100%">`;

    // Y-axis label
    svg += `<text x="10" y="${PAD.t + cH/2}" text-anchor="middle" transform="rotate(-90,10,${PAD.t + cH/2})" fill="#888" font-size="11">cells</text>`;

    // Axes
    svg += `<line x1="${PAD.l}" y1="${PAD.t}" x2="${PAD.l}" y2="${PAD.t+cH}" stroke="#444" stroke-width="1"/>`;
    svg += `<line x1="${PAD.l}" y1="${PAD.t+cH}" x2="${PAD.l+cW}" y2="${PAD.t+cH}" stroke="#444" stroke-width="1"/>`;

    // Y ticks
    [0, Math.round(maxCount/2), maxCount].forEach(v => {
      const y = PAD.t + cH - (v / maxCount) * cH;
      svg += `<line x1="${PAD.l-4}" y1="${y}" x2="${PAD.l}" y2="${y}" stroke="#555"/>`;
      svg += `<text x="${PAD.l-6}" y="${y+4}" text-anchor="end" fill="#888" font-size="10">${v}</text>`;
    });

    // Bars
    items.forEach((d, i) => {
      const x = PAD.l + (i / items.length) * cW;
      const bh = (d.cell_count / maxCount) * cH;
      const y = PAD.t + cH - bh;
      svg += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${bh.toFixed(1)}" fill="#5b9bd5" opacity="0.8" rx="1"/>`;
    });

    // X ticks (AP index labels, every ~8 items)
    const step = Math.max(1, Math.round(items.length / 8));
    items.filter((_, i) => i % step === 0).forEach((d, _, arr) => {
      const idx = items.indexOf(d);
      const x = PAD.l + (idx / items.length) * cW + barW / 2;
      svg += `<text x="${x.toFixed(1)}" y="${PAD.t+cH+16}" text-anchor="middle" fill="#888" font-size="10">${d.ap_index}</text>`;
    });
    svg += `<text x="${PAD.l + cW/2}" y="${H-2}" text-anchor="middle" fill="#888" font-size="11">AP index</text>`;

    svg += '</svg>';
    chartDiv.innerHTML = svg;
  } catch { section.style.display = 'none'; }
}

// ================================================================
// CO-EXPRESSION TABLE
// ================================================================
async function refreshCoexpression() {
  const section = document.getElementById('coexpressionSection');
  const tableDiv = document.getElementById('coexpressionTable');
  if (!section || !tableDiv) return;
  try {
    const r = await fetch(withActiveJobQuery('/api/outputs/coexpression'));
    if (!r.ok) { section.style.display = 'none'; return; }
    const j = await r.json();
    if (!j.ok || !Array.isArray(j.regions) || !j.regions.length) {
      section.style.display = 'none'; return;
    }
    section.style.display = '';
    const rows = j.regions.slice(0, 200);
    let html = `<table class="results-table"><thead><tr>
      <th>${t('coexpr.th.region')}</th>
      ${j.channel_red_available ? `<th>${t('coexpr.th.red')}</th>` : ''}
      ${j.channel_green_available ? `<th>${t('coexpr.th.green')}</th>` : ''}
    </tr></thead><tbody>`;
    rows.forEach(r => {
      const label = r.name ? `${r.acronym} — ${r.name}` : (r.acronym || '—');
      html += `<tr>
        <td>${label}</td>
        ${j.channel_red_available ? `<td style="text-align:right">${Math.round(r.count_red)}</td>` : ''}
        ${j.channel_green_available ? `<td style="text-align:right">${Math.round(r.count_green)}</td>` : ''}
      </tr>`;
    });
    html += '</tbody></table>';
    tableDiv.innerHTML = html;
  } catch { section.style.display = 'none'; }
}

// ================================================================
// PROJECTS & BATCH QUEUE
// ================================================================
let _batchPollTimer = null;

function _statusBadge(status) {
  const map = { done: '#2e7d32', running: '#1565c0', queued: '#e65100', pending: '#555', error: '#b71c1c' };
  const bg = map[status] || '#333';
  const key = `sample.status.${status}`;
  return `<span style="background:${bg};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.75em;white-space:nowrap">${t(key) || status}</span>`;
}

async function loadProjects() {
  const container = document.getElementById('projectsList');
  if (!container) return;
  try {
    const res = await fetch('/api/projects').then(r => r.json());
    if (!res.ok || !Array.isArray(res.projects) || !res.projects.length) {
      container.innerHTML = `<div class="empty-hint">${t('projects.empty')}</div>`;
      return;
    }
    container.innerHTML = '';
    res.projects.forEach(proj => {
      const card = document.createElement('div');
      card.style.cssText = 'border:1px solid #2a2a2a;border-radius:8px;margin-bottom:10px;overflow:hidden';
      card.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:#1a1a1a;cursor:pointer" onclick="toggleProjectSamples(this,'${proj.id}')">
          <div>
            <strong style="color:#e0e0e0">${proj.name}</strong>
            ${proj.description ? `<span style="color:#666;font-size:0.85em;margin-left:8px">${proj.description}</span>` : ''}
          </div>
          <span style="color:#555;font-size:0.85em">▶</span>
        </div>
        <div class="project-samples" id="proj-samples-${proj.id}" style="display:none;padding:10px 14px">
          <div class="empty-hint" style="padding:6px 0">Loading…</div>
        </div>
        <div style="padding:8px 14px;border-top:1px solid #222;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
          <input class="search-input" id="sample-name-${proj.id}" placeholder="${t('sample.namePh')}" style="flex:1;max-width:180px;padding:4px 8px;font-size:0.85em"/>
          <input class="search-input" id="sample-cfg-${proj.id}" placeholder="${t('sample.configPh')}" style="flex:2;max-width:280px;padding:4px 8px;font-size:0.85em"/>
          <input class="search-input" id="sample-dir-${proj.id}" placeholder="${t('sample.inputPh')}" style="flex:2;max-width:280px;padding:4px 8px;font-size:0.85em"/>
          <button class="btn-secondary" style="padding:4px 10px;font-size:0.85em" onclick="addSampleToProject('${proj.id}')">${t('sample.addBtn')}</button>
        </div>`;
      container.appendChild(card);
    });
  } catch (err) {
    container.innerHTML = `<div style="color:#e53935">Failed to load projects: ${err?.message || err}</div>`;
  }
}

async function toggleProjectSamples(header, projectId) {
  const panel = document.getElementById(`proj-samples-${projectId}`);
  if (!panel) return;
  if (panel.style.display === 'none') {
    panel.style.display = '';
    await loadProjectSamples(projectId);
  } else {
    panel.style.display = 'none';
  }
}

async function loadProjectSamples(projectId) {
  const panel = document.getElementById(`proj-samples-${projectId}`);
  if (!panel) return;
  try {
    const res = await fetch(`/api/projects/${projectId}/samples`).then(r => r.json());
    if (!res.ok || !Array.isArray(res.samples) || !res.samples.length) {
      panel.innerHTML = `<div class="empty-hint" style="padding:4px 0">No samples yet.</div>`;
      return;
    }
    panel.innerHTML = '';
    res.samples.forEach(s => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:10px;padding:5px 0;border-bottom:1px solid #1e1e1e;flex-wrap:wrap';
      row.innerHTML = `
        <span style="flex:2;color:#ccc;font-size:0.88em">${s.name || s.id}</span>
        ${_statusBadge(s.status || 'pending')}
        <span style="flex:3;color:#555;font-size:0.78em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${s.config_path || ''}">${s.config_path || '—'}</span>
        <button class="btn-ghost" style="padding:3px 8px;font-size:0.8em" onclick="loadSampleAndRun('${s.config_path || ''}','${s.input_dir || ''}')">
          ${t('sample.run')}
        </button>
        <button class="btn-ghost" style="padding:3px 8px;font-size:0.8em;color:#e53935" onclick="enqueueSample('${s.id}','${s.config_path || ''}','${s.input_dir || ''}')">
          ${t('batch.enqueue')}
        </button>`;
      panel.appendChild(row);
    });
  } catch { panel.innerHTML = '<div class="empty-hint" style="padding:4px 0">Load failed.</div>'; }
}

async function addSampleToProject(projectId) {
  const name = document.getElementById(`sample-name-${projectId}`)?.value.trim();
  const cfg = document.getElementById(`sample-cfg-${projectId}`)?.value.trim();
  const dir = document.getElementById(`sample-dir-${projectId}`)?.value.trim();
  if (!name || !cfg) { showToast('Sample name and config path required.', 'warning'); return; }
  try {
    const res = await fetch(`/api/projects/${projectId}/samples`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, config_path: cfg, input_dir: dir }),
    }).then(r => r.json());
    if (res.ok) {
      showToast('Sample added.', 'success');
      await loadProjectSamples(projectId);
    } else {
      showToast(`Failed: ${res.error || '?'}`, 'error');
    }
  } catch (err) { showToast(`Error: ${err?.message}`, 'error'); }
}

function loadSampleAndRun(configPath, inputDir) {
  if (configPath) {
    const cfgEl = document.querySelector('[data-field="configPath"]') || document.getElementById('configPathInput');
    if (cfgEl) cfgEl.value = configPath;
  }
  if (inputDir) {
    const dirEl = document.getElementById('inputDir');
    if (dirEl) dirEl.value = inputDir;
  }
  // Switch to workflow tab
  document.querySelector('.nav-btn[data-tab="workflow"]')?.click();
}

async function enqueueSample(sampleId, configPath, inputDir) {
  try {
    const res = await fetch('/api/batch/enqueue', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id: sampleId, config_path: configPath, input_dir: inputDir }),
    }).then(r => r.json());
    if (res.ok) { showToast('Enqueued.', 'success'); refreshBatchQueue(); }
    else showToast(`Enqueue failed: ${res.error || '?'}`, 'error');
  } catch (err) { showToast(`Error: ${err?.message}`, 'error'); }
}

async function refreshBatchQueue() {
  const container = document.getElementById('batchQueueTable');
  if (!container) return;
  try {
    const res = await fetch('/api/batch/status').then(r => r.json());
    if (!res.ok) { container.innerHTML = `<div class="empty-hint">${t('batch.empty')}</div>`; return; }
    const active = res.active;
    const queued = Array.isArray(res.queued) ? res.queued : [];
    if (!active && !queued.length) {
      container.innerHTML = `<div class="empty-hint">${t('batch.empty')}</div>`;
      clearInterval(_batchPollTimer); _batchPollTimer = null; return;
    }
    // Start polling if queue is active
    if (!_batchPollTimer) {
      _batchPollTimer = setInterval(() => {
        if (document.getElementById('tab-projects')?.classList.contains('active')) refreshBatchQueue();
        else { clearInterval(_batchPollTimer); _batchPollTimer = null; }
      }, 10000);
    }
    let html = '<table style="width:100%;border-collapse:collapse;font-size:0.85em"><thead><tr>';
    ['Sample', 'Status', 'Queued At', 'Action'].forEach(h => {
      html += `<th style="text-align:left;padding:6px 8px;border-bottom:1px solid #333;color:#aaa">${h}</th>`;
    });
    html += '</tr></thead><tbody>';
    const allItems = active ? [{ ...active, _isActive: true }, ...queued] : queued;
    allItems.forEach(item => {
      const cancelBtn = !item._isActive
        ? `<button class="btn-ghost" style="padding:2px 8px;font-size:0.8em;color:#e53935" onclick="cancelBatchItem('${item.sample_id || item.id}')">${t('batch.cancel')}</button>`
        : '';
      html += `<tr><td style="padding:5px 8px;color:#ccc">${item.sample_id || item.id || '—'}</td><td style="padding:5px 8px">${_statusBadge(item.status || (item._isActive ? 'running' : 'queued'))}</td><td style="padding:5px 8px;color:#666;font-size:0.8em">${item.queued_at || item.created_at || '—'}</td><td style="padding:5px 8px">${cancelBtn}</td></tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
  } catch { container.innerHTML = `<div class="empty-hint">${t('batch.empty')}</div>`; }
}

async function cancelBatchItem(sampleId) {
  try {
    const res = await fetch('/api/batch/cancel', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id: sampleId }),
    }).then(r => r.json());
    if (res.ok) { showToast('Cancelled.', 'info'); refreshBatchQueue(); }
    else showToast(`Cancel failed: ${res.error || '?'}`, 'error');
  } catch (err) { showToast(`Error: ${err?.message}`, 'error'); }
}

async function createProject() {
  const name = document.getElementById('newProjectName')?.value.trim();
  const desc = document.getElementById('newProjectDesc')?.value.trim();
  if (!name) { showToast('Project name required.', 'warning'); return; }
  try {
    const res = await fetch('/api/projects', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description: desc }),
    }).then(r => r.json());
    if (res.ok) {
      document.getElementById('newProjectName').value = '';
      document.getElementById('newProjectDesc').value = '';
      showToast('Project created.', 'success');
      await loadProjects();
    } else {
      showToast(`Failed: ${res.error || '?'}`, 'error');
    }
  } catch (err) { showToast(`Error: ${err?.message}`, 'error'); }
}

document.getElementById('createProjectBtn')?.addEventListener('click', createProject);
document.getElementById('refreshProjectsBtn')?.addEventListener('click', () => { loadProjects(); refreshBatchQueue(); });

document.getElementById('pixelSizeUm')?.addEventListener('input', function() {
  this.dataset.userModified = '1';
  const warn = document.getElementById('pixelSizeWarning');
  if (warn) warn.classList.add('hidden');
});

// ===================== Training Tab Logic =====================

async function loadTrainingSet() {
  try {
    var res = await fetch('/api/cellpose/training-set');
    var data = await res.json();
    if (!data.ok) return;

    document.getElementById('tsImageCount').textContent = data.stats.totalImages;
    document.getElementById('tsCellCount').textContent = data.stats.totalCells;
    document.getElementById('tsAvgCells').textContent = data.stats.avgCellsPerImage;
    document.getElementById('tsReadiness').textContent = data.ready ? 'Ready' : 'Need more';
    document.getElementById('tsReadiness').style.color = data.ready ? '#81C784' : '#e94560';

    var listEl = document.getElementById('trainingSetList');
    listEl.innerHTML = '';
    for (var i = 0; i < data.samples.length; i++) {
      var s = data.samples[i];
      var card = document.createElement('div');
      card.className = 'training-sample-card';
      card.innerHTML = '<span class="ts-name">' + s.name + '</span>' +
        '<span class="ts-cells">' + s.cellCount + ' cells</span>' +
        '<button class="ts-delete" data-name="' + s.name + '" title="Remove">&times;</button>';
      listEl.appendChild(card);
    }

    // Bind delete buttons
    listEl.querySelectorAll('.ts-delete').forEach(function(btn) {
      btn.onclick = async function() {
        var name = this.getAttribute('data-name');
        if (!confirm('Remove ' + name + ' from training set?')) return;
        await fetch('/api/cellpose/training-set/' + encodeURIComponent(name), { method: 'DELETE' });
        loadTrainingSet();
      };
    });
  } catch (err) {
    console.error('Failed to load training set:', err);
  }
}

var _trainPollTimer = null;

async function startTraining() {
  var modelName = document.getElementById('trainModelName').value.trim();
  var baseModel = document.getElementById('trainBaseModel').value;
  var epochs = parseInt(document.getElementById('trainEpochs').value, 10) || 100;
  var gpu = document.getElementById('trainGpu').checked;

  if (!modelName) {
    modelName = 'brainfast_' + Date.now();
    document.getElementById('trainModelName').value = modelName;
  }

  try {
    var res = await fetch('/api/cellpose/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ baseModel: baseModel, modelName: modelName, epochs: epochs, gpu: gpu }),
    });
    var data = await res.json();
    if (!data.ok) {
      showToast('Training failed: ' + data.error, 'error');
      return;
    }

    showToast('Training started: ' + modelName, 'success');
    document.getElementById('startTrainingBtn').disabled = true;
    document.getElementById('cancelTrainingBtn').style.display = '';
    document.getElementById('trainingProgressCard').style.display = '';
    document.getElementById('trainingResultCard').style.display = 'none';

    // Start polling
    _trainPollTimer = setInterval(pollTrainingStatus, 2000);
    pollTrainingStatus();
  } catch (err) {
    showToast('Training error: ' + err.message, 'error');
  }
}

async function pollTrainingStatus() {
  try {
    var res = await fetch('/api/cellpose/train-status');
    var data = await res.json();
    if (!data.ok) return;

    var pct = data.totalEpochs > 0 ? Math.round(data.epoch / data.totalEpochs * 100) : 0;
    document.getElementById('trainProgressBar').style.width = pct + '%';
    document.getElementById('trainEpochText').textContent = 'Epoch ' + data.epoch + ' / ' + data.totalEpochs;
    document.getElementById('trainLossText').textContent = 'Loss: ' + (data.trainLoss != null ? data.trainLoss.toFixed(4) : '--');
    document.getElementById('trainEtaText').textContent = 'ETA: ' + (data.estimatedTimeRemaining || '--');

    if (data.status === 'completed') {
      clearInterval(_trainPollTimer);
      _trainPollTimer = null;
      document.getElementById('startTrainingBtn').disabled = false;
      document.getElementById('cancelTrainingBtn').style.display = 'none';
      document.getElementById('trainingResultCard').style.display = '';

      var summary = 'Model: <strong>' + data.modelName + '</strong><br>';
      summary += 'Final train loss: ' + (data.trainLoss != null ? data.trainLoss.toFixed(4) : '--') + '<br>';
      summary += 'Final test loss: ' + (data.testLoss != null ? data.testLoss.toFixed(4) : '--') + '<br>';
      summary += 'Model path: ' + data.modelPath;
      document.getElementById('trainResultSummary').innerHTML = summary;

      showToast('Training completed: ' + data.modelName, 'success', 5000);

      // Refresh model list in detection panel
      if (typeof loadCellposeModels === 'function') loadCellposeModels();
    } else if (data.status === 'failed') {
      clearInterval(_trainPollTimer);
      _trainPollTimer = null;
      document.getElementById('startTrainingBtn').disabled = false;
      document.getElementById('cancelTrainingBtn').style.display = 'none';

      var msgEl = document.getElementById('trainStatusMsg');
      msgEl.className = 'training-status-msg error';
      msgEl.textContent = 'Training failed: ' + data.error;

      showToast('Training failed: ' + data.error, 'error');
    } else if (data.status === 'cancelled') {
      clearInterval(_trainPollTimer);
      _trainPollTimer = null;
      document.getElementById('startTrainingBtn').disabled = false;
      document.getElementById('cancelTrainingBtn').style.display = 'none';

      var msgEl2 = document.getElementById('trainStatusMsg');
      msgEl2.className = 'training-status-msg';
      msgEl2.textContent = 'Training cancelled.';
    }
  } catch (err) {
    console.error('Training poll error:', err);
  }
}

async function cancelTraining() {
  await fetch('/api/cellpose/train-cancel', { method: 'POST' });
  showToast('Training cancelled', 'warning');
}

async function applyTrainedModel() {
  try {
    var res = await fetch('/api/cellpose/train-status');
    var data = await res.json();
    if (!data.modelName) {
      showToast('No model to apply', 'warning');
      return;
    }

    var res2 = await fetch('/api/cellpose/apply-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelName: data.modelName }),
    });
    var result = await res2.json();
    if (result.ok) {
      showToast('Model applied: ' + result.appliedModel, 'success', 4000);
      if (typeof loadCellposeModels === 'function') loadCellposeModels();
    } else {
      showToast('Failed to apply model: ' + result.error, 'error');
    }
  } catch (err) {
    showToast('Apply model error: ' + err.message, 'error');
  }
}

// Bind training tab events
document.getElementById('refreshTrainingSetBtn').addEventListener('click', loadTrainingSet);
document.getElementById('startTrainingBtn').addEventListener('click', startTraining);
document.getElementById('cancelTrainingBtn').addEventListener('click', cancelTraining);
document.getElementById('applyModelBtn').addEventListener('click', applyTrainedModel);

// Auto-load training set when tab is shown
document.querySelectorAll('.nav-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    if (btn.getAttribute('data-tab') === 'training') {
      loadTrainingSet();
    }
  });
});

// ================================================================
// GUIDED TOUR
// ================================================================
(function() {
  const TOUR_KEY = 'idlebrain.tourDone';

  const STEPS = [
    {
      target: '#oneClickSourcePath',
      titleKey: 'tour.step1.title',
      bodyKey:  'tour.step1.body',
      tab: 'workflow',
    },
    {
      target: '#oneClickAtlasVersion',
      titleKey: 'tour.step2.title',
      bodyKey:  'tour.step2.body',
      tab: 'workflow',
    },
    {
      target: '#oneClickRegMode',
      titleKey: 'tour.step3.title',
      bodyKey:  'tour.step3.body',
      tab: 'workflow',
    },
    {
      target: '#oneClickStartBtn',
      titleKey: 'tour.step4.title',
      bodyKey:  'tour.step4.body',
      tab: 'workflow',
    },
    {
      target: '#exportBtn',
      titleKey: 'tour.step5.title',
      bodyKey:  'tour.step5.body',
      tab: 'results',
    },
  ];

  let _overlay = null;
  let _highlight = null;
  let _tooltip = null;
  let _stepIdx = 0;

  function _isVisibleTourTarget(el) {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return r.width > 0 && r.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
  }

  function _handleTourKeydown(ev) {
    if (ev.key === 'Escape') _endTour(false);
  }

  function _switchTab(tabName) {
    document.querySelectorAll('.nav-btn[data-tab]').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    const btn = document.querySelector(`.nav-btn[data-tab="${tabName}"]`);
    const pane = document.getElementById(`tab-${tabName}`);
    if (btn) btn.classList.add('active');
    if (pane) pane.classList.add('active');
  }

  function _positionTooltip(targetEl, tooltipEl) {
    const r = targetEl.getBoundingClientRect();
    const tw = tooltipEl.offsetWidth || 310;
    const th = tooltipEl.offsetHeight || 160;
    const margin = 16;
    let top = r.bottom + margin;
    let left = r.left;
    if (top + th > window.innerHeight - margin) top = r.top - th - margin;
    if (left + tw > window.innerWidth - margin) left = window.innerWidth - tw - margin;
    if (left < margin) left = margin;
    if (top < margin) top = margin;
    tooltipEl.style.top = top + 'px';
    tooltipEl.style.left = left + 'px';
  }

  function _showStep(idx) {
    _stepIdx = idx;
    const step = STEPS[idx];
    if (!step) { _endTour(true); return; }

    if (step.tab) _switchTab(step.tab);

    const targetEl = document.querySelector(step.target);
    if (!targetEl) { _showStep(idx + 1); return; }  // skip missing elements
    if (!_isVisibleTourTarget(targetEl)) { _showStep(idx + 1); return; }

    targetEl.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'auto' });

    // Position highlight
    setTimeout(() => {
      const r = targetEl.getBoundingClientRect();
      const pad = 6;
      _highlight.style.top    = (r.top - pad) + 'px';
      _highlight.style.left   = (r.left - pad) + 'px';
      _highlight.style.width  = (r.width + pad * 2) + 'px';
      _highlight.style.height = (r.height + pad * 2) + 'px';

      // Build tooltip
      const isLast = idx === STEPS.length - 1;
      _tooltip.innerHTML = `
        <h3>${t(step.titleKey)}</h3>
        <p>${t(step.bodyKey)}</p>
        <div class="tour-tooltip-footer">
          <span class="tour-step-counter">${idx + 1} / ${STEPS.length}</span>
          <div class="tour-btn-row">
            <button class="tour-btn tour-btn-skip" id="_tourSkip">${t('tour.skip')}</button>
            <button class="tour-btn tour-btn-next" id="_tourNext">${isLast ? t('tour.done') : t('tour.next')}</button>
          </div>
        </div>`;
      document.getElementById('_tourSkip').onclick = () => _endTour(false);
      document.getElementById('_tourNext').onclick = () => (isLast ? _endTour(true) : _showStep(idx + 1));

      _positionTooltip(targetEl, _tooltip);
    }, step.tab ? 120 : 0);
  }

  function startTour() {
    if (!_overlay) {
      _overlay   = document.createElement('div');
      _highlight = document.createElement('div');
      _tooltip   = document.createElement('div');
      _overlay.className   = 'tour-overlay';
      _highlight.className = 'tour-highlight';
      _tooltip.className   = 'tour-tooltip';
      _overlay.addEventListener('click', () => _endTour(false));
      document.body.append(_overlay, _highlight, _tooltip);
    }
    _overlay.style.display = _highlight.style.display = _tooltip.style.display = '';
    document.addEventListener('keydown', _handleTourKeydown);
    _showStep(0);
  }

  function _endTour(completed) {
    if (_overlay) { _overlay.style.display = _highlight.style.display = _tooltip.style.display = 'none'; }
    document.removeEventListener('keydown', _handleTourKeydown);
    if (completed) localStorage.setItem(TOUR_KEY, '1');
  }

  // Tour is opt-in via the "?" button in the sidebar. Auto-start made first-run
  // users think the app was stuck when a target was offscreen or hidden.
  document.getElementById('startTourBtn')?.addEventListener('click', startTour);
})();

// ==========================================================================
// Phase β — 3D Landmark Liquify tab
// ==========================================================================
// Self-contained IIFE so it does not collide with the 2D manual-landmark
// state. Backend: /api/liquify-3d/* (see api_liquify_3d.py).
(function () {
  const tabEl = document.getElementById('tab-liquify3d');
  if (!tabEl) return; // HTML not present — defensive guard

  const el = (id) => document.getElementById(id);
  const state = {
    jobId: '',
    sliceFiles: [],        // e.g. ['slice_0000_overlay.png', ...]
    currentZ: 0,
    pendingAtlas: null,    // { y, x } in image-native pixel coords
    pendingReal: null,
    clickMode: 'atlas',
    imageNaturalSize: { h: 0, w: 0 },
    pairs: [],
  };

  // localStorage key for tab state — survives browser reloads and tab
  // switches so the user does not lose their place between corrections.
  const _LS_KEY = 'liquify3d.state.v1';
  function _saveTabState() {
    try {
      localStorage.setItem(
        _LS_KEY,
        JSON.stringify({
          jobId: el('liq3dJobId')?.value || '',
          className: el('liq3dClassName')?.value || '',
          currentZ: state.currentZ,
          clickMode: state.clickMode,
        }),
      );
    } catch (_) { /* private mode etc. — ignore */ }
  }
  function _loadTabState() {
    try {
      const raw = localStorage.getItem(_LS_KEY);
      if (!raw) return;
      const s = JSON.parse(raw);
      if (s.jobId && el('liq3dJobId')) el('liq3dJobId').value = s.jobId;
      if (s.className && el('liq3dClassName')) el('liq3dClassName').value = s.className;
      if (typeof s.currentZ === 'number') state.currentZ = s.currentZ;
      if (s.clickMode === 'atlas' || s.clickMode === 'real') {
        state.clickMode = s.clickMode;
        const r = document.querySelector(
          `input[name="liq3dClickMode"][value="${state.clickMode}"]`,
        );
        if (r) r.checked = true;
      }
    } catch (_) { /* malformed JSON — ignore */ }
  }
  _loadTabState();

  const jobInput   = el('liq3dJobId');
  const reloadBtn  = el('liq3dReloadBtn');
  const sliceStatus= el('liq3dSliceStatus');
  const zRange     = el('liq3dZ');
  const zNum       = el('liq3dZNum');
  const zMax       = el('liq3dZMax');
  const pendingStatus = el('liq3dPendingStatus');
  const img        = el('liq3dImg');
  const overlayImg = el('liq3dOverlayImg');
  const canvas     = el('liq3dCanvas');
  const ctx        = canvas.getContext('2d');
  const applyBtn   = el('liq3dApplyBtn');
  const clearBtn   = el('liq3dClearBtn');
  const applyStatus= el('liq3dApplyStatus');
  const pairsBody  = el('liq3dPairsBody');
  const pairCountEl = el('liq3dPairCount');
  const overlayRow      = el('liq3dOverlayRow');
  const overlayToggle   = el('liq3dOverlayToggle');
  const overlayChannel  = el('liq3dOverlayChannel');
  const overlayColor    = el('liq3dOverlayColor');
  const overlayOpacity  = el('liq3dOverlayOpacity');
  const overlayOpacityNum = el('liq3dOverlayOpacityNum');

  // ------ util: job id resolution ------
  function currentJobId() {
    return (jobInput.value || '').trim() || 'default';
  }

  async function refreshState() {
    try {
      const resp = await fetch(`/api/liquify-3d/state?job=${encodeURIComponent(currentJobId())}`);
      const data = await resp.json();
      if (!data.ok) throw new Error(data.error || 'state failed');
      state.pairs = data.pairs || [];
      state.annotationShape = data.annotation_shape || null;
      renderPairsTable();
      // Empty-state guidance: if the source annotation isn't available for
      // this job, surface an actionable message instead of a blank canvas.
      if (data.guidance) {
        sliceStatus.textContent = data.guidance;
        sliceStatus.style.color = 'var(--warn, #ffa726)';
      } else {
        sliceStatus.style.color = '';
      }
      _renderCoordBadge();
    } catch (err) {
      console.warn('[liquify3d] state fetch failed:', err);
    }
  }

  // #12 — surface the canvas-pixel → annotation-voxel rescale that the
  // backend silently performs on each /add-pair POST. Without this badge
  // a user wouldn't know that their click at (1500, 800) actually became
  // annotation-voxel (300, 160) inside Brainfast.
  function _renderCoordBadge() {
    const badge = el('liq3dCoordBadge');
    if (!badge) return;
    const ann = state.annotationShape;          // [D, H, W]
    const img = state.imageNaturalSize;          // {h, w}
    if (!ann || !img.h || !img.w) {
      badge.textContent = '';
      return;
    }
    const sy = (ann[1] / img.h).toFixed(3);
    const sx = (ann[2] / img.w).toFixed(3);
    badge.textContent =
      `image ${img.w}×${img.h} px  →  annotation grid ${ann[2]}×${ann[1]} ` +
      `(rescale x=${sx}, y=${sy}). Clicks are auto-rescaled before storage.`;
  }

  // Poll /api/liquify-3d/progress every 2s while a long op is running.
  // Returns a cancel function.
  function startProgressPoll(onUpdate) {
    let cancelled = false;
    const tick = async () => {
      if (cancelled) return;
      try {
        const resp = await fetch(`/api/liquify-3d/progress?job=${encodeURIComponent(currentJobId())}`);
        const data = await resp.json();
        if (data && typeof data.percent === 'number') {
          onUpdate(data);
        }
      } catch (e) { /* ignore transient errors */ }
      if (!cancelled) setTimeout(tick, 2000);
    };
    tick();
    return () => { cancelled = true; };
  }

  function renderPairsTable() {
    pairsBody.innerHTML = '';
    state.pairs.forEach((p, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="padding:4px;">${i + 1}</td>
        <td style="padding:4px;">${p.z}</td>
        <td style="padding:4px;">(${p.atlas_y.toFixed(1)}, ${p.atlas_x.toFixed(1)})</td>
        <td style="padding:4px;">(${p.real_y.toFixed(1)}, ${p.real_x.toFixed(1)})</td>
        <td style="padding:4px;">
          <button type="button" data-idx="${i}" class="liq3d-remove-btn"
            style="background:transparent;color:var(--danger,#f55);border:none;cursor:pointer;">&times;</button>
        </td>`;
      pairsBody.appendChild(tr);
    });
    pairCountEl.textContent = String(state.pairs.length);
    applyBtn.disabled = state.pairs.length === 0;
    clearBtn.disabled = state.pairs.length === 0;
    pairsBody.querySelectorAll('.liq3d-remove-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const idx = parseInt(btn.getAttribute('data-idx'), 10);
        try {
          await fetch(`/api/liquify-3d/pair/${idx}?job=${encodeURIComponent(currentJobId())}`, {
            method: 'DELETE',
          });
          refreshState();
          redraw();
        } catch (e) { console.error(e); }
      });
    });
  }

  // ------ slice list ------
  async function reloadSliceList() {
    try {
      // Pass the Job ID from the liquify panel's own input so the slice
      // overlay list matches the job the user just typed. Previously the
      // fetch had no ?job= param and always returned the default job's
      // slices, making the panel claim "No registered slices" even after
      // the user filled a valid completed job id.
      const jid = currentJobId();
      const url = jid && jid !== 'default'
        ? `/api/outputs/reg-slice-list?job=${encodeURIComponent(jid)}`
        : '/api/outputs/reg-slice-list';
      const resp = await fetch(url);
      const data = await resp.json();
      state.sliceFiles = data.files || [];
      if (!state.sliceFiles.length) {
        sliceStatus.textContent = 'No registered slices available — run pipeline through ANTs first.';
        return;
      }
      sliceStatus.textContent = `${state.sliceFiles.length} slice overlays available`;
      const n = state.sliceFiles.length - 1;
      zRange.max = n;
      zNum.max = n;
      zMax.textContent = `/ ${n}`;
      // Default to middle
      state.currentZ = Math.floor(state.sliceFiles.length / 2);
      zRange.value = state.currentZ;
      zNum.value = state.currentZ;
      loadSliceImage();
    } catch (err) {
      sliceStatus.textContent = 'Slice list fetch failed: ' + err.message;
    }
  }

  function _sliceImgUrl(fname) {
    // Same dual-source logic as reloadSliceList so the slice PNG comes from
    // the job we just typed rather than the default job.
    const jid = currentJobId();
    const base = `/api/outputs/reg-slice/${fname}`;
    const suffix = `ts=${Date.now()}`;
    if (jid && jid !== 'default') {
      return `${base}?job=${encodeURIComponent(jid)}&${suffix}`;
    }
    return `${base}?${suffix}`;
  }

  function loadSliceImage() {
    if (!state.sliceFiles.length) return;
    const fname = state.sliceFiles[state.currentZ];
    if (!fname) return;
    img.src = _sliceImgUrl(fname);
    img.style.display = '';
    img.onload = () => {
      state.imageNaturalSize = { h: img.naturalHeight, w: img.naturalWidth };
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.style.width = img.clientWidth + 'px';
      canvas.style.height = img.clientHeight + 'px';
      redraw();
      _renderCoordBadge();
      _refreshOverlayImage();
    };
  }

  // ------ Dual-channel overlay (Phase 4) --------------------------------
  // Pull ch_<N>_<z>.tif for the selected second channel, render it tinted,
  // and stack on top of the main overlay via CSS mix-blend-mode:screen.
  // Completely passive: when the toggle is off, the overlay img is hidden.
  function _refreshOverlayImage() {
    if (!overlayImg || !overlayToggle) return;
    if (!overlayToggle.checked) {
      overlayImg.style.display = 'none';
      overlayImg.src = '';
      return;
    }
    const ch = (overlayChannel?.value || '').trim();
    if (!ch || !state.sliceFiles.length) {
      overlayImg.style.display = 'none';
      return;
    }
    const tint = (overlayColor?.value || 'ffffff').trim();
    const opacity = Math.max(0, Math.min(100, Number(overlayOpacity?.value || 60))) / 100;
    const jid = currentJobId();
    const url = `/api/outputs/raw-channel-slice?job=${encodeURIComponent(jid)}`
              + `&z=${state.currentZ}&channel=${encodeURIComponent(ch)}&tint=${encodeURIComponent(tint)}`
              + `&ts=${Date.now()}`;
    overlayImg.onload = () => {
      overlayImg.style.display = '';
      overlayImg.style.opacity = String(opacity);
      // Match the main image's rendered size so the two overlays line up.
      overlayImg.style.width = img.clientWidth + 'px';
    };
    overlayImg.onerror = () => {
      overlayImg.style.display = 'none';
    };
    overlayImg.src = url;
  }

  async function _refreshChannelOptions() {
    if (!overlayChannel || !overlayRow) return;
    try {
      const resp = await fetch(
        `/api/outputs/channel-info?job=${encodeURIComponent(currentJobId())}`
      );
      const data = await resp.json();
      const list = (data && Array.isArray(data.channels)) ? data.channels : [];
      overlayChannel.innerHTML = '';
      for (const name of list) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        overlayChannel.appendChild(opt);
      }
      // Only show the whole row if the job has ≥2 distinct channels so the
      // control stays out of the way for single-channel runs.
      overlayRow.style.display = list.length >= 2 ? '' : 'none';
      // Default to the 2nd channel in the list (first one is usually the
      // reporter C0 which is already the base image).
      if (list.length >= 2) overlayChannel.value = list[1];
    } catch (_) {
      overlayRow.style.display = 'none';
    }
  }

  overlayToggle?.addEventListener('change', _refreshOverlayImage);
  overlayChannel?.addEventListener('change', _refreshOverlayImage);
  overlayColor?.addEventListener('change', _refreshOverlayImage);
  overlayOpacity?.addEventListener('input', () => {
    if (overlayOpacityNum) overlayOpacityNum.textContent = overlayOpacity.value + '%';
    _refreshOverlayImage();
  });

  // ------ canvas interaction ------
  function canvasToImageCoords(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) * canvas.width) / rect.width,
      y: ((e.clientY - rect.top) * canvas.height) / rect.height,
    };
  }

  canvas.addEventListener('click', async (e) => {
    if (!state.sliceFiles.length) return;
    const { x, y } = canvasToImageCoords(e);
    if (state.clickMode === 'atlas') {
      state.pendingAtlas = { x, y };
      state.clickMode = 'real';
      document.querySelector('input[name="liq3dClickMode"][value="real"]').checked = true;
      pendingStatus.textContent = `Atlas @ (${y.toFixed(0)}, ${x.toFixed(0)}). Now click real target.`;
    } else {
      state.pendingReal = { x, y };
      pendingStatus.textContent = `Pair ready — posting…`;
      const atlas = state.pendingAtlas;
      const real = state.pendingReal;
      state.pendingAtlas = null;
      state.pendingReal = null;
      try {
        const resp = await fetch('/api/liquify-3d/add-pair', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jobId: currentJobId(),
            z: state.currentZ,
            atlas: [atlas.y, atlas.x],
            real: [real.y, real.x],
            image_dims_yx: [state.imageNaturalSize.h, state.imageNaturalSize.w],
          }),
        });
        const data = await resp.json();
        if (!data.ok) throw new Error(data.error || 'add failed');
        pendingStatus.textContent = `Pair added (${data.pair_count} total).`;
        state.clickMode = 'atlas';
        document.querySelector('input[name="liq3dClickMode"][value="atlas"]').checked = true;
        refreshState();
      } catch (err) {
        pendingStatus.textContent = 'Add failed: ' + err.message;
      }
    }
    redraw();
  });

  function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const DOT = 8;
    // Existing pairs in current z (± 0 window for now; could widen later)
    state.pairs.forEach((p, i) => {
      if (p.z !== state.currentZ) return;
      // atlas dot (orange)
      ctx.fillStyle = '#ffcc00'; ctx.strokeStyle = '#000'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p.atlas_x, p.atlas_y, DOT, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      // real dot (green)
      ctx.fillStyle = '#00ff88';
      ctx.beginPath(); ctx.arc(p.real_x, p.real_y, DOT, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      // arrow atlas→real
      ctx.strokeStyle = '#ffffff';
      ctx.beginPath(); ctx.moveTo(p.atlas_x, p.atlas_y); ctx.lineTo(p.real_x, p.real_y); ctx.stroke();
      // label
      ctx.fillStyle = '#fff'; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(String(i + 1), (p.atlas_x + p.real_x) / 2, (p.atlas_y + p.real_y) / 2 - 10);
    });
    if (state.pendingAtlas) {
      ctx.fillStyle = '#ff8800'; ctx.strokeStyle = '#000'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(state.pendingAtlas.x, state.pendingAtlas.y, DOT, 0, 2 * Math.PI);
      ctx.fill(); ctx.stroke();
    }
  }

  // ------ z slider ------
  function onZChange(newZ) {
    const n = state.sliceFiles.length;
    if (!n) return;
    newZ = Math.max(0, Math.min(n - 1, parseInt(newZ, 10) || 0));
    state.currentZ = newZ;
    zRange.value = newZ;
    zNum.value = newZ;
    loadSliceImage();
  }
  zRange.addEventListener('input', (e) => { onZChange(e.target.value); _saveTabState(); });
  zNum.addEventListener('change', (e) => { onZChange(e.target.value); _saveTabState(); });

  // ------ click mode toggle ------
  document.querySelectorAll('input[name="liq3dClickMode"]').forEach((r) => {
    r.addEventListener('change', (e) => {
      state.clickMode = e.target.value;
      _saveTabState();
    });
  });

  // Persist job/class on change so reloads / tab switches don't wipe them.
  el('liq3dJobId')?.addEventListener('change', _saveTabState);
  el('liq3dClassName')?.addEventListener('change', _saveTabState);

  // ------ buttons ------
  reloadBtn.addEventListener('click', () => {
    reloadSliceList();
    refreshState();
    _refreshChannelOptions();
  });

  clearBtn.addEventListener('click', async () => {
    if (!confirm('Clear all landmark pairs for this job?')) return;
    try {
      await fetch('/api/liquify-3d/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId: currentJobId() }),
      });
      // Review finding 3 — invalidate the current (job, class) auto-warm-start
      // signature so the next refreshState sees this as a fresh empty job
      // and re-attempts auto-apply. Otherwise a user who clears their own
      // manual pairs would never see the class prior auto-seed.
      try { _autoWarmStartTried.delete(_autoWarmStartSignature()); } catch (_) {}
      await refreshState();
      redraw();
      _autoWarmStartIfEmpty();
    } catch (err) { applyStatus.textContent = 'Clear failed: ' + err.message; }
  });

  // Undo: remove the most recently added pair via DELETE /pair/<index>.
  // Triggered by either the toolbar button or Ctrl+Z while the tab is focused.
  async function undoLastPair() {
    if (state.pairs.length === 0) {
      pendingStatus.textContent = 'Nothing to undo.';
      return;
    }
    const lastIdx = state.pairs.length - 1;
    try {
      const resp = await fetch(
        `/api/liquify-3d/pair/${lastIdx}?job=${encodeURIComponent(currentJobId())}`,
        { method: 'DELETE' },
      );
      const data = await resp.json();
      if (!data.ok) throw new Error(data.error || 'undo failed');
      pendingStatus.textContent = `Undid pair #${lastIdx + 1}.`;
      refreshState();
      redraw();
    } catch (err) {
      pendingStatus.textContent = 'Undo failed: ' + err.message;
    }
  }

  el('liq3dUndoBtn')?.addEventListener('click', undoLastPair);

  document.addEventListener('keydown', (e) => {
    // Only react when the 3D Liquify tab is the visible one — avoid stealing
    // Ctrl+Z from inputs in other tabs.
    if (!tabEl.classList.contains('active')) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !e.shiftKey) {
      // Don't hijack Ctrl+Z while the user is editing a text input.
      const tag = (document.activeElement?.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea') return;
      e.preventDefault();
      undoLastPair();
    }
  });

  applyBtn.addEventListener('click', async () => {
    applyStatus.textContent = 'Starting Laplacian warp…';
    applyBtn.disabled = true;
    const cancelPoll = startProgressPoll((p) => {
      applyStatus.textContent =
        `[${p.percent}%] ${p.stage || ''}: ${p.message || ''}`.trim();
    });
    try {
      const resp = await fetch('/api/liquify-3d/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId: currentJobId() }),
      });
      const data = await resp.json();
      if (!data.ok) throw new Error(data.error || 'apply failed');
      applyStatus.textContent =
        `Done. ${data.pair_count} pairs → max displacement ${data.displacement_max_voxels.toFixed(2)} voxels. ` +
        `Refined annotation saved at ${data.output_path}`;
    } catch (err) {
      applyStatus.textContent = 'Apply failed: ' + err.message;
    } finally {
      cancelPoll();
      applyBtn.disabled = state.pairs.length === 0;
    }
  });

  // ------------- Close the loop: finalize → cell counts -------------
  const finalizeBtn = el('liq3dFinalizeBtn');
  const finalizeStatus = el('liq3dFinalizeStatus');

  finalizeBtn?.addEventListener('click', async () => {
    finalizeBtn.disabled = true;
    finalizeStatus.textContent = 'Starting finalize…';
    const cancelPoll = startProgressPoll((p) => {
      finalizeStatus.textContent =
        `[${p.percent}%] ${p.stage || ''}: ${p.message || ''}`.trim();
    });
    try {
      const resp = await fetch('/api/liquify-3d/finalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId: currentJobId() }),
      });
      const data = await resp.json();
      if (!data.ok) {
        finalizeStatus.textContent = 'Finalize failed: ' + (data.error || resp.status);
        return;
      }
      finalizeStatus.textContent =
        `Done. ${data.mapped_count} cells re-mapped → ` +
        `cell_counts_hierarchy_liquify3d.csv. Results tab will now prefer the refined counts.`;
    } catch (err) {
      finalizeStatus.textContent = 'Finalize failed: ' + err.message;
    } finally {
      cancelPoll();
      finalizeBtn.disabled = false;
    }
  });

  // ------------- #11 QC done -------------
  const qcDoneBtn = el('liq3dQcDoneBtn');
  const qcStatus  = el('liq3dQcStatus');

  async function refreshQcStatus() {
    if (!qcStatus) return;
    try {
      const resp = await fetch(`/api/liquify-3d/qc-status?job=${encodeURIComponent(currentJobId())}`);
      const data = await resp.json();
      if (data.done) {
        const ts = data.timestamp ? new Date(data.timestamp * 1000).toLocaleString() : '';
        const m = data.metrics || {};
        const summary = ['NCC', 'Dice', 'SSIM']
          .filter((k) => typeof m[k] === 'number')
          .map((k) => `${k}=${m[k].toFixed(3)}`)
          .join(' ');
        qcStatus.textContent = `✓ QC done @ ${ts}${summary ? ' · ' + summary : ''}`;
        qcStatus.style.color = 'var(--success,#4caf50)';
      } else {
        qcStatus.textContent = 'Not yet signed off.';
        qcStatus.style.color = '';
      }
    } catch (_) { /* ignore */ }
  }

  qcDoneBtn?.addEventListener('click', async () => {
    const note = el('liq3dQcNote')?.value?.trim() || '';
    qcDoneBtn.disabled = true;
    qcStatus.textContent = 'Recording sign-off…';
    try {
      // Best-effort: include class name + last known liquify metrics if available.
      const className = el('liq3dClassName')?.value?.trim() || null;
      const resp = await fetch('/api/liquify-3d/qc-done', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jobId: currentJobId(),
          className,
          note,
          metrics: {},  // wire to live metrics in a future iteration
        }),
      });
      const data = await resp.json();
      if (!data.ok) {
        qcStatus.textContent = 'QC done failed: ' + (data.error || resp.status);
        return;
      }
      await refreshQcStatus();
    } catch (err) {
      qcStatus.textContent = 'QC done failed: ' + err.message;
    } finally {
      qcDoneBtn.disabled = false;
    }
  });

  // ------------- #6 Class dropdown + #7 auto-detect -------------
  const classList = el('liq3dClassList');

  async function refreshClassList() {
    if (!classList) return;
    try {
      const resp = await fetch('/api/liquify-3d/class-registry/list');
      const data = await resp.json();
      classList.innerHTML = '';
      for (const c of data.classes || []) {
        const opt = document.createElement('option');
        opt.value = c;
        classList.appendChild(opt);
      }
    } catch (_) { /* ignore */ }
  }

  async function autoDetectClass() {
    const cn = el('liq3dClassName');
    const jid = currentJobId();
    if (!cn || cn.value.trim()) return;  // user already set one
    try {
      const resp = await fetch(`/api/liquify-3d/class-registry/detect?sampleId=${encodeURIComponent(jid)}`);
      const data = await resp.json();
      if (data && data.class) {
        cn.value = data.class;
        cn.dispatchEvent(new Event('change'));  // triggers prior banner refresh
      }
    } catch (_) { /* ignore */ }
  }

  // ------------- Phase γ: class-prior controls -------------
  const classInput       = el('liq3dClassName');
  const priorStatusBtn   = el('liq3dPriorStatusBtn');
  const saveToPriorBtn   = el('liq3dSaveToPriorBtn');
  const loadFromPriorBtn = el('liq3dLoadFromPriorBtn');
  const priorBanner      = el('liq3dPriorBanner');

  function currentClassName() {
    return (classInput.value || '').trim();
  }

  async function refreshPriorBanner() {
    const cls = currentClassName();
    if (!cls) { priorBanner.textContent = ''; return; }
    try {
      const resp = await fetch(`/api/liquify-3d/class-prior/status?class=${encodeURIComponent(cls)}`);
      const data = await resp.json();
      if (!data.ok) { priorBanner.textContent = data.error || 'prior status error'; return; }
      const ready = data.ready_for_warm_start ? '✓' : '⚠';
      priorBanner.textContent =
        `${ready} ${cls}: ${data.sample_count} samples, ${data.entry_count} landmarks` +
        (data.ready_for_warm_start
          ? ' — ready to warm-start new runs'
          : ` — needs ≥ ${data.min_samples_for_apply} samples before auto-apply`);
    } catch (err) {
      priorBanner.textContent = 'Prior status fetch failed: ' + err.message;
    }
  }

  classInput?.addEventListener('change', async () => {
    await refreshPriorBanner();
    _autoWarmStartIfEmpty();
  });
  priorStatusBtn?.addEventListener('click', refreshPriorBanner);
  // Review finding 3 — jobInput change must retrigger too; switching jobs
  // without clearing the tried-set would otherwise leave the new job stuck
  // on the first job's auto-apply decision.
  jobInput?.addEventListener('change', async () => {
    await refreshState();
    await refreshPriorBanner();
    _autoWarmStartIfEmpty();
  });

  // Task 3 — one automatic warm-start hook: when a class is set, the prior
  // is ready, and the job has ZERO existing landmark pairs, apply the
  // warm-start automatically. Never force-overwrites; a job with any manual
  // pairs routes to the explicit button path (which still supports force=true
  // via the confirm() flow below).
  //
  // Review finding 3 — the guard is scoped by (jobId|className) signature
  // rather than a lifetime-of-page boolean, so switching jobs or classes
  // (or clearing pairs, see clearBtn handler) triggers a fresh attempt.
  const _autoWarmStartTried = new Set();
  function _autoWarmStartSignature() {
    return `${currentJobId()}|${currentClassName()}`;
  }
  async function _autoWarmStartIfEmpty() {
    const cls = currentClassName();
    if (!cls) return;
    const sig = _autoWarmStartSignature();
    if (_autoWarmStartTried.has(sig)) return;
    // Only trigger when liquify state has been loaded (state.pairs is an array)
    if (!Array.isArray(state.pairs)) return;
    if (state.pairs.length > 0) {
      // Manual work present — banner is informative only.
      priorBanner.textContent += ' · manual overwrite required (click "Warm-start from prior")';
      return;
    }
    let statusData;
    try {
      const r = await fetch(
        `/api/liquify-3d/class-prior/status?class=${encodeURIComponent(cls)}`
      );
      statusData = await r.json();
    } catch (_) { return; }
    if (!statusData?.ok || !statusData.ready_for_warm_start) return;
    _autoWarmStartTried.add(sig);
    try {
      const resp = await fetch('/api/liquify-3d/class-prior/apply-warm-start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId: currentJobId(), class: cls, force: false }),
      });
      if (resp.status === 409) {
        priorBanner.textContent +=
          ' · manual overwrite required (job already has pairs)';
        return;
      }
      const data = await resp.json();
      if (!data?.ok) return;
      priorBanner.textContent =
        `✓ ${cls}: auto-applied ${data.pair_count} prior landmark(s) to this empty job.`;
      await refreshState();
      redraw();
    } catch (_) {
      // Best-effort: auto-apply failures never block the UI.
    }
  }

  // ----- #8 class-prior coverage heatmap -----
  const priorHeatBtn = el('liq3dPriorHeatBtn');
  const priorHeatRow = el('liq3dPriorHeatRow');
  const priorHeatCanvas = el('liq3dPriorHeatCanvas');

  async function renderPriorHeatmap() {
    if (!priorHeatCanvas) return;
    const cls = currentClassName();
    if (!cls) {
      priorHeatRow.style.display = 'none';
      priorBanner.textContent = 'Enter a class name first.';
      return;
    }
    let bins = [];
    let maxZ = 0;
    try {
      const resp = await fetch(`/api/liquify-3d/class-prior/coverage?class=${encodeURIComponent(cls)}`);
      const data = await resp.json();
      if (!data.ok) throw new Error(data.error || 'coverage failed');
      bins = data.bins || [];
    } catch (err) {
      priorBanner.textContent = 'Coverage fetch failed: ' + err.message;
      return;
    }
    priorHeatRow.style.display = '';
    const W = priorHeatCanvas.width;
    const H = priorHeatCanvas.height;
    const ctx2 = priorHeatCanvas.getContext('2d');
    ctx2.clearRect(0, 0, W, H);

    if (bins.length === 0) {
      ctx2.fillStyle = '#888';
      ctx2.font = '12px sans-serif';
      ctx2.fillText('No landmarks in this class prior yet.', 10, H / 2 + 4);
      return;
    }
    // Use the upper bound of available z; pad to current sliceFiles count if known.
    maxZ = Math.max(state.sliceFiles.length || 1, ...bins.map((b) => b.z + 1));
    const maxN = Math.max(1, ...bins.map((b) => b.count));
    // Build a per-z count array so visualization shows gaps as gaps.
    const counts = new Array(maxZ).fill(0);
    bins.forEach((b) => { counts[b.z] = b.count; });

    // Each pixel column is one bar; map z 0…maxZ-1 to canvas x 0…W
    for (let x = 0; x < W; x++) {
      const z = Math.floor((x / W) * maxZ);
      const c = counts[z];
      if (!c) continue;
      const h = Math.round((c / maxN) * (H - 4));
      // Greener for denser; semitransparent so overlap is visible
      ctx2.fillStyle = `rgba(76, 175, 80, ${0.35 + 0.65 * (c / maxN)})`;
      ctx2.fillRect(x, H - h - 2, 1, h);
    }
    // Legend
    ctx2.fillStyle = '#bbb';
    ctx2.font = '10px sans-serif';
    ctx2.fillText(`${cls} · ${bins.length} entries · max samples/voxel = ${maxN}`, 6, 12);
  }

  priorHeatBtn?.addEventListener('click', renderPriorHeatmap);

  saveToPriorBtn?.addEventListener('click', async () => {
    const cls = currentClassName();
    if (!cls) { priorBanner.textContent = 'Enter a class name first (e.g. ChATe27)'; return; }
    if (state.pairs.length === 0) {
      priorBanner.textContent = 'No landmark pairs to contribute.';
      return;
    }
    try {
      const resp = await fetch('/api/liquify-3d/class-prior/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId: currentJobId(), class: cls, sampleId: currentJobId() }),
      });
      const data = await resp.json();
      if (!data.ok) throw new Error(data.error || 'save failed');
      priorBanner.textContent =
        `Saved ${data.merged_pair_count} pair(s) to ${cls}. Class prior: ${data.sample_count} sample(s) total.`;
    } catch (err) {
      priorBanner.textContent = 'Save to prior failed: ' + err.message;
    }
  });

  loadFromPriorBtn?.addEventListener('click', async () => {
    const cls = currentClassName();
    if (!cls) { priorBanner.textContent = 'Enter a class name first (e.g. ChATe27)'; return; }
    let force = false;
    const run = async () => {
      const resp = await fetch('/api/liquify-3d/class-prior/apply-warm-start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId: currentJobId(), class: cls, force }),
      });
      const data = await resp.json();
      if (resp.status === 409 && !force) {
        if (confirm('This job already has landmark pairs. Overwrite with class prior?')) {
          force = true;
          return await run();
        }
        priorBanner.textContent = 'Warm-start canceled (existing pairs kept).';
        return;
      }
      if (!data.ok) {
        priorBanner.textContent = 'Warm-start failed: ' + (data.error || 'unknown');
        return;
      }
      priorBanner.textContent = `Warm-started job with ${data.pair_count} prior landmark(s).`;
      refreshState();
      redraw();
    };
    try { await run(); } catch (err) { priorBanner.textContent = 'Warm-start failed: ' + err.message; }
  });

  // Auto-load the first time the tab is shown
  document.querySelector('.nav-btn[data-tab="liquify3d"]')?.addEventListener('click', async () => {
    if (!state.sliceFiles.length) {
      reloadSliceList();
      await refreshState();
    } else {
      await refreshState();
    }
    refreshClassList();
    await autoDetectClass();
    await refreshPriorBanner();
    refreshQcStatus();
    // Task 3 — attempt a single auto warm-start once liquify state + class
    // have both been resolved. Safe-guarded by _autoWarmStartTried +
    // the empty-pair check inside the helper.
    _autoWarmStartIfEmpty();
  });
})();

// ==========================================================================
// Onboarding wizard (New Sample tab) — gets a brand-new user from
// "I have raw TIFFs" to "the pipeline is running" without CLI use.
// ==========================================================================
(function () {
  const tab = document.getElementById('tab-newsample');
  if (!tab) return;

  const $ = (id) => document.getElementById(id);
  const sourceInput = $('wizSourcePath');
  const inspectBtn  = $('wizInspectBtn');
  const inspectStatus = $('wizInspectStatus');
  const sampleId   = $('wizSampleId');
  const pixelUm    = $('wizPixelUm');
  const zUm        = $('wizZUm');
  const hemi       = $('wizHemi');
  const channel    = $('wizChannel');
  const launchBtn  = $('wizLaunchBtn');
  const launchStatus = $('wizLaunchStatus');
  // Dual-channel controls (Phase 5)
  const addSecondToggle = $('wizAddSecondChannel');
  const secondFields    = $('wizSecondChannelFields');
  const source2Input    = $('wiz2SourcePath');
  const inspect2Btn     = $('wiz2InspectBtn');
  const inspect2Status  = $('wiz2InspectStatus');
  const channel2        = $('wiz2Channel');

  let lastInspect = null;
  let lastInspect2 = null;

  // ── Source path UX helpers ────────────────────────────────────────────
  // (1) Strip quotes on paste — Windows "Copy as path" yields `"D:\..."`,
  //     and a leading/trailing `"` makes Path() fail downstream.
  const _stripQuotes = (s) => (s || '').replace(/^"+|"+$/g, '').trim();
  sourceInput?.addEventListener('paste', (e) => {
    const txt = (e.clipboardData || window.clipboardData).getData('text');
    if (txt && (txt.startsWith('"') || txt.endsWith('"'))) {
      e.preventDefault();
      sourceInput.value = _stripQuotes(txt);
    }
  });
  source2Input?.addEventListener('paste', (e) => {
    const txt = (e.clipboardData || window.clipboardData).getData('text');
    if (txt && (txt.startsWith('"') || txt.endsWith('"'))) {
      e.preventDefault();
      source2Input.value = _stripQuotes(txt);
    }
  });

  // (2) Browse buttons — pop the OS-native file/folder picker via the
  //     existing /api/browse endpoints (tkinter on the server side, which
  //     for Brainfast is the user's own local machine).
  $('wizBrowseDirBtn')?.addEventListener('click', async () => {
    try {
      const resp = await fetch('/api/browse/folder', { method: 'POST' });
      const data = await resp.json();
      if (data.ok && data.path) sourceInput.value = data.path;
    } catch (err) {
      if (inspectStatus) inspectStatus.textContent = 'Browse failed: ' + err.message;
    }
  });
  $('wizBrowseFileBtn')?.addEventListener('click', async () => {
    try {
      const resp = await fetch('/api/browse/file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filetypes: 'tif,tiff' }),
      });
      const data = await resp.json();
      if (data.ok && data.path) sourceInput.value = data.path;
    } catch (err) {
      if (inspectStatus) inspectStatus.textContent = 'Browse failed: ' + err.message;
    }
  });

  // (3) Extract-slices button — visible only when Inspect detects a
  //     multi-page TIFF.  Calls /api/wizard/extract-multipage-tiff and on
  //     success rewrites sourceInput to the new slice dir + re-runs Inspect
  //     so the launch button can light up.
  const extractRow    = $('wizExtractRow');
  const extractOutDir = $('wizExtractOutDir');
  const extractBtn    = $('wizExtractBtn');
  const extractStatus = $('wizExtractStatus');

  extractBtn?.addEventListener('click', async () => {
    const src = _stripQuotes(sourceInput.value);
    const outDir = _stripQuotes(extractOutDir.value);
    if (!src || !outDir) {
      if (extractStatus) extractStatus.textContent = 'Provide both src + output dir.';
      return;
    }
    extractBtn.disabled = true;
    if (extractStatus) extractStatus.textContent = 'Extracting…';
    try {
      const resp = await fetch('/api/wizard/extract-multipage-tiff', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ src, outDir, channel: 0, everyN: 1 }),
      });
      const data = await resp.json();
      if (!data.ok) {
        if (extractStatus) extractStatus.textContent = 'Extract failed: ' + (data.error || resp.status);
        extractBtn.disabled = false;
        return;
      }
      if (extractStatus) {
        extractStatus.textContent = `✓ Wrote ${data.writtenCount ?? '?'} slices.`;
      }
      sourceInput.value = data.outDir;
      // Re-run Inspect on the new directory so launch button can enable
      inspectBtn?.click();
    } catch (err) {
      if (extractStatus) extractStatus.textContent = 'Extract failed: ' + err.message;
    } finally {
      extractBtn.disabled = false;
    }
  });

  addSecondToggle?.addEventListener('change', () => {
    if (secondFields) {
      secondFields.style.display = addSecondToggle.checked ? '' : 'none';
    }
  });

  inspect2Btn?.addEventListener('click', async () => {
    const sp = (source2Input?.value || '').trim();
    if (!sp) {
      inspect2Status.textContent = 'Please enter the 2nd channel source path.';
      inspect2Status.style.color = 'var(--warn,#ffa726)';
      return;
    }
    inspect2Status.textContent = 'Inspecting 2nd channel…';
    inspect2Status.style.color = '';
    try {
      const resp = await fetch('/api/wizard/inspect-source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sourcePath: sp }),
      });
      const data = await resp.json();
      if (!data.ok) {
        inspect2Status.textContent = '2nd inspect failed: ' + (data.error || resp.status);
        inspect2Status.style.color = 'var(--danger,#f55)';
        lastInspect2 = null;
        return;
      }
      lastInspect2 = data;
      const lines = [`Detected: ${data.kind}`];
      if (data.kind === 'multipage_tiff') {
        lines.push(`pages=${data.n_pages}`);
        if (data.needs_extraction) {
          lines.push('⚠ needs extract_zstack first');
        }
      } else {
        lines.push(`files=${data.n_files}`);
      }
      if (data.sample_shape) lines.push(`shape=${JSON.stringify(data.sample_shape)}`);
      // Consistency check against 1st channel: same shape + same page count.
      if (lastInspect && data.sample_shape && lastInspect.sample_shape) {
        if (JSON.stringify(data.sample_shape) !== JSON.stringify(lastInspect.sample_shape)) {
          lines.push('⚠ shape differs from 1st channel — registration reuse may break');
        }
      }
      inspect2Status.textContent = lines.join('  |  ');
    } catch (err) {
      inspect2Status.textContent = '2nd inspect failed: ' + err.message;
      inspect2Status.style.color = 'var(--danger,#f55)';
      lastInspect2 = null;
    }
  });

  inspectBtn.addEventListener('click', async () => {
    const sp = (sourceInput.value || '').trim();
    if (!sp) {
      inspectStatus.textContent = 'Please enter a source path.';
      inspectStatus.style.color = 'var(--warn,#ffa726)';
      return;
    }
    inspectStatus.textContent = 'Inspecting…';
    inspectStatus.style.color = '';
    try {
      const resp = await fetch('/api/wizard/inspect-source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sourcePath: sp }),
      });
      const data = await resp.json();
      if (!data.ok) {
        inspectStatus.textContent = 'Inspect failed: ' + (data.error || resp.status);
        inspectStatus.style.color = 'var(--danger,#f55)';
        launchBtn.disabled = true;
        return;
      }
      lastInspect = data;
      // Auto-fill form fields with detected defaults
      if (data.suggested_sample_id && !sampleId.value) sampleId.value = data.suggested_sample_id;
      if (typeof data.suggested_pixel_um_xy === 'number') pixelUm.value = data.suggested_pixel_um_xy;
      if (typeof data.suggested_z_spacing_um === 'number') zUm.value = data.suggested_z_spacing_um;

      const lines = [];
      lines.push(`Detected: ${data.kind}`);
      if (data.kind === 'multipage_tiff') {
        lines.push(`pages=${data.n_pages}`);
        if (data.needs_extraction) {
          lines.push('⚠ multi-page TIFF needs extraction — run `python scripts/extract_zstack.py` first');
        }
      } else {
        lines.push(`files=${data.n_files}`);
      }
      if (data.sample_shape) lines.push(`shape=${JSON.stringify(data.sample_shape)}`);
      if (data.dtype) lines.push(`dtype=${data.dtype}`);
      lines.push(`px=${data.suggested_pixel_um_xy}µm, z=${data.suggested_z_spacing_um}µm`);
      inspectStatus.textContent = lines.join('  |  ');
      inspectStatus.style.color = '';

      // Allow launch only for directory inputs (multi-page needs extract first)
      launchBtn.disabled = data.kind !== 'directory' || (data.n_files || 0) === 0;
      if (launchBtn.disabled) {
        launchStatus.textContent = data.kind === 'multipage_tiff'
          ? 'Multi-page TIFF detected — fill output dir below + click Extract slices, then it will auto-Inspect.'
          : 'No TIFF slices found in directory.';
      } else {
        launchStatus.textContent = '';
      }

      // Reveal Extract row + suggest a default output dir when input is multi-page TIFF
      if (data.kind === 'multipage_tiff') {
        if (extractRow) extractRow.style.display = '';
        if (extractOutDir && !extractOutDir.value) {
          // Suggest <parent>/<basename>_slices
          const src = _stripQuotes(sourceInput.value);
          const lastSlash = Math.max(src.lastIndexOf('/'), src.lastIndexOf('\\'));
          const dir = lastSlash >= 0 ? src.slice(0, lastSlash) : '.';
          const name = lastSlash >= 0 ? src.slice(lastSlash + 1) : src;
          const stem = name.replace(/\.tiff?$/i, '');
          // Use whatever separator the user already used (defaults to forward slash)
          const sep = src.includes('\\') ? '\\' : '/';
          extractOutDir.value = `${dir}${sep}${stem}_slices`;
        }
      } else if (extractRow) {
        extractRow.style.display = 'none';
      }
    } catch (err) {
      inspectStatus.textContent = 'Inspect failed: ' + err.message;
      inspectStatus.style.color = 'var(--danger,#f55)';
      launchBtn.disabled = true;
    }
  });

  launchBtn.addEventListener('click', async () => {
    if (!sampleId.value || !sourceInput.value || !pixelUm.value || !zUm.value) {
      launchStatus.textContent = 'Fill sample id, source path, pixel + z spacing first.';
      return;
    }
    // Build channel list + per-channel input dirs
    const ch1 = channel.value;
    const channels = [ch1];
    const inputDirs = { [ch1]: sourceInput.value.trim() };
    if (addSecondToggle?.checked) {
      const ch2 = channel2.value;
      const src2 = (source2Input.value || '').trim();
      if (!src2) {
        launchStatus.textContent = '2nd channel enabled but source path is empty.';
        return;
      }
      if (ch2 === ch1) {
        launchStatus.textContent = '2nd channel must differ from the 1st.';
        return;
      }
      channels.push(ch2);
      inputDirs[ch2] = src2;
    }
    launchBtn.disabled = true;
    launchStatus.textContent = channels.length > 1
      ? `Launching dual-channel pipeline (${channels.join(' + ')})…`
      : 'Launching pipeline…';
    try {
      const advAntsTransform = document.getElementById('wizAntsTransform')?.value || 'SyNRA';
      const advFixedMaxDimRaw = document.getElementById('wizFixedMaxDim')?.value;
      const advFixedMaxDim = advFixedMaxDimRaw ? parseInt(advFixedMaxDimRaw, 10) : null;
      const advAxisAlign = !!document.getElementById('wizAxisAlign')?.checked;
      const advCellToCcf = !!document.getElementById('wizCellToCcf')?.checked;
      const payload = {
        sampleId: sampleId.value.trim(),
        pixelSizeUm: parseFloat(pixelUm.value),
        zSpacingUm: parseFloat(zUm.value),
        channels,
        atlasHemisphere: hemi.value,
        antsTransform: advAntsTransform,
        axisAlignmentEnabled: advAxisAlign,
        useCellToCcfMapping: advCellToCcf,
      };
      if (advFixedMaxDim && Number.isFinite(advFixedMaxDim) && advFixedMaxDim > 0) {
        payload.fixedMaxDim = advFixedMaxDim;
      }
      // Keep single-channel shape backwards compatible: inputDir (str) for
      // legacy jobs; inputDirs (dict) when multiple channels are declared.
      if (channels.length > 1) {
        payload.inputDirs = inputDirs;
        payload.inputDir = inputDirs[ch1]; // fallback for old handlers
      } else {
        payload.inputDir = inputDirs[ch1];
      }
      const resp = await fetch('/api/wizard/launch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!data.ok) {
        launchStatus.textContent = 'Launch failed: ' + (data.error || resp.status);
        launchBtn.disabled = false;
        return;
      }
      const dualSuffix = channels.length > 1
        ? ` · 2nd channel reuses registration (≈15 min after 1st completes)`
        : '';
      launchStatus.textContent =
        `✓ Pipeline started for jobId="${data.jobId}".${dualSuffix} ` +
        `Check the Registration Workflow tab for progress.`;
      launchStatus.style.color = 'var(--success,#4caf50)';
    } catch (err) {
      launchStatus.textContent = 'Launch failed: ' + err.message;
      launchBtn.disabled = false;
    }
  });
})();

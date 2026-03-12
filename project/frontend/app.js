const state = { channel: 'red', progress: 0, running: false, runAll: false };
const logBox = document.getElementById('logBox');
const barFill = document.getElementById('barFill');
const stepText = document.getElementById('stepText');
const statusBadge = document.getElementById('statusBadge');
const rows = document.getElementById('resultRows');
const qcGrid = document.getElementById('qcGrid');
const compareRows = document.getElementById('compareRows');
const historyList = document.getElementById('historyList');
const versionText = document.getElementById('versionText');

function log(msg){
  const ts = new Date().toLocaleTimeString();
  logBox.textContent += `\n[${ts}] ${msg}`;
  logBox.scrollTop = logBox.scrollHeight;
}

function setProgress(p, text){
  state.progress = p;
  barFill.style.width = `${p}%`;
  stepText.textContent = text;
}

function setRunning(r){
  state.running = r;
  statusBadge.textContent = r ? 'RUNNING' : 'IDLE';
  statusBadge.style.background = r ? '#3ba55d' : '#4e5058';
}

document.querySelectorAll('.pill[data-channel]').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.pill[data-channel]').forEach(x => x.classList.remove('active'));
    btn.classList.add('active');
    state.channel = btn.dataset.channel;
    state.runAll = false;
    document.getElementById('batchAll').classList.remove('active');
    log(`channel selected: ${state.channel}`);
  };
});

document.getElementById('batchAll').onclick = () => {
  state.runAll = !state.runAll;
  document.getElementById('batchAll').classList.toggle('active', state.runAll);
  log(state.runAll ? 'batch mode: red+green+farred' : 'batch mode off');
};

function savePreset(){
  const preset = {
    inputDir: document.getElementById('inputDir').value,
    outputDir: document.getElementById('outputDir').value,
    atlasPath: document.getElementById('atlasPath').value,
    structPath: document.getElementById('structPath').value,
    pixelSizeUm: document.getElementById('pixelSizeUm')?.value || '0.65',
    rotateAtlas: document.getElementById('rotateAtlas')?.value || '0',
    flipAtlas: document.getElementById('flipAtlas')?.value || 'none',
    slicingPlane: document.getElementById('slicingPlane')?.value || 'coronal',
    majorTopK: document.getElementById('majorTopK')?.value || '12',
    fitMode: document.getElementById('fitMode')?.value || 'contain',
    channel: state.channel,
    runAll: state.runAll
  };
  localStorage.setItem('braincount.preset', JSON.stringify(preset));
  log('preset saved');
}

function loadPreset(){
  const raw = localStorage.getItem('braincount.preset');
  if(!raw){ log('no preset found'); return; }
  const p = JSON.parse(raw);
  ['inputDir','outputDir','atlasPath','structPath','pixelSizeUm','rotateAtlas','flipAtlas','slicingPlane','majorTopK','fitMode'].forEach(k => {
    const el = document.getElementById(k);
    if(el && p[k] !== undefined) el.value = p[k];
  });
  const target = document.querySelector(`.pill[data-channel="${p.channel||'red'}"]`);
  if(target) target.click();
  state.runAll = !!p.runAll;
  document.getElementById('batchAll').classList.toggle('active', state.runAll);
  log('preset loaded');
}

document.getElementById('savePreset').onclick = savePreset;
document.getElementById('loadPreset').onclick = loadPreset;

document.getElementById('exportBtn').onclick = () => {
  window.open('/api/outputs/leaf', '_blank');
};

document.getElementById('openOutputsBtn').onclick = async () => {
  const info = await fetch('/api/info').then(r=>r.json());
  alert(`Outputs folder:\n${info.outputs}`);
};

document.getElementById('guideBtn').onclick = () => {
  alert([
    'First-run checklist:',
    '1) Fill Input TIFF folder',
    '2) Fill Atlas annotation path',
    '3) Fill Structure mapping CSV path',
    '4) Choose channel (or Run All)',
    '5) Click Run Pipeline',
    '6) Check Output Snapshot + QC Preview'
  ].join('\n'));
};

document.getElementById('aiAlignBtn').onclick = async () => {
  try {
    const paths = {
      realPath: document.getElementById('realSlicePath').value || '../outputs/test_real.tif',
      atlasPath: document.getElementById('atlasLabelPath').value || '../outputs/test_label.tif'
    };
    const alignMode = document.getElementById('alignMode').value;

    const lm = await fetch('/api/align/landmarks', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({
        ...paths,
        maxPoints: Number(document.getElementById('maxPoints').value || 30),
        minDistance: Number(document.getElementById('minDistance').value || 12),
        ransacResidual: Number(document.getElementById('ransacResidual').value || 8),
      })
    }).then(r=>r.json());
    if(!lm.ok){
      alert(lm.error || 'AI landmark proposal failed');
      return;
    }

    const ep = alignMode === 'nonlinear' ? '/api/align/nonlinear' : '/api/align/apply';
    const ap = await fetch(ep, {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({
        realPath: paths.realPath,
        atlasLabelPath: paths.atlasPath
      })
    }).then(r=>r.json());
    if(!ap.ok){
      const extra = ap.failLog ? `\nFail log: ${ap.failLog}` : '';
      alert((ap.error || 'AI align apply failed') + extra);
      log(`AI ${alignMode} failed: ${ap.error || 'unknown'}${ap.failLog ? ` | failLog=${ap.failLog}` : ''}`);
      return;
    }

    log(`AI landmarks: pairs=${lm.landmark_pairs}/${lm.raw_pairs||lm.landmark_pairs}, score=${Number(lm.score).toFixed(4)}`);
    log(`AI ${alignMode} score raw: ${Number(ap.beforeScore).toFixed(4)} -> ${Number(ap.afterScore).toFixed(4)}`);
    log(`AI ${alignMode} score edge(SSIM): ${Number(ap.beforeEdgeScore).toFixed(4)} -> ${Number(ap.afterEdgeScore).toFixed(4)}`);
    if(ap.scoreWarning) log('warning: edge SSIM did not improve');

    qcGrid.innerHTML = '';
    const img = document.createElement('img');
    img.src = alignMode === 'nonlinear' ? `/api/outputs/overlay-compare-nonlinear?${Date.now()}` : `/api/outputs/overlay-compare?${Date.now()}`;
    qcGrid.appendChild(img);

    alert(`AI ${alignMode} done. edge SSIM ${Number(ap.beforeEdgeScore).toFixed(4)} -> ${Number(ap.afterEdgeScore).toFixed(4)}`);
  } catch(e){
    alert('AI landmark align failed');
  }
};


document.getElementById('landmarkViewBtn').onclick = async () => {
  try {
    const realPath = document.getElementById('realSlicePath').value || '../outputs/test_real.tif';
    const atlasPath = document.getElementById('atlasLabelPath').value || '../outputs/test_label.tif';
    const p = await fetch('/api/align/landmark-preview', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ realPath, atlasPath })
    }).then(r=>r.json());
    if(!p.ok){ alert(p.error || 'landmark preview failed'); return; }
    qcGrid.innerHTML = '';
    const img = document.createElement('img');
    img.src = `/api/outputs/landmark-preview?${Date.now()}`;
    qcGrid.appendChild(img);
    log(`landmark preview points=${p.points}`);
  } catch(e){
    alert('landmark preview failed');
  }
};

document.getElementById('cancelBtn').onclick = async () => {
  const res = await fetch('/api/cancel', { method: 'POST' });
  const js = await res.json();
  if(js.ok) {
    log('cancel requested');
    setRunning(false);
    setProgress(0, 'cancelled');
  } else {
    alert(js.error || 'No running task to cancel');
  }
};

function parseCsv(text){
  const lines = text.trim().split(/\r?\n/);
  const head = lines.shift().split(',');
  return lines.map(line => {
    const cols = line.split(',');
    const obj = {};
    head.forEach((h,i)=>obj[h]=cols[i]);
    return obj;
  });
}

async function refreshHistory(){
  try {
    const h = await fetch('/api/history').then(r=>r.json());
    historyList.innerHTML = '';
    (h.history||[]).slice().reverse().forEach(item => {
      const li = document.createElement('li');
      const status = item.ok ? 'OK' : `ERR: ${item.error||'unknown'}`;
      li.textContent = `[${status}] channels=${(item.channels||[]).join('+')} logs=${item.logCount||0}`;
      historyList.appendChild(li);
    });
  } catch {}
}

async function refreshOutputs(){
  try {
    const leaf = await fetch('/api/outputs/leaf').then(r=>r.text());
    const data = parseCsv(leaf);
    rows.innerHTML='';
    data.slice(0,20).forEach(d=>{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${d.region_name||d.region||''}</td><td>${d.count||''}</td><td>${d.confidence||''}</td>`;
      rows.appendChild(tr);
    });

    compareRows.innerHTML = '';
    for (const ch of ['red','green','farred']) {
      try {
        const txt = await fetch(`/api/outputs/leaf/${ch}`).then(r => r.ok ? r.text() : '');
        if (!txt) continue;
        const arr = parseCsv(txt);
        const total = arr.reduce((s, x) => s + Number(x.count || 0), 0);
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${ch}</td><td>${total}</td>`;
        compareRows.appendChild(tr);
      } catch {}
    }

    qcGrid.innerHTML = '';
    for(let i=0;i<5;i++){
      const img = document.createElement('img');
      img.src = `../outputs/qc_overlays/overlay_${String(i).padStart(3,'0')}.png?${Date.now()}`;
      img.onerror = ()=> img.remove();
      qcGrid.appendChild(img);
    }
    await refreshHistory();
    log('outputs refreshed');
  } catch (e){
    log('refresh failed: outputs not ready');
  }
}

document.getElementById('refreshBtn').onclick = refreshOutputs;
document.getElementById('refreshPreviewBtn').onclick = refreshOverlayPreview;
document.getElementById('autoPickBtn').onclick = async () => {
  try {
    const realPath = document.getElementById('realSlicePath').value || '../outputs/test_real.tif';
    const annotationPath = document.getElementById('atlasPath').value || '';
    const pixelSizeUm = Number(document.getElementById('pixelSizeUm')?.value || 0.65);
    const slicingPlane = document.getElementById('slicingPlane')?.value || 'coronal';
    const r = await fetch('/api/atlas/autopick-z', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ realPath, annotationPath, zStep: 1, pixelSizeUm, slicingPlane })
    }).then(x=>x.json());
    if(!r.ok){ alert(r.error || 'auto-pick failed'); return; }
    document.getElementById('atlasLabelPath').value = r.label_slice_tif;
    log(`auto-pick plane=${r.slicing_plane||slicingPlane}, z=${r.best_z}, score=${Number(r.best_score).toFixed(4)}`);
    alert(`Auto-pick done. plane=${r.slicing_plane||slicingPlane}, z=${r.best_z}`);
  } catch(e){
    alert('auto-pick failed');
  }
};

async function validatePaths(showAlert=true){
  const q = new URLSearchParams({
    inputDir: document.getElementById('inputDir').value,
    atlasPath: document.getElementById('atlasPath').value,
    structPath: document.getElementById('structPath').value,
  });
  const res = await fetch(`/api/validate?${q.toString()}`).then(r=>r.json());
  if(!res.ok){
    if(showAlert) alert('Path validation failed:\n' + res.issues.join('\n'));
    statusBadge.textContent = 'PATH_ERR';
    statusBadge.style.background = '#ed4245';
    log('validation failed');
  } else {
    if(!state.running){
      statusBadge.textContent = 'IDLE';
      statusBadge.style.background = '#4e5058';
    }
    log('validation passed');
  }
  return res.ok;
}

async function pollLogsUntilDone(){
  while(true){
    const s = await fetch('/api/status').then(r=>r.json());
    const logs = await fetch('/api/logs').then(r=>r.json());
    logBox.textContent = logs.logs.join('\n');
    logBox.scrollTop = logBox.scrollHeight;
    if(s.running){
      setProgress(Math.min(95, 20 + Math.floor((s.logCount || 0) * 0.8)), `running ${s.currentChannel||''}`.trim());
    }
    if(!s.running){
      if(s.done) log('pipeline done');
      if(s.error) alert(`Pipeline error: ${s.error}`);
      break;
    }
    await new Promise(r=>setTimeout(r, 1200));
  }
}

async function initInfo(){
  try {
    const info = await fetch('/api/info').then(r=>r.json());
    versionText.textContent = `v${info.version || '0.0.0'}`;
    if(!document.getElementById('outputDir').value) {
      document.getElementById('outputDir').value = info.outputs || '';
    }
  } catch {}
}

initInfo();

async function refreshOverlayPreview(){
  try {
    const realPath = document.getElementById('inputDir').value;
    if(!realPath) return;
    const alpha = Number(document.getElementById('alphaRange').value)/100;
    const modeEl = document.getElementById('overlayMode');
    let mode = modeEl.value;

    const fitMode = document.getElementById('fitMode')?.value || 'contain';
    if(fitMode === 'cover') log('warning: fitMode=cover may crop important brain regions');

    const payload = {
      realPath: document.getElementById('realSlicePath').value || '../outputs/test_real.tif',
      labelPath: document.getElementById('atlasLabelPath').value || '../outputs/test_label.tif',
      structureCsv: document.getElementById('structPath').value || '',
      minMeanThreshold: Number(document.getElementById('minMeanThreshold').value || 8),
      pixelSizeUm: Number(document.getElementById('pixelSizeUm')?.value || 0.65),
      rotateAtlas: Number(document.getElementById('rotateAtlas')?.value || 0),
      flipAtlas: document.getElementById('flipAtlas')?.value || 'none',
      majorTopK: Number(document.getElementById('majorTopK')?.value || 12),
      fitMode,
      alpha,
      mode,
    };

    let res = await fetch('/api/overlay/preview', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
    });
    let respJson = null;

    if(!res.ok){
      let msg = 'Overlay preview failed';
      try {
        const j = await res.json();
        msg = j.error || msg;
      } catch {}

      if(mode !== 'contour'){
        mode = 'contour';
        modeEl.value = 'contour';
        const fb = { ...payload, mode: 'contour' };
        res = await fetch('/api/overlay/preview', {
          method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(fb)
        });
        if(res.ok){
          respJson = await res.json();
          alert('Fill预览失败，已自动切换到Contour模式并成功显示。');
          log(`overlay fallback to contour: ${msg}`);
        } else {
          alert(`预览失败：${msg}`);
          log(`overlay preview failed: ${msg}`);
          return;
        }
      } else {
        alert(`预览失败：${msg}`);
        log(`overlay preview failed: ${msg}`);
        return;
      }
    }

    if(!respJson) respJson = await res.json();
    const dg = respJson?.diagnostic || null;
    if(dg){
      log(`diag: fit=${dg.fitMode}, real_aspect=${Number(dg.real_aspect||0).toFixed(3)}, atlas_aspect=${Number((dg.atlas_aspect ?? dg.atlas_aspect_before) || 0).toFixed(3)}, roi_bbox=${JSON.stringify(dg.roi_bbox||[])}, roi_err=${Number(dg.roi_roundtrip_error||0).toFixed(3)}`);
      const ra = Number(dg.real_aspect || 0);
      const aa = Number((dg.atlas_aspect ?? dg.atlas_aspect_before) || 0);
      if(ra > 0 && aa > 0){
        const ratioGap = Math.abs(ra / aa - 1.0);
        if(ratioGap > 0.35){
          alert(`Aspect ratio warning: real=${ra.toFixed(3)} vs atlas=${aa.toFixed(3)} (gap=${(ratioGap*100).toFixed(1)}%). 建议使用ROI或调整fit mode。`);
        }
      }
    }

    qcGrid.innerHTML = '';
    const img = document.createElement('img');
    img.src = `/api/outputs/overlay-preview?${Date.now()}`;
    img.onerror = ()=>{};
    qcGrid.appendChild(img);
    log(`overlay preview updated (alpha=${alpha.toFixed(2)}, mode=${mode})`);
  } catch(e){
    alert('预览失败：请检查切片路径与atlas路径是否正确。');
    log('overlay preview failed');
  }
}

document.getElementById('alphaRange').oninput = refreshOverlayPreview;
document.getElementById('overlayMode').onchange = refreshOverlayPreview;

['inputDir','atlasPath','structPath','realSlicePath','atlasLabelPath'].forEach(id=>{
  const el = document.getElementById(id);
  if(el) el.addEventListener('change', ()=> validatePaths(false));
});

document.getElementById('runBtn').onclick = async () => {
  if(state.running) return;
  if(!(await validatePaths(true))) return;
  setRunning(true);
  setProgress(5, 'queued');

  const channels = state.runAll ? ['red','green','farred'] : [state.channel];
  const payload = {
    configPath: '../configs/run_config.template.json',
    inputDir: document.getElementById('inputDir').value,
    channels,
  };
  const res = await fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const js = await res.json();
  if(!js.ok){
    alert(`Run failed to start: ${js.error||'unknown'}`);
    setRunning(false);
    return;
  }

  setProgress(25, `running ${channels.join('+')}`);
  await pollLogsUntilDone();
  setProgress(100, 'finished');
  await refreshOutputs();
  setRunning(false);
};

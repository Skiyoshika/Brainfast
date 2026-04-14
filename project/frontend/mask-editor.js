/**
 * mask-editor.js — Cellpose mask annotation canvas
 *
 * Two-layer HTML5 Canvas editor for correcting cell instance masks.
 * Tools: brush, eraser, add cell, delete, merge, split, pan.
 * Saves corrected masks to the backend for Cellpose training.
 */

/* global showToast, t, getOverlayJobId, pako */

const MaskEditor = (() => {
  'use strict';

  // --- State ---
  let state = {
    imageData: null,       // Uint8Array grayscale (display-resolution)
    maskData: null,        // Int32Array instance mask
    width: 0,
    height: 0,
    nextCellId: 1,
    selectedCell: 0,
    currentTool: 'brush',
    brushSize: 10,
    maskOpacity: 0.5,
    zoom: 1.0,
    panX: 0, panY: 0,
    isPanning: false,
    isDrawing: false,
    lastX: -1, lastY: -1,
    imagePath: '',
    undoStack: [],
    redoStack: [],
    maxUndo: 20,
    mergeFirst: null,       // first cell ID for merge operation
  };

  // Color LUT: cell ID -> [r,g,b]
  const COLOR_LUT = [];
  function ensureColor(id) {
    while (COLOR_LUT.length <= id) {
      // Generate distinct colors using golden-ratio hue spacing
      const hue = (COLOR_LUT.length * 137.508) % 360;
      const s = 0.7, l = 0.55;
      const c = (1 - Math.abs(2*l - 1)) * s;
      const x = c * (1 - Math.abs((hue/60) % 2 - 1));
      const m = l - c/2;
      let r, g, b;
      if (hue < 60)       { r=c; g=x; b=0; }
      else if (hue < 120) { r=x; g=c; b=0; }
      else if (hue < 180) { r=0; g=c; b=x; }
      else if (hue < 240) { r=0; g=x; b=c; }
      else if (hue < 300) { r=x; g=0; b=c; }
      else                { r=c; g=0; b=x; }
      COLOR_LUT.push([
        Math.round((r+m)*255),
        Math.round((g+m)*255),
        Math.round((b+m)*255),
      ]);
    }
    return COLOR_LUT[id];
  }

  // --- DOM refs (resolved on first open) ---
  let dom = {};
  function resolveDom() {
    dom.modal = document.getElementById('maskEditorModal');
    dom.imageCanvas = document.getElementById('meImageCanvas');
    dom.maskCanvas = document.getElementById('meMaskCanvas');
    dom.canvasWrap = document.getElementById('meCanvasWrap');
    dom.canvasArea = document.getElementById('meCanvasArea');
    dom.brushSize = document.getElementById('meBrushSize');
    dom.brushSizeVal = document.getElementById('meBrushSizeVal');
    dom.maskOpacity = document.getElementById('meMaskOpacity');
    dom.maskOpacityVal = document.getElementById('meMaskOpacityVal');
    dom.cellList = document.getElementById('meCellList');
    dom.statusLeft = document.getElementById('meStatusLeft');
    dom.statusRight = document.getElementById('meStatusRight');
    dom.undoBtn = document.getElementById('meUndoBtn');
    dom.redoBtn = document.getElementById('meRedoBtn');
    dom.saveBtn = document.getElementById('meSaveTrainingBtn');
    dom.closeBtn = document.getElementById('maskEditorClose');
  }

  // --- Canvas Rendering ---
  function renderImage() {
    if (!state.imageData) return;
    const ctx = dom.imageCanvas.getContext('2d');
    const imgData = ctx.createImageData(state.width, state.height);
    for (let i = 0; i < state.imageData.length; i++) {
      const v = state.imageData[i];
      imgData.data[i*4] = v;
      imgData.data[i*4+1] = v;
      imgData.data[i*4+2] = v;
      imgData.data[i*4+3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }

  function renderMask() {
    if (!state.maskData) return;
    const ctx = dom.maskCanvas.getContext('2d');
    const imgData = ctx.createImageData(state.width, state.height);
    const alpha = Math.round(state.maskOpacity * 255);
    for (let i = 0; i < state.maskData.length; i++) {
      const cellId = state.maskData[i];
      if (cellId <= 0) {
        imgData.data[i*4+3] = 0; // transparent background
      } else {
        const c = ensureColor(cellId);
        imgData.data[i*4] = c[0];
        imgData.data[i*4+1] = c[1];
        imgData.data[i*4+2] = c[2];
        imgData.data[i*4+3] = (cellId === state.selectedCell) ? Math.min(alpha + 60, 255) : alpha;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  function updateTransform() {
    dom.canvasWrap.style.transform =
      'translate(' + state.panX + 'px, ' + state.panY + 'px) scale(' + state.zoom + ')';
  }

  function render() {
    renderMask();
    updateStatus();
    updateCellList();
  }

  // --- Undo/Redo ---
  function pushUndo() {
    if (state.undoStack.length >= state.maxUndo) {
      state.undoStack.shift();
    }
    state.undoStack.push(new Int32Array(state.maskData));
    state.redoStack = [];
  }

  function undo() {
    if (state.undoStack.length === 0) return;
    state.redoStack.push(new Int32Array(state.maskData));
    state.maskData = state.undoStack.pop();
    state.nextCellId = Math.max(1, maxCellId() + 1);
    render();
  }

  function redo() {
    if (state.redoStack.length === 0) return;
    state.undoStack.push(new Int32Array(state.maskData));
    state.maskData = state.redoStack.pop();
    state.nextCellId = Math.max(1, maxCellId() + 1);
    render();
  }

  function maxCellId() {
    let max = 0;
    for (let i = 0; i < state.maskData.length; i++) {
      if (state.maskData[i] > max) max = state.maskData[i];
    }
    return max;
  }

  // --- Tools ---
  function paintCircle(cx, cy, radius, value) {
    const r2 = radius * radius;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx*dx + dy*dy > r2) continue;
        const px = cx + dx, py = cy + dy;
        if (px < 0 || px >= state.width || py < 0 || py >= state.height) continue;
        state.maskData[py * state.width + px] = value;
      }
    }
  }

  function paintLine(x0, y0, x1, y1, radius, value) {
    const dx = Math.abs(x1-x0), dy = Math.abs(y1-y0);
    const steps = Math.max(dx, dy, 1);
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const px = Math.round(x0 + (x1-x0)*t);
      const py = Math.round(y0 + (y1-y0)*t);
      paintCircle(px, py, radius, value);
    }
  }

  function deleteCellById(cellId) {
    pushUndo();
    for (let i = 0; i < state.maskData.length; i++) {
      if (state.maskData[i] === cellId) state.maskData[i] = 0;
    }
    if (state.selectedCell === cellId) state.selectedCell = 0;
    render();
  }

  function mergeCells(idA, idB) {
    if (idA === idB) return;
    pushUndo();
    const target = Math.min(idA, idB);
    const source = Math.max(idA, idB);
    for (let i = 0; i < state.maskData.length; i++) {
      if (state.maskData[i] === source) state.maskData[i] = target;
    }
    state.selectedCell = target;
    render();
  }

  function splitCellAtLine(x0, y0, x1, y1) {
    const idx = y0 * state.width + x0;
    const cellId = state.maskData[idx];
    if (cellId <= 0) return;

    pushUndo();

    // Erase the line through the cell
    paintLine(x0, y0, x1, y1, 1, 0);

    // Flood-fill connected components from the original cell
    const visited = new Uint8Array(state.maskData.length);
    const newId = state.nextCellId++;
    let foundFirst = false;

    for (let i = 0; i < state.maskData.length; i++) {
      if (state.maskData[i] !== cellId || visited[i]) continue;

      // BFS flood fill
      const queue = [i];
      visited[i] = 1;
      const component = [];
      while (queue.length > 0) {
        const ci = queue.shift();
        component.push(ci);
        const cx = ci % state.width, cy = Math.floor(ci / state.width);
        for (const [ddx, ddy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
          const nx = cx+ddx, ny = cy+ddy;
          if (nx<0 || nx>=state.width || ny<0 || ny>=state.height) continue;
          const ni = ny*state.width + nx;
          if (!visited[ni] && state.maskData[ni] === cellId) {
            visited[ni] = 1;
            queue.push(ni);
          }
        }
      }

      if (!foundFirst) {
        foundFirst = true;
      } else {
        for (const pi of component) {
          state.maskData[pi] = newId;
        }
      }
    }
    render();
  }

  // --- Mouse Handlers ---
  function canvasCoords(e) {
    const rect = dom.maskCanvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / state.zoom);
    const y = Math.floor((e.clientY - rect.top) / state.zoom);
    return { x: x, y: y };
  }

  function onMouseDown(e) {
    if (e.button === 1 || (e.button === 0 && state.currentTool === 'pan') || e.shiftKey) {
      state.isPanning = true;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      return;
    }

    const pos = canvasCoords(e);

    if (state.currentTool === 'delete') {
      const cellId = state.maskData[pos.y * state.width + pos.x];
      if (cellId > 0) deleteCellById(cellId);
      return;
    }

    if (state.currentTool === 'merge') {
      const cellId = state.maskData[pos.y * state.width + pos.x];
      if (cellId <= 0) return;
      if (state.mergeFirst === null) {
        state.mergeFirst = cellId;
        state.selectedCell = cellId;
        render();
        updateStatus();
        return;
      } else {
        mergeCells(state.mergeFirst, cellId);
        state.mergeFirst = null;
        return;
      }
    }

    if (state.currentTool === 'split') {
      state.isDrawing = true;
      state.lastX = pos.x;
      state.lastY = pos.y;
      return;
    }

    if (state.currentTool === 'brush' || state.currentTool === 'eraser' || state.currentTool === 'addcell') {
      state.isDrawing = true;
      pushUndo();

      if (state.currentTool === 'addcell') {
        state.selectedCell = state.nextCellId++;
      }

      var value = state.currentTool === 'eraser' ? 0 : state.selectedCell;
      if (value <= 0 && state.currentTool !== 'eraser') {
        var cellId = state.maskData[pos.y * state.width + pos.x];
        if (cellId > 0) {
          state.selectedCell = cellId;
        } else {
          state.selectedCell = state.nextCellId++;
        }
      }

      var paintVal = state.currentTool === 'eraser' ? 0 : state.selectedCell;
      paintCircle(pos.x, pos.y, state.brushSize, paintVal);
      state.lastX = pos.x;
      state.lastY = pos.y;
      renderMask();
    }
  }

  function onMouseMove(e) {
    if (state.isPanning) {
      state.panX += e.clientX - state.lastX;
      state.panY += e.clientY - state.lastY;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      updateTransform();
      return;
    }

    if (!state.isDrawing) return;
    const pos = canvasCoords(e);

    if (state.currentTool === 'split') {
      return;
    }

    var value = state.currentTool === 'eraser' ? 0 : state.selectedCell;
    paintLine(state.lastX, state.lastY, pos.x, pos.y, state.brushSize, value);
    state.lastX = pos.x;
    state.lastY = pos.y;
    renderMask();
  }

  function onMouseUp(e) {
    if (state.isPanning) {
      state.isPanning = false;
      return;
    }

    if (state.isDrawing && state.currentTool === 'split') {
      const pos = canvasCoords(e);
      splitCellAtLine(state.lastX, state.lastY, pos.x, pos.y);
    }

    state.isDrawing = false;
    if (state.currentTool !== 'split') {
      updateCellList();
      updateStatus();
    }
  }

  function onWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.1, Math.min(10, state.zoom * delta));

    const rect = dom.canvasArea.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    state.panX = cx - (cx - state.panX) * (newZoom / state.zoom);
    state.panY = cy - (cy - state.panY) * (newZoom / state.zoom);
    state.zoom = newZoom;
    updateTransform();
    updateStatus();
  }

  // --- UI Updates ---
  function updateStatus() {
    var toolName = state.currentTool.charAt(0).toUpperCase() + state.currentTool.slice(1);
    var left = toolName + ': ' + state.brushSize + 'px';
    if (state.selectedCell > 0) left += ' | Cell #' + state.selectedCell;
    if (state.currentTool === 'merge' && state.mergeFirst !== null) {
      left += ' | Click second cell to merge with #' + state.mergeFirst;
    }
    dom.statusLeft.textContent = left;
    dom.statusRight.textContent = 'Zoom: ' + Math.round(state.zoom * 100) + '% | ' + maxCellId() + ' cells';
  }

  function updateCellList() {
    var cellIds = new Set();
    for (let i = 0; i < state.maskData.length; i++) {
      if (state.maskData[i] > 0) cellIds.add(state.maskData[i]);
    }
    dom.cellList.innerHTML = '';
    var sorted = Array.from(cellIds).sort(function(a,b) { return a-b; });
    for (var j = 0; j < sorted.length; j++) {
      var id = sorted[j];
      var c = ensureColor(id);
      var div = document.createElement('div');
      div.className = 'me-cell-item' + (id === state.selectedCell ? ' selected' : '');
      div.innerHTML = '<span class="me-cell-dot" style="background:rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')"></span> Cell ' + id;
      div.setAttribute('data-cell-id', id);
      div.onclick = (function(cid) { return function() { state.selectedCell = cid; render(); }; })(id);
      dom.cellList.appendChild(div);
    }
  }

  // --- Keyboard Shortcuts ---
  function onKeyDown(e) {
    if (!dom.modal || !dom.modal.classList.contains('active')) return;

    var key = e.key.toLowerCase();
    if (key === 'b') setTool('brush');
    else if (key === 'e') setTool('eraser');
    else if (key === 'n') setTool('addcell');
    else if (key === 'd') setTool('delete');
    else if (key === '[') {
      state.brushSize = Math.max(1, state.brushSize - 1);
      dom.brushSize.value = state.brushSize;
      dom.brushSizeVal.textContent = state.brushSize;
      updateStatus();
    }
    else if (key === ']') {
      state.brushSize = Math.min(50, state.brushSize + 1);
      dom.brushSize.value = state.brushSize;
      dom.brushSizeVal.textContent = state.brushSize;
      updateStatus();
    }
    else if (key === 'z' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
    }
    else if (key === 'escape') close();
  }

  function setTool(tool) {
    state.currentTool = tool;
    state.mergeFirst = null;
    document.querySelectorAll('.me-tool-btn').forEach(function(btn) {
      btn.classList.toggle('active', btn.dataset.meTool === tool);
    });
    updateStatus();
  }

  // --- Save to Training Set ---
  async function saveToTrainingSet() {
    if (!state.maskData || !state.imagePath) return;

    var compressed = pako.deflate(new Uint8Array(state.maskData.buffer));
    var hexStr = Array.from(compressed, function(b) { return b.toString(16).padStart(2, '0'); }).join('');

    try {
      var res = await fetch('/api/cellpose/save-training-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imagePath: state.imagePath,
          maskHex: hexStr,
          width: state.width,
          height: state.height,
        }),
      });
      var data = await res.json();
      if (data.ok) {
        if (typeof showToast === 'function') {
          showToast('Saved! Training set: ' + data.trainingSetStats.totalImages + ' images, ' + data.trainingSetStats.totalCells + ' cells', 'success', 4000);
        }
      } else {
        if (typeof showToast === 'function') showToast('Save failed: ' + data.error, 'error');
      }
    } catch (err) {
      if (typeof showToast === 'function') showToast('Save failed: ' + err.message, 'error');
    }
  }

  // --- Open/Close ---
  async function open(imagePath, jobId) {
    resolveDom();
    state.imagePath = imagePath;
    state.undoStack = [];
    state.redoStack = [];
    state.mergeFirst = null;
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    state.selectedCell = 1;

    // Fetch masks from backend
    try {
      var res = await fetch('/api/detect/preview/masks?jobId=' + encodeURIComponent(jobId));
      var data = await res.json();
      if (!data.ok) {
        if (typeof showToast === 'function') showToast('No masks available. Run detection with returnMasks first.', 'warning');
        return;
      }

      state.width = data.width;
      state.height = data.height;
      state.nextCellId = data.cellCount + 1;

      // Decompress mask
      var hexBytes = new Uint8Array(data.maskHex.match(/.{1,2}/g).map(function(b) { return parseInt(b, 16); }));
      var decompressed = pako.inflate(hexBytes);
      state.maskData = new Int32Array(decompressed.buffer);

    } catch (err) {
      if (typeof showToast === 'function') showToast('Failed to load masks: ' + err.message, 'error');
      return;
    }

    // Fetch original image as grayscale
    try {
      var imgRes = await fetch('/api/detect/preview/overlay?jobId=' + encodeURIComponent(jobId));
      var blob = await imgRes.blob();
      var bmp = await createImageBitmap(blob);

      var offscreen = document.createElement('canvas');
      offscreen.width = bmp.width;
      offscreen.height = bmp.height;
      var offCtx = offscreen.getContext('2d');
      offCtx.drawImage(bmp, 0, 0);
      var pixels = offCtx.getImageData(0, 0, bmp.width, bmp.height);

      state.imageData = new Uint8Array(state.width * state.height);
      var srcW = bmp.width, srcH = bmp.height;
      for (var y = 0; y < state.height; y++) {
        for (var x = 0; x < state.width; x++) {
          var sx = Math.floor(x * srcW / state.width);
          var sy = Math.floor(y * srcH / state.height);
          state.imageData[y * state.width + x] = pixels.data[(sy * srcW + sx) * 4];
        }
      }
    } catch (err) {
      state.imageData = new Uint8Array(state.width * state.height);
    }

    // Setup canvases
    dom.imageCanvas.width = state.width;
    dom.imageCanvas.height = state.height;
    dom.maskCanvas.width = state.width;
    dom.maskCanvas.height = state.height;
    dom.canvasWrap.style.width = state.width + 'px';
    dom.canvasWrap.style.height = state.height + 'px';

    // Center in viewport
    var area = dom.canvasArea.getBoundingClientRect();
    var fitZoom = Math.min(area.width / state.width, (area.height - 30) / state.height) * 0.9;
    state.zoom = fitZoom;
    state.panX = (area.width - state.width * fitZoom) / 2;
    state.panY = (area.height - state.height * fitZoom) / 2;

    renderImage();
    render();
    updateTransform();
    setTool('brush');

    // Show modal
    dom.modal.classList.add('active');

    // Bind events
    dom.maskCanvas.addEventListener('mousedown', onMouseDown);
    dom.maskCanvas.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    dom.canvasArea.addEventListener('wheel', onWheel, { passive: false });
  }

  function close() {
    dom.modal.classList.remove('active');
    dom.maskCanvas.removeEventListener('mousedown', onMouseDown);
    dom.maskCanvas.removeEventListener('mousemove', onMouseMove);
    window.removeEventListener('mouseup', onMouseUp);
    dom.canvasArea.removeEventListener('wheel', onWheel);
    state.maskData = null;
    state.imageData = null;
  }

  // --- Event Bindings (deferred until DOM ready) ---
  function init() {
    resolveDom();

    // Tool buttons
    document.querySelectorAll('.me-tool-btn').forEach(function(btn) {
      btn.addEventListener('click', function() { setTool(btn.dataset.meTool); });
    });

    // Sliders
    dom.brushSize.oninput = function() {
      state.brushSize = parseInt(this.value, 10);
      dom.brushSizeVal.textContent = this.value;
      updateStatus();
    };
    dom.maskOpacity.oninput = function() {
      state.maskOpacity = parseInt(this.value, 10) / 100;
      dom.maskOpacityVal.textContent = this.value + '%';
      renderMask();
    };

    // Buttons
    dom.undoBtn.addEventListener('click', undo);
    dom.redoBtn.addEventListener('click', redo);
    dom.saveBtn.addEventListener('click', saveToTrainingSet);
    dom.closeBtn.addEventListener('click', close);

    // Keyboard
    document.addEventListener('keydown', onKeyDown);
  }

  // Init when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { open: open, close: close };
})();

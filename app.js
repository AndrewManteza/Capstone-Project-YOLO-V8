/* ------------------------------------------------------------------
   Weapon Detection Scan — client-side YOLO inference with ONNX Runtime Web
   ------------------------------------------------------------------
   This file has three parts:
   1. UI wiring (dropzone, canvas, modal, settings)
   2. Model loading
   3. Preprocessing -> inference -> postprocessing (NMS) -> drawing

   If your exported model's output shape doesn't match the assumption
   below, only `postprocessYolo()` needs to change.
------------------------------------------------------------------- */

const els = {
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  stage: document.getElementById('stage'),
  canvas: document.getElementById('canvas'),
  scanLine: document.getElementById('scanLine'),
  statusText: document.getElementById('statusText'),
  scanBtn: document.getElementById('scanBtn'),
  rescanBtn: document.getElementById('rescanBtn'),
  modelStatus: document.getElementById('modelStatus'),
  modelStatusText: document.getElementById('modelStatusText'),
  modelPath: document.getElementById('modelPath'),
  confSlider: document.getElementById('confSlider'),
  confValue: document.getElementById('confValue'),
  sizeSlider: document.getElementById('sizeSlider'),
  sizeValue: document.getElementById('sizeValue'),
  modalOverlay: document.getElementById('modalOverlay'),
  modalConfidence: document.getElementById('modalConfidence'),
  modalDismiss: document.getElementById('modalDismiss'),
};

const ctx = els.canvas.getContext('2d');

// The class label(s) in your model that should trigger the alert.
// If your model has multiple classes, list every label that counts as a weapon.
const ALERT_LABELS = ['gun', 'pistol', 'firearm', 'weapon', 'rifle'];

// If your model only has a single class (common for a dedicated "gun detector"),
// index 0 is treated as that class regardless of name.
const CLASS_NAMES = ['gun'];

let session = null;
let currentImage = null;
let modelInputSize = parseInt(els.sizeSlider.value, 10);

init();

async function init() {
  bindUI();
  await loadModel(els.modelPath.value.trim());
}

function bindUI() {
  els.dropzone.addEventListener('click', () => els.fileInput.click());
  els.dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    els.dropzone.classList.add('drag');
  });
  els.dropzone.addEventListener('dragleave', () => els.dropzone.classList.remove('drag'));
  els.dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    els.dropzone.classList.remove('drag');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  els.fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  els.scanBtn.addEventListener('click', runScan);
  els.rescanBtn.addEventListener('click', () => {
    els.fileInput.value = '';
    els.fileInput.click();
  });

  els.confSlider.addEventListener('input', () => {
    els.confValue.textContent = parseFloat(els.confSlider.value).toFixed(2);
  });

  els.sizeSlider.addEventListener('input', () => {
    modelInputSize = parseInt(els.sizeSlider.value, 10);
    els.sizeValue.textContent = modelInputSize;
  });

  els.modelPath.addEventListener('change', () => {
    loadModel(els.modelPath.value.trim());
  });

  els.modalDismiss.addEventListener('click', () => {
    els.modalOverlay.classList.remove('visible');
  });
  els.modalOverlay.addEventListener('click', (e) => {
    if (e.target === els.modalOverlay) els.modalOverlay.classList.remove('visible');
  });
}

async function loadModel(path) {
  setModelStatus('loading', 'Loading model…');
  try {
    session = await ort.InferenceSession.create(path, {
      executionProviders: ['wasm'],
    });
    setModelStatus('ready', 'Model loaded');
  } catch (err) {
    console.error('Model load failed:', err);
    session = null;
    setModelStatus('missing', 'No model found — add one at ' + path);
  }
}

function setModelStatus(state, text) {
  els.modelStatus.classList.toggle('model-missing', state !== 'ready');
  els.modelStatusText.textContent = text;
}

function handleFile(file) {
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    els.canvas.width = img.width;
    els.canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    els.stage.classList.add('visible');
    els.statusText.textContent = 'Ready to scan';
    els.statusText.className = 'status-text';
  };
  img.src = URL.createObjectURL(file);
}

async function runScan() {
  if (!currentImage) return;
  if (!session) {
    els.statusText.textContent = 'No model loaded — check model settings below';
    els.statusText.className = 'status-text alert';
    return;
  }

  els.scanBtn.disabled = true;
  els.rescanBtn.disabled = true;
  els.scanLine.classList.add('active');
  els.statusText.textContent = 'Scanning…';
  els.statusText.className = 'status-text';

  try {
    const { tensor, scale, padX, padY } = preprocess(currentImage, modelInputSize);
    const feeds = { [session.inputNames[0]]: tensor };
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];

    const confThreshold = parseFloat(els.confSlider.value);
    const detections = postprocessYolo(output, confThreshold, scale, padX, padY);

    // redraw the base image, then overlay boxes
    ctx.drawImage(currentImage, 0, 0);
    drawDetections(detections);

    const alertHits = detections.filter((d) => isAlertClass(d.classId));

    if (alertHits.length > 0) {
      const topHit = alertHits.reduce((a, b) => (b.score > a.score ? b : a));
      els.statusText.textContent = `${alertHits.length} match(es) found`;
      els.statusText.className = 'status-text alert';
      showAlertModal(topHit.score);
    } else {
      els.statusText.textContent = 'Clear — no weapons detected';
      els.statusText.className = 'status-text clear';
    }
  } catch (err) {
    console.error('Inference failed:', err);
    els.statusText.textContent = 'Scan failed — see console for details';
    els.statusText.className = 'status-text alert';
  } finally {
    els.scanLine.classList.remove('active');
    els.scanBtn.disabled = false;
    els.rescanBtn.disabled = false;
  }
}

function isAlertClass(classId) {
  const name = (CLASS_NAMES[classId] || '').toLowerCase();
  if (CLASS_NAMES.length === 1) return true; // single-class model: any detection counts
  return ALERT_LABELS.includes(name);
}

/* ---------------- preprocessing ---------------- */

function preprocess(img, size) {
  // letterbox resize into a size x size square, keeping aspect ratio
  const off = document.createElement('canvas');
  off.width = size;
  off.height = size;
  const offCtx = off.getContext('2d');
  offCtx.fillStyle = '#727272';
  offCtx.fillRect(0, 0, size, size);

  const scale = Math.min(size / img.width, size / img.height);
  const newW = Math.round(img.width * scale);
  const newH = Math.round(img.height * scale);
  const padX = Math.floor((size - newW) / 2);
  const padY = Math.floor((size - newH) / 2);

  offCtx.drawImage(img, 0, 0, img.width, img.height, padX, padY, newW, newH);

  const { data } = offCtx.getImageData(0, 0, size, size);
  const floatData = new Float32Array(size * size * 3);

  // HWC RGBA -> CHW RGB, normalized 0..1
  for (let i = 0; i < size * size; i++) {
    floatData[i] = data[i * 4] / 255;                         // R
    floatData[size * size + i] = data[i * 4 + 1] / 255;       // G
    floatData[2 * size * size + i] = data[i * 4 + 2] / 255;   // B
  }

  const tensor = new ort.Tensor('float32', floatData, [1, 3, size, size]);
  return { tensor, scale, padX, padY };
}

/* ---------------- postprocessing ----------------
   Assumes a YOLOv8-style export with output shape [1, 4+numClasses, numBoxes]
   (box coords are cx, cy, w, h in the model's input-size space).
   Adjust this function if your export shape differs (e.g. YOLOv5's
   [1, numBoxes, 5+numClasses] with an explicit objectness score).
------------------------------------------------------------------- */

function postprocessYolo(output, confThreshold, scale, padX, padY) {
  const dims = output.dims; // e.g. [1, 5, 8400] for a single-class model
  const data = output.data;
  const numAttrs = dims[1];
  const numBoxes = dims[2];
  const numClasses = numAttrs - 4;

  const candidates = [];

  for (let i = 0; i < numBoxes; i++) {
    let bestScore = 0;
    let bestClass = 0;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numBoxes + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }
    if (bestScore < confThreshold) continue;

    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];

    // undo letterbox padding + scale to get back to original image coordinates
    const x1 = (cx - w / 2 - padX) / scale;
    const y1 = (cy - h / 2 - padY) / scale;
    const x2 = (cx + w / 2 - padX) / scale;
    const y2 = (cy + h / 2 - padY) / scale;

    candidates.push({ x1, y1, x2, y2, score: bestScore, classId: bestClass });
  }

  return nonMaxSuppression(candidates, 0.45);
}

function nonMaxSuppression(boxes, iouThreshold) {
  boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  const used = new Array(boxes.length).fill(false);

  for (let i = 0; i < boxes.length; i++) {
    if (used[i]) continue;
    kept.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (used[j]) continue;
      if (boxes[i].classId !== boxes[j].classId) continue;
      if (iou(boxes[i], boxes[j]) > iouThreshold) used[j] = true;
    }
  }
  return kept;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

/* ---------------- drawing ---------------- */

function drawDetections(detections) {
  detections.forEach((d) => {
    const alert = isAlertClass(d.classId);
    const color = alert ? '#e8433a' : '#3b82f6';
    const label = `${CLASS_NAMES[d.classId] || 'object'} ${(d.score * 100).toFixed(0)}%`;

    ctx.lineWidth = Math.max(2, currentImage.width / 300);
    ctx.strokeStyle = color;
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);

    ctx.font = `${Math.max(14, currentImage.width / 60)}px -apple-system, sans-serif`;
    const textWidth = ctx.measureText(label).width;
    const textHeight = Math.max(14, currentImage.width / 60) + 8;

    ctx.fillStyle = color;
    ctx.fillRect(d.x1, Math.max(0, d.y1 - textHeight), textWidth + 12, textHeight);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, d.x1 + 6, Math.max(textHeight - 6, d.y1 - 6));
  });
}

function showAlertModal(score) {
  els.modalConfidence.textContent = `Confidence: ${(score * 100).toFixed(1)}%`;
  els.modalOverlay.classList.add('visible');
}

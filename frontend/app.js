/**
 * app.js — AI Text Detector Frontend
 * Supports: paste text mode + file upload (PDF/DOCX/TXT)
 * Vanilla JS, zero dependencies.
 */

'use strict';

// ─────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────
const CONFIG = {
  API_BASE_URL:        'http://localhost:8000',
  MIN_CHARS:           20,
  MAX_CHARS:           50_000,
  MAX_FILE_MB:         20,
  ALLOWED_EXTENSIONS:  ['pdf', 'docx', 'txt'],
  CHAR_WARN_RATIO:     0.9,
};

const MAX_FILE_BYTES = CONFIG.MAX_FILE_MB * 1024 * 1024;

// ─────────────────────────────────────────────────────────────
// DOM REFS
// ─────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const dom = {
  // Tabs
  tabText:        $('tab-text'),
  tabFile:        $('tab-file'),
  panelText:      $('panel-text'),
  panelFile:      $('panel-file'),

  // Text input
  textInput:      $('text-input'),
  charCount:      $('char-count'),
  inputHint:      $('input-hint'),
  btnClear:       $('btn-clear'),
  btnPaste:       $('btn-paste'),

  // File input
  fileInput:      $('file-input'),
  dropZone:       $('drop-zone'),
  btnBrowse:      $('btn-browse'),
  filePreview:    $('file-preview'),
  fileTypeIcon:   $('file-type-icon'),
  fileName:       $('file-name'),
  fileMeta:       $('file-meta'),
  btnFileRemove:  $('btn-file-remove'),

  // Shared
  inputError:     $('input-error'),
  inputErrorText: $('input-error-text'),
  btnDetect:      $('btn-detect'),

  // Results
  resultSection:  $('result-section'),
  resultEmpty:    $('result-empty'),
  resultLoading:  $('result-loading'),
  resultCard:     $('result-card'),
  resultError:    $('result-error'),
  loadingText:    $('loading-text'),
  loadingSub:     $('loading-sub'),

  // Result content
  resultVerdict:    $('result-verdict'),
  resultIcon:       $('result-icon'),
  resultPrediction: $('result-prediction'),
  gaugeHumanPct:    $('gauge-human-pct'),
  gaugeAiPct:       $('gauge-ai-pct'),
  gaugeHumanFill:   $('gauge-human-fill'),
  gaugeAiFill:      $('gauge-ai-fill'),
  confidenceValue:  $('confidence-value'),
  confidenceBar:    $('confidence-bar'),
  confBarTrack:     document.querySelector('.confidence-bar-track'),

  // Document info
  docInfoPanel: $('doc-info-panel'),
  docFilename:  $('doc-filename'),
  docPages:     $('doc-pages'),
  docWords:     $('doc-words'),
  docChars:     $('doc-chars'),

  // Buttons
  btnReset:  $('btn-reset'),
  btnRetry:  $('btn-retry'),
  resultErrorText: $('result-error-text'),
};

// ─────────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────────
const state = {
  mode:         'text',    // 'text' | 'file'
  loading:      false,
  selectedFile: null,
  lastText:     '',
  lastFile:     null,
};

// ─────────────────────────────────────────────────────────────
// FILE TYPE ICONS
// ─────────────────────────────────────────────────────────────
const FILE_ICONS = {
  pdf:  `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#f472b6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="16" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>`,
  docx: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>`,
  txt:  `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>`,
};

// ─────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

function showError(msg) {
  dom.inputError.hidden = false;
  dom.inputErrorText.textContent = msg;
}

function hideError() {
  dom.inputError.hidden = true;
}

// ─────────────────────────────────────────────────────────────
// RESULT PANEL CONTROLLER
// ─────────────────────────────────────────────────────────────
const ResultPanel = {
  showEmpty() {
    dom.resultEmpty.hidden   = false;
    dom.resultLoading.hidden = true;
    dom.resultCard.hidden    = true;
    dom.resultError.hidden   = true;
  },
  showLoading(isFile = false) {
    dom.resultEmpty.hidden   = true;
    dom.resultLoading.hidden = false;
    dom.resultCard.hidden    = true;
    dom.resultError.hidden   = true;
    if (isFile) {
      dom.loadingText.textContent = 'Membaca dan menganalisis dokumen...';
      dom.loadingSub.textContent  = 'Mengekstrak teks → menghitung 15 fitur + TF-IDF';
    } else {
      dom.loadingText.textContent = 'Menganalisis pola linguistik...';
      dom.loadingSub.textContent  = 'Menghitung 15 fitur + TF-IDF (300 token)';
    }
  },
  showResult(data, documentInfo = null) {
    dom.resultEmpty.hidden   = true;
    dom.resultLoading.hidden = true;
    dom.resultCard.hidden    = false;
    dom.resultError.hidden   = true;
    _renderResult(data, documentInfo);
  },
  showError(message) {
    dom.resultEmpty.hidden   = true;
    dom.resultLoading.hidden = true;
    dom.resultCard.hidden    = true;
    dom.resultError.hidden   = false;
    dom.resultErrorText.textContent = message;
  },
};

// ─────────────────────────────────────────────────────────────
// RENDER RESULT
// ─────────────────────────────────────────────────────────────
function _renderResult(data, documentInfo) {
  const isAI = data.prediction === 'AI Generated';
  const pHuman = data.probabilities.human;
  const pAI    = data.probabilities.ai;

  // Verdict banner
  dom.resultVerdict.className = `result-verdict ${isAI ? 'is-ai' : 'is-human'}`;
  dom.resultIcon.textContent  = isAI ? '🤖' : '👤';
  dom.resultPrediction.textContent = data.prediction;

  // Dual gauge — reset first, then animate
  dom.gaugeHumanPct.textContent = `${pHuman.toFixed(1)}%`;
  dom.gaugeAiPct.textContent    = `${pAI.toFixed(1)}%`;

  dom.gaugeHumanFill.style.width = '0%';
  dom.gaugeAiFill.style.width    = '0%';

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      dom.gaugeHumanFill.style.width = `${pHuman}%`;
      dom.gaugeAiFill.style.width    = `${pAI}%`;
    });
  });

  // Confidence bar
  dom.confidenceValue.textContent = `${data.confidence.toFixed(1)}%`;
  dom.confidenceBar.className = `confidence-bar ${isAI ? 'is-ai' : 'is-human'}`;
  dom.confidenceBar.style.width = '0%';
  dom.confBarTrack?.setAttribute('aria-valuenow', data.confidence);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      dom.confidenceBar.style.width = `${data.confidence}%`;
    });
  });

  // Document info panel (file uploads only)
  if (documentInfo) {
    dom.docInfoPanel.hidden = false;
    dom.docFilename.textContent = documentInfo.filename || '—';
    dom.docPages.textContent    = documentInfo.pages != null ? documentInfo.pages : '—';
    dom.docWords.textContent    = (documentInfo.words || 0).toLocaleString('id-ID');
    dom.docChars.textContent    = (documentInfo.characters || 0).toLocaleString('id-ID');
  } else {
    dom.docInfoPanel.hidden = true;
  }
}

// ─────────────────────────────────────────────────────────────
// TAB SWITCHER
// ─────────────────────────────────────────────────────────────
function switchTab(mode) {
  state.mode = mode;
  hideError();

  const isText = mode === 'text';

  // Tabs
  dom.tabText.classList.toggle('mode-tab--active', isText);
  dom.tabFile.classList.toggle('mode-tab--active', !isText);
  dom.tabText.setAttribute('aria-selected', isText);
  dom.tabFile.setAttribute('aria-selected', !isText);

  // Panels
  dom.panelText.classList.toggle('input-panel--hidden', !isText);
  dom.panelFile.classList.toggle('input-panel--hidden', isText);

  // Reset results
  ResultPanel.showEmpty();

  // Update detect button
  updateDetectButton();
}

// ─────────────────────────────────────────────────────────────
// TEXT INPUT STATE
// ─────────────────────────────────────────────────────────────
function updateDetectButton() {
  if (state.mode === 'text') {
    const len = dom.textInput.value.trim().length;
    dom.btnDetect.disabled = state.loading || len < CONFIG.MIN_CHARS;
  } else {
    dom.btnDetect.disabled = state.loading || !state.selectedFile;
  }
}

function onTextInput() {
  const raw  = dom.textInput.value;
  const len  = raw.length;
  const trim = raw.trim().length;

  // Char count
  dom.charCount.textContent = `${len.toLocaleString('id-ID')} / ${CONFIG.MAX_CHARS.toLocaleString('id-ID')}`;
  dom.charCount.classList.toggle('is-warning', len > CONFIG.MAX_CHARS * CONFIG.CHAR_WARN_RATIO);

  // Hint
  dom.inputHint.textContent = trim > 0 && trim < CONFIG.MIN_CHARS
    ? `Butuh ${CONFIG.MIN_CHARS - trim} karakter lagi`
    : 'Minimal 20 karakter';

  // Clear button
  dom.btnClear.disabled = len === 0;

  hideError();
  updateDetectButton();
}

// ─────────────────────────────────────────────────────────────
// FILE HANDLING
// ─────────────────────────────────────────────────────────────
function getFileExt(filename) {
  return filename.split('.').pop().toLowerCase();
}

function validateFile(file) {
  const ext = getFileExt(file.name);
  if (!CONFIG.ALLOWED_EXTENSIONS.includes(ext)) {
    showError(`Format tidak didukung: .${ext}. Gunakan PDF, DOCX, atau TXT.`);
    return false;
  }
  if (file.size > MAX_FILE_BYTES) {
    showError(`File terlalu besar: ${formatBytes(file.size)}. Maksimal ${CONFIG.MAX_FILE_MB} MB.`);
    return false;
  }
  return true;
}

function applyFile(file) {
  if (!file) return;
  if (!validateFile(file)) return;

  state.selectedFile = file;
  state.lastFile     = file;
  hideError();

  const ext = getFileExt(file.name);
  dom.fileTypeIcon.innerHTML = FILE_ICONS[ext] || FILE_ICONS.txt;
  dom.fileName.textContent   = file.name;
  dom.fileMeta.textContent   = `${ext.toUpperCase()} · ${formatBytes(file.size)}`;

  dom.filePreview.hidden = false;
  updateDetectButton();
}

function clearFile() {
  state.selectedFile = null;
  dom.fileInput.value = '';
  dom.filePreview.hidden = true;
  hideError();
  updateDetectButton();
}

// ─────────────────────────────────────────────────────────────
// API CALLS
// ─────────────────────────────────────────────────────────────
function setLoading(on) {
  state.loading = on;
  dom.btnDetect.classList.toggle('is-loading', on);
  dom.btnDetect.disabled = on;
  dom.btnClear.disabled  = on;
}

async function runTextDetection() {
  const text = dom.textInput.value.trim();

  if (!text) { showError('Teks tidak boleh kosong.'); return; }
  if (text.length < CONFIG.MIN_CHARS) {
    showError(`Teks terlalu pendek. Minimal ${CONFIG.MIN_CHARS} karakter.`);
    return;
  }
  if (text.length > CONFIG.MAX_CHARS) {
    showError(`Teks terlalu panjang. Maksimal ${CONFIG.MAX_CHARS.toLocaleString('id-ID')} karakter.`);
    return;
  }

  state.lastText = text;
  setLoading(true);
  ResultPanel.showLoading(false);

  try {
    const res = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const detail = await _parseError(res);
      ResultPanel.showError(detail);
      return;
    }

    const data = await res.json();
    ResultPanel.showResult(data, null);

  } catch (err) {
    console.error('[AIDetect] Text predict error:', err);
    ResultPanel.showError(_networkErrorMsg());
  } finally {
    setLoading(false);
    updateDetectButton();
  }
}

async function runFileDetection() {
  const file = state.selectedFile;
  if (!file) { showError('Pilih file terlebih dahulu.'); return; }

  setLoading(true);
  ResultPanel.showLoading(true);

  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${CONFIG.API_BASE_URL}/predict-file`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const detail = await _parseError(res);
      ResultPanel.showError(detail);
      return;
    }

    const data = await res.json();
    // data = { prediction, confidence, probabilities, document: { filename, pages, words, characters, ocr_used } }
    ResultPanel.showResult(data, data.document || null);

  } catch (err) {
    console.error('[AIDetect] File predict error:', err);
    ResultPanel.showError(_networkErrorMsg());
  } finally {
    setLoading(false);
    updateDetectButton();
  }
}

async function _parseError(res) {
  try {
    const body = await res.json();
    if (body?.detail?.error) return body.detail.error;
    if (typeof body?.detail === 'string') return body.detail;
    if (typeof body?.error === 'string') return body.error;
  } catch (_) { /* ignore */ }
  if (res.status === 503) return 'Model belum siap. Jalankan: python3 scripts/build_preprocessor.py';
  if (res.status === 400) return 'Input tidak valid. Pastikan file mengandung teks yang bisa dibaca.';
  return `Server error (HTTP ${res.status}).`;
}

function _networkErrorMsg() {
  return `Tidak dapat terhubung ke server. Pastikan backend berjalan di ${CONFIG.API_BASE_URL}.`;
}

// ─────────────────────────────────────────────────────────────
// EVENT LISTENERS
// ─────────────────────────────────────────────────────────────

// ── Tab switch
dom.tabText.addEventListener('click', () => switchTab('text'));
dom.tabFile.addEventListener('click', () => switchTab('file'));

// ── Text panel
dom.textInput.addEventListener('input', onTextInput);

dom.btnClear.addEventListener('click', () => {
  dom.textInput.value = '';
  dom.textInput.focus();
  onTextInput();
  ResultPanel.showEmpty();
});

dom.btnPaste.addEventListener('click', async () => {
  try {
    const text = await navigator.clipboard.readText();
    dom.textInput.value = text;
    dom.textInput.focus();
    onTextInput();
  } catch {
    dom.textInput.focus();
    dom.textInput.select();
  }
});

// Ctrl+Enter shortcut
dom.textInput.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    if (!dom.btnDetect.disabled) runTextDetection();
  }
});

// ── File panel
dom.btnBrowse.addEventListener('click', () => dom.fileInput.click());

dom.dropZone.addEventListener('click', (e) => {
  // Don't trigger if clicking the Browse button (it has its own listener)
  if (e.target === dom.btnBrowse || dom.btnBrowse.contains(e.target)) return;
  dom.fileInput.click();
});

dom.dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    dom.fileInput.click();
  }
});

dom.fileInput.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) applyFile(file);
});

dom.btnFileRemove.addEventListener('click', (e) => {
  e.stopPropagation();
  clearFile();
  ResultPanel.showEmpty();
});

// Drag & Drop
['dragenter', 'dragover'].forEach(evt => {
  dom.dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dom.dropZone.classList.add('is-drag-over');
  });
});
['dragleave', 'drop'].forEach(evt => {
  dom.dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dom.dropZone.classList.remove('is-drag-over');
  });
});
dom.dropZone.addEventListener('drop', (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) applyFile(file);
});

// ── Detect button
dom.btnDetect.addEventListener('click', () => {
  if (state.mode === 'text') {
    runTextDetection();
  } else {
    runFileDetection();
  }
});

// ── Reset button
dom.btnReset.addEventListener('click', () => {
  dom.textInput.value = '';
  clearFile();
  hideError();
  ResultPanel.showEmpty();
  onTextInput();
});

// ── Retry button
dom.btnRetry.addEventListener('click', () => {
  if (state.mode === 'text' && state.lastText) {
    dom.textInput.value = state.lastText;
    onTextInput();
    runTextDetection();
  } else if (state.mode === 'file' && state.lastFile) {
    state.selectedFile = state.lastFile;
    runFileDetection();
  } else {
    ResultPanel.showEmpty();
  }
});

// ─────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────
(function init() {
  onTextInput();
  ResultPanel.showEmpty();

  // Silent health check
  fetch(`${CONFIG.API_BASE_URL}/health`, { signal: AbortSignal.timeout(3000) })
    .then(r => { if (r.ok) console.info('[AIDetect] Backend ready.'); })
    .catch(() => { console.warn(`[AIDetect] Backend offline at ${CONFIG.API_BASE_URL}`); });
})();
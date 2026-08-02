/**
 * Shadowspace Workbench — Sprint 7 Fashion-MNIST Belief Space & UX Polish
 * main.js — Multi-dataset switching, Projection Catalog view selection,
 * interactive zoom/pan camera navigation, local integrity diagnostics,
 * rich metadata inspection, saved-view atlas, and reproducible exports.
 */

"use strict";

// ─── State ────────────────────────────────────────────────────────────────────

let currentDataset = "calibration_3class"; // active dataset key
let fixtureData    = null;                 // JSON from /api/fixture?dataset=...
let currentRep     = "probability";        // active representation key
let currentMetric  = "euclidean";          // active metric key
let currentViewId  = "pca_corners";        // active catalog view ID
let selectedIdx    = 0;                    // selected point index (default 0)
let hoveredIdx     = null;                 // currently hovered point index
let currentK       = 3;                    // neighborhood size k
let diagResult     = null;                 // live diagnostic payload from /api/diagnostics
let savedViews     = [];                   // list of SavedView objects from /api/saved-views

// Zoom & Pan Camera State
let zoomLevel  = 1.0;
let panOffsetX = 0;
let panOffsetY = 0;
let isPanning  = false;
let startPanX  = 0;
let startPanY  = 0;

// Grand Tour Animation State
let tourFrames       = null;
let tourBases        = null;
let isTourPlaying    = false;
let tourFrameIdx     = 0;
let tourSpeed        = 1.0;
let tourAnimHandle   = null;
let lastTourTime     = 0;
let tourGlobalBounds = null; // fixed [-1.1, 1.1] x [-1.1, 1.1] since server normalises frames
let savedZoom        = 1.0;  // saved before tour, restored on pause
let savedPanX        = 0;
let savedPanY        = 0;

// Multi-Selection Marquee Box State
let selectedIndices  = [];     // array of selected point indices
let isBoxSelecting   = false;  // active Shift+Drag box selection
let boxStartX        = 0;
let boxStartY        = 0;
let boxCurX          = 0;
let boxCurY          = 0;

// Representation Morph State
let isMorphing       = false;  // true during representation morph animation

// Dual View Split-Screen State
let isDualView       = false;  // active split-screen comparison mode
let syncCameras      = true;   // synchronized zoom & pan across View A and View B
let canvasB          = null;   // DOM canvas element B
let ctxB             = null;   // 2D rendering context B
let viewCoordsB      = null;   // 2D coordinates array for View B
let zoomLevelB       = 1.0;
let panOffsetXB      = 0;
let panOffsetYB      = 0;

// Sprint 13 Geometric Analysis State
let showTopologyGraph = false; // global k-NN edge overlay
let showDistortionMap = false; // spatial distortion heatmap
let topologyEdges     = null;  // cached edge list from /api/topology
let distortionGrid    = null;  // cached grid from /api/distortion-grid
let subspaceAngles    = null;  // cached angles from /api/subspace-angles

// Sprint 14 Stability & Rashomon Atlas State
let showStabilityMap   = false; // point stability overlay
let pointStabilityData = null;  // cached payload from /api/point-stability
let rashomonSetData    = null;  // cached payload from /api/rashomon-set

const POINT_RADIUS        = 7;
const POINT_RADIUS_HOVER  = 10;
const POINT_RADIUS_SELECT = 10;
const AXIS_PADDING        = 48;
const GRID_LINES          = 6;

// ─── DOM refs ─────────────────────────────────────────────────────────────────

const canvas            = document.getElementById("scatter-canvas");
const ctx               = canvas.getContext("2d");
const tooltip           = document.getElementById("canvas-tooltip");
const statusDot         = document.querySelector(".status-dot");
const statusLabel       = document.querySelector(".status-label");
const datasetSelect     = document.getElementById("dataset-select");
const catalogSelect     = document.getElementById("catalog-select");
const metricSelect      = document.getElementById("metric-select");
const kSlider            = document.getElementById("k-slider");
const kValLabel          = document.getElementById("k-val-label");
const toggleReducedMotion = document.getElementById("toggle-reduced-motion");
const btnSaveView       = document.getElementById("btn-save-view");
const atlasViewName     = document.getElementById("atlas-view-name");
const atlasViewNote     = document.getElementById("atlas-view-note");
const atlasList         = document.getElementById("atlas-list");

// Zoom controls
const btnZoomIn    = document.getElementById("btn-zoom-in");
const btnZoomOut   = document.getElementById("btn-zoom-out");
const btnZoomFocus = document.getElementById("btn-zoom-focus");
const btnZoomReset = document.getElementById("btn-zoom-reset");

// ─── Boot ─────────────────────────────────────────────────────────────────────

async function boot() {
  await loadAvailableDatasets();
  initDatasetSelector();
  initCatalogSelector();
  initRepSelector();
  initMetricSelector();
  initKSlider();
  initAtlasControls();
  initZoomControls();
  initHelpModal();
  initImportModal();
  initReportExport();
  initStressHeatmapToggle();
  initTourControls();
  initDualViewControls();
  initSprint13Controls();
  initOptimizerControls();
  initSprint15UIControls();
  initAccessibility();
  initKeyboardShortcuts();
  initCanvasInteraction();
  await loadDataset(currentDataset);
  await fetchSavedViews();
}

// ─── Dataset Loader ───────────────────────────────────────────────────────────

async function loadDataset(datasetKey) {
  try {
    setStatus("loading", "Loading dataset…");
    // Invalidate Sprint 13 overlay caches — they are dataset/view specific
    topologyEdges  = null;
    distortionGrid = null;
    subspaceAngles = null;
    updateSubspaceAnglePanel();

    const res = await fetch(`/api/fixture?dataset=${encodeURIComponent(datasetKey)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    fixtureData = await res.json();
    currentDataset = datasetKey;

    const nPts = fixtureData.object_ids.length;
    const nFeat = fixtureData.feature_names ? fixtureData.feature_names.length : 3;

    document.getElementById("scatter-main-title").textContent = `PCA Tour — ${fixtureData.display_name}`;
    document.getElementById("scatter-subtitle").textContent  = `${nPts} objects · ${nFeat} features`;

    updateCatalogSelector();

    kSlider.max = Math.max(1, nPts - 1);
    if (currentK >= nPts) currentK = Math.min(3, nPts - 1);
    kSlider.value = currentK;
    kValLabel.textContent = currentK;

    resetZoom();

    setStatus("ready", `${fixtureData.display_name} loaded`);
    updateVarianceBars();
    updateSemanticBadge();
    updateLegendList();
    selectPoint(0);
    await loadTourPath();
  } catch (err) {
    setStatus("error", "Failed to load dataset");
    console.error("Dataset load error:", err);
  }
}

function updateCatalogSelector() {
  if (!catalogSelect || !fixtureData) return;
  const cat = fixtureData.catalog || {};
  const viewKeys = Object.keys(cat);
  if (viewKeys.length > 0) {
    catalogSelect.innerHTML = viewKeys.map(vk =>
      `<option value="${escapeHtml(vk)}">${escapeHtml(cat[vk].display_name || vk)}</option>`
    ).join("");
    if (cat[currentViewId]) {
      catalogSelect.value = currentViewId;
    } else {
      currentViewId = viewKeys[0];
      catalogSelect.value = currentViewId;
    }
  } else {
    catalogSelect.innerHTML = `<option value="pca_corners">PCA Corner View</option>`;
    currentViewId = "pca_corners";
  }
}

function initDatasetSelector() {
  if (datasetSelect) {
    datasetSelect.addEventListener("change", (e) => {
      loadDataset(e.target.value);
    });
  }
}

function initCatalogSelector() {
  if (catalogSelect) {
    catalogSelect.addEventListener("change", (e) => {
      const newViewId = e.target.value;
      // If switching away from a Rashomon injected view, remove injected options
      [...catalogSelect.options]
        .filter(o => o.dataset.rashomon && o.value !== newViewId)
        .forEach(o => o.remove());
      currentViewId = newViewId;
      // Invalidate view-specific overlay caches
      topologyEdges  = null;
      distortionGrid = null;
      updateSemanticBadge();
      fetchDiagnostics();
      if (showTopologyGraph) loadTopologyEdges();
      if (showDistortionMap) loadDistortionGrid();
      if (isDualView) loadSubspaceAngles();
      renderScatter();
    });
  }
}

// ─── Status bar ───────────────────────────────────────────────────────────────

function setStatus(state, label) {
  statusDot.className   = "status-dot " + state;
  statusLabel.textContent = label;
}

// ─── Representation selector ──────────────────────────────────────────────────

const REP_METRICS = {
  probability: [
    { id: "euclidean", label: "Euclidean" },
    { id: "hellinger", label: "Hellinger" },
    { id: "fisher_rao", label: "Fisher-Rao" },
    { id: "aitchison", label: "Aitchison" },
  ],
  sqrt_probability: [
    { id: "euclidean", label: "Euclidean" },
    { id: "fisher_rao", label: "Fisher-Rao" },
  ],
  clr_probability: [
    { id: "aitchison", label: "Aitchison" },
    { id: "euclidean", label: "Euclidean" },
  ],
};

function updateMetricOptions() {
  const allowed = REP_METRICS[currentRep] || REP_METRICS.probability;
  metricSelect.innerHTML = allowed
    .map(m => `<option value="${m.id}">${m.label}</option>`)
    .join("");

  if (allowed.some(m => m.id === currentMetric)) {
    metricSelect.value = currentMetric;
  } else {
    currentMetric = allowed[0].id;
    metricSelect.value = currentMetric;
  }
}

function initRepSelector() {
  document.querySelectorAll('input[name="representation"]').forEach(radio => {
    radio.addEventListener("change", (e) => {
      setRepresentation(e.target.value);
    });
  });
  updateMetricOptions();
}

async function setRepresentation(repId) {
  const oldRep = currentRep;
  currentRep = repId;
  document.querySelectorAll(".radio-card").forEach(card => {
    const radio = card.querySelector("input");
    if (radio) {
      radio.checked = radio.value === currentRep;
      card.classList.toggle("active", radio.value === currentRep);
    }
  });
  hoveredIdx = null;
  tourFrameIdx = 0;
  if (isTourPlaying) {
    pauseTour();
  }
  updateMetricOptions();
  updateVarianceBars();
  updateSemanticBadge();
  renderScatter();

  await loadTourPath();
  if (isDualView) {
    await loadDualViewBData();
  }
  if (showTopologyGraph) loadTopologyEdges();
  if (showDistortionMap) loadDistortionGrid();
  await fetchDiagnostics();
  renderScatter();
}

// ─── Metric & k selectors ─────────────────────────────────────────────────────

function initMetricSelector() {
  metricSelect.addEventListener("change", async (e) => {
    currentMetric = e.target.value;
    if (showDistortionMap) loadDistortionGrid();
    await fetchDiagnostics();
    renderScatter();
  });
}

function initKSlider() {
  kSlider.addEventListener("input", (e) => {
    currentK = parseInt(e.target.value, 10);
    kValLabel.textContent = currentK;
    fetchDiagnostics();
  });
}

// ─── Interactive Zoom & Pan Camera Controls ───────────────────────────────────

function resetZoom() {
  zoomLevel  = 1.0;
  panOffsetX = 0;
  panOffsetY = 0;
  renderScatter();
}

function zoomBy(factor, pivotScreenX, pivotScreenY) {
  if (!fixtureData) return;
  const coords = getCoords();
  if (!coords || coords.length === 0) return;

  const oldZoom = zoomLevel;
  const newZoom = Math.max(0.8, Math.min(25.0, oldZoom * factor));
  if (newZoom === oldZoom) return;

  const xs = coords.map(c => c[0]);
  const ys = coords.map(c => c[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const pad = AXIS_PADDING;
  const w = canvas.width - pad * 2;
  const h = canvas.height - pad * 2;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;

  const baseScale = Math.min(w / rangeX, h / rangeY) * 0.82;
  const oldScale  = baseScale * oldZoom;

  let px, py, dataX, dataY;

  if (pivotScreenX !== undefined && pivotScreenY !== undefined) {
    // Pivot at specified screen coordinate (e.g. mouse position)
    px = pivotScreenX;
    py = pivotScreenY;
    dataX = (px - (pad + w / 2) - panOffsetX) / oldScale + midX;
    dataY = midY - (py - (pad + h / 2) - panOffsetY) / oldScale;
  } else if (selectedIdx !== null && selectedIdx < coords.length) {
    // Pivot at selected node's data position
    dataX = coords[selectedIdx][0];
    dataY = coords[selectedIdx][1];
    px = (pad + w / 2) + (dataX - midX) * oldScale + panOffsetX;
    py = (pad + h / 2) - (dataY - midY) * oldScale + panOffsetY;
  } else {
    // Pivot at screen center
    px = pad + w / 2;
    py = pad + h / 2;
    dataX = midX;
    dataY = midY;
  }

  zoomLevel = newZoom;
  const newScale = baseScale * newZoom;

  panOffsetX = px - (pad + w / 2) - (dataX - midX) * newScale;
  panOffsetY = py - (pad + h / 2) + (dataY - midY) * newScale;

  renderScatter();
}

function focusNeighborhood() {
  if (!fixtureData || selectedIdx === null) return;
  const coords = getCoords();
  const targetId = fixtureData.object_ids[selectedIdx];
  const idToIdx = new Map(fixtureData.object_ids.map((id, idx) => [id, idx]));

  const activeIds = [targetId];
  if (diagResult) {
    activeIds.push(...diagResult.preserved, ...diagResult.torn, ...diagResult.false_neighbors);
  }

  const activeCoords = activeIds
    .map(id => idToIdx.get(id))
    .filter(idx => idx !== undefined && idx < coords.length)
    .map(idx => coords[idx]);

  if (activeCoords.length === 0) return;

  const xs = activeCoords.map(c => c[0]);
  const ys = activeCoords.map(c => c[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const dx = Math.max(0.08, maxX - minX);
  const dy = Math.max(0.08, maxY - minY);

  const allXs = coords.map(c => c[0]), allYs = coords.map(c => c[1]);
  const fullRangeX = Math.max(...allXs) - Math.min(...allXs) || 1;
  const fullRangeY = Math.max(...allYs) - Math.min(...allYs) || 1;
  const fullCx = (Math.min(...allXs) + Math.max(...allXs)) / 2;
  const fullCy = (Math.min(...allYs) + Math.max(...allYs)) / 2;

  const targetZoom = Math.min(15.0, Math.max(1.5, 0.7 / Math.max(dx / fullRangeX, dy / fullRangeY)));
  zoomLevel = targetZoom;

  const baseScale = Math.min(canvas.width / fullRangeX, canvas.height / fullRangeY) * 0.82;
  const scale = baseScale * zoomLevel;
  panOffsetX = (cx - fullCx) * scale;
  panOffsetY = -(cy - fullCy) * scale;

  renderScatter();
}

function initZoomControls() {
  if (btnZoomIn)    btnZoomIn.addEventListener("click", () => zoomBy(1.3));
  if (btnZoomOut)   btnZoomOut.addEventListener("click", () => zoomBy(1 / 1.3));
  if (btnZoomFocus) btnZoomFocus.addEventListener("click", () => focusNeighborhood());
  if (btnZoomReset) btnZoomReset.addEventListener("click", () => resetZoom());

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;

    if (selectedIdx !== null) {
      zoomBy(factor);
    } else {
      zoomBy(factor, mouseX, mouseY);
    }
  }, { passive: false });
}

function initHelpModal() {
  const modal       = document.getElementById("help-modal");
  const btnOpen     = document.getElementById("btn-open-help");
  const btnClose    = document.getElementById("btn-close-help");
  const btnDismiss  = document.getElementById("btn-dismiss-help");

  if (!modal) return;

  const openModal  = () => modal.classList.remove("hidden");
  const closeModal = () => modal.classList.add("hidden");

  if (btnOpen)    btnOpen.addEventListener("click", openModal);
  if (btnClose)   btnClose.addEventListener("click", closeModal);
  if (btnDismiss) btnDismiss.addEventListener("click", closeModal);

  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) {
      closeModal();
    }
  });
}

async function loadAvailableDatasets() {
  if (!datasetSelect) return;
  try {
    const res = await fetch("/api/datasets");
    if (!res.ok) return;
    const datasets = await res.json();
    const badgeMap = {
      synthetic: "⬡ ",
      generated: "✦ ",
      imported: "↑ ",
      fetched: "📦 ",
    };
    datasetSelect.innerHTML = datasets.map(ds => {
      const badge = badgeMap[ds.source_type] || "";
      return `<option value="${escapeHtml(ds.key)}">${badge}${escapeHtml(ds.display_name)}</option>`;
    }).join("");
    datasetSelect.value = currentDataset;
  } catch (err) {
    console.error("Failed to load dataset list:", err);
  }
}

function initImportModal() {
  const modal       = document.getElementById("import-modal");
  const btnOpen     = document.getElementById("btn-open-import");
  const btnClose    = document.getElementById("btn-close-import");
  const btnCancel   = document.getElementById("btn-cancel-import");
  const btnSubmit   = document.getElementById("btn-submit-import");
  const form        = document.getElementById("import-form");
  const errorMsg    = document.getElementById("import-error-msg");

  if (!modal) return;

  const openModal  = () => {
    errorMsg.classList.add("hidden");
    errorMsg.textContent = "";
    modal.classList.remove("hidden");
  };
  const closeModal = () => modal.classList.add("hidden");

  if (btnOpen)   btnOpen.addEventListener("click", openModal);
  if (btnClose)  btnClose.addEventListener("click", closeModal);
  if (btnCancel) btnCancel.addEventListener("click", closeModal);

  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) {
      closeModal();
    }
  });

  if (btnSubmit) {
    btnSubmit.addEventListener("click", async () => {
      const fileInput = document.getElementById("import-file");
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        errorMsg.textContent = "Please select a CSV or Parquet file to upload.";
        errorMsg.classList.remove("hidden");
        return;
      }

      const formData = new FormData(form);
      btnSubmit.disabled = true;
      btnSubmit.textContent = "Uploading & Processing…";
      errorMsg.classList.add("hidden");

      try {
        const res = await fetch("/api/import-dataset", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || `HTTP ${res.status}`);
        }

        closeModal();
        form.reset();
        await loadAvailableDatasets();
        datasetSelect.value = data.dataset_key;
        await loadDataset(data.dataset_key);

      } catch (err) {
        errorMsg.textContent = err.message || "Failed to import dataset.";
        errorMsg.classList.remove("hidden");
      } finally {
        btnSubmit.disabled = false;
        btnSubmit.textContent = "Import & Launch";
      }
    });
  }
}

// ─── Accessibility & Keyboard ─────────────────────────────────────────────────

function initAccessibility() {
  if (toggleReducedMotion) {
    toggleReducedMotion.addEventListener("change", (e) => {
      document.body.classList.toggle("reduced-motion", e.target.checked);
    });
  }
}

function initKeyboardShortcuts() {
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") {
      return;
    }
    if (!fixtureData) return;

    if (e.key === "ArrowRight") {
      e.preventDefault();
      selectPoint((selectedIdx + 1) % fixtureData.object_ids.length);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      const n = fixtureData.object_ids.length;
      selectPoint((selectedIdx - 1 + n) % n);
    } else if (e.key === "1") {
      setRepresentation("probability");
    } else if (e.key === "2") {
      setRepresentation("sqrt_probability");
    } else if (e.key === "+" || e.key === "=") {
      zoomBy(1.2);
    } else if (e.key === "-") {
      zoomBy(1 / 1.2);
    } else if (e.key === "0") {
      resetZoom();
    } else if (e.key.toLowerCase() === "f") {
      focusNeighborhood();
    }
  });
}

// ─── Variance bars & Legend list ──────────────────────────────────────────────

function updateVarianceBars() {
  if (!fixtureData) return;
  const rep   = fixtureData.representations[currentRep];
  const evs   = rep ? rep.eigenvalues : [0.5, 0.5];
  const total = evs[0] + evs[1];
  const pct1  = total > 0 ? (evs[0] / total * 100) : 50;
  const pct2  = total > 0 ? (evs[1] / total * 100) : 50;

  document.getElementById("var-pc1").style.width = pct1.toFixed(1) + "%";
  document.getElementById("var-pc2").style.width = pct2.toFixed(1) + "%";
  document.getElementById("var-pc1-label").textContent = pct1.toFixed(0) + "%";
  document.getElementById("var-pc2-label").textContent = pct2.toFixed(0) + "%";
}

function updateLegendList() {
  const legendStrip = document.getElementById("canvas-legend-strip");
  if (!legendStrip || !fixtureData) return;

  const palette = [
    "#6ee7b7", "#93c5fd", "#c4b5fd", "#f472b6", "#fbbf24",
    "#a7f3d0", "#f87171", "#60a5fa", "#e879f9", "#38bdf8"
  ];

  if (currentDataset === "synthetic_4class" || currentDataset === "calibration_3class") {
    const isSynth = currentDataset === "synthetic_4class";
    legendStrip.innerHTML = `
      <div class="canvas-legend-item"><span class="canvas-legend-dot" style="background:#6ee7b7"></span>Corner cluster</div>
      <div class="canvas-legend-item"><span class="canvas-legend-dot" style="background:#93c5fd"></span>Midpoint</div>
      <div class="canvas-legend-item"><span class="canvas-legend-dot" style="background:#fbbf24"></span>Uniform center</div>
      <div class="canvas-legend-item"><span class="canvas-legend-dot" style="background:${isSynth ? '#f472b6' : '#c4b5fd'}"></span>${isSynth ? 'Outlier' : 'Interior'}</div>
    `;
    return;
  }

  const featNames = fixtureData.feature_names || [];
  legendStrip.innerHTML = featNames
    .map((name, i) => {
      const cleanName = name.replace(/^p_/, '');
      const color = palette[i % palette.length];
      return `<div class="canvas-legend-item"><span class="canvas-legend-dot" style="background:${color}"></span>${escapeHtml(cleanName)}</div>`;
    })
    .join("");
}

function getPointMeta(idx) {
  if (!fixtureData || idx === null || idx < 0 || idx >= fixtureData.object_ids.length) return null;

  const id = fixtureData.object_ids[idx];
  const raw = fixtureData.raw_matrix[idx];
  const featNames = fixtureData.feature_names || [];

  let maxVal = -1;
  let maxIdx = 0;
  if (raw && raw.length > 0) {
    for (let i = 0; i < raw.length; i++) {
      if (raw[i] > maxVal) {
        maxVal = raw[i];
        maxIdx = i;
      }
    }
  }

  const rawPredName = featNames[maxIdx] ? featNames[maxIdx].replace(/^p_/, '') : `class_${maxIdx}`;
  const confidence = maxVal >= 0 ? maxVal : 0;
  const meta = fixtureData.objects_meta ? fixtureData.objects_meta[idx] : null;

  const predClassName = (meta && (meta.pred_class_name || meta.predicted_label)) || rawPredName;
  const trueClassName = (meta && (meta.true_class_name || meta.true_label)) || null;
  const isCorrect = (meta && meta.is_correct !== undefined)
    ? meta.is_correct
    : ((meta && meta.correct !== undefined) ? meta.correct : (trueClassName ? String(trueClassName).replace(/^p_/, '') === String(predClassName).replace(/^p_/, '') : true));
  const entropy = (meta && meta.entropy !== undefined) ? meta.entropy : null;

  return {
    id,
    predClassName: String(predClassName).replace(/^p_/, ''),
    trueClassName: trueClassName ? String(trueClassName).replace(/^p_/, '') : null,
    confidence,
    isCorrect,
    entropy,
    maxClassIdx: maxIdx,
    raw,
  };
}

// ─── Semantic Badge Panel ─────────────────────────────────────────────────────

function updateSemanticBadge() {
  const badgeTag    = document.getElementById("semantic-kind-tag");
  const badgeDesc   = document.getElementById("semantic-desc");
  const badgeStatus = document.getElementById("semantic-status-badge");

  if (isMorphing) {
    badgeTag.textContent = "representation_morph";
    badgeStatus.textContent = "Intermediate Morph";
    badgeStatus.className = "panel-badge badge-morph";
    badgeDesc.textContent = "Procrustes-aligned transition between geometric representations. Intermediate frames are semantically invalid projections.";
    return;
  }

  const catEntry = fixtureData && fixtureData.catalog ? fixtureData.catalog[currentViewId] : null;

  if (catEntry && catEntry.is_misleading) {
    badgeTag.textContent = catEntry.view_id;
    badgeStatus.textContent = "Misleading View Warning";
    badgeStatus.className = "panel-badge badge-placeholder";
    badgeStatus.style.borderColor = "#f87171";
    badgeStatus.style.color = "#f87171";
    badgeDesc.textContent = catEntry.warning_note || catEntry.description;
  } else if (currentRep === "probability") {
    badgeTag.textContent = currentViewId;
    badgeStatus.textContent = "Valid Projection";
    badgeStatus.className = "panel-badge badge-active";
    badgeStatus.style.borderColor = "";
    badgeStatus.style.color = "";
    badgeDesc.textContent = catEntry ? catEntry.description : "Orthogonal 2D linear projection Y = X F. Intermediate frames preserve exact linear geometry.";
  } else {
    badgeTag.textContent = `${currentViewId} (Fisher-Rao)`;
    badgeStatus.textContent = "Valid Projection";
    badgeStatus.className = "panel-badge badge-active";
    badgeStatus.style.borderColor = "";
    badgeStatus.style.color = "";
    badgeDesc.textContent = "Orthogonal 2D projection on square-root probability coordinates. Preserves Fisher-Rao Riemannian geometry.";
  }
}

// ─── Diagnostics API Fetch ────────────────────────────────────────────────────

async function fetchDiagnostics() {
  if (!fixtureData || selectedIdx === null) return;
  const targetId = fixtureData.object_ids[selectedIdx];
  const url = `/api/diagnostics?dataset=${encodeURIComponent(currentDataset)}&target_id=${encodeURIComponent(targetId)}&representation=${encodeURIComponent(currentRep)}&metric=${encodeURIComponent(currentMetric)}&view_id=${encodeURIComponent(currentViewId)}&k=${currentK}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    diagResult = await res.json();
    updateIntegrityPanel(diagResult);
    renderScatter();
  } catch (err) {
    console.error("Diagnostics fetch error:", err);
    diagResult = null;
    renderScatter();
  }
}

async function loadTopologyEdges() {
  if (!fixtureData) return;
  const url = `/api/topology?dataset=${encodeURIComponent(currentDataset)}&representation=${encodeURIComponent(currentRep)}&metric=${encodeURIComponent(currentMetric)}&view_id=${encodeURIComponent(currentViewId)}&k=${currentK}`;
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();
    topologyEdges = data.edges;
    renderScatter();
  } catch (err) {
    console.error("Failed to load topology edges:", err);
  }
}

async function loadDistortionGrid() {
  if (!fixtureData) return;
  const url = `/api/distortion-grid?dataset=${encodeURIComponent(currentDataset)}&representation=${encodeURIComponent(currentRep)}&metric=${encodeURIComponent(currentMetric)}&view_id=${encodeURIComponent(currentViewId)}&resolution=32`;
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    distortionGrid = await res.json();
    renderScatter();
  } catch (err) {
    console.error("Failed to load distortion grid:", err);
  }
}

async function loadSubspaceAngles() {
  if (!fixtureData || !isDualView) return;
  const viewBName = "fisher_lda";
  const url = `/api/subspace-angles?dataset=${encodeURIComponent(currentDataset)}&representation=${encodeURIComponent(currentRep)}&view_a=${encodeURIComponent(currentViewId)}&view_b=${encodeURIComponent(viewBName)}`;
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    subspaceAngles = await res.json();
    updateSubspaceAnglePanel();
  } catch (err) {
    console.error("Failed to load subspace angles:", err);
  }
}

function updateSubspaceAnglePanel() {
  const panel = document.getElementById("subspace-angle-panel");
  if (!panel) return;
  if (!isDualView || !subspaceAngles) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");

  const elT1 = document.getElementById("angle-theta1");
  const elT2 = document.getElementById("angle-theta2");
  if (elT1) elT1.textContent = `${subspaceAngles.theta_1_deg.toFixed(1)}°`;
  if (elT2) elT2.textContent = `${subspaceAngles.theta_2_deg.toFixed(1)}°`;

  const gElem = document.getElementById("angle-grassmannian");
  if (gElem) {
    gElem.textContent = `${subspaceAngles.grassmannian_dist_deg.toFixed(1)}° (${subspaceAngles.interpretation})`;
    gElem.className = `angle-val-badge ${subspaceAngles.interpretation}`;
  }
}

// ─── Sprint 14 Stability & Rashomon Atlas ──────────────────────────────────────

async function loadPointStability() {
  if (!fixtureData) return;
  const url = `/api/point-stability?dataset=${encodeURIComponent(currentDataset)}&representation=${encodeURIComponent(currentRep)}&metric=${encodeURIComponent(currentMetric)}&k=${currentK}`;
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    pointStabilityData = await res.json();
    updateStabilityPanel();
    renderScatter();
  } catch (err) {
    console.error("Failed to load point stability:", err);
  }
}

async function loadRashomonSet() {
  if (!fixtureData) return;
  const url = `/api/rashomon-set?dataset=${encodeURIComponent(currentDataset)}&representation=${encodeURIComponent(currentRep)}&view_id=${encodeURIComponent(currentViewId)}&threshold=0.30`;
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    rashomonSetData = await res.json();
    updateStabilityPanel();
  } catch (err) {
    console.error("Failed to load Rashomon set:", err);
  }
}

function updateStabilityPanel() {
  const panel = document.getElementById("stability-panel");
  if (!panel) return;
  if (!showStabilityMap) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");

  if (pointStabilityData) {
    const pIdxLabel = document.getElementById("stability-index-label");
    const gaugeBar  = document.getElementById("stability-gauge-bar");
    const pct       = Math.round((pointStabilityData.persistence_index || 0) * 100);
    if (pIdxLabel) pIdxLabel.textContent = `${pct}% Persistent`;
    if (gaugeBar)  gaugeBar.style.width   = `${pct}%`;
  }

  updateSelectedPointStabilityCard();
  updateRashomonListUI();
}

function updateSelectedPointStabilityCard() {
  const card = document.getElementById("selected-stability-card");
  if (!card) return;
  if (!showStabilityMap || !pointStabilityData || !pointStabilityData.stability_scores) {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");

  const score = pointStabilityData.stability_scores[selectedIdx] ?? 1.0;
  const tag = document.getElementById("selected-point-stab-tag");
  const desc = document.getElementById("selected-point-stab-desc");

  if (score >= 0.65) {
    if (tag) {
      tag.textContent = "Perspective-Invariant";
      tag.className = "card-status-tag";
    }
    if (desc) desc.textContent = `High stability (${Math.round(score * 100)}% overlap). Neighborhood persists across candidate 2D views.`;
  } else if (score >= 0.35) {
    if (tag) {
      tag.textContent = "Moderate Sensitivity";
      tag.className = "card-status-tag";
      tag.style.background = "rgba(251, 191, 36, 0.2)";
      tag.style.color = "#fbbf24";
    }
    if (desc) desc.textContent = `Moderate stability (${Math.round(score * 100)}% overlap). Neighborhood varies across some projection planes.`;
  } else {
    if (tag) {
      tag.textContent = "Perspective-Sensitive";
      tag.className = "card-status-tag volatile";
    }
    if (desc) desc.textContent = `Low stability (${Math.round(score * 100)}% overlap). 2D placement is an orthographic projection artifact. True high-D distance is preserved.`;
  }
}

function updateRashomonListUI() {
  const list = document.getElementById("rashomon-list");
  if (!list) return;
  if (!rashomonSetData || !rashomonSetData.candidates || rashomonSetData.candidates.length === 0) {
    list.innerHTML = `<div class="rashomon-empty">No candidates met threshold.</div>`;
    return;
  }

  list.innerHTML = rashomonSetData.candidates.map(cand => `
    <div class="rashomon-card">
      <div class="rashomon-card-header">
        <span class="rashomon-name">${escapeHtml(cand.display_name)}</span>
        <span class="rashomon-t-score">T: ${(cand.trustworthiness * 100).toFixed(1)}%</span>
      </div>
      <div class="rashomon-card-meta">
        <span>Dist: ${cand.grassmannian_dist_deg}°</span>
      </div>
      <div class="rashomon-actions">
        <button class="btn-rashomon-action" onclick="jumpToRashomonCandidate('${escapeHtml(cand.id)}')">▶ Jump to View</button>
        <button class="btn-rashomon-action" onclick="compareRashomonCandidate('${escapeHtml(cand.id)}')">🗖 Dual View</button>
      </div>
    </div>
  `).join("");
}

function updateIntegrityPanel(data) {
  if (!data) return;
  document.getElementById("diag-precision").textContent = (data.precision * 100).toFixed(0) + "%";
  document.getElementById("diag-recall").textContent    = (data.recall * 100).toFixed(0) + "%";
  
  const trustElem = document.getElementById("diag-trust");
  const trustGauge = document.getElementById("trust-gauge-bar");
  const trustPct  = (data.trustworthiness * 100).toFixed(0);
  trustElem.textContent = `${trustPct}%`;
  if (trustGauge) trustGauge.style.width = `${trustPct}%`;

  if (data.trustworthiness >= 0.90) {
    trustElem.style.color = "#6ee7b7";
    if (trustGauge) trustGauge.style.backgroundColor = "#34d399";
  } else if (data.trustworthiness < 0.80) {
    trustElem.style.color = "#f87171";
    if (trustGauge) trustGauge.style.backgroundColor = "#f43f5e";
  } else {
    trustElem.style.color = "#fbbf24";
    if (trustGauge) trustGauge.style.backgroundColor = "#fbbf24";
  }

  document.getElementById("diag-stress").textContent = data.stress.toFixed(2);

  const nPreserved = data.preserved.length;
  const nTorn      = data.torn.length;
  const nFalse     = data.false_neighbors.length;
  const totalBreakdown = nPreserved + nTorn + nFalse || 1;

  document.getElementById("count-preserved").textContent = nPreserved;
  document.getElementById("count-torn").textContent      = nTorn ? `${nTorn} (${data.torn.join(", ")})` : "0";
  document.getElementById("count-false").textContent     = nFalse ? `${nFalse} (${data.false_neighbors.join(", ")})` : "0";

  const barPreserved = document.getElementById("bar-seg-preserved");
  const barTorn      = document.getElementById("bar-seg-torn");
  const barFalse     = document.getElementById("bar-seg-false");

  if (barPreserved) barPreserved.style.width = `${(nPreserved / totalBreakdown) * 100}%`;
  if (barTorn)      barTorn.style.width      = `${(nTorn / totalBreakdown) * 100}%`;
  if (barFalse)     barFalse.style.width     = `${(nFalse / totalBreakdown) * 100}%`;
}

function initSprint15UIControls() {
  const btnTabSetup = document.getElementById("tab-btn-setup");
  const btnTabAnalysis = document.getElementById("tab-btn-analysis");
  const tabContentSetup = document.getElementById("sidebar-tab-setup");
  const tabContentAnalysis = document.getElementById("sidebar-tab-analysis");

  if (btnTabSetup && btnTabAnalysis) {
    btnTabSetup.addEventListener("click", () => {
      btnTabSetup.classList.add("active");
      btnTabAnalysis.classList.remove("active");
      if (tabContentSetup) tabContentSetup.classList.remove("hidden");
      if (tabContentAnalysis) tabContentAnalysis.classList.add("hidden");
    });
    btnTabAnalysis.addEventListener("click", () => {
      btnTabAnalysis.classList.add("active");
      btnTabSetup.classList.remove("active");
      if (tabContentAnalysis) tabContentAnalysis.classList.remove("hidden");
      if (tabContentSetup) tabContentSetup.classList.add("hidden");
    });
  }



  document.querySelectorAll(".panel.collapsible .panel-header").forEach((header) => {
    header.addEventListener("click", () => {
      const panel = header.closest(".panel");
      if (panel) {
        panel.classList.toggle("collapsed");
        const icon = panel.querySelector(".panel-toggle-icon");
        if (icon) {
          icon.textContent = panel.classList.contains("collapsed") ? "▶" : "▼";
        }
      }
    });
  });
}

function toggleSplitView(forceState) {
  if (typeof forceState === "boolean") {
    isDualView = forceState;
  } else {
    isDualView = !isDualView;
  }
  const btnSplit = document.getElementById("btn-split-view");
  const wrap = document.getElementById("canvas-wrap");
  const containerB = document.getElementById("canvas-container-b");

  if (isDualView) {
    if (btnSplit) { btnSplit.classList.add("active"); btnSplit.textContent = "🗖 Single View"; }
    if (wrap) wrap.classList.add("dual-view");
    if (containerB) containerB.classList.remove("hidden");
    loadDualViewBData();
  } else {
    if (btnSplit) { btnSplit.classList.remove("active"); btnSplit.textContent = "🗖 Split View"; }
    if (wrap) wrap.classList.remove("dual-view");
    if (containerB) containerB.classList.add("hidden");
  }
  updateSubspaceAnglePanel();
  renderScatter();
}

function _projectBasis(basis) {
  // Project the current representation matrix onto a (p, 2) basis,
  // returning normalised 2D coordinates [[x,y], ...]
  if (!fixtureData) return null;
  const rawMat = fixtureData.raw_matrix;
  if (!rawMat || !basis || !basis.length) return null;
  const projected = rawMat.map(row =>
    [row.reduce((s, v, j) => s + v * basis[j][0], 0),
     row.reduce((s, v, j) => s + v * basis[j][1], 0)]
  );
  const xs = projected.map(c => c[0]), ys = projected.map(c => c[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rx = maxX - minX || 1, ry = maxY - minY || 1;
  return projected.map(([x, y]) => [
    ((x - minX) / rx) * 2 - 1,
    ((y - minY) / ry) * 2 - 1
  ]);
}

function jumpToRashomonCandidate(candId) {
  const cand = (rashomonSetData?.candidates || []).find(c => c.id === candId);
  if (!cand) return;

  // Stop any running Grand Tour first
  if (isTourPlaying) pauseTour();

  // If this view exists in the catalog, switch to it normally
  if (fixtureData?.catalog && fixtureData.catalog[candId]) {
    currentViewId = candId;
    if (catalogSelect) catalogSelect.value = candId;
    topologyEdges  = null;
    distortionGrid = null;
    updateSemanticBadge();
    fetchDiagnostics();
    if (showTopologyGraph) loadTopologyEdges();
    if (showDistortionMap) loadDistortionGrid();
    if (showStabilityMap)  loadRashomonSet();
    renderScatter();
    return;
  }

  // For Haar / custom bases: directly project dataset onto basis and display
  if (cand.basis) {
    const coords = _projectBasis(cand.basis);
    if (coords) {
      // Temporarily inject into catalog so getCoords() picks it up
      if (!fixtureData.catalog) fixtureData.catalog = {};
      fixtureData.catalog[candId] = { coords, description: cand.display_name };
      if (!fixtureData.representations[currentRep]) {
        fixtureData.representations[currentRep] = {};
      }
      currentViewId = candId;
      if (catalogSelect) {
        const opt = document.createElement("option");
        opt.value = candId;
        opt.textContent = cand.display_name;
        opt.dataset.rashomon = "true";
        // Remove stale Rashomon options first
        [...catalogSelect.options].filter(o => o.dataset.rashomon).forEach(o => o.remove());
        catalogSelect.appendChild(opt);
        catalogSelect.value = candId;
      }
      topologyEdges = null;
      distortionGrid = null;
      updateSemanticBadge();
      fetchDiagnostics();
      renderScatter();
    }
  }
}

function compareRashomonCandidate(candId) {
  const cand = (rashomonSetData?.candidates || []).find(c => c.id === candId);
  if (!cand) return;

  if (!isDualView) {
    toggleSplitView(true);
  }

  if (fixtureData) {
    if (fixtureData.catalog && fixtureData.catalog[candId]) {
      viewCoordsB = fixtureData.catalog[candId].coords;
    } else if (cand.basis && fixtureData.raw_matrix) {
      const rawMat = fixtureData.raw_matrix;
      const basis = cand.basis;
      const projected = rawMat.map(row => [
        row.reduce((sum, val, j) => sum + val * basis[j][0], 0),
        row.reduce((sum, val, j) => sum + val * basis[j][1], 0)
      ]);
      const xs = projected.map(c => c[0]), ys = projected.map(c => c[1]);
      const minX = Math.min(...xs), maxX = Math.max(...xs) || 1;
      const minY = Math.min(...ys), maxY = Math.max(...ys) || 1;
      const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
      viewCoordsB = projected.map(([x, y]) => [
        ((x - minX) / rangeX) * 2 - 1,
        ((y - minY) / rangeY) * 2 - 1
      ]);
    }

    const tagB = document.getElementById("view-tag-b");
    if (tagB) tagB.textContent = `View B: ${cand.display_name}`;
    loadSubspaceAngles();
    renderScatter();
  }
}

window.toggleSplitView = toggleSplitView;
window.jumpToRashomonCandidate = jumpToRashomonCandidate;
window.compareRashomonCandidate = compareRashomonCandidate;

function initSprint13Controls() {
  const btnTopo = document.getElementById("btn-topology");
  const btnDist = document.getElementById("btn-distortion");
  const btnStab = document.getElementById("btn-stability");

  if (btnTopo) {
    btnTopo.addEventListener("click", () => {
      showTopologyGraph = !showTopologyGraph;
      btnTopo.classList.toggle("active", showTopologyGraph);
      if (showTopologyGraph && !topologyEdges) {
        loadTopologyEdges();
      } else {
        renderScatter();
      }
    });
  }

  if (btnDist) {
    btnDist.addEventListener("click", () => {
      showDistortionMap = !showDistortionMap;
      btnDist.classList.toggle("active", showDistortionMap);
      if (showDistortionMap && !distortionGrid) {
        loadDistortionGrid();
      } else {
        renderScatter();
      }
    });
  }

  if (btnStab) {
    btnStab.addEventListener("click", () => {
      showStabilityMap = !showStabilityMap;
      btnStab.classList.toggle("active", showStabilityMap);
      updateStabilityPanel();
      if (showStabilityMap && !pointStabilityData) {
        loadPointStability();
        loadRashomonSet();
      } else {
        renderScatter();
      }
    });
  }
}

function updateIntegrityPanel(data) {
  if (!data) return;
  document.getElementById("diag-precision").textContent = (data.precision * 100).toFixed(0) + "%";
  document.getElementById("diag-recall").textContent    = (data.recall * 100).toFixed(0) + "%";
  
  const trustElem = document.getElementById("diag-trust");
  trustElem.textContent = data.trustworthiness.toFixed(2);
  if (data.trustworthiness >= 0.90) {
    trustElem.style.color = "#6ee7b7";
  } else if (data.trustworthiness < 0.80) {
    trustElem.style.color = "#f87171";
  } else {
    trustElem.style.color = "#fbbf24";
  }

  document.getElementById("diag-stress").textContent = data.stress.toFixed(2);

  const countPreserved = document.getElementById("count-preserved");
  const countTorn      = document.getElementById("count-torn");
  const countFalse     = document.getElementById("count-false");

  countPreserved.textContent = data.preserved.length;
  countTorn.textContent      = data.torn.length ? `${data.torn.length} (${data.torn.join(", ")})` : "0";
  countFalse.textContent     = data.false_neighbors.length ? `${data.false_neighbors.length} (${data.false_neighbors.join(", ")})` : "0";
}

// ─── Saved-View Atlas (Sprint 5) ─────────────────────────────────────────────

function initAtlasControls() {
  if (btnSaveView) {
    btnSaveView.addEventListener("click", saveCurrentView);
  }
}

async function fetchSavedViews() {
  try {
    const res = await fetch("/api/saved-views");
    if (!res.ok) return;
    savedViews = await res.json();
    renderAtlasList();
  } catch (err) {
    console.error("Failed to fetch saved views:", err);
  }
}

async function saveCurrentView() {
  const name = atlasViewName.value.trim() || `View ${savedViews.length + 1}`;
  const note = atlasViewNote.value.trim();
  const targetId = fixtureData ? fixtureData.object_ids[selectedIdx] : "corner_0";

  const payload = {
    name,
    note,
    dataset: currentDataset,
    representation_id: currentRep,
    metric_id: currentMetric,
    k: currentK,
    target_id: targetId,
  };

  try {
    const res = await fetch("/api/saved-views", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to save view");
    atlasViewName.value = "";
    atlasViewNote.value = "";
    await fetchSavedViews();
  } catch (err) {
    console.error("Error saving view:", err);
  }
}

async function deleteSavedView(viewId, event) {
  event.stopPropagation();
  try {
    const res = await fetch(`/api/saved-views?id=${encodeURIComponent(viewId)}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Failed to delete view");
    await fetchSavedViews();
  } catch (err) {
    console.error("Error deleting view:", err);
  }
}

async function restoreSavedView(view) {
  if (view.metadata && view.metadata.dataset && view.metadata.dataset !== currentDataset) {
    await loadDataset(view.metadata.dataset);
    if (datasetSelect) datasetSelect.value = view.metadata.dataset;
  }
  setRepresentation(view.representation_id);
  currentMetric = view.metric_id;
  metricSelect.value = currentMetric;
  currentK = view.k;
  kSlider.value = currentK;
  kValLabel.textContent = currentK;

  if (fixtureData) {
    const idx = fixtureData.object_ids.indexOf(view.target_id);
    if (idx !== -1) {
      selectPoint(idx);
    }
  }
}

function renderAtlasList() {
  if (!atlasList) return;
  if (savedViews.length === 0) {
    atlasList.innerHTML = '<p class="panel-hint">No saved views yet.</p>';
    return;
  }

  atlasList.innerHTML = savedViews
    .map(v => `
      <div class="atlas-card" data-id="${v.id}">
        <div class="atlas-card-header">
          <span class="atlas-card-title">${escapeHtml(v.name)}</span>
          <button class="btn-delete-view" data-id="${v.id}" title="Delete view">✕</button>
        </div>
        <div class="atlas-card-meta">${v.representation_id} · ${v.metric_id} · k=${v.k} · ${v.target_id}</div>
        ${v.note ? `<div class="atlas-card-note">${escapeHtml(v.note)}</div>` : ""}
      </div>
    `).join("");

  atlasList.querySelectorAll(".atlas-card").forEach(card => {
    card.addEventListener("click", () => {
      const vId = card.getAttribute("data-id");
      const found = savedViews.find(sv => sv.id === vId);
      if (found) restoreSavedView(found);
    });
  });

  atlasList.querySelectorAll(".btn-delete-view").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const vId = btn.getAttribute("data-id");
      deleteSavedView(vId, e);
    });
  });
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ─── Grand Tour Controller ───────────────────────────────────────────────────

async function loadTourPath() {
  if (!fixtureData) return;
  try {
    const res = await fetch(`/api/tour-path?dataset=${encodeURIComponent(currentDataset)}&representation=${encodeURIComponent(currentRep)}&n_frames=180`);
    if (!res.ok) return;
    const data = await res.json();
    tourFrames = data.frames;
    tourBases = data.bases;
    tourFrameIdx = 0;
    // Server globally normalises all frames to [-1, 1] — use fixed viewport.
    tourGlobalBounds = { minX: -1.1, maxX: 1.1, minY: -1.1, maxY: 1.1 };
    const scrubber = document.getElementById("tour-scrubber");
    if (scrubber) {
      scrubber.max = Math.max(0, tourFrames.length - 1);
      scrubber.value = 0;
    }
    updateTourFrameLabel();
  } catch (err) {
    console.error("Failed to load tour path:", err);
  }
}

function initTourControls() {
  const btnPlay    = document.getElementById("btn-tour-play");
  const scrubber   = document.getElementById("tour-scrubber");
  const speedSelect = document.getElementById("tour-speed-select");
  if (btnPlay)    btnPlay.addEventListener("click", () => toggleTourPlayback());
  if (scrubber)   scrubber.addEventListener("input", (e) => seekTourFrame(parseInt(e.target.value, 10)));
  if (speedSelect) speedSelect.addEventListener("change", (e) => { tourSpeed = parseFloat(e.target.value) || 1.0; });
}

function initOptimizerControls() {
  const btnLda = document.getElementById("btn-opt-lda");
  const btnIntegrity = document.getElementById("btn-opt-integrity");
  if (btnLda)       btnLda.addEventListener("click", () => optimizeSubspaceView("class_separation"));
  if (btnIntegrity) btnIntegrity.addEventListener("click", () => optimizeSubspaceView("neighborhood_integrity"));
}

async function optimizeSubspaceView(criterion) {
  if (!fixtureData) return;
  const targetId = (fixtureData.object_ids && fixtureData.object_ids[selectedIdx]) || (fixtureData.object_ids && fixtureData.object_ids[0]) || "target_0";
  try {
    setStatus("loading", "Optimizing projection subspace...");
    const url = `/api/optimize-view?dataset=${encodeURIComponent(currentDataset)}&representation=${encodeURIComponent(currentRep)}&criterion=${encodeURIComponent(criterion)}&target_id=${encodeURIComponent(targetId)}&n_frames=60`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Optimization failed");
    const data = await res.json();
    tourFrames = data.frames;
    tourBases = data.bases;
    tourFrameIdx = 0;
    tourGlobalBounds = { minX: -1.1, maxX: 1.1, minY: -1.1, maxY: 1.1 };
    const scrubber = document.getElementById("tour-scrubber");
    if (scrubber) {
      scrubber.max = Math.max(0, tourFrames.length - 1);
      scrubber.value = 0;
    }
    updateTourFrameLabel();
    setStatus("ready", `Subspace optimized (${criterion})`);
    playTour();
  } catch (err) {
    console.error("Error optimizing subspace:", err);
    setStatus("error", "Subspace optimization failed");
  }
}

function updateMultiInspector(indices) {
  const multiPanel = document.getElementById("panel-multi-selection");
  if (!multiPanel) return;
  if (!indices || indices.length < 2) {
    multiPanel.classList.add("hidden");
    return;
  }
  multiPanel.classList.remove("hidden");

  document.getElementById("multi-count").textContent = indices.length;

  let totalConf = 0;
  const classesSet = new Set();
  indices.forEach(idx => {
    const meta = getPointMeta(idx);
    if (meta) {
      totalConf += meta.confidence;
      classesSet.add(meta.predClassName);
    }
  });

  const avgConf = (totalConf / indices.length * 100).toFixed(0) + "%";
  document.getElementById("multi-avg-conf").textContent = avgConf;
  document.getElementById("multi-classes").textContent = Array.from(classesSet).join(", ");
}

function toggleTourPlayback() {
  if (isTourPlaying) pauseTour(); else playTour();
}

function playTour() {
  if (!tourFrames || tourFrames.length === 0) return;
  if (tourAnimHandle) { cancelAnimationFrame(tourAnimHandle); tourAnimHandle = null; }
  savedZoom  = zoomLevel;  savedPanX = panOffsetX;  savedPanY = panOffsetY;
  zoomLevel  = 1.0;        panOffsetX = 0;           panOffsetY = 0;
  isTourPlaying = true;
  const btnPlay = document.getElementById("btn-tour-play");
  if (btnPlay) { btnPlay.textContent = "⏸ Pause Tour"; btnPlay.classList.add("playing"); }
  lastTourTime = performance.now();
  stepTourAnimation();
}

function pauseTour() {
  isTourPlaying = false;
  if (tourAnimHandle) { cancelAnimationFrame(tourAnimHandle); tourAnimHandle = null; }
  zoomLevel  = savedZoom;  panOffsetX = savedPanX;  panOffsetY = savedPanY;
  const btnPlay = document.getElementById("btn-tour-play");
  if (btnPlay) { btnPlay.textContent = "▶ Play Tour"; btnPlay.classList.remove("playing"); }
  renderScatter();
}

function stepTourAnimation(timestamp) {
  if (!isTourPlaying || !tourFrames) return;
  const now      = timestamp || performance.now();
  const interval = (1000 / 20) / tourSpeed;
  if (now - lastTourTime >= interval) {
    tourFrameIdx = (tourFrameIdx + 1) % tourFrames.length;
    const scrubber = document.getElementById("tour-scrubber");
    if (scrubber) scrubber.value = tourFrameIdx;
    updateTourFrameLabel();
    renderScatter();
    lastTourTime = now;
  }
  tourAnimHandle = requestAnimationFrame(stepTourAnimation);
}

function seekTourFrame(idx) {
  if (!tourFrames || idx < 0 || idx >= tourFrames.length) return;
  tourFrameIdx = idx;
  updateTourFrameLabel();
  renderScatter();
}

function updateTourFrameLabel() {
  const lbl = document.getElementById("tour-frame-label");
  if (lbl && tourFrames) lbl.textContent = `Frame ${tourFrameIdx + 1} / ${tourFrames.length}`;
}

// ─── Scatter renderer ─────────────────────────────────────────────────────────

function computeViewCoords(repId, viewId) {
  if (!fixtureData) return [];

  let basis = null;
  if (fixtureData.catalog && fixtureData.catalog[viewId]) {
    basis = fixtureData.catalog[viewId].basis;
  } else if (fixtureData.representations && fixtureData.representations[repId]) {
    basis = fixtureData.representations[repId].basis;
  }

  if (!basis || !fixtureData.raw_matrix) {
    if (fixtureData.representations && fixtureData.representations[repId]) {
      return fixtureData.representations[repId].coords;
    }
    return [];
  }

  let mat = fixtureData.raw_matrix;
  if (repId === "sqrt_probability") {
    mat = mat.map(row => row.map(v => Math.sqrt(Math.max(0, v))));
  } else if (repId === "clr_probability") {
    mat = mat.map(row => {
      const floor = row.map(v => Math.max(v, 1e-300));
      const logVec = floor.map(v => Math.log(v));
      const meanLog = logVec.reduce((a, b) => a + b, 0) / logVec.length;
      return logVec.map(v => v - meanLog);
    });
  }

  const n = mat.length;
  const k = mat[0].length;
  const means = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      means[j] += mat[i][j];
    }
  }
  for (let j = 0; j < k; j++) {
    means[j] /= n;
  }

  const coords = new Array(n);
  for (let i = 0; i < n; i++) {
    let x = 0, y = 0;
    for (let j = 0; j < k; j++) {
      const cVal = mat[i][j] - means[j];
      x += cVal * basis[j][0];
      y += cVal * basis[j][1];
    }
    coords[i] = [x, y];
  }
  return coords;
}

function getCoords() {
  if (!fixtureData) return [];
  if (tourFrames && tourFrameIdx < tourFrames.length && (isTourPlaying || tourFrameIdx > 0)) {
    return tourFrames[tourFrameIdx];
  }
  return computeViewCoords(currentRep, currentViewId);
}

function computeScale(coords, targetCanvas = canvas, zoom = zoomLevel, px = panOffsetX, py = panOffsetY) {
  if (!coords || coords.length === 0) return { toScreen: () => [0, 0], pad: 48, w: 500, h: 500 };
  const dpr  = window.devicePixelRatio || 1;
  const cssW = (targetCanvas || canvas).width / dpr;
  const cssH = (targetCanvas || canvas).height / dpr;
  const pad  = AXIS_PADDING;
  const w    = cssW - pad * 2;
  const h    = cssH - pad * 2;

  // During tour playback or scrubbing use the fixed [-1.1, 1.1] viewport. Server
  // globally normalises all frames so this always contains all points.
  if ((isTourPlaying || (tourFrames && tourFrameIdx > 0)) && tourGlobalBounds) {
    const b      = tourGlobalBounds;
    const rangeX = b.maxX - b.minX;
    const rangeY = b.maxY - b.minY;
    const scale  = Math.min(w / rangeX, h / rangeY) * 0.88;
    const cx     = (b.minX + b.maxX) / 2;
    const cy     = (b.minY + b.maxY) / 2;
    return {
      toScreen: (x, y) => [
        pad + w / 2 + (x - cx) * scale,
        pad + h / 2 - (y - cy) * scale,
      ],
      scale, minX: b.minX, maxX: b.maxX, minY: b.minY, maxY: b.maxY, cx, cy, pad, w, h,
    };
  }

  const xs     = coords.map(c => c[0]);
  const ys     = coords.map(c => c[1]);
  const minX   = Math.min(...xs), maxX = Math.max(...xs);
  const minY   = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const baseScale = Math.min(w / rangeX, h / rangeY) * 0.82;
  const scale     = baseScale * zoom;
  const cx        = (minX + maxX) / 2 - (px / scale);
  const cy        = (minY + maxY) / 2 + (py / scale);

  return {
    toScreen: (x, y) => [
      pad + w / 2 + (x - cx) * scale,
      pad + h / 2 - (y - cy) * scale,
    ],
    scale, minX, maxX, minY, maxY, cx, cy, pad, w, h,
  };
}

function renderScatter() {
  if (!fixtureData) return;
  const coordsA = getCoords();
  const wrap = document.getElementById("canvas-wrap");

  if (!wrap) return;
  const availW = isDualView ? (wrap.clientWidth - 32) / 2 : wrap.clientWidth - 16;
  const availH = wrap.clientHeight - 16;
  const size   = Math.max(100, Math.floor(Math.min(availW, availH)));

  renderScatterCanvas(canvas, ctx, coordsA, zoomLevel, panOffsetX, panOffsetY, size, false);

  if (isDualView && canvasB && ctxB) {
    const coordsB = viewCoordsB || coordsA;
    const zB  = syncCameras ? zoomLevel  : zoomLevelB;
    const pxB = syncCameras ? panOffsetX : panOffsetXB;
    const pyB = syncCameras ? panOffsetY : panOffsetYB;
    renderScatterCanvas(canvasB, ctxB, coordsB, zB, pxB, pyB, size, true);
  }
}

function drawDistortionHeatmap(sc2, cCtx, W, H) {
  if (!distortionGrid || !distortionGrid.grid) return;
  const grid   = distortionGrid.grid;
  const res    = distortionGrid.resolution || 32;
  const bounds = distortionGrid.bounds;
  if (!bounds) return;

  const spanX   = bounds.xMax - bounds.xMin;
  const spanY   = bounds.yMax - bounds.yMin;
  const cellW   = spanX / res;
  const cellH   = spanY / res;

  cCtx.save();
  for (let r = 0; r < res; r++) {
    const yMax = bounds.yMax - r * cellH;
    const yMin = bounds.yMax - (r + 1) * cellH;
    for (let c = 0; c < res; c++) {
      const val = grid[r][c];
      if (val === null || val === undefined) continue;

      const xMin = bounds.xMin + c * cellW;
      const xMax = bounds.xMin + (c + 1) * cellW;

      const [sX1, sY1] = sc2.toScreen(xMin, yMax);
      const [sX2, sY2] = sc2.toScreen(xMax, yMin);
      const rectW = Math.abs(sX2 - sX1);
      const rectH = Math.abs(sY2 - sY1);
      const rx = Math.min(sX1, sX2);
      const ry = Math.min(sY1, sY2);

      let alpha = Math.min(0.45, Math.abs(val - 1.0) * 0.5 + 0.10);
      let colorStr;
      if (val < 1.0) {
        colorStr = `rgba(56, 189, 248, ${alpha.toFixed(2)})`;   // blue = compression
      } else {
        colorStr = `rgba(248, 113, 113, ${alpha.toFixed(2)})`;  // red = expansion
      }

      cCtx.fillStyle = colorStr;
      cCtx.fillRect(rx, ry, rectW + 0.5, rectH + 0.5);
    }
  }
  cCtx.restore();
}

function drawTopologyGraph(coords, sc2, cCtx) {
  if (!topologyEdges || !fixtureData) return;
  const objectIds = fixtureData.object_ids;
  if (!objectIds) return;

  const idToIdx = {};
  objectIds.forEach((id, i) => { idToIdx[id] = i; });

  cCtx.save();
  for (const edge of topologyEdges) {
    const i1 = idToIdx[edge.source];
    const i2 = idToIdx[edge.target];
    if (i1 === undefined || i2 === undefined || i1 >= coords.length || i2 >= coords.length) continue;

    const [x1, y1] = sc2.toScreen(coords[i1][0], coords[i1][1]);
    const [x2, y2] = sc2.toScreen(coords[i2][0], coords[i2][1]);

    cCtx.beginPath();
    cCtx.moveTo(x1, y1);
    cCtx.lineTo(x2, y2);

    if (edge.type === "preserved") {
      cCtx.strokeStyle = "rgba(52, 211, 153, 0.28)";
      cCtx.lineWidth   = 1.2;
      cCtx.setLineDash([]);
    } else if (edge.type === "torn") {
      cCtx.strokeStyle = "rgba(248, 113, 113, 0.35)";
      cCtx.lineWidth   = 1.0;
      cCtx.setLineDash([3, 3]);
    } else {
      cCtx.strokeStyle = "rgba(251, 191, 36, 0.35)";
      cCtx.lineWidth   = 1.0;
      cCtx.setLineDash([2, 2]);
    }
    cCtx.stroke();
  }
  cCtx.restore();
}

function renderScatterCanvas(cEl, cCtx, coords, zoom, px, py, size, isViewB) {
  if (!cEl || !cCtx || !coords) return;
  const colors = fixtureData.colors;
  const dpr    = window.devicePixelRatio || 1;

  cEl.style.width  = size + "px";
  cEl.style.height = size + "px";
  cEl.width  = Math.floor(size * dpr);
  cEl.height = Math.floor(size * dpr);
  cCtx.setTransform(1, 0, 0, 1, 0, 0);
  cCtx.scale(dpr, dpr);

  const sc2 = computeScale(coords, cEl, zoom, px, py);
  const W = size, H = size;

  // Background
  cCtx.fillStyle = "#111827";
  cCtx.fillRect(0, 0, W, H);

  // Spatial Distortion Heatmap Background Overlay
  if (showDistortionMap && distortionGrid) {
    drawDistortionHeatmap(sc2, cCtx, W, H);
  }

  // Grid
  cCtx.strokeStyle = "rgba(255,255,255,0.04)";
  cCtx.lineWidth   = 1;
  for (let i = 0; i <= GRID_LINES; i++) {
    const t = i / GRID_LINES;
    const gx = sc2.pad + t * sc2.w;
    const gy = sc2.pad + t * sc2.h;
    cCtx.beginPath(); cCtx.moveTo(gx, sc2.pad); cCtx.lineTo(gx, sc2.pad + sc2.h); cCtx.stroke();
    cCtx.beginPath(); cCtx.moveTo(sc2.pad, gy); cCtx.lineTo(sc2.pad + sc2.w, gy); cCtx.stroke();
  }

  // Axis labels & zoom / tour indicator
  cCtx.font      = "10px 'JetBrains Mono', monospace";
  cCtx.textAlign = "center";
  if (isTourPlaying) {
    cCtx.fillStyle = "rgba(52,211,153,0.7)";
    cCtx.fillText("Grand Tour — Proj. X →", W / 2, H - 8);
    cCtx.save();
    cCtx.translate(12, H / 2);
    cCtx.rotate(-Math.PI / 2);
    cCtx.fillText("Proj. Y ↑", 0, 0);
    cCtx.restore();
  } else {
    cCtx.fillStyle = isViewB ? "rgba(56,189,248,0.5)" : "rgba(148,163,184,0.5)";
    cCtx.fillText(isViewB ? "Fisher LDA 1 →" : "PC1 →", W / 2, H - 8);
    cCtx.save();
    cCtx.translate(12, H / 2);
    cCtx.rotate(-Math.PI / 2);
    cCtx.fillText(isViewB ? "Fisher LDA 2 ↑" : "PC2 ↑", 0, 0);
    cCtx.restore();
    if (zoom !== 1.0) {
      cCtx.textAlign = "left";
      cCtx.fillStyle = isViewB ? "#38bdf8" : "#34d399";
      cCtx.fillText(`${zoom.toFixed(1)}x Zoom`, 48, H - 10);
    }
  }

  // Rep label
  cCtx.font      = "11px Inter, sans-serif";
  cCtx.fillStyle = "rgba(148,163,184,0.4)";
  cCtx.textAlign = "right";
  cCtx.fillText(isViewB ? "Fisher LDA View B" : `${currentViewId} · ${currentRep.replace("_", " ")}`, W - 12, H - 10);

  // Global k-NN Topology Graph Overlay
  if (showTopologyGraph && topologyEdges) {
    drawTopologyGraph(coords, sc2, cCtx);
  }

  // Simplex structure lines (if 3-class calibration)
  if (coords.length === 15) {
    drawSimplexEdges(coords, sc2, cCtx);
  }

  // Diagnostic overlay lines & neighbor rings from selected point
  if (selectedIdx !== null && diagResult) {
    drawDiagnosticOverlay(coords, sc2, cCtx);
  }

  const stabVolatileCutoff = 0.40;

  // Draw points
  coords.forEach((c, i) => {
    const [sx, sy] = sc2.toScreen(c[0], c[1]);
    const color    = (colors && colors[i]) ? colors[i] : "#6ee7b7";
    const isSel    = (i === selectedIdx);
    const isHov    = (i === hoveredIdx);
    const r        = (isSel || isHov) ? POINT_RADIUS_HOVER : POINT_RADIUS;

    cCtx.beginPath();
    cCtx.arc(sx, sy, r, 0, Math.PI * 2);

    if (isSel) {
      cCtx.fillStyle   = color;
      cCtx.shadowColor = color;
      cCtx.shadowBlur  = 16;
    } else if (isHov) {
      cCtx.fillStyle   = color;
      cCtx.shadowColor = color;
      cCtx.shadowBlur  = 10;
    } else {
      cCtx.fillStyle   = color;
      cCtx.shadowBlur  = 0;
    }
    cCtx.fill();

    cCtx.strokeStyle = isSel ? "#ffffff" : (isHov ? "#ffffff" : "rgba(255,255,255,0.15)");
    cCtx.lineWidth   = isSel ? 2.5 : 1.0;
    cCtx.stroke();

    // High-Performance Volatile & Persistent Point Stability Aura
    if (showStabilityMap && pointStabilityData && pointStabilityData.stability_scores) {
      const stabScore = pointStabilityData.stability_scores[i] ?? 1.0;
      if (stabScore <= stabVolatileCutoff) {
        // Volatile / Unstable Points: Fast Electric Orange Pulsing Hazard Ring
        cCtx.save();
        cCtx.beginPath();
        cCtx.arc(sx, sy, r + 4, 0, Math.PI * 2);
        cCtx.strokeStyle = "#f97316";
        cCtx.lineWidth = 2.2;
        cCtx.setLineDash([4, 3]);
        cCtx.stroke();

        cCtx.beginPath();
        cCtx.arc(sx, sy, r + 7, 0, Math.PI * 2);
        cCtx.strokeStyle = "rgba(249, 115, 22, 0.5)";
        cCtx.lineWidth = 1.0;
        cCtx.stroke();
        cCtx.restore();
      } else if (stabScore >= 0.70) {
        // Persistent / Highly Stable Points: Fast Cyber Cyan Shield Ring
        cCtx.save();
        cCtx.beginPath();
        cCtx.arc(sx, sy, r + 3.5, 0, Math.PI * 2);
        cCtx.strokeStyle = "#38bdf8";
        cCtx.lineWidth = 1.8;
        cCtx.stroke();

        cCtx.beginPath();
        cCtx.arc(sx, sy, r + 6.5, 0, Math.PI * 2);
        cCtx.strokeStyle = "rgba(56, 189, 248, 0.4)";
        cCtx.lineWidth = 1.0;
        cCtx.stroke();
        cCtx.restore();
      }
    }
  });
  cCtx.shadowBlur = 0;

  // ID & Label on hover or select
  const activeIdx = hoveredIdx !== null ? hoveredIdx : selectedIdx;
  if (activeIdx !== null && activeIdx < coords.length) {
    const [sx, sy] = sc2.toScreen(coords[activeIdx][0], coords[activeIdx][1]);
    const pMeta = getPointMeta(activeIdx);
    let labelText = pMeta.trueClassName
      ? `${pMeta.id} • ${pMeta.trueClassName} (${(pMeta.confidence * 100).toFixed(0)}% ${pMeta.predClassName})`
      : `${pMeta.id} • ${pMeta.predClassName} (${(pMeta.confidence * 100).toFixed(0)}%)`;

    if (showStabilityMap && pointStabilityData && pointStabilityData.stability_scores) {
      const sVal = pointStabilityData.stability_scores[activeIdx];
      if (sVal !== undefined) {
        labelText += ` • 🛡️ ${(sVal * 100).toFixed(0)}% Stable`;
      }
    }

    cCtx.font      = "10px 'JetBrains Mono', monospace";
    cCtx.fillStyle = "rgba(255,255,255,0.95)";
    cCtx.textAlign = "center";
    const labelY = sy - POINT_RADIUS_HOVER - 6;
    cCtx.fillText(labelText, sx, labelY);
  }

  // Draw Shift+Drag Marquee Selection Box
  if (isBoxSelecting) {
    const bx = Math.min(boxStartX, boxCurX), bw = Math.abs(boxCurX - boxStartX);
    const by = Math.min(boxStartY, boxCurY), bh = Math.abs(boxCurY - boxStartY);
    cCtx.save();
    cCtx.fillStyle = "rgba(52, 211, 153, 0.12)";
    cCtx.strokeStyle = "#34d399";
    cCtx.lineWidth = 1.5;
    cCtx.setLineDash([4, 4]);
    cCtx.fillRect(bx, by, bw, bh);
    cCtx.strokeRect(bx, by, bw, bh);
    cCtx.restore();
  }

  // Active Feature Loadings HUD during Grand Tour
  if (isTourPlaying || (tourFrames && tourFrames.length > 0 && tourFrameIdx > 0)) {
    drawFeatureLoadingsHUD(W, H, cCtx);
  }
}

function drawFeatureLoadingsHUD(W, H, targetCtx) {
  const activeCtx = targetCtx || ctx;
  if (!tourBases || tourFrameIdx >= tourBases.length) return;
  const basis = tourBases[tourFrameIdx];
  const featNames = fixtureData ? fixtureData.feature_names : [];
  if (!basis || !featNames) return;

  const loadings = [];
  for (let i = 0; i < basis.length; i++) {
    const vx = basis[i][0];
    const vy = basis[i][1];
    const mag = Math.hypot(vx, vy);
    const name = featNames[i] ? featNames[i].replace(/^p_/, "") : `f${i}`;
    loadings.push({ name, mag });
  }

  loadings.sort((a, b) => b.mag - a.mag);
  const top = loadings.slice(0, 4);

  const hudW  = 184;
  const itemH = 16;
  const hudH  = 26 + top.length * itemH;
  const hudX  = W - hudW - 16;
  const hudY  = 16;

  activeCtx.save();
  // Glass panel container
  activeCtx.fillStyle   = "rgba(15, 23, 42, 0.88)";
  activeCtx.strokeStyle = "rgba(52, 211, 153, 0.35)";
  activeCtx.lineWidth   = 1;
  activeCtx.beginPath();
  if (activeCtx.roundRect) {
    activeCtx.roundRect(hudX, hudY, hudW, hudH, 6);
  } else {
    activeCtx.rect(hudX, hudY, hudW, hudH);
  }
  activeCtx.fill();
  activeCtx.stroke();

  // HUD Title
  activeCtx.font      = "600 9px 'JetBrains Mono', monospace";
  activeCtx.fillStyle = "#34d399";
  activeCtx.textAlign = "left";
  activeCtx.fillText("TOP FEATURE LOADINGS", hudX + 10, hudY + 16);

  // Bars
  const barMaxW = 55;
  top.forEach((item, idx) => {
    const y = hudY + 32 + idx * itemH;

    // Feature Name
    activeCtx.font      = "10px Inter, sans-serif";
    activeCtx.fillStyle = "rgba(226, 232, 240, 0.9)";
    activeCtx.textAlign = "left";
    const displayName = item.name.length > 12 ? item.name.substring(0, 11) + "…" : item.name;
    activeCtx.fillText(displayName, hudX + 10, y);

    // Track Background
    const barX = hudX + 88;
    const barY = y - 8;
    activeCtx.fillStyle = "rgba(255, 255, 255, 0.08)";
    activeCtx.fillRect(barX, barY, barMaxW, 6);

    // Active Bar
    const barW = Math.min(barMaxW, Math.max(2, item.mag * barMaxW));
    activeCtx.fillStyle = "#34d399";
    activeCtx.fillRect(barX, barY, barW, 6);

    // Numeric Value
    activeCtx.font      = "9px 'JetBrains Mono', monospace";
    activeCtx.fillStyle = "rgba(148, 163, 184, 0.85)";
    activeCtx.textAlign = "right";
    activeCtx.fillText(item.mag.toFixed(2), hudX + hudW - 8, y);
  });

  activeCtx.restore();
}

function drawSimplexEdges(coords, sc, targetCtx) {
  const activeCtx = targetCtx || ctx;
  const edgePairs = [[0, 3], [0, 4], [1, 3], [1, 5], [2, 4], [2, 5], [3, 6], [4, 6], [5, 6]];
  activeCtx.strokeStyle = "rgba(255,255,255,0.05)";
  activeCtx.lineWidth   = 0.8;
  edgePairs.forEach(([a, b]) => {
    if (a >= coords.length || b >= coords.length) return;
    const [ax, ay] = sc.toScreen(coords[a][0], coords[a][1]);
    const [bx, by] = sc.toScreen(coords[b][0], coords[b][1]);
    activeCtx.beginPath();
    activeCtx.moveTo(ax, ay);
    activeCtx.lineTo(bx, by);
    activeCtx.stroke();
  });
}

function drawDiagnosticOverlay(coords, sc, targetCtx) {
  const activeCtx = targetCtx || ctx;
  if (!diagResult || selectedIdx >= coords.length) return;
  const [srcX, srcY] = sc.toScreen(coords[selectedIdx][0], coords[selectedIdx][1]);
  const idToIdx = new Map(fixtureData.object_ids.map((id, idx) => [id, idx]));

  const drawLink = (targetId, color, dashPattern) => {
    const tIdx = idToIdx.get(targetId);
    if (tIdx === undefined || tIdx >= coords.length) return;
    const [tx, ty] = sc.toScreen(coords[tIdx][0], coords[tIdx][1]);

    activeCtx.save();
    activeCtx.strokeStyle = color;
    activeCtx.lineWidth   = 2.5;
    activeCtx.setLineDash(dashPattern);
    activeCtx.beginPath();
    activeCtx.moveTo(srcX, srcY);
    activeCtx.lineTo(tx, ty);
    activeCtx.stroke();
    activeCtx.restore();
  };

  const drawNeighborRing = (targetId, color, ringRadius) => {
    const tIdx = idToIdx.get(targetId);
    if (tIdx === undefined || tIdx >= coords.length) return;
    const [tx, ty] = sc.toScreen(coords[tIdx][0], coords[tIdx][1]);

    activeCtx.save();
    activeCtx.shadowColor = color;
    activeCtx.shadowBlur  = 12;
    activeCtx.strokeStyle = color;
    activeCtx.lineWidth   = 2.0;
    activeCtx.beginPath();
    activeCtx.arc(tx, ty, ringRadius, 0, Math.PI * 2);
    activeCtx.stroke();
    activeCtx.restore();
  };

  diagResult.preserved.forEach(id => drawLink(id, "#34d399", []));
  diagResult.torn.forEach(id => drawLink(id, "#f87171", [6, 4]));
  diagResult.false_neighbors.forEach(id => drawLink(id, "#fbbf24", [2, 3]));

  diagResult.preserved.forEach(id => drawNeighborRing(id, "#34d399", 12));
  diagResult.torn.forEach(id => drawNeighborRing(id, "#f87171", 14));
  diagResult.false_neighbors.forEach(id => drawNeighborRing(id, "#fbbf24", 14));
}

// ─── Canvas interaction & Pan ─────────────────────────────────────────────────

function initCanvasInteraction() {
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mouseleave", () => {
    isPanning = false;
    isBoxSelecting = false;
    hoveredIdx = null;
    tooltip.classList.add("hidden");
    canvas.style.cursor = "crosshair";
    renderScatter();
  });
  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("click", onMouseClick);

  canvasB = document.getElementById("scatter-canvas-b");
  if (canvasB) {
    ctxB = canvasB.getContext("2d");
    canvasB.addEventListener("mousemove", onMouseMove);
    canvasB.addEventListener("mouseleave", () => {
      isPanning = false;
      isBoxSelecting = false;
      hoveredIdx = null;
      tooltip.classList.add("hidden");
      canvasB.style.cursor = "crosshair";
      renderScatter();
    });
    canvasB.addEventListener("mousedown", onMouseDown);
    canvasB.addEventListener("mouseup", onMouseUp);
    canvasB.addEventListener("click", onMouseClick);
  }

  window.addEventListener("resize", () => renderScatter());
}

function initDualViewControls() {
  const btnSplit = document.getElementById("btn-split-view");
  const chkSync  = document.getElementById("chk-sync-cameras");

  if (chkSync) {
    chkSync.addEventListener("change", (e) => {
      syncCameras = e.target.checked;
      if (syncCameras) {
        zoomLevelB = zoomLevel;
        panOffsetXB = panOffsetX;
        panOffsetYB = panOffsetY;
      }
      renderScatter();
    });
  }

  if (btnSplit) {
    btnSplit.addEventListener("click", () => {
      isDualView = !isDualView;
      const wrap = document.getElementById("canvas-wrap");
      const containerB = document.getElementById("canvas-container-b");

      if (isDualView) {
        btnSplit.classList.add("active");
        btnSplit.textContent = "🗖 Single View";
        if (wrap) wrap.classList.add("dual-view");
        if (containerB) containerB.classList.remove("hidden");

        loadDualViewBData();
      } else {
        btnSplit.classList.remove("active");
        btnSplit.textContent = "🗖 Split View";
        if (wrap) wrap.classList.remove("dual-view");
        if (containerB) containerB.classList.add("hidden");
      }
      setTimeout(() => renderScatter(), 50);
    });
  }
}

async function loadDualViewBData() {
  if (!fixtureData) return;
  viewCoordsB = getCoords();
  renderScatter();

  try {
    const targetId = (fixtureData.object_ids && fixtureData.object_ids[selectedIdx]) || "target_0";
    const res = await fetch(`/api/optimize-view?dataset=${encodeURIComponent(currentDataset)}&representation=${encodeURIComponent(currentRep)}&criterion=class_separation&target_id=${encodeURIComponent(targetId)}&n_frames=2`);
    if (res.ok) {
      const data = await res.json();
      if (data.frames && data.frames.length > 0) {
        viewCoordsB = data.frames[data.frames.length - 1];
        const tagB = document.getElementById("view-tag-b");
        if (tagB) tagB.textContent = "View B: Fisher LDA";
        renderScatter();
      }
    }
  } catch (err) {
    console.error("Failed to load dual view B coords:", err);
  }
  await loadSubspaceAngles();
}

function onMouseDown(e) {
  const targetEl = (isDualView && e.target === canvasB) ? canvasB : canvas;
  const rect = targetEl.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;

  if (e.shiftKey) {
    isBoxSelecting = true;
    boxStartX = mx;
    boxStartY = my;
    boxCurX = mx;
    boxCurY = my;
    targetEl.style.cursor = "crosshair";
    return;
  }

  const idx = getHitIndex(e.clientX, e.clientY, targetEl);
  if (idx === null || e.button === 1) {
    isPanning  = true;
    startPanX  = (targetEl === canvasB && !syncCameras) ? e.clientX - panOffsetXB : e.clientX - panOffsetX;
    startPanY  = (targetEl === canvasB && !syncCameras) ? e.clientY - panOffsetYB : e.clientY - panOffsetY;
    targetEl.style.cursor = "grabbing";
  }
}

function onMouseUp(e) {
  const targetEl = (isDualView && e.target === canvasB) ? canvasB : canvas;
  if (isBoxSelecting) {
    isBoxSelecting = false;
    const minX = Math.min(boxStartX, boxCurX), maxX = Math.max(boxStartX, boxCurX);
    const minY = Math.min(boxStartY, boxCurY), maxY = Math.max(boxStartY, boxCurY);
    const coords = (targetEl === canvasB) ? (viewCoordsB || getCoords()) : getCoords();
    const z = (targetEl === canvasB) ? (syncCameras ? zoomLevel : zoomLevelB) : zoomLevel;
    const px = (targetEl === canvasB) ? (syncCameras ? panOffsetX : panOffsetXB) : panOffsetX;
    const py = (targetEl === canvasB) ? (syncCameras ? panOffsetY : panOffsetYB) : panOffsetY;
    const sc = computeScale(coords, targetEl, z, px, py);

    selectedIndices = [];
    for (let i = 0; i < coords.length; i++) {
      const [sx, sy] = sc.toScreen(coords[i][0], coords[i][1]);
      if (sx >= minX && sx <= maxX && sy >= minY && sy <= maxY) {
        selectedIndices.push(i);
      }
    }
    updateMultiInspector(selectedIndices);
    renderScatter();
    targetEl.style.cursor = "crosshair";
    return;
  }

  if (isPanning) {
    isPanning = false;
    targetEl.style.cursor = "crosshair";
  }
}

function getHitIndex(clientX, clientY, targetEl) {
  if (!fixtureData) return null;
  const cEl = (targetEl === canvasB) ? canvasB : canvas;
  const rect = cEl.getBoundingClientRect();
  const mx   = clientX - rect.left;
  const my   = clientY - rect.top;

  const isB    = (cEl === canvasB);
  const coords = isB ? (viewCoordsB || getCoords()) : getCoords();
  const z      = isB ? (syncCameras ? zoomLevel : zoomLevelB) : zoomLevel;
  const px     = isB ? (syncCameras ? panOffsetX : panOffsetXB) : panOffsetX;
  const py     = isB ? (syncCameras ? panOffsetY : panOffsetYB) : panOffsetY;
  const sc     = computeScale(coords, cEl, z, px, py);

  for (let i = coords.length - 1; i >= 0; i--) {
    const [sx, sy] = sc.toScreen(coords[i][0], coords[i][1]);
    const dist = Math.hypot(mx - sx, my - sy);
    if (dist <= POINT_RADIUS_HOVER + 4) return i;
  }
  return null;
}

function onMouseMove(e) {
  const targetEl = (isDualView && e.target === canvasB) ? canvasB : canvas;
  const rect = targetEl.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;

  if (isBoxSelecting) {
    boxCurX = mx;
    boxCurY = my;
    renderScatter();
    return;
  }

  if (isPanning) {
    if (targetEl === canvasB && !syncCameras) {
      panOffsetXB = e.clientX - startPanX;
      panOffsetYB = e.clientY - startPanY;
    } else {
      panOffsetX = e.clientX - startPanX;
      panOffsetY = e.clientY - startPanY;
      if (syncCameras) {
        panOffsetXB = panOffsetX;
        panOffsetYB = panOffsetY;
      }
    }
    renderScatter();
    return;
  }

  const idx = getHitIndex(e.clientX, e.clientY, targetEl);
  if (idx !== hoveredIdx) {
    hoveredIdx = idx;
    renderScatter();
  }

  if (idx !== null) {
    const pMeta = getPointMeta(idx);
    const confPct = (pMeta.confidence * 100).toFixed(0);
    const statusIcon = pMeta.isCorrect ? "✓" : "✕";
    const trueLabelStr = pMeta.trueClassName ? ` [True: ${escapeHtml(pMeta.trueClassName)}]` : "";

    tooltip.innerHTML = `<strong>${escapeHtml(pMeta.id)}</strong> &nbsp; Predict: <span style="color:#6ee7b7">${escapeHtml(pMeta.predClassName)}</span> (${confPct}%)${trueLabelStr} ${statusIcon}`;

    const tx = e.clientX - rect.left + 14;
    const ty = e.clientY - rect.top  - 12;
    tooltip.style.left = tx + "px";
    tooltip.style.top  = ty + "px";
    tooltip.classList.remove("hidden");
    targetEl.style.cursor = "pointer";
  } else {
    tooltip.classList.add("hidden");
    targetEl.style.cursor = "crosshair";
  }
}

function onMouseClick(e) {
  if (isPanning) return;
  const idx = getHitIndex(e.clientX, e.clientY);
  if (idx === null) return;
  selectPoint(idx);
}

function selectPoint(idx) {
  selectedIdx = idx;
  updateInspector(idx);
  fetchDiagnostics();
  updateSelectedPointStabilityCard();
}

// ─── Source inspector ─────────────────────────────────────────────────────────

function updateInspector(idx) {
  if (!fixtureData || idx >= fixtureData.object_ids.length) return;
  const pMeta = getPointMeta(idx);
  if (!pMeta) return;

  const raw       = pMeta.raw;
  const coords    = getCoords()[idx];
  const featNames = fixtureData.feature_names || raw.map((_, i) => `class_${i}`);
  const dl        = document.getElementById("inspector-list");

  let html = "";

  const confPct = (pMeta.confidence * 100).toFixed(1);
  const correctBadge = pMeta.isCorrect
    ? '<span style="color:#6ee7b7; font-weight:700">✓ Correct</span>'
    : '<span style="color:#f87171; font-weight:700">✕ Error</span>';

  html += `<dt>Object ID</dt><dd class="highlight">${escapeHtml(pMeta.id)}</dd>`;
  html += `<dt>Prediction</dt><dd style="color:#6ee7b7; font-weight:600">${escapeHtml(pMeta.predClassName)} (${confPct}%)</dd>`;
  if (pMeta.trueClassName) {
    html += `<dt>True Label</dt><dd>${escapeHtml(pMeta.trueClassName)} &nbsp; ${correctBadge}</dd>`;
  }
  if (pMeta.entropy !== null) {
    html += `<dt>Entropy</dt><dd>${pMeta.entropy.toFixed(3)} nats</dd>`;
  }

  const palette = [
    "#6ee7b7", "#93c5fd", "#c4b5fd", "#f472b6", "#fbbf24",
    "#a7f3d0", "#f87171", "#60a5fa", "#e879f9", "#38bdf8"
  ];

  featNames.forEach((name, i) => {
    const cleanName = name.replace(/^p_/, '');
    const val = raw[i] !== undefined ? raw[i].toFixed(4) : "—";
    const dotColor = palette[i % palette.length];
    const isMax = i === pMeta.maxClassIdx;
    const valStyle = isMax ? 'style="color:#6ee7b7; font-weight:600"' : '';

    html += `<dt><span class="legend-dot" style="display:inline-block; width:6px; height:6px; background:${dotColor}; margin-right:4px"></span>${escapeHtml(cleanName)}</dt><dd ${valStyle}>${val}</dd>`;
  });

  if (coords) {
    html += `<dt>PC1</dt><dd>${coords[0].toFixed(4)}</dd>`;
    html += `<dt>PC2</dt><dd>${coords[1].toFixed(4)}</dd>`;
  }

  dl.innerHTML = html;

  // Fetch payload image thumbnail if available
  fetchPayloadImage(pMeta.id);
}

async function fetchPayloadImage(targetId) {
  const container = document.getElementById("inspector-payload-container");
  if (!container) return;

  if (!fixtureData || !fixtureData.has_payloads) {
    container.classList.add("hidden");
    container.innerHTML = "";
    return;
  }

  try {
    const res = await fetch(`/api/object-payload?dataset=${encodeURIComponent(currentDataset)}&target_id=${encodeURIComponent(targetId)}`);
    if (res.ok && res.headers.get("content-type")?.includes("image/png")) {
      const blob = await res.blob();
      const imgUrl = URL.createObjectURL(blob);
      container.innerHTML = `<img src="${imgUrl}" alt="${escapeHtml(targetId)}" /><span class="payload-label">Raw Sample Image (8x8)</span>`;
      container.classList.remove("hidden");
    } else {
      container.classList.add("hidden");
      container.innerHTML = "";
    }
  } catch (err) {
    container.classList.add("hidden");
    container.innerHTML = "";
  }
}

function initReportExport() {
  const btn = document.getElementById("btn-export-record");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Generating Report…";
    try {
      const targetId = (fixtureData && selectedIdx !== null) ? fixtureData.object_ids[selectedIdx] : "";
      const payload = {
        dataset: currentDataset,
        target_id: targetId,
        representation: currentRep,
        metric: currentMetric,
        view_id: currentViewId,
        k: currentK,
        saved_views: savedViews,
      };

      const res = await fetch("/api/export-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `shadowspace-investigation-${currentDataset}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export report error:", err);
      alert("Failed to generate investigation report.");
    } finally {
      btn.disabled = false;
      btn.textContent = "↓ Export Investigation Record";
    }
  });
}

function initStressHeatmapToggle() {
  const toggle = document.getElementById("toggle-stress-heatmap");
  if (!toggle) return;
  toggle.addEventListener("change", () => {
    renderScatter();
  });
}

// ─── Go ───────────────────────────────────────────────────────────────────────

boot();

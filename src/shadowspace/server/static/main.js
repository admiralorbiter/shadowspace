/**
 * Shadowspace Workbench — Sprint 4
 * main.js — Fetches fixture data, renders PCA scatter on canvas,
 * handles representation switching, point selection, and live local integrity overlays.
 */

"use strict";

// ─── State ────────────────────────────────────────────────────────────────────

let fixtureData    = null;          // raw JSON from /api/fixture
let currentRep     = "probability"; // active representation key
let currentMetric  = "euclidean";   // active metric key
let selectedIdx    = 0;             // selected point index (default 0)
let hoveredIdx     = null;          // currently hovered point index
let currentK       = 3;             // neighborhood size k
let diagResult     = null;          // live diagnostic payload from /api/diagnostics

const POINT_RADIUS        = 7;
const POINT_RADIUS_HOVER  = 10;
const POINT_RADIUS_SELECT = 10;
const AXIS_PADDING        = 48;
const GRID_LINES          = 6;

// ─── DOM refs ─────────────────────────────────────────────────────────────────

const canvas      = document.getElementById("scatter-canvas");
const ctx         = canvas.getContext("2d");
const tooltip     = document.getElementById("canvas-tooltip");
const statusDot   = document.querySelector(".status-dot");
const statusLabel = document.querySelector(".status-label");
const metricSelect = document.getElementById("metric-select");
const kSlider      = document.getElementById("k-slider");
const kValLabel    = document.getElementById("k-val-label");

// ─── Boot ─────────────────────────────────────────────────────────────────────

async function boot() {
  try {
    const res = await fetch("/api/fixture");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    fixtureData = await res.json();

    setStatus("ready", "Fixture loaded — 15 objects");
    initRepSelector();
    initMetricSelector();
    initKSlider();
    updateVarianceBars();
    initCanvasInteraction();
    selectPoint(0);
  } catch (err) {
    setStatus("error", "Failed to load fixture");
    console.error("Boot error:", err);
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
  ],
};

function updateMetricOptions() {
  const allowed = REP_METRICS[currentRep] || REP_METRICS.probability;
  metricSelect.innerHTML = allowed
    .map(m => `<option value="${m.id}">${m.label}</option>`)
    .join("");

  // Keep current metric if allowed, otherwise fall back to first
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
      currentRep = e.target.value;
      document.querySelectorAll(".radio-card").forEach(card => {
        card.classList.toggle("active", card.querySelector("input").value === currentRep);
      });
      hoveredIdx = null;
      updateMetricOptions();
      updateVarianceBars();
      fetchDiagnostics();
    });
  });
  updateMetricOptions();
}

// ─── Metric & k selectors ─────────────────────────────────────────────────────

function initMetricSelector() {
  metricSelect.addEventListener("change", (e) => {
    currentMetric = e.target.value;
    fetchDiagnostics();
  });
}

function initKSlider() {
  kSlider.addEventListener("input", (e) => {
    currentK = parseInt(e.target.value, 10);
    kValLabel.textContent = currentK;
    fetchDiagnostics();
  });
}

// ─── Variance bars ────────────────────────────────────────────────────────────

function updateVarianceBars() {
  if (!fixtureData) return;
  const rep   = fixtureData.representations[currentRep];
  const evs   = rep.eigenvalues;
  const total = evs[0] + evs[1];
  const pct1  = total > 0 ? (evs[0] / total * 100) : 50;
  const pct2  = total > 0 ? (evs[1] / total * 100) : 50;

  document.getElementById("var-pc1").style.width = pct1.toFixed(1) + "%";
  document.getElementById("var-pc2").style.width = pct2.toFixed(1) + "%";
  document.getElementById("var-pc1-label").textContent = pct1.toFixed(0) + "%";
  document.getElementById("var-pc2-label").textContent = pct2.toFixed(0) + "%";
}

// ─── Diagnostics API Fetch ────────────────────────────────────────────────────

async function fetchDiagnostics() {
  if (!fixtureData || selectedIdx === null) return;
  const targetId = fixtureData.object_ids[selectedIdx];
  const url = `/api/diagnostics?target_id=${encodeURIComponent(targetId)}&representation=${encodeURIComponent(currentRep)}&metric=${encodeURIComponent(currentMetric)}&k=${currentK}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    diagResult = await res.json();
    updateIntegrityPanel(diagResult);
    renderScatter();
  } catch (err) {
    console.error("Diagnostics fetch error:", err);
  }
}

function updateIntegrityPanel(data) {
  if (!data) return;
  document.getElementById("diag-precision").textContent = (data.precision * 100).toFixed(0) + "%";
  document.getElementById("diag-recall").textContent    = (data.recall * 100).toFixed(0) + "%";
  document.getElementById("diag-trust").textContent     = data.trustworthiness.toFixed(2);
  document.getElementById("diag-stress").textContent    = data.stress.toFixed(2);

  document.getElementById("count-preserved").textContent = data.preserved.length;
  document.getElementById("count-torn").textContent      = data.torn.length;
  document.getElementById("count-false").textContent     = data.false_neighbors.length;
}

// ─── Scatter renderer ─────────────────────────────────────────────────────────

function getCoords() {
  return fixtureData.representations[currentRep].coords;
}

function computeScale(coords) {
  const xs     = coords.map(c => c[0]);
  const ys     = coords.map(c => c[1]);
  const minX   = Math.min(...xs), maxX = Math.max(...xs);
  const minY   = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const pad    = AXIS_PADDING;
  const w      = canvas.width  - pad * 2;
  const h      = canvas.height - pad * 2;

  const scale = Math.min(w / rangeX, h / rangeY) * 0.82;
  const cx    = (minX + maxX) / 2;
  const cy    = (minY + maxY) / 2;

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

  const coords = getCoords();
  const colors = fixtureData.colors;
  const sc     = computeScale(coords);
  const dpr    = window.devicePixelRatio || 1;

  const wrap  = document.getElementById("canvas-wrap");
  const size  = Math.min(wrap.clientWidth, wrap.clientHeight, 580);
  canvas.style.width  = size + "px";
  canvas.style.height = size + "px";
  canvas.width  = size * dpr;
  canvas.height = size * dpr;
  ctx.scale(dpr, dpr);

  const sc2 = computeScale(coords);
  const W = size, H = size;

  // Background
  ctx.fillStyle = "#111827";
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth   = 1;
  for (let i = 0; i <= GRID_LINES; i++) {
    const t = i / GRID_LINES;
    const gx = sc2.pad + t * sc2.w;
    const gy = sc2.pad + t * sc2.h;
    ctx.beginPath(); ctx.moveTo(gx, sc2.pad); ctx.lineTo(gx, sc2.pad + sc2.h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sc2.pad, gy); ctx.lineTo(sc2.pad + sc2.w, gy); ctx.stroke();
  }

  // Axis labels
  ctx.font      = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(148,163,184,0.5)";
  ctx.textAlign = "center";
  ctx.fillText("PC1 →", W / 2, H - 8);
  ctx.save();
  ctx.translate(12, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("PC2 ↑", 0, 0);
  ctx.restore();

  // Rep label
  ctx.font      = "11px Inter, sans-serif";
  ctx.fillStyle = "rgba(148,163,184,0.4)";
  ctx.textAlign = "right";
  ctx.fillText(currentRep.replace("_", " "), W - 12, H - 10);

  // Simplex structure lines
  drawSimplexEdges(coords, sc2);

  // Diagnostic overlay lines from selected point
  if (selectedIdx !== null && diagResult) {
    drawDiagnosticOverlay(coords, sc2);
  }

  // Draw points
  for (let i = 0; i < coords.length; i++) {
    const [sx, sy] = sc2.toScreen(coords[i][0], coords[i][1]);
    const isSelected = i === selectedIdx;
    const isHovered  = i === hoveredIdx;
    const r    = isSelected ? POINT_RADIUS_SELECT : (isHovered ? POINT_RADIUS_HOVER : POINT_RADIUS);
    const col  = colors[i];

    if (isSelected || isHovered) {
      ctx.shadowColor = isSelected ? "#34d399" : col;
      ctx.shadowBlur  = 16;
    } else {
      ctx.shadowBlur = 0;
    }

    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();

    ctx.strokeStyle = isSelected ? "#34d399" : (isHovered ? "#ffffff" : "rgba(255,255,255,0.15)");
    ctx.lineWidth   = isSelected ? 2.5 : (isHovered ? 1.5 : 0.8);
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // ID label on hover or select
  const activeIdx = hoveredIdx !== null ? hoveredIdx : selectedIdx;
  if (activeIdx !== null) {
    const [sx, sy] = sc2.toScreen(coords[activeIdx][0], coords[activeIdx][1]);
    const id = fixtureData.object_ids[activeIdx];
    ctx.font      = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.textAlign = "center";
    const labelY = sy - POINT_RADIUS_HOVER - 6;
    ctx.fillText(id, sx, labelY);
  }
}

function drawSimplexEdges(coords, sc) {
  const edgePairs = [[0, 3], [0, 4], [1, 3], [1, 5], [2, 4], [2, 5], [3, 6], [4, 6], [5, 6]];
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth   = 0.8;
  edgePairs.forEach(([a, b]) => {
    if (a >= coords.length || b >= coords.length) return;
    const [ax, ay] = sc.toScreen(coords[a][0], coords[a][1]);
    const [bx, by] = sc.toScreen(coords[b][0], coords[b][1]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  });
}

function drawDiagnosticOverlay(coords, sc) {
  const [srcX, srcY] = sc.toScreen(coords[selectedIdx][0], coords[selectedIdx][1]);
  const idToIdx = new Map(fixtureData.object_ids.map((id, idx) => [id, idx]));

  // Helper to draw link line
  const drawLink = (targetId, color, dashPattern) => {
    const tIdx = idToIdx.get(targetId);
    if (tIdx === undefined) return;
    const [tx, ty] = sc.toScreen(coords[tIdx][0], coords[tIdx][1]);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.0;
    ctx.setLineDash(dashPattern);
    ctx.beginPath();
    ctx.moveTo(srcX, srcY);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.restore();
  };

  // Preserved (Solid Green)
  diagResult.preserved.forEach(id => drawLink(id, "#34d399", []));

  // Torn (Dashed Red)
  diagResult.torn.forEach(id => drawLink(id, "#f87171", [6, 4]));

  // False (Dotted Amber)
  diagResult.false_neighbors.forEach(id => drawLink(id, "#fbbf24", [2, 3]));
}

// ─── Canvas interaction ───────────────────────────────────────────────────────

function initCanvasInteraction() {
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mouseleave", () => {
    hoveredIdx = null;
    tooltip.classList.add("hidden");
    renderScatter();
  });
  canvas.addEventListener("click", onMouseClick);
  window.addEventListener("resize", () => renderScatter());
}

function getHitIndex(clientX, clientY) {
  if (!fixtureData) return null;
  const rect   = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width  / (window.devicePixelRatio || 1);
  const scaleY = canvas.height / rect.height / (window.devicePixelRatio || 1);
  const mx = (clientX - rect.left) * scaleX;
  const my = (clientY - rect.top)  * scaleY;

  const coords = getCoords();
  const sc     = computeScale(coords);

  for (let i = coords.length - 1; i >= 0; i--) {
    const [sx, sy] = sc.toScreen(coords[i][0], coords[i][1]);
    const dist = Math.hypot(mx - sx, my - sy);
    if (dist <= POINT_RADIUS_HOVER + 4) return i;
  }
  return null;
}

function onMouseMove(e) {
  const idx = getHitIndex(e.clientX, e.clientY);
  if (idx !== hoveredIdx) {
    hoveredIdx = idx;
    renderScatter();
  }

  if (idx !== null) {
    const id   = fixtureData.object_ids[idx];
    const raw  = fixtureData.raw_matrix[idx];
    tooltip.innerHTML = `<strong>${id}</strong> &nbsp; [${raw.map(v => v.toFixed(3)).join(", ")}]`;
    const rect = canvas.getBoundingClientRect();
    const tx = e.clientX - rect.left + 14;
    const ty = e.clientY - rect.top  - 12;
    tooltip.style.left = tx + "px";
    tooltip.style.top  = ty + "px";
    tooltip.classList.remove("hidden");
    canvas.style.cursor = "pointer";
  } else {
    tooltip.classList.add("hidden");
    canvas.style.cursor = "crosshair";
  }
}

function onMouseClick(e) {
  const idx = getHitIndex(e.clientX, e.clientY);
  if (idx === null) return;
  selectPoint(idx);
}

function selectPoint(idx) {
  selectedIdx = idx;
  updateInspector(idx);
  fetchDiagnostics();
}

// ─── Source inspector ─────────────────────────────────────────────────────────

function updateInspector(idx) {
  const id     = fixtureData.object_ids[idx];
  const raw    = fixtureData.raw_matrix[idx];
  const coords = getCoords()[idx];

  document.getElementById("insp-id").textContent  = id;
  document.getElementById("insp-p0").textContent  = raw[0].toFixed(4);
  document.getElementById("insp-p1").textContent  = raw[1].toFixed(4);
  document.getElementById("insp-p2").textContent  = raw[2].toFixed(4);
  document.getElementById("insp-pc1").textContent = coords[0].toFixed(4);
  document.getElementById("insp-pc2").textContent = coords[1].toFixed(4);
}

// ─── Go ───────────────────────────────────────────────────────────────────────

boot();

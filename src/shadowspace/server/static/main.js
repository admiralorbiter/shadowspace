/**
 * Shadowspace Workbench — Sprint 3b
 * main.js — Fetches fixture data, renders PCA scatter on canvas,
 * handles representation switching and point inspection.
 */

"use strict";

// ─── State ────────────────────────────────────────────────────────────────────

let fixtureData = null;          // raw JSON from /api/fixture
let currentRep  = "probability"; // active representation key
let hoveredIdx  = null;          // currently hovered point index

const POINT_RADIUS        = 7;
const POINT_RADIUS_HOVER  = 10;
const POINT_RADIUS_SELECT = 10;
const AXIS_PADDING        = 48; // px from canvas edge to data area
const GRID_LINES          = 6;

// ─── DOM refs ─────────────────────────────────────────────────────────────────

const canvas      = document.getElementById("scatter-canvas");
const ctx         = canvas.getContext("2d");
const tooltip     = document.getElementById("canvas-tooltip");
const statusDot   = document.querySelector(".status-dot");
const statusLabel = document.querySelector(".status-label");

// ─── Boot ─────────────────────────────────────────────────────────────────────

async function boot() {
  try {
    const res = await fetch("/api/fixture");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    fixtureData = await res.json();

    setStatus("ready", "Fixture loaded — 15 objects");
    initRepSelector();
    updateVarianceBars();
    renderScatter();
    initCanvasInteraction();
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

function initRepSelector() {
  document.querySelectorAll('input[name="representation"]').forEach(radio => {
    radio.addEventListener("change", (e) => {
      currentRep = e.target.value;
      // Update card active state
      document.querySelectorAll(".radio-card").forEach(card => {
        card.classList.toggle("active", card.querySelector("input").value === currentRep);
      });
      hoveredIdx = null;
      clearInspector();
      updateVarianceBars();
      renderScatter();
    });
  });
}

// ─── Variance bars ────────────────────────────────────────────────────────────

function updateVarianceBars() {
  if (!fixtureData) return;
  const rep  = fixtureData.representations[currentRep];
  const evs  = rep.eigenvalues;  // [λ1, λ2]
  const total = evs[0] + evs[1];
  const pct1  = total > 0 ? (evs[0] / total * 100) : 50;
  const pct2  = total > 0 ? (evs[1] / total * 100) : 50;

  document.getElementById("var-pc1").style.width = pct1.toFixed(1) + "%";
  document.getElementById("var-pc2").style.width = pct2.toFixed(1) + "%";
  document.getElementById("var-pc1-label").textContent = pct1.toFixed(0) + "%";
  document.getElementById("var-pc2-label").textContent = pct2.toFixed(0) + "%";
}

// ─── Scatter renderer ─────────────────────────────────────────────────────────

function getCoords() {
  return fixtureData.representations[currentRep].coords; // [[x,y], ...]
}

function computeScale(coords) {
  const xs  = coords.map(c => c[0]);
  const ys  = coords.map(c => c[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const pad = AXIS_PADDING;
  const w   = canvas.width  - pad * 2;
  const h   = canvas.height - pad * 2;

  // Uniform scale to keep aspect ratio
  const scale = Math.min(w / rangeX, h / rangeY) * 0.82;
  const cx    = (minX + maxX) / 2;
  const cy    = (minY + maxY) / 2;

  return {
    toScreen: (x, y) => [
      pad + w / 2 + (x - cx) * scale,
      pad + h / 2 - (y - cy) * scale,   // flip y
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

  // Resize canvas to container
  const wrap  = document.getElementById("canvas-wrap");
  const size  = Math.min(wrap.clientWidth, wrap.clientHeight, 580);
  canvas.style.width  = size + "px";
  canvas.style.height = size + "px";
  canvas.width  = size * dpr;
  canvas.height = size * dpr;
  ctx.scale(dpr, dpr);

  // Recompute scale with new canvas size
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

  // Axis labels (PC1, PC2)
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

  // Draw edges between nearest-ish points (optional: simplex structure lines)
  drawSimplexEdges(coords, sc2, colors);

  // Draw points
  for (let i = 0; i < coords.length; i++) {
    const [sx, sy] = sc2.toScreen(coords[i][0], coords[i][1]);
    const r    = i === hoveredIdx ? POINT_RADIUS_HOVER : POINT_RADIUS;
    const col  = colors[i];
    const isHovered = i === hoveredIdx;

    // Glow on hover
    if (isHovered) {
      ctx.shadowColor = col;
      ctx.shadowBlur  = 16;
    } else {
      ctx.shadowBlur = 0;
    }

    // Fill
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();

    // Border
    ctx.strokeStyle = isHovered ? "#ffffff" : "rgba(255,255,255,0.15)";
    ctx.lineWidth   = isHovered ? 1.5 : 0.8;
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // ID label on hover
  if (hoveredIdx !== null) {
    const [sx, sy] = sc2.toScreen(coords[hoveredIdx][0], coords[hoveredIdx][1]);
    const id = fixtureData.object_ids[hoveredIdx];
    ctx.font      = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.textAlign = "center";
    const labelY = sy - POINT_RADIUS_HOVER - 6;
    ctx.fillText(id, sx, labelY);
  }
}

function drawSimplexEdges(coords, sc, colors) {
  // Connect corners, midpoints, center with faint lines for structure
  const cornerIdx   = [0, 1, 2];
  const edgePairs   = [[0, 3], [0, 4], [1, 3], [1, 5], [2, 4], [2, 5], [3, 6], [4, 6], [5, 6]];

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
    // Position tooltip near cursor
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
  updateInspector(idx);
}

// ─── Source inspector ─────────────────────────────────────────────────────────

function updateInspector(idx) {
  const id    = fixtureData.object_ids[idx];
  const raw   = fixtureData.raw_matrix[idx];
  const coords = getCoords()[idx];

  document.getElementById("insp-id").textContent  = id;
  document.getElementById("insp-p0").textContent  = raw[0].toFixed(4);
  document.getElementById("insp-p1").textContent  = raw[1].toFixed(4);
  document.getElementById("insp-p2").textContent  = raw[2].toFixed(4);
  document.getElementById("insp-pc1").textContent = coords[0].toFixed(4);
  document.getElementById("insp-pc2").textContent = coords[1].toFixed(4);

  // Highlight
  document.querySelectorAll(".inspector-list dd").forEach(el => el.classList.add("highlight"));
  setTimeout(() => {
    document.querySelectorAll(".inspector-list dd").forEach(el => el.classList.remove("highlight"));
  }, 800);
}

function clearInspector() {
  ["insp-id", "insp-p0", "insp-p1", "insp-p2", "insp-pc1", "insp-pc2"].forEach(id => {
    document.getElementById(id).textContent = "—";
  });
}

// ─── Go ───────────────────────────────────────────────────────────────────────

boot();

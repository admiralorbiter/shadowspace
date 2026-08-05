"""Build JSON payload and self-contained interactive Doppelgänger Atlas HTML visualizer."""

import os
import json
import polars as pl

CANONICAL_PATH = "data/chaosnli/processed/canonical_items.parquet"
STRICT_PAIRS_PATH = "results/ambiguity_atlas/strict_pairs.parquet"
POSTERIOR_PATH = "results/ambiguity_atlas/posterior_stability.parquet"
RETENTION_PATH = "results/ambiguity_atlas/model_retention.parquet"
OUTPUT_PAYLOAD_PATH = "results/ambiguity_atlas/atlas_payload.json"
WEB_PAYLOAD_PATH = "docs/viz/ambiguity_atlas/atlas_payload.json"
WEB_HTML_PATH = "docs/viz/ambiguity_atlas/index.html"


def get_html_template(payload_json_str: str) -> str:
    """Return complete self-contained HTML visualizer with responsive CSS and error-safe JS."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ambiguity Doppelgänger Atlas: What Confidence & Entropy Cannot Tell You</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script>
        window.ATLAS_PAYLOAD = {payload_json_str};
    </script>
    <style>
        :root {{
            --bg-dark: #0b0f19;
            --panel-bg: rgba(17, 24, 39, 0.85);
            --panel-border: rgba(255, 255, 255, 0.12);
            --accent-cyan: #38bdf8;
            --accent-indigo: #818cf8;
            --accent-emerald: #34d399;
            --accent-amber: #fbbf24;
            --accent-rose: #f43f5e;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --font-title: 'Outfit', sans-serif;
            --font-body: 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: var(--font-body);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.12) 0%, transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(129, 140, 248, 0.12) 0%, transparent 45%);
        }}

        header {{
            padding: 12px 20px;
            background: rgba(11, 15, 25, 0.9);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--panel-border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }}

        .header-title h1 {{
            font-family: var(--font-title);
            font-size: 1.3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header-title p {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .mode-toggle {{
            display: flex;
            gap: 6px;
            background: rgba(0, 0, 0, 0.4);
            padding: 4px;
            border-radius: 8px;
            border: 1px solid var(--panel-border);
        }}
        .mode-btn {{
            padding: 6px 14px;
            border-radius: 6px;
            border: none;
            background: transparent;
            color: var(--text-muted);
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .mode-btn.active {{
            background: var(--accent-indigo);
            color: #fff;
            box-shadow: 0 2px 8px rgba(129, 140, 248, 0.4);
        }}

        main {{
            display: grid;
            grid-template-columns: 280px minmax(360px, 1fr) 380px;
            flex: 1;
            overflow: hidden;
            min-height: 0;
        }}

        /* Left Panel */
        .left-panel {{
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            border-right: 1px solid var(--panel-border);
            display: flex;
            flex-direction: column;
            padding: 14px;
            gap: 12px;
            overflow: hidden;
        }}

        .filter-section h3 {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }}

        select, input[type="range"] {{
            width: 100%;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid var(--panel-border);
            color: var(--text-main);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 0.82rem;
        }}

        .pair-list {{
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding-right: 2px;
        }}
        .pair-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 8px 10px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .pair-card:hover {{
            border-color: var(--accent-cyan);
            background: rgba(56, 189, 248, 0.08);
        }}
        .pair-card.active {{
            border-color: var(--accent-indigo);
            background: rgba(129, 140, 248, 0.18);
            box-shadow: 0 0 10px rgba(129, 140, 248, 0.25);
        }}

        .pair-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            margin-bottom: 2px;
        }}
        .badge {{
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-robust {{ background: rgba(52, 211, 153, 0.2); color: var(--accent-emerald); }}
        .badge-probable {{ background: rgba(56, 189, 248, 0.2); color: var(--accent-cyan); }}
        .badge-uncertain {{ background: rgba(251, 191, 36, 0.2); color: var(--accent-amber); }}
        .badge-point {{ background: rgba(244, 63, 94, 0.2); color: var(--accent-rose); }}

        .pair-title {{
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-main);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        /* Center Simplex View */
        .center-panel {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 16px;
            position: relative;
            min-width: 0;
            overflow: hidden;
        }}

        .simplex-container {{
            width: 100%;
            height: 100%;
            max-width: 580px;
            max-height: 500px;
            position: relative;
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 12px;
            box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.6);
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        svg.simplex-svg {{
            width: 100%;
            height: 100%;
            max-height: 480px;
        }}

        .simplex-legend {{
            position: absolute;
            bottom: 14px;
            left: 14px;
            background: rgba(11, 15, 25, 0.92);
            backdrop-filter: blur(8px);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 4px;
            pointer-events: none;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-dot {{
            height: 8px;
            width: 8px;
            border-radius: 50%;
        }}

        /* Right Panel Inspector */
        .right-panel {{
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            border-left: 1px solid var(--panel-border);
            display: flex;
            flex-direction: column;
            padding: 16px;
            gap: 12px;
            overflow-y: auto;
        }}

        .reveal-banner {{
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.15));
            border: 1px solid rgba(129, 140, 248, 0.3);
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .reveal-btn {{
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-indigo));
            border: none;
            color: #fff;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.8rem;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(56, 189, 248, 0.3);
        }}

        .item-card {{
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .item-tag {{
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--accent-cyan);
            letter-spacing: 0.5px;
        }}
        .item-text {{
            font-size: 0.82rem;
            line-height: 1.35;
        }}

        .summary-box {{
            background: rgba(0, 0, 0, 0.45);
            border: 1px dashed var(--panel-border);
            border-radius: 8px;
            padding: 10px 12px;
            font-family: var(--font-mono);
            font-size: 0.78rem;
            color: var(--accent-amber);
            line-height: 1.45;
        }}

        .distribution-bar {{
            display: flex;
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 4px;
        }}
        .bar-e {{ background: var(--accent-emerald); }}
        .bar-n {{ background: var(--accent-indigo); }}
        .bar-c {{ background: var(--accent-rose); }}

        /* Game Mode Styles */
        .game-container {{
            display: none;
            flex-direction: column;
            gap: 14px;
            align-items: center;
            justify-content: center;
            height: 100%;
        }}
        .game-card {{
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 20px;
            width: 100%;
            max-width: 480px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            text-align: center;
        }}
        .game-options {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .game-opt-btn {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--panel-border);
            color: var(--text-main);
            padding: 10px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .game-opt-btn:hover {{
            border-color: var(--accent-cyan);
            background: rgba(56, 189, 248, 0.1);
        }}

        .hidden-eval {{ display: none; }}
    </style>
</head>
<body>

<header>
    <div class="header-title">
        <h1>Ambiguity Doppelgänger Atlas</h1>
        <p>What Confidence & Shannon Entropy Cannot Tell You About Human Disagreement</p>
    </div>
    <div class="mode-toggle">
        <button class="mode-btn active" id="btn-mode-explorer" onclick="setMode('explorer')">Atlas Explorer</button>
        <button class="mode-btn" id="btn-mode-game" onclick="setMode('game')">Game Mode</button>
    </div>
</header>

<main>
    <!-- Left Filter Panel -->
    <div class="left-panel">
        <div class="filter-section">
            <h3>Majority Class</h3>
            <select id="select-majority" onchange="applyFilters()">
                <option value="ALL">All Classes</option>
                <option value="entailment">Entailment</option>
                <option value="neutral">Neutral</option>
                <option value="contradiction">Contradiction</option>
            </select>
        </div>

        <div class="filter-section">
            <h3>Posterior Stability</h3>
            <select id="select-stability" onchange="applyFilters()">
                <option value="ALL">All Categories</option>
                <option value="ROBUST_COLLISION">Robust Collision (>= 90%)</option>
                <option value="PROBABLE_COLLISION">Probable Collision (>= 80%)</option>
                <option value="UNCERTAIN_COLLISION">Uncertain Collision</option>
                <option value="POINT_ESTIMATE_ONLY">Point Estimate Only</option>
            </select>
        </div>

        <div class="filter-section">
            <h3>Hellinger Separation: <span id="dist-val">0.00</span></h3>
            <input type="range" id="range-dist" min="0.0" max="0.7" step="0.05" value="0.0" oninput="document.getElementById('dist-val').innerText = this.value; applyFilters();">
        </div>

        <div class="filter-section">
            <h3>Doppelgänger Pair Index (<span id="pair-count">0</span>)</h3>
        </div>

        <div class="pair-list" id="pair-list-container">
            <!-- Dynamic pair cards -->
        </div>
    </div>

    <!-- Center Simplex Canvas -->
    <div class="center-panel" id="explorer-view">
        <div class="simplex-container">
            <svg class="simplex-svg" viewBox="0 0 600 520" id="simplex-svg">
                <!-- Iso-entropy Contours -->
                <circle cx="300" cy="310" r="80" fill="none" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3,3" />
                <circle cx="300" cy="310" r="140" fill="none" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3,3" />
                <circle cx="300" cy="310" r="200" fill="none" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3,3" />

                <!-- Simplex Outer Boundary Triangle -->
                <polygon points="300,50 80,440 520,440" fill="rgba(17, 24, 39, 0.75)" stroke="rgba(255, 255, 255, 0.35)" stroke-width="2.5" />

                <!-- Axis Altitude Lines -->
                <line x1="300" y1="50" x2="300" y2="440" stroke="rgba(255, 255, 255, 0.3)" stroke-dasharray="4,4" stroke-width="1.5" />
                <line x1="80" y1="440" x2="410" y2="245" stroke="rgba(255, 255, 255, 0.12)" stroke-dasharray="4,4" />
                <line x1="520" y1="440" x2="190" y2="245" stroke="rgba(255, 255, 255, 0.12)" stroke-dasharray="4,4" />

                <!-- Vertex Badges -->
                <circle cx="300" cy="50" r="12" fill="#34d399" />
                <text x="300" y="28" fill="#34d399" font-size="14" font-weight="800" text-anchor="middle">ENTAILMENT (E)</text>

                <circle cx="80" cy="440" r="12" fill="#818cf8" />
                <text x="65" y="470" fill="#818cf8" font-size="14" font-weight="800" text-anchor="end">NEUTRAL (N)</text>

                <circle cx="520" cy="440" r="12" fill="#f43f5e" />
                <text x="535" y="470" fill="#f43f5e" font-size="14" font-weight="800" text-anchor="start">CONTRADICTION (C)</text>

                <!-- Dynamic Doppelgänger Pair Overlay -->
                <line id="svg-line" stroke="#38bdf8" stroke-width="3.5" stroke-dasharray="4,3" opacity="0" />
                
                <circle id="svg-pt-a" r="10" fill="#38bdf8" stroke="#ffffff" stroke-width="2.5" opacity="0" />
                <text id="svg-txt-a" fill="#38bdf8" font-size="13" font-weight="800" opacity="0">ITEM A</text>

                <circle id="svg-pt-b" r="10" fill="#f43f5e" stroke="#ffffff" stroke-width="2.5" opacity="0" />
                <text id="svg-txt-b" fill="#f43f5e" font-size="13" font-weight="800" opacity="0">ITEM B</text>
            </svg>
            <div class="simplex-legend">
                <div style="font-weight: 700; color: var(--accent-cyan); margin-bottom: 2px;">Simplex Topology & Legend</div>
                <div class="legend-item"><div class="legend-dot" style="background: var(--accent-cyan)"></div> Item A (High Minority Class 1)</div>
                <div class="legend-item"><div class="legend-dot" style="background: var(--accent-rose)"></div> Item B (High Minority Class 2)</div>
                <div class="legend-item"><div class="legend-dot" style="background: rgba(255,255,255,0.4)"></div> Mirror Altitude Axis (δ = 0)</div>
            </div>
        </div>
    </div>

    <!-- Game Mode View -->
    <div class="center-panel game-container" id="game-view">
        <div class="game-card">
            <h2 style="font-family: var(--font-title); color: var(--accent-cyan);">Doppelgänger Diagnostic Challenge</h2>
            <p style="font-size: 0.88rem; color: var(--text-muted);">Compare these two summary cards from standard evaluation dashboards:</p>
            <div class="summary-box" id="game-summary-box">
                Dashboard Summary Card:<br>
                Majority: Entailment<br>
                Confidence: 60.0%<br>
                Entropy: 1.295 bits
            </div>
            <p style="font-size: 0.9rem; font-weight: 500;">Do items A and B express the exact same human disagreement structure?</p>
            <div class="game-options">
                <button class="game-opt-btn" onclick="answerGame('same')">Yes, identical disagreement</button>
                <button class="game-opt-btn" onclick="answerGame('diff')">No, opposite minority orientation</button>
                <button class="game-opt-btn" onclick="answerGame('unknown')">Not enough information in 1D summary</button>
            </div>
            <div id="game-result" style="font-size: 0.88rem; margin-top: 6px;"></div>
        </div>
    </div>

    <!-- Right Inspector Panel -->
    <div class="right-panel">
        <div class="reveal-banner">
            <div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">Summary Dashboard Mode</div>
                <div style="font-size: 0.88rem; font-weight: 700;" id="reveal-status">1D Summary Active</div>
            </div>
            <button class="reveal-btn" id="btn-reveal" onclick="toggleReveal()">Reveal Geometry</button>
        </div>

        <!-- 1D Summary Card -->
        <div class="summary-box" id="dashboard-card">
            -- Select a Doppelgänger Pair --
        </div>

        <!-- Item A -->
        <div class="item-card">
            <div class="item-tag" id="tag-item-a">ITEM A (ID: --)</div>
            <div class="item-text" id="text-premise-a"><strong>Premise:</strong> Select a pair...</div>
            <div class="item-text" id="text-hypo-a"><strong>Hypothesis:</strong> Select a pair...</div>
            <div class="hidden-eval" id="eval-a">
                <div style="font-size: 0.78rem; margin-top: 4px;">Vote Distribution: <span id="counts-a">--</span></div>
                <div class="distribution-bar">
                    <div class="bar-e" id="bar-a-e" style="width: 33%"></div>
                    <div class="bar-n" id="bar-a-n" style="width: 33%"></div>
                    <div class="bar-c" id="bar-a-c" style="width: 33%"></div>
                </div>
            </div>
        </div>

        <!-- Item B -->
        <div class="item-card">
            <div class="item-tag" id="tag-item-b">ITEM B (ID: --)</div>
            <div class="item-text" id="text-premise-b"><strong>Premise:</strong> Select a pair...</div>
            <div class="item-text" id="text-hypo-b"><strong>Hypothesis:</strong> Select a pair...</div>
            <div class="hidden-eval" id="eval-b">
                <div style="font-size: 0.78rem; margin-top: 4px;">Vote Distribution: <span id="counts-b">--</span></div>
                <div class="distribution-bar">
                    <div class="bar-e" id="bar-b-e" style="width: 33%"></div>
                    <div class="bar-n" id="bar-b-n" style="width: 33%"></div>
                    <div class="bar-c" id="bar-b-c" style="width: 33%"></div>
                </div>
            </div>
        </div>

        <!-- Metrics Inspection -->
        <div class="item-card hidden-eval" id="metrics-card">
            <div class="item-tag">GEOMETRIC & STABILITY DIAGNOSTICS</div>
            <div style="font-size: 0.82rem; display: flex; flex-direction: column; gap: 4px;">
                <div>Hellinger Distance: <strong id="val-dh">--</strong></div>
                <div>Posterior Stability: <strong id="val-post">--</strong></div>
                <div>Orientation Retention (RoBERTa): <strong id="val-model-ret">--</strong></div>
            </div>
        </div>
    </div>
</main>

<script>
    let atlasData = window.ATLAS_PAYLOAD;
    let filteredPairs = [];
    let currentPair = null;
    let isRevealed = false;

    // Simplex vertices in SVG coordinates
    const V_E = {{ x: 300, y: 50 }};
    const V_N = {{ x: 80, y: 440 }};
    const V_C = {{ x: 520, y: 440 }};

    function escapeText(str) {{
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }}

    function probToSvg(p) {{
        if (!p || p.length < 3) return {{ x: 300, y: 250 }};
        const x = p[0] * V_E.x + p[1] * V_N.x + p[2] * V_C.x;
        const y = p[0] * V_E.y + p[1] * V_N.y + p[2] * V_C.y;
        return {{ x, y }};
    }}

    function initData() {{
        try {{
            if (!atlasData || !atlasData.pairs) return;
            applyFilters();
            if (filteredPairs.length > 0) {{
                selectPair(filteredPairs[0].pair_id);
            }}
        }} catch (err) {{
            console.error("Error initializing atlas data:", err);
        }}
    }}

    function setMode(mode) {{
        document.getElementById('btn-mode-explorer').classList.toggle('active', mode === 'explorer');
        document.getElementById('btn-mode-game').classList.toggle('active', mode === 'game');
        document.getElementById('explorer-view').style.display = (mode === 'explorer') ? 'flex' : 'none';
        document.getElementById('game-view').style.display = (mode === 'game') ? 'flex' : 'none';
    }}

    function applyFilters() {{
        if (!atlasData || !atlasData.pairs) return;
        const maj = document.getElementById('select-majority').value;
        const stab = document.getElementById('select-stability').value;
        const minDist = parseFloat(document.getElementById('range-dist').value);

        filteredPairs = atlasData.pairs.filter(p => {{
            if (maj !== 'ALL' && p.majority_label !== maj) return false;
            if (stab !== 'ALL' && p.posterior.stability_category !== stab) return false;
            if (p.d_hellinger < minDist) return false;
            return true;
        }});

        document.getElementById('pair-count').innerText = filteredPairs.length;
        renderPairList();
        renderSimplex();
    }}

    function renderPairList() {{
        const container = document.getElementById('pair-list-container');
        if (!container) return;
        container.innerHTML = '';

        filteredPairs.slice(0, 100).forEach(pair => {{
            const card = document.createElement('div');
            card.className = `pair-card ${{currentPair && currentPair.pair_id === pair.pair_id ? 'active' : ''}}`;
            card.onclick = () => selectPair(pair.pair_id);

            const badgeClass = {{
                'ROBUST_COLLISION': 'badge-robust',
                'PROBABLE_COLLISION': 'badge-probable',
                'UNCERTAIN_COLLISION': 'badge-uncertain',
                'POINT_ESTIMATE_ONLY': 'badge-point',
            }}[pair.posterior.stability_category] || 'badge-point';

            const cleanPremise = escapeText(pair.item_a.premise);
            const stabText = escapeText(pair.posterior.stability_category.replace('_COLLISION', ''));

            card.innerHTML = `
                <div class="pair-header">
                    <span class="badge ${{badgeClass}}">${{stabText}}</span>
                    <span style="color: var(--accent-cyan); font-weight: 600;">d_H: ${{pair.d_hellinger.toFixed(3)}}</span>
                </div>
                <div class="pair-title">${{cleanPremise}}</div>
            `;
            container.appendChild(card);
        }});
    }}

    function selectPair(pairId) {{
        if (!atlasData || !atlasData.pairs) return;
        currentPair = atlasData.pairs.find(p => p.pair_id === pairId);
        renderPairList();
        updateInspector();
        renderSimplex();
    }}

    function toggleReveal() {{
        isRevealed = !isRevealed;
        document.getElementById('btn-reveal').innerText = isRevealed ? 'Hide Geometry' : 'Reveal Geometry';
        document.getElementById('reveal-status').innerText = isRevealed ? 'Full 2D Simplex Geometry' : '1D Summary Active';
        
        document.querySelectorAll('.hidden-eval').forEach(el => {{
            el.style.display = isRevealed ? 'block' : 'none';
        }});
        renderSimplex();
    }}

    function updateInspector() {{
        if (!currentPair) return;
        const p = currentPair;
        
        document.getElementById('dashboard-card').innerHTML = `
            DASHBOARD SUMMARY CARD<br>
            ---------------------------<br>
            Majority Class: <strong style="color:#fff">${{escapeText(p.majority_label.toUpperCase())}}</strong><br>
            Confidence: <strong style="color:#fff">${{(p.majority_probability * 100).toFixed(1)}}%</strong> (${{p.majority_count}} votes)<br>
            Shannon Entropy: <strong style="color:#fff">${{p.entropy_bits.toFixed(3)}} bits</strong>
        `;

        document.getElementById('tag-item-a').innerText = `ITEM A (${{escapeText(p.item_a.source.toUpperCase())}}: ${{escapeText(p.item_a.object_id)}})`;
        document.getElementById('text-premise-a').innerHTML = `<strong>Premise:</strong> ${{escapeText(p.item_a.premise)}}`;
        document.getElementById('text-hypo-a').innerHTML = `<strong>Hypothesis:</strong> ${{escapeText(p.item_a.hypothesis)}}`;
        document.getElementById('counts-a').innerText = `E: ${{p.item_a.counts[0]}} | N: ${{p.item_a.counts[1]}} | C: ${{p.item_a.counts[2]}}`;
        
        const sumA = p.item_a.counts.reduce((a,b)=>a+b, 0) || 1;
        document.getElementById('bar-a-e').style.width = (p.item_a.counts[0]/sumA * 100) + '%';
        document.getElementById('bar-a-n').style.width = (p.item_a.counts[1]/sumA * 100) + '%';
        document.getElementById('bar-a-c').style.width = (p.item_a.counts[2]/sumA * 100) + '%';

        document.getElementById('tag-item-b').innerText = `ITEM B (${{escapeText(p.item_b.source.toUpperCase())}}: ${{escapeText(p.item_b.object_id)}})`;
        document.getElementById('text-premise-b').innerHTML = `<strong>Premise:</strong> ${{escapeText(p.item_b.premise)}}`;
        document.getElementById('text-hypo-b').innerHTML = `<strong>Hypothesis:</strong> ${{escapeText(p.item_b.hypothesis)}}`;
        document.getElementById('counts-b').innerText = `E: ${{p.item_b.counts[0]}} | N: ${{p.item_b.counts[1]}} | C: ${{p.item_b.counts[2]}}`;
        
        const sumB = p.item_b.counts.reduce((a,b)=>a+b, 0) || 1;
        document.getElementById('bar-b-e').style.width = (p.item_b.counts[0]/sumB * 100) + '%';
        document.getElementById('bar-b-n').style.width = (p.item_b.counts[1]/sumB * 100) + '%';
        document.getElementById('bar-b-c').style.width = (p.item_b.counts[2]/sumB * 100) + '%';

        document.getElementById('val-dh').innerText = `${{p.d_hellinger.toFixed(3)}} (Fisher-Rao: ${{p.d_fisher_rao.toFixed(3)}})`;
        document.getElementById('val-post').innerText = `${{escapeText(p.posterior.stability_category)}} (Pr[Same Maj]: ${{(p.posterior.prob_same_majority * 100).toFixed(0)}}%)`;
        
        let robStr = 'N/A';
        try {{
            if (p.models && p.models['roberta-large'] && p.models['roberta-large']['raw']) {{
                const rob = p.models['roberta-large']['raw'];
                const rVal = (typeof rob.retention_ratio === 'number') ? rob.retention_ratio.toFixed(2) : 'N/A';
                robStr = `${{escapeText(rob.retention_category)}} (R = ${{rVal}})`;
            }}
        }} catch (e) {{ console.warn("Model metrics format warning:", e); }}
        
        document.getElementById('val-model-ret').innerText = robStr;
    }}

    function renderSimplex() {{
        try {{
            const line = document.getElementById('svg-line');
            const ptA = document.getElementById('svg-pt-a');
            const txtA = document.getElementById('svg-txt-a');
            const ptB = document.getElementById('svg-pt-b');
            const txtB = document.getElementById('svg-txt-b');

            if (!line || !ptA || !ptB) return;

            if (currentPair) {{
                const pA = probToSvg(currentPair.item_a.p);
                const pB = probToSvg(currentPair.item_b.p);

                line.setAttribute('x1', pA.x);
                line.setAttribute('y1', pA.y);
                line.setAttribute('x2', pB.x);
                line.setAttribute('y2', pB.y);
                line.setAttribute('opacity', '1');

                ptA.setAttribute('cx', pA.x);
                ptA.setAttribute('cy', pA.y);
                ptA.setAttribute('opacity', '1');

                txtA.setAttribute('x', pA.x - 15);
                txtA.setAttribute('y', pA.y - 15);
                txtA.setAttribute('opacity', '1');

                ptB.setAttribute('cx', pB.x);
                ptB.setAttribute('cy', pB.y);
                ptB.setAttribute('opacity', '1');

                txtB.setAttribute('x', pB.x + 15);
                txtB.setAttribute('y', pB.y - 15);
                txtB.setAttribute('opacity', '1');
            }} else {{
                line.setAttribute('opacity', '0');
                ptA.setAttribute('opacity', '0');
                txtA.setAttribute('opacity', '0');
                ptB.setAttribute('opacity', '0');
                txtB.setAttribute('opacity', '0');
            }}
        }} catch (err) {{
            console.error("Error rendering simplex:", err);
        }}
    }}

    function answerGame(choice) {{
        const resDiv = document.getElementById('game-result');
        if (choice === 'unknown') {{
            resDiv.innerHTML = `<strong style="color: var(--accent-emerald);">CORRECT!</strong> A 1D summary (Majority, Confidence, Entropy) maps 2-to-1 and cannot distinguish whether minority disagreement favors Neutral vs Contradiction.`;
        }} else {{
            resDiv.innerHTML = `<strong style="color: var(--accent-rose);">INCORRECT.</strong> Look closer: 1D scalar summaries discard the signed minority orientation bit \\delta.`;
        }}
    }}

    document.addEventListener("DOMContentLoaded", initData);
</script>
</body>
</html>
"""


def build_atlas_payload():
    """Package dataset, posterior, and model outputs into a single JSON payload."""
    print("=== Building Doppelgänger Atlas Interactive Payload & Inlined HTML ===")
    
    df_canon = pl.read_parquet(CANONICAL_PATH)
    df_pairs = pl.read_parquet(STRICT_PAIRS_PATH)
    df_post = pl.read_parquet(POSTERIOR_PATH)
    df_ret = pl.read_parquet(RETENTION_PATH)
    
    canon_dict = {row["object_id"]: row for row in df_canon.to_dicts()}
    post_dict = {row["pair_id"]: row for row in df_post.to_dicts()}
    
    ret_dict = {}
    for row in df_ret.to_dicts():
        ret_dict[(row["pair_id"], row["model_name"], row["tier"])] = row

    pairs_list = []
    
    for pair in df_pairs.to_dicts():
        p_id = pair["pair_id"]
        item_a = canon_dict[pair["object_id_a"]]
        item_b = canon_dict[pair["object_id_b"]]
        post_data = post_dict.get(p_id, {})
        
        models_data = {}
        for (p_key, model_name, tier), ret_row in ret_dict.items():
            if p_key == p_id:
                if model_name not in models_data:
                    models_data[model_name] = {}
                models_data[model_name][tier] = {
                    "retention_ratio": ret_row["retention_ratio"],
                    "dh_model": ret_row["dh_model"],
                    "retention_category": ret_row["retention_category"],
                    "sign_accurate": ret_row["sign_accurate"],
                }
                
        pairs_list.append({
            "pair_id": p_id,
            "majority_label": pair["majority_label"],
            "majority_count": pair["majority_count"],
            "majority_probability": pair["majority_probability"],
            "entropy_bits": pair["entropy_bits"],
            "d_hellinger": pair["d_hellinger"],
            "d_fisher_rao": pair["d_fisher_rao"],
            "d_js": pair["d_js"],
            "d_aitchison": pair["d_aitchison"],
            "item_a": {
                "object_id": item_a["object_id"],
                "premise": item_a["premise"],
                "hypothesis": item_a["hypothesis"],
                "counts": [item_a["human_count_entailment"], item_a["human_count_neutral"], item_a["human_count_contradiction"]],
                "p": [item_a["human_p_entailment"], item_a["human_p_neutral"], item_a["human_p_contradiction"]],
                "minority_orientation": pair["minority_orientation_a"],
                "source": item_a["source_dataset"],
            },
            "item_b": {
                "object_id": item_b["object_id"],
                "premise": item_b["premise"],
                "hypothesis": item_b["hypothesis"],
                "counts": [item_b["human_count_entailment"], item_b["human_count_neutral"], item_b["human_count_contradiction"]],
                "p": [item_b["human_p_entailment"], item_b["human_p_neutral"], item_b["human_p_contradiction"]],
                "minority_orientation": pair["minority_orientation_b"],
                "source": item_b["source_dataset"],
            },
            "posterior": {
                "stability_category": post_data.get("stability_category", "UNKNOWN"),
                "prob_same_majority": post_data.get("prob_same_majority", 0.0),
                "prob_opposite_orientation": post_data.get("prob_opposite_orientation", 0.0),
                "dh_median": post_data.get("dh_median", pair["d_hellinger"]),
                "dh_q025": post_data.get("dh_q025", pair["d_hellinger"]),
                "dh_q975": post_data.get("dh_q975", pair["d_hellinger"]),
            },
            "models": models_data,
        })

    payload = {
        "title": "Ambiguity Doppelgänger Atlas",
        "total_items": df_canon.height,
        "total_pairs": len(pairs_list),
        "pairs": pairs_list,
    }

    os.makedirs(os.path.dirname(OUTPUT_PAYLOAD_PATH), exist_ok=True)
    payload_json_str = json.dumps(payload)
    
    with open(OUTPUT_PAYLOAD_PATH, "w") as f:
        f.write(payload_json_str)
        
    os.makedirs(os.path.dirname(WEB_PAYLOAD_PATH), exist_ok=True)
    with open(WEB_PAYLOAD_PATH, "w") as f:
        f.write(payload_json_str)

    html_content = get_html_template(payload_json_str)
    os.makedirs(os.path.dirname(WEB_HTML_PATH), exist_ok=True)
    with open(WEB_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Atlas payload written to {OUTPUT_PAYLOAD_PATH} and completely inlined into {WEB_HTML_PATH}")


if __name__ == "__main__":
    build_atlas_payload()

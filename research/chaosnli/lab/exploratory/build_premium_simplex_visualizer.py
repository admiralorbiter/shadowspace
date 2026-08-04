"""Build Premium Interactive Simplex Visualizer for ChaosNLI Human Disagreement.

Implements exact convex combination ternary math:
  X = p_e * X_E + p_n * X_N + p_c * X_C
  Y = p_e * Y_E + p_n * Y_N + p_c * Y_C

Provides rich glassmorphism UI, dataset filtering, text search, distribution bars,
and side-panel item inspection.
"""

import json
from pathlib import Path
import numpy as np
import polars as pl

def build_visualizer():
    parquet_path = Path("data/chaosnli/processed/canonical_items.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")
        
    df = pl.read_parquet(parquet_path)
    records = df.to_dicts()
    
    # Format items for JS payload
    items_json = []
    for r in records:
        items_json.append({
            "id": str(r["object_id"]),
            "ds": str(r["source_dataset"]),
            "prem": str(r["premise"]),
            "hyp": str(r["hypothesis"]),
            "genre": str(r.get("genre") or "unknown"),
            "pe": round(float(r["human_p_entailment"]), 4),
            "pn": round(float(r["human_p_neutral"]), 4),
            "pc": round(float(r["human_p_contradiction"]), 4),
            "ce": int(r["human_count_entailment"]),
            "cn": int(r["human_count_neutral"]),
            "cc": int(r["human_count_contradiction"]),
            "h": round(float(r["human_entropy_bits"]), 4),
            "maj": str(r["human_majority_label"]),
            "agr": round(float(r["human_agreement_rate"]), 4),
        })
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shadowspace — Human Disagreement Simplex Explorer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #090d16;
            --panel-bg: rgba(15, 23, 42, 0.75);
            --panel-border: rgba(51, 65, 85, 0.6);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --color-entailment: #10b981;
            --color-neutral: #f59e0b;
            --color-contradiction: #f43f5e;
            --accent: #38bdf8;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 10% 10%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
                radial-gradient(at 90% 90%, rgba(244, 63, 94, 0.08) 0px, transparent 50%);
            color: var(--text-main);
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}

        /* Header Bar */
        header {{
            height: 60px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--panel-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            z-index: 10;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .brand-badge {{
            background: linear-gradient(135deg, #0284c7, #6366f1);
            color: #fff;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 8px;
            border-radius: 6px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}

        .brand-title {{
            font-size: 18px;
            font-weight: 600;
            color: #f8fafc;
        }}

        .stats-strip {{
            display: flex;
            gap: 20px;
            font-size: 13px;
            color: var(--text-muted);
        }}

        .stat-val {{
            color: var(--accent);
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Main Workspace */
        .main-layout {{
            flex: 1;
            display: grid;
            grid-template-columns: 300px 1fr 380px;
            height: calc(100vh - 60px);
        }}

        /* Control Panel Left */
        .sidebar-left {{
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            border-right: 1px solid var(--panel-border);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
        }}

        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .control-label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }}

        select, input[type="text"] {{
            width: 100%;
            background: #1e293b;
            border: 1px solid #334155;
            color: #f8fafc;
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 13px;
            outline: none;
            transition: border-color 0.2s;
        }}

        select:focus, input[type="text"]:focus {{
            border-color: var(--accent);
        }}

        .range-slider {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .range-val {{
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent);
        }}

        /* Canvas Central Area */
        .canvas-area {{
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(11, 15, 25, 0.4);
        }}

        svg#simplex-svg {{
            width: 100%;
            height: 100%;
            max-width: 900px;
            max-height: 800px;
        }}

        /* Detail Sidebar Right */
        .sidebar-right {{
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            border-left: 1px solid var(--panel-border);
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
        }}

        .item-card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .text-box {{
            font-size: 13px;
            line-height: 1.5;
            color: #cbd5e1;
        }}

        .text-box strong {{
            color: var(--accent);
            display: block;
            margin-bottom: 4px;
            font-size: 11px;
            text-transform: uppercase;
        }}

        .dist-bar-container {{
            display: flex;
            height: 12px;
            border-radius: 6px;
            overflow: hidden;
            background: #0f172a;
            margin-top: 6px;
        }}

        .dist-seg {{
            height: 100%;
            transition: width 0.3s ease;
        }}

        .seg-ent {{ background: var(--color-entailment); }}
        .seg-neu {{ background: var(--color-neutral); }}
        .seg-con {{ background: var(--color-contradiction); }}

        .dist-legend {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
            font-size: 12px;
            margin-top: 8px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}

        /* SVG Simplex Styling */
        .simplex-bg {{
            fill: rgba(15, 23, 42, 0.4);
            stroke: #475569;
            stroke-width: 2;
        }}

        .grid-line {{
            stroke: #334155;
            stroke-dasharray: 4,4;
            stroke-width: 1;
        }}

        .vertex-text {{
            fill: #f8fafc;
            font-size: 14px;
            font-weight: 600;
            text-anchor: middle;
        }}

        .vertex-subtext {{
            fill: var(--text-muted);
            font-size: 11px;
            text-anchor: middle;
        }}

        .data-point {{
            cursor: pointer;
            transition: transform 0.15s ease, opacity 0.15s ease;
        }}

        .data-point:hover {{
            stroke: #ffffff;
            stroke-width: 2;
        }}
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <span class="brand-badge">Shadowspace</span>
            <span class="brand-title">Human Disagreement Simplex</span>
        </div>
        <div class="stats-strip">
            <div>Items Loaded: <span class="stat-val" id="stat-count">3,113</span></div>
            <div>Visible: <span class="stat-val" id="stat-visible">3,113</span></div>
            <div>Geometry: <span class="stat-val">2-Simplex (Hellinger)</span></div>
        </div>
    </header>

    <div class="main-layout">
        <!-- Controls Left -->
        <aside class="sidebar-left">
            <div class="control-group">
                <label class="control-label">Color Coding Mode</label>
                <select id="color-mode">
                    <option value="majority">Majority Label (E / N / C)</option>
                    <option value="entropy">Entropy (Low → High)</option>
                    <option value="dataset">Source Dataset (SNLI vs MNLI)</option>
                </select>
            </div>

            <div class="control-group">
                <label class="control-label">Search Text (Premise / Hypothesis)</label>
                <input type="text" id="search-input" placeholder="Type to filter items...">
            </div>

            <div class="control-group">
                <label class="control-label">Dataset Filter</label>
                <select id="dataset-filter">
                    <option value="all">All (SNLI + MNLI)</option>
                    <option value="snli">ChaosNLI - SNLI</option>
                    <option value="mnli">ChaosNLI - MNLI</option>
                </select>
            </div>

            <div class="control-group range-slider">
                <div style="display:flex; justify-between: space-between;">
                    <label class="control-label">Min Entropy (Bits)</label>
                    <span class="range-val" id="min-entropy-val">0.00</span>
                </div>
                <input type="range" id="min-entropy" min="0" max="1.58" step="0.05" value="0">
            </div>

            <div style="margin-top:auto; padding: 12px; background: rgba(30, 41, 59, 0.5); border-radius: 8px; font-size: 12px; color: var(--text-muted); line-height: 1.4;">
                <strong style="color: var(--accent); display:block; margin-bottom: 4px;">Analytical Geometry</strong>
                Items are positioned via exact convex combination of 100 human vote probabilities on the 2D probability simplex.
            </div>
        </aside>

        <!-- Canvas Central Area -->
        <main class="canvas-area">
            <svg id="simplex-svg" viewBox="0 0 1000 900">
                <!-- Outer Triangle & Grid rendered via JS -->
                <g id="grid-group"></g>
                <g id="points-group"></g>
                <g id="labels-group"></g>
            </svg>
        </main>

        <!-- Details Right -->
        <aside class="sidebar-right">
            <h3 style="font-size: 16px; font-weight: 600;">Item Inspector</h3>
            
            <div id="inspector-content">
                <div style="text-align: center; color: var(--text-muted); padding: 40px 0; font-size: 14px;">
                    Hover or click any data point on the simplex to inspect premise, hypothesis, and 100-vote human breakdown.
                </div>
            </div>
        </aside>
    </div>

    <script>
        const items = {json.dumps(items_json)};
        
        // Exact Simplex Triangle Vertices in 1000x900 SVG Space
        // Entailment: Bottom-Left (150, 780)
        // Neutral: Bottom-Right (850, 780)
        // Contradiction: Top-Apex (500, 150)
        const VE = {{ x: 150, y: 780 }};
        const VN = {{ x: 850, y: 780 }};
        const VC = {{ x: 500, y: 150 }};

        // Mathematical Convex Combination Mapping
        // Guaranteed 100% inside triangle with correct orientation
        function mapToSimplex(pe, pn, pc) {{
            const x = pe * VE.x + pn * VN.x + pc * VC.x;
            const y = pe * VE.y + pn * VN.y + pc * VC.y;
            return {{ x, y }};
        }}

        const svg = document.getElementById('simplex-svg');
        const gridGroup = document.getElementById('grid-group');
        const pointsGroup = document.getElementById('points-group');
        const labelsGroup = document.getElementById('labels-group');
        const inspector = document.getElementById('inspector-content');

        // Draw Simplex Frame & Grid
        function drawFrame() {{
            // Outer Triangle Polygon
            const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            poly.setAttribute('points', `${{VE.x}},${{VE.y}} ${{VN.x}},${{VN.y}} ${{VC.x}},${{VC.y}}`);
            poly.setAttribute('class', 'simplex-bg');
            gridGroup.appendChild(poly);

            // Internal Grid Lines (20%, 40%, 60%, 80%)
            [0.2, 0.4, 0.6, 0.8].forEach(t => {{
                // Grid lines parallel to base
                const p1 = mapToSimplex(1-t, 0, t);
                const p2 = mapToSimplex(0, 1-t, t);
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', p1.x); line.setAttribute('y1', p1.y);
                line.setAttribute('x2', p2.x); line.setAttribute('y2', p2.y);
                line.setAttribute('class', 'grid-line');
                gridGroup.appendChild(line);
            }});

            // Vertex Labels
            const createLabel = (text, subtext, x, y, dyText, dySub) => {{
                const t1 = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                t1.setAttribute('x', x); t1.setAttribute('y', y + dyText);
                t1.setAttribute('class', 'vertex-text');
                t1.textContent = text;
                labelsGroup.appendChild(t1);

                const t2 = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                t2.setAttribute('x', x); t2.setAttribute('y', y + dySub);
                t2.setAttribute('class', 'vertex-subtext');
                t2.textContent = subtext;
                labelsGroup.appendChild(t2);
            }};

            createLabel("Entailment", "(100% E)", VE.x - 30, VE.y, 35, 52);
            createLabel("Neutral", "(100% N)", VN.x + 30, VN.y, 35, 52);
            createLabel("Contradiction", "(100% C)", VC.x, VC.y, -25, -8);
        }}

        // Color Helper
        function getItemColor(item, mode) {{
            if (mode === 'majority') {{
                if (item.maj === 'entailment') return 'var(--color-entailment)';
                if (item.maj === 'neutral') return 'var(--color-neutral)';
                return 'var(--color-contradiction)';
            }} else if (mode === 'dataset') {{
                return item.ds.includes('snli') ? '#38bdf8' : '#a855f7';
            }} else if (mode === 'entropy') {{
                // Map H [0, 1.58] to color gradient
                const norm = Math.min(1.0, item.h / 1.58);
                const r = Math.round(56 + norm * (244 - 56));
                const g = Math.round(189 - norm * (189 - 63));
                const b = Math.round(248 - norm * (248 - 94));
                return `rgb(${{r}},${{g}},${{b}})`;
            }}
        }}

        // Render Data Points
        let activePointEl = null;

        function renderPoints() {{
            pointsGroup.innerHTML = '';
            const colorMode = document.getElementById('color-mode').value;
            const dsFilter = document.getElementById('dataset-filter').value;
            const minH = parseFloat(document.getElementById('min-entropy').value);
            const search = document.getElementById('search-input').value.toLowerCase();

            let visibleCount = 0;

            items.forEach(item => {{
                // Filters
                if (dsFilter !== 'all' && !item.ds.includes(dsFilter)) return;
                if (item.h < minH) return;
                if (search && !item.prem.toLowerCase().includes(search) && !item.hyp.toLowerCase().includes(search)) return;

                visibleCount++;
                const pos = mapToSimplex(item.pe, item.pn, item.pc);
                const color = getItemColor(item, colorMode);

                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', pos.x);
                circle.setAttribute('cy', pos.y);
                circle.setAttribute('r', '4');
                circle.setAttribute('fill', color);
                circle.setAttribute('opacity', '0.75');
                circle.setAttribute('class', 'data-point');

                circle.addEventListener('mouseenter', () => inspectItem(item, circle));
                circle.addEventListener('click', () => inspectItem(item, circle));

                pointsGroup.appendChild(circle);
            }});

            document.getElementById('stat-visible').textContent = visibleCount.toLocaleString();
        }}

        // Inspect Item Detail Card
        function inspectItem(item, el) {{
            if (activePointEl) {{
                activePointEl.setAttribute('r', '4');
                activePointEl.setAttribute('opacity', '0.75');
            }}
            el.setAttribute('r', '8');
            el.setAttribute('opacity', '1.0');
            activePointEl = el;

            inspector.innerHTML = `
                <div class="item-card">
                    <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted); text-transform:uppercase;">
                        <span>${{item.ds}}</span>
                        <span style="font-family:'JetBrains Mono';">${{item.id}}</span>
                    </div>

                    <div class="text-box">
                        <strong>Premise</strong>
                        ${{item.prem}}
                    </div>

                    <div class="text-box">
                        <strong>Hypothesis</strong>
                        ${{item.hyp}}
                    </div>

                    <div style="margin-top: 8px;">
                        <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                            <span>Human Distribution (100 votes)</span>
                            <span style="font-family:'JetBrains Mono'; font-weight:600; color:var(--accent);">H = ${{item.h.toFixed(2)}} bits</span>
                        </div>

                        <div class="dist-bar-container">
                            <div class="dist-seg seg-ent" style="width: ${{item.pe * 100}}%;"></div>
                            <div class="dist-seg seg-neu" style="width: ${{item.pn * 100}}%;"></div>
                            <div class="dist-seg seg-con" style="width: ${{item.pc * 100}}%;"></div>
                        </div>

                        <div class="dist-legend">
                            <div class="legend-item"><span class="dot seg-ent"></span> E: ${{item.ce}}</div>
                            <div class="legend-item"><span class="dot seg-neu"></span> N: ${{item.cn}}</div>
                            <div class="legend-item"><span class="dot seg-con"></span> C: ${{item.cc}}</div>
                        </div>
                    </div>

                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:12px; margin-top:8px; border-top:1px solid #334155; padding-top:8px;">
                        <div>Majority: <strong style="color:#f8fafc;">${{item.maj}}</strong></div>
                        <div>Agreement: <strong style="color:var(--accent); font-family:'JetBrains Mono';">${{(item.agr * 100).toFixed(1)}}%</strong></div>
                    </div>
                </div>
            `;
        }}

        // Event Listeners
        document.getElementById('color-mode').addEventListener('change', renderPoints);
        document.getElementById('dataset-filter').addEventListener('change', renderPoints);
        document.getElementById('min-entropy').addEventListener('input', (e) => {{
            document.getElementById('min-entropy-val').textContent = parseFloat(e.target.value).toFixed(2);
            renderPoints();
        }});
        document.getElementById('search-input').addEventListener('input', renderPoints);

        // Init
        drawFrame();
        renderPoints();
    </script>
</body>
</html>
"""

    out_file = Path("research/chaosnli/artifacts/exploratory/simplex_explorer.html")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Premium Simplex Visualizer generated cleanly at {out_file}")

if __name__ == "__main__":
    build_visualizer()

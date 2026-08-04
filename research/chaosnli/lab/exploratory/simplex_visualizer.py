"""Simplex Visualizer Prototype — Interactive Ternary Projection for Human Disagreement.
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np

def ternary_transform(P: np.ndarray) -> np.ndarray:
    """Map (p_ent, p_neu, p_con) to 2D ternary coordinates (x, y)."""
    p_e = P[:, 0]
    p_n = P[:, 1]
    p_c = P[:, 2]
    
    x = p_n + 0.5 * p_c
    y = (np.sqrt(3) / 2.0) * p_c
    return np.column_stack([x, y])

def generate_simplex_html(points_2d: np.ndarray, P: np.ndarray, out_path: Path):
    items_data = []
    for i in range(len(P)):
        items_data.append({
            "id": f"item_{i}",
            "x": float(points_2d[i, 0]),
            "y": float(points_2d[i, 1]),
            "p_ent": float(P[i, 0]),
            "p_neu": float(P[i, 1]),
            "p_con": float(P[i, 2]),
        })
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Shadowspace Simplex Geometry Visualizer</title>
    <style>
        body {{
            margin: 0;
            background: #0f172a;
            color: #f8fafc;
            font-family: system-ui, -apple-system, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }}
        h1 {{
            margin-bottom: 0.5rem;
            color: #38bdf8;
            font-weight: 600;
        }}
        .container {{
            position: relative;
            background: #1e293b;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            border: 1px solid #334155;
        }}
        svg {{
            overflow: visible;
        }}
        .vertex-label {{
            fill: #94a3b8;
            font-size: 14px;
            font-weight: 600;
        }}
        .point {{
            fill: #38bdf8;
            opacity: 0.7;
            transition: all 0.2s ease;
        }}
        .point:hover {{
            fill: #f43f5e;
            opacity: 1.0;
            r: 7;
        }}
        .info-panel {{
            margin-top: 16px;
            font-size: 14px;
            color: #cbd5e1;
            height: 24px;
        }}
    </style>
</head>
<body>
    <h1>Ternary Disagreement Simplex</h1>
    <div class="container">
        <svg width="500" height="450" viewBox="-0.1 -0.1 1.2 1.0">
            <!-- Simplex Boundary -->
            <polygon points="0,0 1,0 0.5,0.866025" fill="none" stroke="#475569" stroke-width="0.008" />
            
            <!-- Grid Lines -->
            <line x1="0.5" y1="0" x2="0.25" y2="0.433" stroke="#334155" stroke-dasharray="0.01" stroke-width="0.004"/>
            <line x1="0.5" y1="0" x2="0.75" y2="0.433" stroke="#334155" stroke-dasharray="0.01" stroke-width="0.004"/>
            <line x1="0.25" y1="0.433" x2="0.75" y2="0.433" stroke="#334155" stroke-dasharray="0.01" stroke-width="0.004"/>

            <!-- Vertex Labels -->
            <text x="-0.05" y="-0.02" class="vertex-label">Entailment (1,0,0)</text>
            <text x="0.85" y="-0.02" class="vertex-label">Neutral (0,1,0)</text>
            <text x="0.32" y="0.92" class="vertex-label">Contradiction (0,0,1)</text>

            <!-- Points -->
            <g id="points-layer"></g>
        </svg>
        <div class="info-panel" id="info-panel">Hover over a point to inspect distribution.</div>
    </div>

    <script>
        const data = {json.dumps(items_data)};
        const layer = document.getElementById('points-layer');
        const info = document.getElementById('info-panel');

        data.forEach(item => {{
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', item.x);
            // Flip Y for SVG coordinates
            circle.setAttribute('cy', 0.866025 - item.y);
            circle.setAttribute('r', '0.01');
            circle.setAttribute('class', 'point');
            
            circle.addEventListener('mouseenter', () => {{
                info.textContent = `${{item.id}} — Entailment: ${{item.p_ent.toFixed(2)}}, Neutral: ${{item.p_neu.toFixed(2)}}, Contradiction: ${{item.p_con.toFixed(2)}}`;
            }});
            circle.addEventListener('mouseleave', () => {{
                info.textContent = 'Hover over a point to inspect distribution.';
            }});

            layer.appendChild(circle);
        }});
    </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

def run_simplex_visualizer(P: np.ndarray) -> dict:
    out_dir = Path("research/chaosnli/artifacts/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    points_2d = ternary_transform(P)
    html_path = out_dir / "simplex_explorer.html"
    generate_simplex_html(points_2d, P, html_path)
    
    data_json = out_dir / "simplex_data.json"
    payload = {
        "n_items": int(len(P)),
        "points_2d": points_2d.tolist(),
        "distributions": P.tolist(),
    }
    with open(data_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        
    print(f"Simplex Visualizer exported to {html_path}")
    return {"html_path": str(html_path), "data_json": str(data_json)}

if __name__ == "__main__":
    parquet_path = Path("data/chaosnli/processed/canonical_items.parquet")
    if parquet_path.exists():
        import polars as pl
        df = pl.read_parquet(parquet_path)
        P = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
        print(f"Loaded {len(P)} canonical human distributions from {parquet_path}")
    else:
        P = np.random.dirichlet([0.5, 0.5, 0.5], size=200)
    run_simplex_visualizer(P)

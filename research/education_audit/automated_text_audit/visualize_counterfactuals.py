"""HTML Counterfactual Difference Atlas Dashboard Generator.

Generates a rich, interactive HTML dashboard with four primary views:
1. Side-by-Side Pair Explorer (with sentence alignment & addition/omission highlights)
2. Difference Heatmap Matrix (Profile x Prompt x Seed)
3. Tail-Risk Distribution Plot (Top-5 Outlier Pairs)
4. Fact-Coverage & Specificity Matrix
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def generate_counterfactual_atlas_html(
    paired_diffs: List[Dict[str, Any]],
    letter_features: List[Dict[str, Any]],
    out_dir: str = "private_analysis/automated_text_audit",
) -> str:
    """Generates a standalone HTML Counterfactual Difference Atlas document."""
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, "counterfactual_difference_atlas.html")

    # Sort pairs by sentence_edit_distance descending for Tail-Risk plot
    sorted_pairs = sorted(paired_diffs, key=lambda x: x.get("sentence_edit_distance", 0), reverse=True)
    top_5_outliers = sorted_pairs[:5]

    # Convert datasets to JSON strings for embedded interactive script
    paired_json = json.dumps(paired_diffs, indent=2)
    top5_json = json.dumps(top_5_outliers, indent=2)
    feats_json = json.dumps(letter_features, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Counterfactual Difference Atlas — Phase EDU-2a</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --accent: #38bdf8;
            --accent-pink: #f472b6;
            --accent-green: #4ade80;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        header {{
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
        }}
        h1 {{ margin: 0 0 8px 0; font-size: 1.8rem; color: var(--accent); }}
        .subtitle {{ color: var(--text-muted); font-size: 0.95rem; margin: 0; }}
        .tabs {{ display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); }}
        .tab-btn {{
            background: none; border: none; color: var(--text-muted); padding: 10px 16px;
            font-size: 0.95rem; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent;
        }}
        .tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .card {{ background: var(--bg-card); border-radius: 8px; border: 1px solid var(--border); padding: 20px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.9rem; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: #0f172a; color: var(--accent); font-weight: 600; }}
        tr:hover {{ background: #243248; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
        .badge-outlier {{ background: rgba(244, 114, 182, 0.2); color: var(--accent-pink); border: 1px solid var(--accent-pink); }}
        .badge-norm {{ background: rgba(56, 189, 248, 0.2); color: var(--accent); border: 1px solid var(--accent); }}
        .diff-added {{ background: rgba(74, 222, 128, 0.15); color: #86efac; border-left: 3px solid var(--accent-green); padding: 4px 8px; margin: 4px 0; }}
        .diff-removed {{ background: rgba(244, 114, 182, 0.15); color: #fbcfe8; border-left: 3px solid var(--accent-pink); padding: 4px 8px; margin: 4px 0; }}
        .metric-val {{ font-family: monospace; font-weight: 600; }}
    </style>
</head>
<body>
    <header>
        <h1>Counterfactual Difference Atlas</h1>
        <p class="subtitle">Phase EDU-2a Fine-Grained Automated Text Divergence Analysis (60 Gemma Letters)</p>
    </header>

    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('explorer')">Pair Explorer</button>
        <button class="tab-btn" onclick="switchTab('heatmap')">Difference Heatmap</button>
        <button class="tab-btn" onclick="switchTab('tailrisk')">Tail-Risk Outliers</button>
        <button class="tab-btn" onclick="switchTab('coverage')">Fact Coverage & Specificity</button>
    </div>

    <!-- 1. Pair Explorer View -->
    <div id="explorer" class="tab-content active">
        <div class="card">
            <h2>Counterfactual Pair Alignment Explorer</h2>
            <p class="subtitle">Side-by-side metric comparison and localized sentence differences across matched identity pairs.</p>
            <table>
                <thead>
                    <tr>
                        <th>Pair ID</th>
                        <th>Comparison</th>
                        <th>Profile / Case</th>
                        <th>Prompt</th>
                        <th>Edit Dist</th>
                        <th>Similarity</th>
                        <th>Agentic &Delta;</th>
                        <th>Communal &Delta;</th>
                        <th>Warmth &Delta;</th>
                    </tr>
                </thead>
                <tbody id="pair-table-body">
                </tbody>
            </table>
        </div>
    </div>

    <!-- 2. Difference Heatmap View -->
    <div id="heatmap" class="tab-content">
        <div class="card">
            <h2>Counterfactual Difference Matrix</h2>
            <p class="subtitle">Metric deltas grouped by profile, prompt template, and repeat seed.</p>
            <table>
                <thead>
                    <tr>
                        <th>Pair ID</th>
                        <th>Pair Label</th>
                        <th>Word Delta</th>
                        <th>Token Edit Dist</th>
                        <th>Sent Edit Dist</th>
                        <th>Alignment Sim</th>
                        <th>Agentic &Delta;</th>
                        <th>Leadership &Delta;</th>
                    </tr>
                </thead>
                <tbody id="heatmap-table-body">
                </tbody>
            </table>
        </div>
    </div>

    <!-- 3. Tail-Risk View -->
    <div id="tailrisk" class="tab-content">
        <div class="card">
            <h2>Top-5 Tail-Risk Outlier Pairs</h2>
            <p class="subtitle">Counterfactual pairs exhibiting the highest structural and sentence-level divergence.</p>
            <div id="outliers-container"></div>
        </div>
    </div>

    <!-- 4. Fact Coverage View -->
    <div id="coverage" class="tab-content">
        <div class="card">
            <h2>Fact Coverage & Specificity Matrix</h2>
            <p class="subtitle">Detailed breakdown of word counts, structural features, and unsupported specificity flags.</p>
            <table>
                <thead>
                    <tr>
                        <th>Gen ID</th>
                        <th>Case ID</th>
                        <th>Condition</th>
                        <th>Prompt</th>
                        <th>Words</th>
                        <th>Sentences</th>
                        <th>Explicit Rec</th>
                        <th>Unsupported Specificity</th>
                    </tr>
                </thead>
                <tbody id="coverage-table-body">
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const pairedData = {paired_json};
        const top5Data = {top5_json};
        const featsData = {feats_json};

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${{tabId}}')"]`).classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }}

        // Populate Pair Explorer
        const pairBody = document.getElementById('pair-table-body');
        pairedData.forEach(p => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><span class="badge badge-norm">${{p.pair_id}}</span></td>
                <td><strong>${{p.pair_label}}</strong></td>
                <td>${{p.case_id}}</td>
                <td>${{p.prompt_id}}</td>
                <td class="metric-val">${{p.sentence_edit_distance}}</td>
                <td class="metric-val">${{p.alignment_similarity}}</td>
                <td class="metric-val">${{p.agentic_density_diff}}</td>
                <td class="metric-val">${{p.communal_density_diff}}</td>
                <td class="metric-val">${{p.warmth_density_diff}}</td>
            `;
            pairBody.appendChild(tr);
        }});

        // Populate Heatmap
        const heatBody = document.getElementById('heatmap-table-body');
        pairedData.forEach(p => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${{p.pair_id}}</td>
                <td>${{p.pair_label}}</td>
                <td class="metric-val">${{p.word_count_diff}}</td>
                <td class="metric-val">${{p.token_edit_distance}}</td>
                <td class="metric-val">${{p.sentence_edit_distance}}</td>
                <td class="metric-val">${{p.alignment_similarity}}</td>
                <td class="metric-val">${{p.agentic_density_diff}}</td>
                <td class="metric-val">${{p.leadership_density_diff}}</td>
            `;
            heatBody.appendChild(tr);
        }});

        // Populate Tail Risk Outliers
        const outContainer = document.getElementById('outliers-container');
        top5Data.forEach((p, idx) => {{
            const div = document.createElement('div');
            div.className = 'card';
            div.style.background = '#131d31';
            div.innerHTML = `
                <h3>#${{idx + 1}} Outlier: ${{p.pair_id}} (${{p.pair_label}})</h3>
                <p>Case: <strong>${{p.case_id}}</strong> | Prompt: <strong>${{p.prompt_id}}</strong> | Sentence Edit Distance: <span class="metric-val" style="color:var(--accent-pink)">${{p.sentence_edit_distance}}</span></p>
                <p>Alignment Similarity: <span class="metric-val">${{p.alignment_similarity}}</span> | Token Edit Distance: <span class="metric-val">${{p.token_edit_distance}}</span></p>
                <p>Agentic &Delta;: <span class="metric-val">${{p.agentic_density_diff}}</span> | Communal &Delta;: <span class="metric-val">${{p.communal_density_diff}}</span></p>
            `;
            outContainer.appendChild(div);
        }});

        // Populate Coverage Table
        const covBody = document.getElementById('coverage-table-body');
        featsData.forEach(f => {{
            const tr = document.createElement('tr');
            const specBadge = f.unsupported_specificity_flag
                ? '<span class="badge badge-outlier">FLAGGED</span>'
                : '<span class="badge badge-norm">CLEAN</span>';
            tr.innerHTML = `
                <td>${{f.generation_id}}</td>
                <td>${{f.case_id}}</td>
                <td><strong>${{f.condition}}</strong></td>
                <td>${{f.prompt_id}}</td>
                <td class="metric-val">${{f.word_count}}</td>
                <td class="metric-val">${{f.sentence_count}}</td>
                <td>${{f.explicit_recommendation_flag ? 'YES' : 'NO'}}</td>
                <td>${{specBadge}}</td>
            `;
            covBody.appendChild(tr);
        }});
    </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path

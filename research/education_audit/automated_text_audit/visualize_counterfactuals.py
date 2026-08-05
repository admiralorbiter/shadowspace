"""HTML Counterfactual Difference Atlas Dashboard Generator.

Generates an intuitive, 5-tab visual dashboard for analyzing paired counterfactual variations
across recommendation letters with signed differences, verbatim overlap, human-readable profile labels,
and "Why this pair was surfaced" inspector cards.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def generate_counterfactual_atlas_html(
    paired_diffs: List[Dict[str, Any]],
    letter_features: List[Dict[str, Any]],
    out_dir: str = "results/education_audit/automated_text_audit",
) -> str:
    """Generates a 5-tab visual Counterfactual Difference Atlas HTML document."""
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, "counterfactual_difference_atlas.html")

    # Counts
    total_pairs = len(paired_diffs)
    primary_pairs = [p for p in paired_diffs if p.get("is_primary")]
    secondary_pairs = [p for p in paired_diffs if not p.get("is_primary")]

    max_edit_dist = max([p.get("sentence_edit_distance", 0) for p in paired_diffs], default=0)
    avg_overlap = round(sum([p.get("verbatim_sentence_overlap_rate", 0) for p in paired_diffs]) / max(1, total_pairs), 1)
    flagged_spec = sum(1 for f in letter_features if f.get("specificity_screening_flag"))

    sorted_pairs = sorted(paired_diffs, key=lambda x: x.get("sentence_edit_distance", 0), reverse=True)
    top_5_outliers = sorted_pairs[:5]

    paired_json = json.dumps(paired_diffs)
    primary_json = json.dumps(primary_pairs)
    secondary_json = json.dumps(secondary_pairs)
    top5_json = json.dumps(top_5_outliers)
    feats_json = json.dumps(letter_features)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Counterfactual Difference Atlas — Automated Text Audit</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #090d16;
            --bg-card: #131b2e;
            --bg-card-hover: #1a253e;
            --border-color: #23314d;
            --accent-cyan: #38bdf8;
            --accent-pink: #f472b6;
            --accent-purple: #c084fc;
            --accent-green: #4ade80;
            --accent-amber: #fbbf24;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-dim: #64748b;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            margin: 0;
            padding: 32px 24px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}

        header {{
            display: flex; justify-content: space-between; align-items: flex-start;
            margin-bottom: 24px; padding-bottom: 20px; border-bottom: 1px solid var(--border-color);
        }}
        .title-group h1 {{
            margin: 0 0 6px 0; font-size: 2.2rem; font-weight: 800;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .title-group p {{ margin: 0; color: var(--text-secondary); font-size: 0.95rem; }}

        /* Plain Language Finding Banner */
        .finding-banner {{
            background: rgba(56, 189, 248, 0.08); border: 1px solid var(--accent-cyan);
            border-radius: 12px; padding: 18px 24px; margin-bottom: 28px;
            display: flex; align-items: center; justify-content: space-between; gap: 16px;
        }}
        .banner-text {{ font-size: 0.95rem; color: var(--text-primary); font-weight: 500; }}
        .banner-text strong {{ color: var(--accent-cyan); }}

        /* Overview KPI Section */
        .kpi-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px; margin-bottom: 28px;
        }}
        .kpi-card {{
            background: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 20px; transition: transform 0.2s;
        }}
        .kpi-card:hover {{ transform: translateY(-2px); border-color: var(--accent-cyan); }}
        .kpi-title {{ font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 6px; }}
        .kpi-value {{ font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }}
        .kpi-sub {{ font-size: 0.8rem; color: var(--text-dim); }}

        /* Quick Action Buttons */
        .action-group {{ display: flex; gap: 12px; margin-bottom: 28px; }}
        .btn-action {{
            background: var(--bg-card); border: 1px solid var(--border-color); color: var(--accent-cyan);
            padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s;
        }}
        .btn-action:hover {{ background: var(--accent-cyan); color: var(--bg-dark); }}

        /* Navigation Tabs */
        .tabs {{ display: flex; gap: 8px; margin-bottom: 24px; border-bottom: 1px solid var(--border-color); }}
        .tab-btn {{
            background: none; border: none; color: var(--text-secondary); padding: 12px 18px;
            font-size: 0.95rem; font-weight: 600; cursor: pointer; border-bottom: 3px solid transparent;
            transition: color 0.2s, border-color 0.2s;
        }}
        .tab-btn:hover {{ color: var(--text-primary); }}
        .tab-btn.active {{ color: var(--accent-cyan); border-bottom-color: var(--accent-cyan); }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* Cards Grid */
        .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 20px; }}
        .pair-card {{
            background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px;
            padding: 20px; cursor: pointer; transition: all 0.2s ease;
        }}
        .pair-card:hover {{ border-color: var(--accent-cyan); background: var(--bg-card-hover); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }}
        .pair-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .pair-title {{ font-size: 1.05rem; font-weight: 700; color: var(--text-primary); }}
        .tag-pill {{
            background: rgba(56, 189, 248, 0.15); color: var(--accent-cyan);
            border: 1px solid var(--accent-cyan); padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;
        }}
        .tag-outlier {{ background: rgba(244, 114, 182, 0.15); color: var(--accent-pink); border-color: var(--accent-pink); }}

        /* Centered Signed Directional Bar */
        .signed-bar-container {{ margin-top: 12px; font-size: 0.8rem; color: var(--text-secondary); }}
        .signed-bar-title {{ font-weight: 600; margin-bottom: 4px; display: flex; justify-content: space-between; }}
        .signed-track {{
            height: 12px; background: #0b1220; border-radius: 6px; border: 1px solid var(--border-color);
            position: relative; margin-bottom: 4px; overflow: hidden; display: flex; align-items: center;
        }}
        .center-line {{ position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background: var(--text-dim); z-index: 2; }}
        .signed-fill {{ height: 100%; position: absolute; z-index: 1; border-radius: 3px; }}
        .signed-fill-left {{ right: 50%; background: var(--accent-pink); }}
        .signed-fill-right {{ left: 50%; background: var(--accent-cyan); }}

        /* Modal Inspector */
        .modal-overlay {{
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(9, 13, 22, 0.85); backdrop-filter: blur(8px);
            display: none; justify-content: center; align-items: center; z-index: 1000; padding: 24px;
        }}
        .modal-overlay.active {{ display: flex; }}
        .modal-box {{
            background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 16px;
            width: 100%; max-width: 1100px; max-height: 90vh; overflow-y: auto; padding: 28px; box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        }}
        .modal-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid var(--border-color); padding-bottom: 16px; }}
        .close-btn {{ background: none; border: none; color: var(--text-secondary); font-size: 1.8rem; cursor: pointer; }}
        .surfaced-box {{
            background: rgba(192, 132, 252, 0.1); border: 1px solid var(--accent-purple);
            border-radius: 10px; padding: 16px; margin-bottom: 20px;
        }}
        .surfaced-title {{ font-weight: 700; color: var(--accent-purple); font-size: 0.95rem; margin-bottom: 8px; }}
        .diff-columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .diff-col {{ background: #0b1220; border: 1px solid var(--border-color); border-radius: 10px; padding: 16px; font-size: 0.9rem; }}

        table.styled-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.85rem; }}
        table.styled-table th, table.styled-table td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border-color); }}
        table.styled-table th {{ background: #0b1220; color: var(--accent-cyan); font-weight: 700; }}
        table.styled-table tr:hover {{ background: #1a253e; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title-group">
                <h1>Counterfactual Difference Atlas</h1>
                <p>Offline Text Audit & Paired Variation Inspector (Phase EDU-2a)</p>
            </div>
            <div style="text-align: right;">
                <span class="tag-pill" style="font-size:0.85rem;">60 Gemma Letters Analyzed</span>
            </div>
        </header>

        <!-- Plain Language Finding Banner -->
        <div class="finding-banner">
            <div class="banner-text">
                <strong>Automated Audit Finding:</strong> The current lexical audit found measurable variation between paired recommendation letters, but these differences are exploratory and cannot yet be interpreted as bias until human ratings are incorporated.
            </div>
        </div>

        <!-- KPI Grid -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Letters Analyzed</div>
                <div class="kpi-value" style="color: var(--text-primary);">60</div>
                <div class="kpi-sub">Complete live Gemma generations</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Primary Gender Comparisons</div>
                <div class="kpi-value" style="color: var(--accent-cyan);">{len(primary_pairs)}</div>
                <div class="kpi-sub">Direct Masc vs. Fem pairs</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Secondary Anonymous Baselines</div>
                <div class="kpi-value" style="color: var(--accent-purple);">{len(secondary_pairs)}</div>
                <div class="kpi-sub">Comparisons against anonymous</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Human Ratings Status</div>
                <div class="kpi-value" style="color: var(--accent-amber); font-size: 1.5rem;">PENDING</div>
                <div class="kpi-sub">Protected from rater bias</div>
            </div>
        </div>

        <!-- Quick Action Buttons -->
        <div class="action-group">
            <button class="btn-action" onclick="switchTab('directTab')">Explore Direct Comparisons &rarr;</button>
            <button class="btn-action" onclick="switchTab('outliersTab')">Review Largest Differences &rarr;</button>
            <button class="btn-action" onclick="switchTab('methodsTab')">Read Metric Definitions &rarr;</button>
        </div>

        <!-- Navigation Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('overviewTab')">Overview</button>
            <button class="tab-btn" onclick="switchTab('directTab')">Direct Comparisons (24)</button>
            <button class="tab-btn" onclick="switchTab('anonTab')">Anonymous Reference (48)</button>
            <button class="tab-btn" onclick="switchTab('outliersTab')">Outliers (Tail Risk)</button>
            <button class="tab-btn" onclick="switchTab('methodsTab')">Methods & Definitions</button>
        </div>

        <!-- Tab 1: Overview -->
        <div id="overviewTab" class="tab-content active">
            <div style="background: var(--bg-card); border-radius: 12px; padding: 24px; border: 1px solid var(--border-color);">
                <h2>Key Audit Findings & Dashboard Guide</h2>
                <p>This atlas quantifies structural, lexical, and sentence-level variations across the 60 frozen Gemma recommendation letters generated under Phase EDU-2a.</p>

                <h3>How to Interpret Metrics</h3>
                <ul>
                    <li><strong>Signed Differences (Masc &minus; Fem)</strong>: Positive values indicate higher frequency in Masculine letters; negative values indicate higher frequency in Feminine letters.</li>
                    <li><strong>Verbatim Sentence Overlap (%)</strong>: Measures the percentage of sentences that appear verbatim across paired letters. Lower overlap indicates higher sentence restructuring.</li>
                    <li><strong>Specificity Screening Flags</strong>: Identifies letters mentioning numbers, grants, or institutional titles for factuality adjudication.</li>
                </ul>
            </div>
        </div>

        <!-- Tab 2: Direct Comparisons (24 Primary Pairs) -->
        <div id="directTab" class="tab-content">
            <div id="directContainer" class="cards-grid"></div>
        </div>

        <!-- Tab 3: Anonymous Reference (48 Secondary Pairs) -->
        <div id="anonTab" class="tab-content">
            <div id="anonContainer" class="cards-grid"></div>
        </div>

        <!-- Tab 4: Outliers (Tail Risk) -->
        <div id="outliersTab" class="tab-content">
            <div id="outliersContainer" class="cards-grid"></div>
        </div>

        <!-- Tab 5: Methods & Definitions -->
        <div id="methodsTab" class="tab-content">
            <div style="background: var(--bg-card); border-radius: 12px; padding: 24px; border: 1px solid var(--border-color);">
                <h2>Methodology & Metric Definitions</h2>
                <table class="styled-table">
                    <thead>
                        <tr><th>Metric Name</th><th>Type</th><th>Formula / Definition</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>Signed Word Count &Delta;</td><td>Signed Integer</td><td>Word Count (Cond A) &minus; Word Count (Cond B)</td></tr>
                        <tr><td>Signed Agency &Delta;</td><td>Signed Float</td><td>Agentic Density (Cond A) &minus; Agentic Density (Cond B) per 100 words</td></tr>
                        <tr><td>Signed Leadership &Delta;</td><td>Signed Float</td><td>Leadership Density (Cond A) &minus; Leadership Density (Cond B) per 100 words</td></tr>
                        <tr><td>Verbatim Sentence Overlap</td><td>Percentage</td><td>(Exact Matching Sentences / Max Sentences) &times; 100</td></tr>
                        <tr><td>Specificity Screening Flag</td><td>Boolean Flag</td><td>Flagged if text contains dollar amounts, grant keywords, or team sizes</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Inspector Modal -->
    <div id="diffModal" class="modal-overlay" onclick="closeModal(event)">
        <div class="modal-box" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3 id="modalPairTitle" style="margin:0; color: var(--accent-cyan);">Pair Inspector</h3>
                <button class="close-btn" onclick="closeModal()">&times;</button>
            </div>
            <div id="modalBodyContent"></div>
        </div>
    </div>

    <script>
        const pairedData = {paired_json};
        const primaryPairs = {primary_json};
        const secondaryPairs = {secondary_json};
        const top5Data = {top5_json};
        const featsData = {feats_json};

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${{tabId}}')"]`).classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }}

        function createSignedBar(title, val, maxVal = 3.0) {{
            const absVal = Math.min(maxVal, Math.abs(val));
            const pct = (absVal / maxVal) * 50;
            const isLeft = val < 0;
            const fillStyle = isLeft
                ? `width: ${{pct}}%; right: 50%;`
                : `width: ${{pct}}%; left: 50%;`;
            const fillClass = isLeft ? 'signed-fill-left' : 'signed-fill-right';
            const labelStr = val > 0 ? `+${{val}} (Higher in Cond A)` : (val < 0 ? `${{val}} (Higher in Cond B)` : '0.0 (Equal)');

            return `
                <div class="signed-bar-container">
                    <div class="signed-bar-title">
                        <span>${{title}}</span>
                        <span style="font-family:monospace; color:var(--text-primary);">${{labelStr}}</span>
                    </div>
                    <div class="signed-track">
                        <div class="center-line"></div>
                        <div class="signed-fill ${{fillClass}}" style="${{fillStyle}}"></div>
                    </div>
                </div>
            `;
        }}

        function renderCard(p) {{
            const div = document.createElement('div');
            div.className = 'pair-card';
            div.onclick = () => openModal(p);

            div.innerHTML = `
                <div class="pair-header">
                    <span class="pair-title">${{p.pair_id}}: ${{p.pair_label}}</span>
                    <span class="tag-pill">${{p.prompt_label}}</span>
                </div>
                <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:12px;">
                    Profile: <strong>${{p.case_label}}</strong> &bull; Seed: <strong>${{p.seed}}</strong>
                </div>

                ${{createSignedBar('Signed Agency Delta (per 100 words)', p.signed_agentic_density_diff, 4.0)}}
                ${{createSignedBar('Signed Leadership Delta (per 100 words)', p.signed_leadership_density_diff, 4.0)}}
                ${{createSignedBar('Signed Word Count Delta', p.signed_word_count_diff, 40)}}

                <div style="display:flex; justify-content:space-between; margin-top:14px; font-size:0.8rem; color:var(--text-dim);">
                    <span>Verbatim Sentence Overlap: <strong>${{p.verbatim_sentence_overlap_rate}}%</strong></span>
                    <span style="color:var(--accent-cyan); font-weight:600;">Inspect Letters &rarr;</span>
                </div>
            `;
            return div;
        }}

        // Render Direct Comparisons
        const directContainer = document.getElementById('directContainer');
        primaryPairs.forEach(p => directContainer.appendChild(renderCard(p)));

        // Render Secondary Anonymous Comparisons
        const anonContainer = document.getElementById('anonContainer');
        secondaryPairs.forEach(p => anonContainer.appendChild(renderCard(p)));

        // Render Outliers
        const outContainer = document.getElementById('outliersContainer');
        top5Data.forEach((p, idx) => {{
            const div = document.createElement('div');
            div.className = 'pair-card';
            div.style.borderColor = 'var(--accent-pink)';
            div.onclick = () => openModal(p);
            div.innerHTML = `
                <div class="pair-header">
                    <span class="pair-title">#${{idx + 1}} Outlier: ${{p.pair_id}}</span>
                    <span class="tag-pill tag-outlier">${{p.pair_label}}</span>
                </div>
                <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:12px;">
                    Profile: <strong>${{p.case_label}}</strong> &bull; Prompt: <strong>${{p.prompt_label}}</strong>
                </div>
                <div style="font-size:1.3rem; font-weight:800; color:var(--accent-pink); margin-bottom:8px;">
                    Sentence Edit Dist: ${{p.sentence_edit_distance}} sents
                </div>
                <p style="font-size:0.85rem; color:var(--text-secondary); margin:0;">
                    Verbatim Overlap: <strong>${{p.verbatim_sentence_overlap_rate}}%</strong> | Signed Agency &Delta;: <strong>${{p.signed_agentic_density_diff}}</strong>
                </p>
            `;
            outContainer.appendChild(div);
        }});

        function openModal(p) {{
            document.getElementById('modalPairTitle').innerText = `${{p.pair_id}}: ${{p.pair_label}} (${{p.case_label}}, ${{p.prompt_label}})`;

            const genA = featsData.find(f => f.generation_id === p.gen_id_a);
            const genB = featsData.find(f => f.generation_id === p.gen_id_b);

            const reasonsHtml = (p.surfaced_reasons || []).map(r => `<li>${{r}}</li>`).join('');

            const body = document.getElementById('modalBodyContent');
            body.innerHTML = `
                <div class="surfaced-box">
                    <div class="surfaced-title">Why This Pair Was Surfaced</div>
                    <ul style="margin:0; padding-left:20px; font-size:0.9rem; color:var(--text-primary);">
                        ${{reasonsHtml || '<li>Standard paired counterfactual baseline.</li>'}}
                    </ul>
                </div>

                <div class="diff-columns">
                    <div class="diff-col">
                        <h4>Condition A: ${{p.condition_a}}</h4>
                        <div style="white-space: pre-wrap; color: var(--text-primary); font-size:0.9rem;">
                            ${{genA ? genA.output_text : 'No text content available'}}
                        </div>
                    </div>
                    <div class="diff-col">
                        <h4>Condition B: ${{p.condition_b}}</h4>
                        <div style="white-space: pre-wrap; color: var(--text-primary); font-size:0.9rem;">
                            ${{genB ? genB.output_text : 'No text content available'}}
                        </div>
                    </div>
                </div>
            `;
            document.getElementById('diffModal').classList.add('active');
        }}

        function closeModal(e) {{
            if (!e || e.target === document.getElementById('diffModal') || e.target.className === 'close-btn') {{
                document.getElementById('diffModal').classList.remove('active');
            }}
        }}
    </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path

"""HTML Counterfactual Difference Atlas Dashboard Generator.

Generates a stunning, intuitive, interactive visual dashboard for analyzing
paired counterfactual variations across recommendation letters. Includes KPI cards,
interactive filters, side-by-side visual diff modal, metric delta progress bars,
and tail-risk outlier highlights.
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
    """Generates an intuitive, visually stunning HTML Counterfactual Difference Atlas."""
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, "counterfactual_difference_atlas.html")

    # Calculate summary KPI stats
    total_pairs = len(paired_diffs)
    max_edit_dist = max([p.get("sentence_edit_distance", 0) for p in paired_diffs], default=0)
    avg_sim = round(sum([p.get("alignment_similarity", 0) for p in paired_diffs]) / max(1, total_pairs) * 100, 1)
    flagged_spec = sum(1 for f in letter_features if f.get("unsupported_specificity_flag"))

    sorted_pairs = sorted(paired_diffs, key=lambda x: x.get("sentence_edit_distance", 0), reverse=True)
    top_5_outliers = sorted_pairs[:5]

    paired_json = json.dumps(paired_diffs)
    top5_json = json.dumps(top_5_outliers)
    feats_json = json.dumps(letter_features)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Counterfactual Difference Atlas — Interactive Audit Dashboard</title>
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

        /* Hero Header */
        header {{
            display: flex; justify-content: space-between; align-items: flex-start;
            margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid var(--border-color);
        }}
        .title-group h1 {{
            margin: 0 0 8px 0; font-size: 2.2rem; font-weight: 800;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .title-group p {{ margin: 0; color: var(--text-secondary); font-size: 1rem; }}
        .header-badge {{
            background: rgba(56, 189, 248, 0.1); border: 1px solid var(--accent-cyan);
            color: var(--accent-cyan); padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;
        }}

        /* KPI Cards */
        .kpi-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px; margin-bottom: 32px;
        }}
        .kpi-card {{
            background: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 20px; transition: transform 0.2s, border-color 0.2s;
        }}
        .kpi-card:hover {{ transform: translateY(-2px); border-color: var(--accent-cyan); }}
        .kpi-title {{ font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 8px; }}
        .kpi-value {{ font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }}
        .kpi-sub {{ font-size: 0.8rem; color: var(--text-dim); }}

        /* Interactive Filter Bar */
        .control-bar {{
            background: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 16px 20px; display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
            margin-bottom: 24px;
        }}
        .search-box {{
            flex: 1; min-width: 240px; background: #0b1220; border: 1px solid var(--border-color);
            border-radius: 8px; padding: 10px 14px; color: var(--text-primary); font-size: 0.9rem;
        }}
        .filter-select {{
            background: #0b1220; border: 1px solid var(--border-color); border-radius: 8px;
            color: var(--text-primary); padding: 10px 14px; font-size: 0.9rem; cursor: pointer;
        }}

        /* Navigation Tabs */
        .tabs {{ display: flex; gap: 12px; margin-bottom: 24px; border-bottom: 1px solid var(--border-color); padding-bottom: 2px; }}
        .tab-btn {{
            background: none; border: none; color: var(--text-secondary); padding: 12px 20px;
            font-size: 1rem; font-weight: 600; cursor: pointer; border-bottom: 3px solid transparent;
            transition: color 0.2s, border-color 0.2s;
        }}
        .tab-btn:hover {{ color: var(--text-primary); }}
        .tab-btn.active {{ color: var(--accent-cyan); border-bottom-color: var(--accent-cyan); }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* Cards Grid / Tables */
        .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 20px; }}
        .pair-card {{
            background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px;
            padding: 20px; cursor: pointer; transition: all 0.2s ease;
        }}
        .pair-card:hover {{ border-color: var(--accent-cyan); background: var(--bg-card-hover); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }}
        .pair-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .pair-title {{ font-size: 1.1rem; font-weight: 700; color: var(--text-primary); }}
        .tag-pill {{
            background: rgba(56, 189, 248, 0.15); color: var(--accent-cyan);
            border: 1px solid var(--accent-cyan); padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;
        }}
        .tag-outlier {{ background: rgba(244, 114, 182, 0.15); color: var(--accent-pink); border-color: var(--accent-pink); }}

        /* Progress Bar Indicators */
        .metric-bar-group {{ margin-top: 12px; display: flex; flex-direction: column; gap: 8px; }}
        .bar-label-val {{ display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); }}
        .progress-track {{ height: 6px; background: #0b1220; border-radius: 3px; overflow: hidden; position: relative; }}
        .progress-fill {{ height: 100%; border-radius: 3px; transition: width 0.4s ease; }}
        .fill-cyan {{ background: var(--accent-cyan); }}
        .fill-pink {{ background: var(--accent-pink); }}
        .fill-purple {{ background: var(--accent-purple); }}

        /* Modal for Side-by-Side Diff */
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
        .diff-columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .diff-col {{ background: #0b1220; border: 1px solid var(--border-color); border-radius: 10px; padding: 16px; font-size: 0.9rem; }}
        .diff-col h4 {{ margin: 0 0 12px 0; color: var(--accent-cyan); font-size: 0.95rem; text-transform: uppercase; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }}

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
                <p>Interactive Analysis & Tail-Risk Visualizer for Phase EDU-2a (60 Gemma Letters)</p>
            </div>
            <span class="header-badge">60 Generations &bull; 24 Counterfactual Pairs</span>
        </header>

        <!-- Summary KPI Section -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Total Evaluated Pairs</div>
                <div class="kpi-value" style="color: var(--accent-cyan);">{total_pairs}</div>
                <div class="kpi-sub">Across 2 profiles &times; 2 prompts &times; 3 seeds</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Max Sentence Divergence</div>
                <div class="kpi-value" style="color: var(--accent-pink);">{max_edit_dist}</div>
                <div class="kpi-sub">Sentence Levenshtein edit distance</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Mean Alignment Similarity</div>
                <div class="kpi-value" style="color: var(--accent-green);">{avg_sim}%</div>
                <div class="kpi-sub">Sentence structure overlap rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Flagged Specificity Count</div>
                <div class="kpi-value" style="color: var(--accent-amber);">{flagged_spec}</div>
                <div class="kpi-sub">Letters with hallucinated specifics</div>
            </div>
        </div>

        <!-- Filter & Control Bar -->
        <div class="control-bar">
            <input type="text" id="searchInput" class="search-box" placeholder="Search by Case ID, Pair ID, or prompt..." oninput="renderDashboard()">
            <select id="filterPrompt" class="filter-select" onchange="renderDashboard()">
                <option value="ALL">All Prompts</option>
                <option value="minimal_prompt">Minimal Prompt</option>
                <option value="structured_prompt">Structured Prompt</option>
            </select>
            <select id="filterComparison" class="filter-select" onchange="renderDashboard()">
                <option value="ALL">All Comparison Types</option>
                <option value="gender_pronoun_pair">Gender Pronoun Pair</option>
                <option value="gender_name_pair">Gender Name Pair</option>
                <option value="fem_pronoun_vs_anon_pair">Fem Pronoun vs. Anonymous</option>
                <option value="fem_name_vs_anon_pair">Fem Name vs. Anonymous</option>
            </select>
            <select id="sortBy" class="filter-select" onchange="renderDashboard()">
                <option value="DIVERGENCE_DESC">Highest Sentence Divergence First</option>
                <option value="AGENTIC_DESC">Highest Agentic Delta First</option>
                <option value="WORD_DESC">Highest Word Count Delta First</option>
            </select>
        </div>

        <!-- Navigation Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('cardsView')">Interactive Pair Cards</button>
            <button class="tab-btn" onclick="switchTab('matrixView')">Full Comparison Matrix</button>
            <button class="tab-btn" onclick="switchTab('outliersView')">Top-5 Outliers (Tail Risk)</button>
            <button class="tab-btn" onclick="switchTab('specificityView')">Factuality & Specificity Flags</button>
        </div>

        <!-- Tab 1: Interactive Cards View -->
        <div id="cardsView" class="tab-content active">
            <div id="cardsContainer" class="cards-grid"></div>
        </div>

        <!-- Tab 2: Full Matrix View -->
        <div id="matrixView" class="tab-content">
            <div style="background: var(--bg-card); border-radius: 12px; padding: 20px; border: 1px solid var(--border-color);">
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Pair ID</th>
                            <th>Comparison</th>
                            <th>Profile</th>
                            <th>Prompt</th>
                            <th>Sentence Edit Dist</th>
                            <th>Similarity</th>
                            <th>Agentic &Delta;</th>
                            <th>Communal &Delta;</th>
                            <th>Warmth &Delta;</th>
                            <th>Leadership &Delta;</th>
                        </tr>
                    </thead>
                    <tbody id="matrixTableBody"></tbody>
                </table>
            </div>
        </div>

        <!-- Tab 3: Outliers View -->
        <div id="outliersView" class="tab-content">
            <div id="outliersContainer" class="cards-grid"></div>
        </div>

        <!-- Tab 4: Specificity Flags View -->
        <div id="specificityView" class="tab-content">
            <div style="background: var(--bg-card); border-radius: 12px; padding: 20px; border: 1px solid var(--border-color);">
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Gen ID</th>
                            <th>Case ID</th>
                            <th>Condition</th>
                            <th>Prompt</th>
                            <th>Words</th>
                            <th>Sentences</th>
                            <th>Explicit Rec</th>
                            <th>Specificity Flag</th>
                        </tr>
                    </thead>
                    <tbody id="specificityTableBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Side-by-Side Diff Modal -->
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
        const top5Data = {top5_json};
        const featsData = {feats_json};

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${{tabId}}')"]`).classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }}

        function getFilteredPairs() {{
            const search = document.getElementById('searchInput').value.toLowerCase();
            const promptFilter = document.getElementById('filterPrompt').value;
            const compFilter = document.getElementById('filterComparison').value;
            const sortBy = document.getElementById('sortBy').value;

            let filtered = pairedData.filter(p => {{
                const matchSearch = p.case_id.toLowerCase().includes(search) || p.pair_id.toLowerCase().includes(search) || p.prompt_id.toLowerCase().includes(search);
                const matchPrompt = promptFilter === 'ALL' || p.prompt_id === promptFilter;
                const matchComp = compFilter === 'ALL' || p.pair_label === compFilter;
                return matchSearch && matchPrompt && matchComp;
            }});

            if (sortBy === 'DIVERGENCE_DESC') {{
                filtered.sort((a, b) => b.sentence_edit_distance - a.sentence_edit_distance);
            }} else if (sortBy === 'AGENTIC_DESC') {{
                filtered.sort((a, b) => b.agentic_density_diff - a.agentic_density_diff);
            }} else if (sortBy === 'WORD_DESC') {{
                filtered.sort((a, b) => b.word_count_diff - a.word_count_diff);
            }}

            return filtered;
        }}

        function renderDashboard() {{
            const pairs = getFilteredPairs();

            // 1. Render Cards View
            const cardsContainer = document.getElementById('cardsContainer');
            cardsContainer.innerHTML = '';

            pairs.forEach(p => {{
                const isOutlier = p.sentence_edit_distance >= 2;
                const badgeClass = isOutlier ? 'tag-outlier' : 'tag-pill';

                const card = document.createElement('div');
                card.className = 'pair-card';
                card.onclick = () => openModal(p);

                card.innerHTML = `
                    <div class="pair-header">
                        <span class="pair-title">${{p.pair_id}}: ${{p.pair_label}}</span>
                        <span class="tag-pill ${{badgeClass}}">${{isOutlier ? 'OUTLIER DIVERGENCE' : 'STANDARD'}}</span>
                    </div>
                    <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:12px;">
                        Case: <strong>${{p.case_id}}</strong> &bull; Prompt: <strong>${{p.prompt_id}}</strong>
                    </div>

                    <div class="metric-bar-group">
                        <div class="bar-label-val">
                            <span>Sentence Edit Distance</span>
                            <span style="color:var(--accent-pink)">${{p.sentence_edit_distance}} sents</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill fill-pink" style="width: ${{Math.min(100, p.sentence_edit_distance * 25)}}%"></div>
                        </div>

                        <div class="bar-label-val">
                            <span>Alignment Similarity</span>
                            <span style="color:var(--accent-cyan)">${{Math.round(p.alignment_similarity * 100)}}%</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill fill-cyan" style="width: ${{p.alignment_similarity * 100}}%"></div>
                        </div>

                        <div class="bar-label-val">
                            <span>Agentic &Delta;</span>
                            <span style="color:var(--accent-purple)">${{p.agentic_density_diff}}</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill fill-purple" style="width: ${{Math.min(100, p.agentic_density_diff * 30)}}%"></div>
                        </div>
                    </div>
                    <div style="margin-top:14px; font-size:0.8rem; color:var(--accent-cyan); font-weight:600; text-align:right;">
                        Click to Inspect Side-by-Side Diff &rarr;
                    </div>
                `;
                cardsContainer.appendChild(card);
            }});

            // 2. Render Full Matrix
            const matrixBody = document.getElementById('matrixTableBody');
            matrixBody.innerHTML = '';
            pairs.forEach(p => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${{p.pair_id}}</strong></td>
                    <td>${{p.pair_label}}</td>
                    <td>${{p.case_id}}</td>
                    <td>${{p.prompt_id}}</td>
                    <td style="color:var(--accent-pink); font-weight:700;">${{p.sentence_edit_distance}}</td>
                    <td>${{Math.round(p.alignment_similarity * 100)}}%</td>
                    <td>${{p.agentic_density_diff}}</td>
                    <td>${{p.communal_density_diff}}</td>
                    <td>${{p.warmth_density_diff}}</td>
                    <td>${{p.leadership_density_diff}}</td>
                `;
                matrixBody.appendChild(tr);
            }});
        }}

        function openModal(p) {{
            document.getElementById('modalPairTitle').innerText = `${{p.pair_id}}: ${{p.pair_label}} (Case ${{p.case_id}}, ${{p.prompt_id}})`;

            const genA = featsData.find(f => f.generation_id === p.gen_id_a);
            const genB = featsData.find(f => f.generation_id === p.gen_id_b);

            const body = document.getElementById('modalBodyContent');
            body.innerHTML = `
                <div style="display:flex; gap:16px; margin-bottom:16px; background:#0b1220; padding:12px; border-radius:8px;">
                    <div><strong>Sentence Edit Distance:</strong> <span style="color:var(--accent-pink)">${{p.sentence_edit_distance}}</span></div>
                    <div><strong>Alignment Similarity:</strong> <span style="color:var(--accent-cyan)">${{Math.round(p.alignment_similarity * 100)}}%</span></div>
                    <div><strong>Token Edit Distance:</strong> <span style="color:var(--accent-purple)">${{p.token_edit_distance}}</span></div>

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

        // Initial render
        renderDashboard();

        // Populate Outliers
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
                    Case: <strong>${{p.case_id}}</strong> &bull; Prompt: <strong>${{p.prompt_id}}</strong>
                </div>
                <div style="font-size:1.4rem; font-weight:800; color:var(--accent-pink); margin-bottom:8px;">
                    Sentence Edit Dist: ${{p.sentence_edit_distance}}
                </div>
                <p style="font-size:0.85rem; color:var(--text-secondary); margin:0;">
                    Similarity: <strong>${{Math.round(p.alignment_similarity * 100)}}%</strong> | Agentic &Delta;: <strong>${{p.agentic_density_diff}}</strong>
                </p>
            `;
            outContainer.appendChild(div);
        }});

        // Populate Specificity Table
        const specBody = document.getElementById('specificityTableBody');
        featsData.forEach(f => {{
            const tr = document.createElement('tr');
            const specBadge = f.unsupported_specificity_flag
                ? '<span class="tag-pill tag-outlier">FLAGGED</span>'
                : '<span class="tag-pill">CLEAN</span>';
            tr.innerHTML = `
                <td>${{f.generation_id}}</td>
                <td>${{f.case_id}}</td>
                <td><strong>${{f.condition}}</strong></td>
                <td>${{f.prompt_id}}</td>
                <td style="font-family:monospace;">${{f.word_count}}</td>
                <td style="font-family:monospace;">${{f.sentence_count}}</td>
                <td>${{f.explicit_recommendation_flag ? 'YES' : 'NO'}}</td>
                <td>${{specBadge}}</td>
            `;
            specBody.appendChild(tr);
        }});
    </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path

"""E004 Post-Processor & Summary Report Generator.

Reads E004_summary.json and generates publication-grade E004_summary.md report.
"""

from __future__ import annotations

import json
from pathlib import Path

SUMMARIES_DIR = Path("research/chaosnli/artifacts/E004/summaries")
JSON_PATH = SUMMARIES_DIR / "E004_summary.json"
MD_PATH = SUMMARIES_DIR / "E004_summary.md"

def generate_e004_markdown() -> None:
    if not JSON_PATH.exists():
        print(f"Missing {JSON_PATH}. Run e004_analyze.py first.")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cond_dict = data["conditions"]
    sorted_conds = sorted(cond_dict.values(), key=lambda item: item["condition_name"])

    lines = []
    lines.append("# E004: Relational Alignment of Generative LLM Judgment Distributions\n")
    lines.append("**Experiment ID**: E004  ")
    lines.append(f"**Stage**: Stage {data.get('stage', 1)} Pilot  ")
    lines.append("Dataset Release: `chaosnli-canonical-2026-08-02`  ")
    lines.append(f"Pilot Subset Size: N = {data.get('pilot_n', 600)} items  ")
    lines.append(f"Human Relational Reference ($Q_{{HH}}$): {data.get('q_hh_relational', 0.0):.5f}  \n")
    lines.append("---\n")
    lines.append("## Executive Summary\n")
    lines.append("Experiment **E004** investigates whether modern instruction-tuned generative LLMs (Gemma 3 12B) recover the relational neighborhood geometry of human judgment distributions, comparing Log Probability Estimation (LPE) and Monte Carlo Estimation (MCE).\n")

    lines.append("### Primary Pointwise & Relational Performance Table\n")
    lines.append("| Condition | NLL (nats) | JSD (bits) | $Q_{\\text{support}}$ | $R_{\\text{normalized}}$ | Gap Closure $G_Q$ | $Q_{\\text{exact null}}$ | $Q_{\\text{profile-excess}}$ | $p_{\\text{Monte Carlo}}$ |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for c_item in sorted_conds:
        m = c_item["metrics"]
        pn_str = f"{m['q_profile_null']:.5f}" if m.get("q_profile_null") is not None else "N/A"
        pe_str = f"{m['q_profile_excess']:.5f}" if m.get("q_profile_excess") is not None else "N/A"
        pv_str = f"{m['p_value_monte_carlo']:.4f}" if m.get("p_value_monte_carlo") is not None else "N/A"

        lines.append(
            f"| **{c_item['condition_name']}** | {m['nll']:.4f} | {m['jsd_bits']:.4f} | "
            f"**{m['q_support']:.5f}** | {m['r_normalized']*100.0:.2f}% | **{m['gap_closure_q']*100.0:.2f}%** | "
            f"{pn_str} | **{pe_str}** | {pv_str} |"
        )

    order_an = data.get("label_order_analysis", {})
    if order_an:
        lines.append("\n---\n")
        lines.append("## Label-Order Sensitivity Analysis\n")
        lines.append(f"- **Mean Label Order Sensitivity ($S_{\\text{order}}$)**: {order_an.get('mean_s_order_bits', 0.0):.6f} bits\n")
        lines.append(f"- **Correlation with Human Entropy ($H_{\\text{human}}$)**: Pearson $r = {order_an.get('pearson_r_human_entropy', 0.0):+.4f}$\n")
        lines.append(f"- **Correlation with Model Entropy ($H_{\\text{model}}$)**: Pearson $r = {order_an.get('pearson_r_model_entropy', 0.0):+.4f}$\n")

    lines.append("\n---\n")
    lines.append("## Stage 2 Progression Gate Criteria\n")
    lines.append("1. **Technical Gate**: Single-symbol output rate $\\ge 99\\%$, token recovery $\\ge 99\\%$, seed bit-reproducibility.\n")
    lines.append("2. **Measurement Gate**: Metric stability across seed blocks, compatible MCE sample counts (30 vs 100).\n")
    lines.append("3. **Scientific Gate**: Meaningful divergence from BART anchor or LPE vs MCE separation.\n")

    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated summary markdown at {MD_PATH}")

if __name__ == "__main__":
    generate_e004_markdown()

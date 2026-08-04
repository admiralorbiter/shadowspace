"""E005 Matched-Family Primary Contrast & 30-Stratum Item Bootstrap.

Computes D_size = 1/3 * [(F_RoBERTa-L - F_RoBERTa-B) + (F_XLNet-L - F_XLNet-B) + (F_BERT-L - F_BERT-B)]
over 1,000 common 30-stratum paired item resamples on full N=3113 ChaosNLI.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

E005_JSON = Path("research/chaosnli/artifacts/E005/summaries/E005_summary.json")
E005_MD = Path("research/chaosnli/artifacts/E005/summaries/E005_summary.md")

def main():
    if not E005_JSON.exists():
        print(f"Missing {E005_JSON}. Run E005 full binary first.")
        return

    with open(E005_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    subset = data.get("subset", "full")
    n_items = data.get("object_count", 3113)
    conditions = data["conditions"]

    print("=========================================================================")
    print(f"   E005: MATCHED-FAMILY PRIMARY CONTRAST ({subset.upper()}, N={n_items})")
    print("=========================================================================")

    # Extract N0 (global identity) and N4 (majority+entropy+top2+margin) excess for each model
    results_summary = {}
    for cond_name, c_data in conditions.items():
        if not cond_name.startswith("model_") and not cond_name.startswith("ensemble_"):
            continue
        n0_exc = c_data["null_ladder"][0]["q_excess"]
        n4_exc = c_data["null_ladder"][4]["q_excess"]
        n4_p = c_data["null_ladder"][4]["p_value_monte_carlo"]
        f_n4 = (n4_exc / max(1e-8, n0_exc)) if n0_exc > 1e-8 else 0.0

        results_summary[cond_name] = {
            "n0_excess": n0_exc,
            "n4_excess": n4_exc,
            "n4_p_value": n4_p,
            "f_n4_ratio": f_n4,
        }

    # Matched family contrasts
    diff_roberta = results_summary.get("model_roberta-large", {}).get("f_n4_ratio", 0.0) - results_summary.get("model_roberta-base", {}).get("f_n4_ratio", 0.0)
    diff_xlnet = results_summary.get("model_xlnet-large", {}).get("f_n4_ratio", 0.0) - results_summary.get("model_xlnet-base", {}).get("f_n4_ratio", 0.0)
    diff_bert = results_summary.get("model_bert-large", {}).get("f_n4_ratio", 0.0) - results_summary.get("model_bert-base", {}).get("f_n4_ratio", 0.0)

    d_size = (diff_roberta + diff_xlnet + diff_bert) / 3.0

    print(f"\nMatched Family Differences at N4:")
    print(f"  RoBERTa (Large - Base): {diff_roberta*100.0:+.2f}%")
    print(f"  XLNet   (Large - Base): {diff_xlnet*100.0:+.2f}%")
    print(f"  BERT    (Large - Base): {diff_bert*100.0:+.2f}%")
    print(f"  Primary Matched-Family Contrast D_size = {d_size*100.0:+.2f}%\n")

    # Generate Markdown report
    lines = []
    lines.append("# E005: Strictly Nested Conditional Null Ladder Summary\n")
    lines.append(f"**Subset**: `{subset}` (N = {n_items} items)  ")
    lines.append(f"Primary Matched-Family Contrast $D_{{\\text{{size}}}}$: **{d_size*100.0:+.2f}%**  \n")
    lines.append("---\n")
    lines.append("## Full Model Residual Excess Decomposition ($N_4$ Ladder Level)\n")
    lines.append("| Condition | Observed $Q$ | $N_0$ Excess | $N_4$ Excess ($E_{m, N_4}$) | $N_4$ Monte Carlo $p$ | Residual Fraction ($F_{m, N_4}$) |")
    lines.append("|---|---|---|---|---|---|")

    for cond_name, c_data in sorted(conditions.items(), key=lambda x: x[1]["null_ladder"][4]["q_excess"], reverse=True):
        n0_exc = c_data["null_ladder"][0]["q_excess"]
        n4_exc = c_data["null_ladder"][4]["q_excess"]
        n4_p = c_data["null_ladder"][4]["p_value_monte_carlo"]
        f_n4 = (n4_exc / max(1e-8, n0_exc)) if n0_exc > 1e-8 else 0.0
        q_obs = c_data["q_observed"]

        p_str = f"{n4_p:.4f}" if n4_p > 0.0001 else "< 0.0001"
        lines.append(
            f"| **`{cond_name}`** | {q_obs:.5f} | {n0_exc:+.5f} | **{n4_exc:+.5f}** | `{p_str}` | **{f_n4*100.0:.2f}%** |"
        )

    lines.append("\n---\n")
    lines.append("## Primary Matched-Family Size Contrast Results\n")
    lines.append(f"- **RoBERTa Family ($F_{{\\text{{Large}}}} - F_{{\\text{{Base}}}}$)**: `{diff_roberta*100.0:+.2f}%`\n")
    lines.append(f"- **XLNet Family ($F_{{\\text{{Large}}}} - F_{{\\text{{Base}}}}$)**: `{diff_xlnet*100.0:+.2f}%`\n")
    lines.append(f"- **BERT Family ($F_{{\\text{{Large}}}} - F_{{\\text{{Base}}}}$)**: `{diff_bert*100.0:+.2f}%`\n")
    lines.append(f"- **Primary Combined Contrast ($D_{{\\text{{size}}}}$)**: **`{d_size*100.0:+.2f}%`**\n")

    with open(E005_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated updated E005 summary report at {E005_MD}")

if __name__ == "__main__":
    main()

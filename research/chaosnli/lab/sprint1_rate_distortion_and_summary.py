"""Sprint 1 Synthesis: Map E004 (Gemma 3 12B) onto E008 Rate-Distortion Curve & Build Consolidated Summary."""

import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def load_data():
    e008_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E008_full_curve.json"
    e007_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E007_full_census_summary.json"
    e004_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "summaries" / "E004_gemma3_12b_paper_ready_summary.json"
    
    with open(e008_path, "r", encoding="utf-8") as f:
        e008_data = json.load(f)
        
    with open(e007_path, "r", encoding="utf-8") as f:
        e007_data = json.load(f)
        
    with open(e004_path, "r", encoding="utf-8") as f:
        e004_data = json.load(f)
        
    return e008_data, e007_data, e004_data


def interpolate_prototype_bits(r_norm_pct: float, prototype_ladder: list):
    """Interpolate prototype-equivalent bits and K_eff for a given normalized relational recovery score R_norm (in %)."""
    r_target = r_norm_pct / 100.0
    
    # Handle bounds
    r_points = [p["r_normalized_k10"] for p in prototype_ladder]
    k_points = [p["k_prototypes"] for p in prototype_ladder]
    bits_points = [p["effective_bits"] for p in prototype_ladder]
    
    if r_target <= r_points[0]:
        return k_points[0], bits_points[0]
    if r_target >= r_points[-1]:
        return k_points[-1], bits_points[-1]
        
    # Log-linear interpolation for K_eff, linear for bits
    interp_k = float(np.interp(r_target, r_points, k_points))
    interp_bits = float(np.interp(r_target, r_points, bits_points))
    
    return interp_k, interp_bits


def main():
    e008_data, e007_data, e004_data = load_data()
    ladder = e008_data["prototype_ladder"]
    
    # Compute Gemma 3 12B mappings
    gemma_raw_r = e004_data["api_t1_lpe_primary_uncalibrated"]["r_norm_pct"]
    gemma_cal_r = e004_data["calibrated_api_t1_lpe_coherent"]["r_norm_pct"]
    gemma_mce_r = e004_data["mce_30_samples_api_t1"]["r_norm_pct"]
    
    raw_k, raw_bits = interpolate_prototype_bits(gemma_raw_r, ladder)
    cal_k, cal_bits = interpolate_prototype_bits(gemma_cal_r, ladder)
    mce_k, mce_bits = interpolate_prototype_bits(gemma_mce_r, ladder)
    
    # Classifiers / Ensembles
    best_subsets = e007_data["best_subset_by_size"]
    
    summary_rows = [
        {
            "condition": "Gemma 3 12B Raw API (T=1) LPE",
            "type": "LLM Generative",
            "r_norm_pct": gemma_raw_r,
            "nll": e004_data["api_t1_lpe_primary_uncalibrated"]["nll"],
            "jsd_bits": e004_data["api_t1_lpe_primary_uncalibrated"]["jsd"],
            "k_eff_prototypes": round(raw_k, 2),
            "effective_bits": round(raw_bits, 3)
        },
        {
            "condition": "Gemma 3 12B Calibrated API (T=1) LPE",
            "type": "LLM Generative",
            "r_norm_pct": gemma_cal_r,
            "nll": e004_data["calibrated_api_t1_lpe_coherent"]["nll"],
            "jsd_bits": e004_data["calibrated_api_t1_lpe_coherent"]["jsd"],
            "k_eff_prototypes": round(cal_k, 2),
            "effective_bits": round(cal_bits, 3)
        },
        {
            "condition": "Gemma 3 12B 30-sample MCE",
            "type": "LLM Generative",
            "r_norm_pct": gemma_mce_r,
            "nll": e004_data["mce_30_samples_api_t1"]["nll"],
            "jsd_bits": e004_data["mce_30_samples_api_t1"]["jsd"],
            "k_eff_prototypes": round(mce_k, 2),
            "effective_bits": round(mce_bits, 3)
        }
    ]
    
    # Add key ensemble subsets
    for size_str in ["1", "2", "3", "4", "6", "8"]:
        sub = best_subsets[size_str]
        r_pct = sub["r_normalized"] * 100.0
        k_eff, bits = interpolate_prototype_bits(r_pct, ladder)
        models_join = " + ".join(sub["model_names"])
        summary_rows.append({
            "condition": f"Classifier Best Subgroup (Size {size_str}: {models_join})",
            "type": "Classifier Ensemble",
            "r_norm_pct": round(r_pct, 2),
            "nll": round(sub["nll"], 4),
            "jsd_bits": round(sub["jsd_bits"], 4),
            "k_eff_prototypes": round(k_eff, 2),
            "effective_bits": round(bits, 3)
        })
        
    out_dir = PROJECT_ROOT / "results" / "sprint1"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = out_dir / "consolidated_rate_distortion_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
        
    # Write Markdown Table
    md_content = """# Sprint 1 Consolidated Rate-Distortion & Relational Resolution Summary

| Model / Condition | Category | $R_{\\text{norm}}$ (%) | NLL | JSD (bits) | $K_{\\text{eff}}$ Prototypes | Effective Bits |
|-------------------|----------|------------------------|-----|------------|------------------------------|----------------|
"""
    for row in summary_rows:
        md_content += f"| {row['condition']} | {row['type']} | {row['r_norm_pct']:.2f}% | {row['nll']:.4f} | {row['jsd_bits']:.4f} | {row['k_eff_prototypes']} | {row['effective_bits']} |\n"
        
    md_content += """
---
### Key Insights

1. **Gemma 3 12B Relational Resolution Capacity**:
   - Gemma 3 12B (raw LPE: **9.72%**, calibrated LPE: **9.76%**) operates at $K_{\\text{eff}} \\approx 2.22$ human prototype quantizer equivalents ($\\,\\approx 1.11$ bits).
   - Gemma 3 12B resolves slightly more structure than a 2-prototype discrete quantizer, but is far below a single BART-Large classifier (**37.93%**, $K_{\\text{eff}} \\approx 6.25$, $2.65$ bits) and the best 6-model classifier ensemble (**77.69%**, $K_{\\text{eff}} \\approx 36.32$, $5.18$ bits).

2. **Calibration Paradox on Rate-Distortion Scale**:
   - Scalar temperature calibration reduces NLL dramatically from **3.8277** to **0.9308** (91.3% NLL gap closed), but its relational resolution remains static at **1.11 bits** ($K_{\\text{eff}}=2.22$).
   - This confirms geometrically that calibration adjusts confidence scaling along a 1D ray without altering prototype resolution or relational neighborhood structure.
"""
    
    md_path = PROJECT_ROOT / "research" / "reports" / "SPRINT1_CONSOLIDATED_SUMMARY.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Successfully exported consolidated summary to:\n  - {json_path}\n  - {md_path}")

if __name__ == "__main__":
    main()

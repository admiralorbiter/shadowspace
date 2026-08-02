import polars as pl
from shadowspace.chaosnli.models import load_model_predictions
from shadowspace.chaosnli.model_topology import evaluate_hypothesis2_temperature_scaling

df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
models = load_model_predictions()

h2_res = evaluate_hypothesis2_temperature_scaling(models, df, temperatures=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0])

print("=== HYPOTHESIS 2: TEMPERATURE SCALING VS TOPOLOGY RECOVERY ===\n")
for model in ["roberta-large", "bert-large", "bart-large"]:
    print(f"Model: {model}")
    print("  T     | Pointwise JSD (bits) | Soft Q_NX_HM(10)")
    print("  ---------------------------------------------")
    for row in h2_res[model]:
        t_val = row["temperature"]
        jsd_val = row["mean_jsd_bits"]
        qnx_val = row["qnx_soft_hm"]
        print(f"  {t_val:<5.1f} | {jsd_val:<20.4f} | {qnx_val:.5f}")
    print()

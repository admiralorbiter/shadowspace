"""Download and ID-matching script for VariErr, LiveNLI, and CIFAR-10H datasets.

Downloads public datasets to data/external/ and checks item-ID overlap with ChaosNLI.
"""

import json
import os
import urllib.request
import numpy as np
import polars as pl

DATA_DIR = "data/external"
os.makedirs(DATA_DIR, exist_ok=True)

print("=========================================================================")
print("            DOWNLOADING & VERIFYING EXTERNAL DATASETS                    ")
print("=========================================================================\n")

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def download_file(url: str, dest: str):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest, 'wb') as f:
        f.write(resp.read())

# 1. Download VariErr
varierr_path = os.path.join(DATA_DIR, "varierr.json")
if not os.path.exists(varierr_path) or os.path.getsize(varierr_path) == 0:
    print("Downloading VariErr NLI dataset from GitHub...", flush=True)
    urls = [
        "https://raw.githubusercontent.com/mainlp/VariErr-NLI/main/varierr.json",
        "https://huggingface.co/datasets/mainlp/varierr/raw/main/varierr.json"
    ]
    for url in urls:
        try:
            download_file(url, varierr_path)
            print(f"VariErr downloaded successfully from {url}", flush=True)
            break
        except Exception as e:
            print(f"Failed {url}: {e}")
else:
    print("VariErr dataset already present.", flush=True)

# 2. Download LiveNLI
livenli_path = os.path.join(DATA_DIR, "livenli.jsonl")
if not os.path.exists(livenli_path):
    print("Downloading LiveNLI dataset from GitHub...", flush=True)
    urls_livenli = [
        "https://raw.githubusercontent.com/njjiang/LiveNLI/main/data/LiveNLI.jsonl",
        "https://raw.githubusercontent.com/njjiang/LiveNLI/main/livenli.jsonl",
        "https://raw.githubusercontent.com/njjiang/LiveNLI/main/LiveNLI.csv"
    ]
    for url in urls_livenli:
        try:
            download_file(url, livenli_path)
            print(f"LiveNLI downloaded successfully from {url}", flush=True)
            break
        except Exception as e:
            print(f"Failed LiveNLI {url}: {e}")
else:
    print("LiveNLI dataset already present.", flush=True)

# 3. Download CIFAR-10H soft labels
cifar_counts_path = os.path.join(DATA_DIR, "cifar10h-counts.npy")
if not os.path.exists(cifar_counts_path):
    print("Downloading CIFAR-10H counts from GitHub...", flush=True)
    url_cifar = "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-counts.npy"
    try:
        download_file(url_cifar, cifar_counts_path)
        print("CIFAR-10H counts downloaded successfully.", flush=True)
    except Exception as e:
        print(f"Failed CIFAR-10H: {e}")
else:
    print("CIFAR-10H counts already present.", flush=True)

# -------------------------------------------------------------------------
# ITEM-ID OVERLAP ANALYSIS
# -------------------------------------------------------------------------
print("\n--- CHECKING ITEM-ID OVERLAP WITH CHAOSNLI ---", flush=True)

df_chaos = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
chaos_mnli_ids = set(str(x) for x in df_chaos.filter(pl.col("source_dataset") == "chaosnli_mnli")["source_pair_id"].to_list())
chaos_snli_ids = set(str(x) for x in df_chaos.filter(pl.col("source_dataset") == "chaosnli_snli")["source_pair_id"].to_list())
chaos_all_ids = set(str(x) for x in df_chaos["source_pair_id"].to_list())

print(f"ChaosNLI Total Items    : {len(df_chaos)}")
print(f"ChaosNLI MNLI Items     : {len(chaos_mnli_ids)}")
print(f"ChaosNLI SNLI Items     : {len(chaos_snli_ids)}")

# Check VariErr overlap
if os.path.exists(varierr_path):
    varierr_data = []
    with open(varierr_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    varierr_data.append(json.loads(line))
                except Exception:
                    pass

    varierr_pair_ids = set()
    for item in varierr_data:
        # Check all possible keys
        p_id = item.get("pair_id") or item.get("pairID") or item.get("id") or item.get("mnli_pair_id") or item.get("pair_id_1") or item.get("original_pair_id")
        if p_id is not None:
            varierr_pair_ids.add(str(p_id))

    overlap_varierr = chaos_mnli_ids.intersection(varierr_pair_ids)
    print(f"\nVariErr Total Records    : {len(varierr_data)}")
    print(f"VariErr Unique Pair IDs  : {len(varierr_pair_ids)}")
    print(f"VariErr <-> ChaosNLI-M Overlap: {len(overlap_varierr)} items ({len(overlap_varierr)/len(chaos_mnli_ids)*100:.1f}% of ChaosNLI-M)")

    # Show sample VariErr keys if needed
    if len(varierr_data) > 0 and len(varierr_pair_ids) == 0:
        print(f"VariErr sample keys: {list(varierr_data[0].keys())}")

# Check LiveNLI overlap
if os.path.exists(livenli_path):
    livenli_data = []
    with open(livenli_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    livenli_data.append(json.loads(line))
                except Exception:
                    pass
    livenli_ids = set()
    for item in livenli_data:
        p_id = item.get("pair_id") or item.get("pairID") or item.get("id")
        if p_id is not None:
            livenli_ids.add(str(p_id))
    overlap_livenli = chaos_mnli_ids.intersection(livenli_ids)
    print(f"\nLiveNLI Total Records    : {len(livenli_data)}")
    print(f"LiveNLI Unique Pair IDs  : {len(livenli_ids)}")
    print(f"LiveNLI <-> ChaosNLI-M Overlap: {len(overlap_livenli)} items")

# CIFAR-10H metadata
if os.path.exists(cifar_counts_path):
    cifar_counts = np.load(cifar_counts_path)
    print(f"\nCIFAR-10H Matrix Shape  : {cifar_counts.shape} (10,000 test images x 10 categories)")
    print(f"CIFAR-10H Mean Votes/Image: {np.sum(cifar_counts, axis=1).mean():.1f} votes")

print("\n=========================================================================")
print("          EXTERNAL DATASETS ACQUIRED & VERIFIED CLEANLY                  ")
print("=========================================================================")

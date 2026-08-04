"""Promote verified exploratory geometry artifacts into git-tracked result and documentation directories:

- HTML Visualizers -> docs/viz/chaosnli/geometry_lens.html
- Summary JSON Packages -> results/exploratory/*.json
"""

import shutil
from pathlib import Path

def promote_artifacts():
    artifacts_dir = Path("research/chaosnli/artifacts/exploratory")
    results_dir = Path("results/exploratory")
    viz_dir = Path("docs/viz/chaosnli")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Promote HTML visualizer
    html_src = artifacts_dir / "simplex_explorer.html"
    if html_src.exists():
        html_dst = viz_dir / "geometry_lens.html"
        shutil.copy2(html_src, html_dst)
        print(f"Promoted visualizer to {html_dst}")
        
    # 2. Promote JSON summaries
    for json_file in artifacts_dir.glob("*.json"):
        dst = results_dir / json_file.name
        shutil.copy2(json_file, dst)
        print(f"Promoted summary to {dst}")

if __name__ == "__main__":
    promote_artifacts()

# ruff: noqa: E402
# %% [markdown]
# # Shadowspace — Sprint 0 Smoke Demo
#
# This script demonstrates the dtour widget with the 15-point calibration fixture.
# Run it as a Jupyter notebook (open with Jupyter, or convert with
# `jupyter nbconvert --to notebook --execute notebooks/smoke_demo.py`).
#
# **What this shows:**
# - The calibration fixture loads correctly as a Polars DataFrame
# - dtour.Widget renders the fixture in the notebook
# - dtour.little_tour() generates a PCA little tour over the fixture

# %% [markdown]
# ## 1. Imports

# %%
import dtour
import polars as pl

from shadowspace.conventions import DTOUR_PINNED_VERSION
from shadowspace.data.calibration import calibration_fixture

print(f"dtour version: {dtour.__version__}")
print(f"Expected:      {DTOUR_PINNED_VERSION}")
assert dtour.__version__ == DTOUR_PINNED_VERSION, (
    f"dtour version mismatch: got {dtour.__version__!r}, "
    f"expected {DTOUR_PINNED_VERSION!r}"
)

# %% [markdown]
# ## 2. Load the calibration fixture

# %%
matrix, ids = calibration_fixture()
print(f"Matrix shape: {matrix.shape}")  # expect (15, 3)
print(f"Row sums:     {matrix.sum(axis=1)}")  # expect all 1.0

# Build a Polars DataFrame — dtour's native format
df = pl.DataFrame(
    {
        "object_id": ids,
        "p0": matrix[:, 0].tolist(),
        "p1": matrix[:, 1].tolist(),
        "p2": matrix[:, 2].tolist(),
    }
)
print(df)

# %% [markdown]
# ## 3. Render with dtour.Widget (Jupyter only)

# %%
# Simple scatter — shows the fixture in the widget
dtour.Widget(data=df.select(["p0", "p1", "p2"]))

# %% [markdown]
# ## 4. PCA little tour over the fixture

# %%
features = ["p0", "p1", "p2"]
tour = dtour.little_tour(df.select(features))

dtour.Widget(
    data=df.select(features),
    tour=tour,
    point_color_by=None,
)

# %% [markdown]
# ## 5. Record the installed dtour wheel hash
#
# Run the cell below and paste the output into `docs/DEPENDENCY_NOTES.md`.

# %%
import subprocess

result = subprocess.run(
    ["pip", "show", "--files", "dtour"],
    capture_output=True,
    text=True,
)
print(result.stdout)

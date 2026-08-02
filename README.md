# Shadowspace

> **Epistemic Reliability in Interactive High-Dimensional Visualization & Belief-Space Exploration**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Hardening Status](https://img.shields.io/badge/Hardening%20Gates-A--E%20Passed-brightgreen.svg)](#)
[![Test Suite](https://img.shields.io/badge/Tests-176%20passed%20%7C%2087.3%25%20coverage-success.svg)](#)

Shadowspace is an interactive visual analytics platform and research system engineered to solve the **epistemic unreliability of low-dimensional projections**. 

While static embedding methods (e.g. t-SNE, UMAP, 2D PCA) dominate exploratory data analysis in machine learning, they frequently induce miscalibration: users interpret projection artifacts—such as false clusters, artificial voids, and distorted distances—as intrinsic domain properties.

Shadowspace shifts exploratory visualization from passive layout viewing to **diagnostically transparent, multi-geometry exploration**. It combines continuous Grassmannian projection tours ($\operatorname{Gr}(2, p)$), live local integrity feedback overlays ($k$-NN intrusion/extrusion), multiple non-Euclidean representation spaces, guided subspace optimization, and cryptographically verified investigation records.

---

## 🌟 Key Capabilities & Features

- 🛸 **Grassmannian Projection Tours ($\operatorname{Gr}(2, p)$)**: Smooth, continuous 2D projection navigation along exact geodesic paths computed via SVD principal angle decomposition (GLERP).
- 🔬 **Live Integrity X-Ray Overlays**: Real-time diagnostic overlays distinguishing **Preserved Neighbors (Solid Green)**, **Torn Extrusions (Dashed Red)**, and **False Intrusions (Dotted Amber)** with live Trustworthiness ($T$) and Kruskal Stress-1 scores.
- 📐 **Multi-Geometry Representation Engine**: Seamlessly transforms coordinates across Raw Probabilities ($P$), Fisher-Rao Spherical Geometry ($\sqrt{P}$), Aitchison Compositional Geometry ($\operatorname{CLR}(P)$), Jensen-Shannon Distance, and Raw Logits.
- 🎯 **Guided Subspace Optimization**: Intent-driven optimization finding maximum class separation (Fisher LDA) or maximum local neighborhood integrity bases via smooth GLERP camera rotations.
- 🏷️ **Semantic Validity Badges**: Active epistemic safeguards that dynamically badge navigation frames, preventing misinterpretation of non-linear morph midpoints as analytical evidence.
- 📦 **Dual Storage Bundle Engine**: Reads and writes multi-file Parquet directory bundles and single-file SQLite vector bundles (`.db`) with zero-infrastructure `sqlite-vec` vector similarity search.
- 🛡️ **Rashomon Atlas & Point Stability**: Computes persistence metrics across candidate projection planes to isolate perspective-invariant structures from projection artifacts.
- 📜 **Cryptographic Investigation Records**: Export and import complete investigation sessions (`/api/export-record`) with SHA-256 matrix, object ID, and feature schema verification.
- 🗃️ **Benchmark & Dataset Suite**: Built-in dataset fetchers and CLI importers for Fashion-MNIST 10-class prediction belief spaces, Iris, Wine, Digits, and custom CSV/Parquet model outputs.

---

## 📐 Mathematical & Theoretical Foundation

### 1. True Geodesic Interpolation on the Grassmannian ($\operatorname{Gr}(2, p)$)
Projection navigation operates on the Grassmannian manifold $\operatorname{Gr}(2, p)$—the space of 2D linear subspaces in $\mathbb{R}^p$. Interpolation between two $p \times 2$ orthonormal basis matrices $B_A, B_B$ is computed via SVD principal angle decomposition of $B_A^\top B_B = U \Sigma V^\top$:

\[
Z_i(\tau) = \cos(\tau \theta_i) Y_{0,i}^* + \sin(\tau \theta_i) Q_i, \quad \tau \in [0, 1]
\]

where $\theta_i = \arccos(\sigma_i)$ are principal angles, $Y_0^* = B_A U$, $Y_1^* = B_B V$, and $Q_i = (Y_{1,i}^* - \cos \theta_i Y_{0,i}^*) / \sin \theta_i$. This guarantees constant angular velocity and exact basis orthonormality ($Z(\tau)^\top Z(\tau) = I_2$) across all intermediate frames.

### 2. Multi-Geometry Probability Manifolds
- **Raw Simplex ($P$)**: Standard probability vectors $p \in \Delta^{p-1}$ subject to Euclidean or Hellinger metrics.
- **Fisher-Rao Riemannian Geometry ($\sqrt{P}$)**: Square-root transformation mapping probability vectors to the positive orthant of a unit sphere $\mathbb{S}_+^{p-1}$ with Fisher-Rao Riemannian distance:
  \[
  d_{\mathrm{FR}}(p, q) = 2 \arccos\left(\sum_{i=1}^p \sqrt{p_i q_i}\right) \in [0, \pi] \text{ radians}
  \]
- **Aitchison Compositional Geometry ($\operatorname{CLR}(P)$)**: Centered Log-Ratio transform with multiplicative zero replacement policy ($p_i^* = \delta$ for exact zeros):
  \[
  \operatorname{clr}(p)_i = \ln p_i^* - \frac{1}{p} \sum_{j=1}^p \ln p_j^* \implies d_{\mathrm{Aitchison}}(p, q) = \|\operatorname{clr}(p) - \operatorname{clr}(q)\|_2
  \]
- **Jensen-Shannon Distance**: Square root of base-2 JS divergence $d_{\mathrm{JS}}(p, q) = \sqrt{\frac{1}{2} D_{\mathrm{KL}}(p \| m) + \frac{1}{2} D_{\mathrm{KL}}(q \| m)} \in [0, 1]$.

### 3. Local Integrity Metrics ($k$-NN Neighborhood Analysis)
For a target object $i$ and neighborhood size $k$:
- **Preserved Neighbors ($N_i^{2D} \cap N_i^{HD}$)**: High-D neighbors correctly preserved in 2D (Green).
- **Torn Extrusions ($N_i^{HD} \setminus N_i^{2D}$)**: True high-D neighbors torn apart by 2D flattening (Red Dashed).
- **False Intrusions ($N_i^{2D} \setminus N_i^{HD}$)**: Unrelated high-D points brought artificially close in 2D (Amber Dotted).
- **Trustworthiness Score ($T(k)$)**:
  \[
  T(k) = 1 - \frac{2}{N k (2N - 3k - 1)} \sum_{i=1}^N \sum_{j \in N_i^{2D} \setminus N_i^{HD}} (r(i, j) - k)
  \]

---

## 🚀 Quickstart & Installation

### Requirements
- **Python**: 3.12 or higher
- **OS**: Windows, macOS, or Linux

### Installation

Clone the repository and install dependencies using standard `pip`:

```bash
git clone https://github.com/admiralorbiter/shadowspace.git
cd shadowspace

# Install requirements
python -m pip install -r requirements.txt

# Or install in editable mode with all extras
python -m pip install -e .[datasets,dev]
```

### Launch Interactive Workbench

Start the web application:

```bash
python app.py
```

Then open your browser to **`http://127.0.0.1:5000`**.

---

## 💻 Command-Line Interface (`shadowspace`)

Shadowspace provides a comprehensive CLI for bundle generation, ingestion, validation, and benchmark management.

```bash
# Generate synthetic 4-class belief space bundle
shadowspace generate synthetic --classes 4 --samples 2000 --output data/bundles/synthetic-v1

# Generate 3-class calibration simplex fixture
shadowspace generate calibration --output data/bundles/calibration-v1

# Import custom CSV classifier outputs into a Shadowspace bundle
shadowspace import-csv --input model_predictions.csv --output data/bundles/my_model --id-col sample_id --label-col target --normalize

# Import Parquet dataset into a Shadowspace bundle
shadowspace import-parquet --input embeddings.parquet --output data/bundles/my_embeddings --id-col id --label-col class

# List benchmark datasets registered in Shadowspace
shadowspace datasets list

# Fetch benchmark dataset (e.g. Fashion-MNIST, Iris, Wine, Digits)
shadowspace datasets fetch --datasets iris,wine,digits --output data/bundles

# Inspect benchmark dataset specs
shadowspace datasets info fashion_mnist

# Validate bundle schema, SHA-256 hashes, and matrix contracts
shadowspace validate-bundle data/bundles/synthetic-v1
```

---

## 🐍 Python API Usage Examples

### 1. Single-File SQLite Bundle Reader & Vector Search

```python
from shadowspace.bundle.sqlite_reader import SQLiteBundleReader

# Load single-file SQLite vector bundle (.db)
with SQLiteBundleReader("data/bundles/synthetic-v1.db") as reader:
    manifest = reader.read_manifest()
    print(f"Dataset: {manifest.dataset_name}, Objects: {manifest.total_objects}")

    # Extract probability matrix and IDs
    matrix, object_ids = reader.get_representation_matrix("probability")
    print(f"Loaded matrix shape: {matrix.shape}")
```

### 2. Geometry Transforms & Metrics

```python
import numpy as np
from shadowspace.math.metrics import pairwise_fisher_rao, pairwise_aitchison
from shadowspace.math.clr import clr_transform

# Sample probability distributions
P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])

# Compute pairwise Fisher-Rao distances (2 * arccos(BC))
d_fr = pairwise_fisher_rao(P)

# Transform to Centered Log-Ratio (CLR) coordinates
clr_coords = clr_transform(P)
d_aitchison = pairwise_aitchison(P)
```

### 3. Grassmannian Geodesics & Rashomon Sets

```python
import numpy as np
from shadowspace.projection.paths import grassmann_geodesic
from shadowspace.math.stability import generate_rashomon_set

# Compute geodesic midpoint on Gr(2, 5) manifold
B_start, _ = np.linalg.qr(np.random.randn(5, 2))
B_end, _ = np.linalg.qr(np.random.randn(5, 2))
B_mid = grassmann_geodesic(B_start, B_end, tau=0.5)

# Generate diverse Rashomon projection candidate set (d_G >= 12 degrees)
X = np.random.dirichlet((1, 1, 1, 1), size=100)
rashomon_views = generate_rashomon_set(X, n_candidates=4, quality_threshold=0.60)
```

---

## 🌐 Web Workbench & REST API

The Flask backend ([`src/shadowspace/server/routes.py`](src/shadowspace/server/routes.py)) exposes rich REST endpoints powering the interactive browser workbench:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Interactive Workbench Web UI (Dual-sidebar, Scatter Canvas, HUD) |
| `/api/datasets` | `GET` | List available benchmark and imported dataset bundles |
| `/api/fixture` | `GET` | Retrieve complete dataset payload, representations, and catalog projections |
| `/api/import-dataset` | `POST` | Dynamically upload and ingest CSV/Parquet datasets (<10 MB) |
| `/api/diagnostics` | `GET` | Compute point-level $k$-NN intrusion/extrusion, Trustworthiness, Stress |
| `/api/topology` | `GET` | Compute complete $k$-NN graph categorized into preserved, torn, and false edges |
| `/api/distortion-grid` | `GET` | Compute spatial 2D viewport distortion heatmaps |
| `/api/subspace-angles` | `GET` | Calculate canonical principal angles $\theta_1, \theta_2$ and Grassmannian distance $d_G$ |
| `/api/object-payload` | `GET` | Inspect raw source ground truth (images, logits, entropy, predictions) |
| `/api/tour-path` | `GET` | Generate GLERP geodesic projection tour keyframes on $\operatorname{Gr}(2, p)$ |
| `/api/optimize-view` | `GET` | Solve Fisher LDA class separability or local covariance integrity bases |
| `/api/point-stability` | `GET` | Evaluate point stability across Rashomon candidate projections |
| `/api/rashomon-set` | `GET` | Generate diverse set of high-quality candidate projections |
| `/api/saved-views` | `GET/POST/DELETE` | Atlas state management for saved analytical views and notes |
| `/api/export-record` | `GET` | Export complete investigation session with SHA-256 cryptographic digests |
| `/api/import-record` | `POST` | Validate SHA-256 hashes against active dataset and restore view states |
| `/api/health` | `GET` | System health check reporting hardening gate milestones & `sqlite-vec` status |

---

## 🎮 Workbench Keyboard Shortcuts

| Input | Action | Description |
| :--- | :--- | :--- |
| **`Mouse Drag`** | Pan Viewport | Pan 2D viewport |
| **`Wheel / + -`** | Zoom | Pivot-zoom centered on cursor or target selection |
| **`Shift + Drag`** | Marquee Selection | Multi-point bounding box selection for aggregate HUD inspection |
| **`🔍 Focus`** | Fit Neighborhood | Smoothly auto-fit camera bounds around target point and its $k$-NN graph |
| **`↺ Reset`** | Reset Camera | Restore default isotropic 1.0x camera zoom and extent |
| **`← / →`** | Step Selection | Cycle target point selection through dataset |
| **`1 / 2`** | Toggle Geometry | Switch between Raw Probability ($P$) and Fisher-Rao ($\sqrt{P}$) spaces |
| **`Space`** | Play / Pause | Toggle continuous Grand Tour projection animation |

---

## 🧪 Testing & Validation Suite

Shadowspace maintains an extensive automated test suite with **176 passed tests** and **87.3% code coverage**.

Run the test suite using `pytest`:

```bash
# Run pytest with coverage report
python -m pytest

# Run specific test module
python -m pytest tests/test_paths.py
```

### Coverage Breakdown

- **Pure Mathematical Core** (`math/`, `projection/`): >90% branch coverage
- **Data Bundle Schemas & Readers** (`bundle/`, `models/`): >90% branch coverage
- **CLI & Importers** (`cli.py`, `importers/`): Clean contract validation
- **Server Routes & API** (`server/routes.py`): Full integration testing

---

## 📚 Documentation Index

Detailed specifications, math papers, decisions, and data contracts are documented in the [`docs/`](docs/) directory:

- [ARCHITECTURE_AND_DATA_CONTRACT.md](docs/ARCHITECTURE_AND_DATA_CONTRACT.md): Comprehensive system architecture, Pydantic schemas, and data bundle contracts.
- [MATHEMATICAL_AND_RESEARCH_KNOWLEDGE_BASE.md](docs/MATHEMATICAL_AND_RESEARCH_KNOWLEDGE_BASE.md): Mathematical definitions, Grassmannian geometry, probability simplex representations, and reading list.
- [PROJECT_DECISIONS.md](docs/PROJECT_DECISIONS.md): Testing strategy, test layers, canonical fixtures A–E, and mathematical invariant tests.
- [RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md): Research identity, novelties, RQ1–RQ4 hypotheses, and near-term experiment sequences.
- [TESTING_AND_VALIDATION.md](docs/TESTING_AND_VALIDATION.md): Test verification targets and fixture specifications.

---

## 📜 License

Shadowspace is licensed under the **[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE)**. You are free to share and adapt the material for non-commercial purposes with attribution. Commercial use requires explicit permission.

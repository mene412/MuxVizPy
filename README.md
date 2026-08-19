# MuxVizPy

**MuxVizPy** is a Python package for multilayer and multiplex network analysis, inspired by the original [MuxViz](http://muxviz.net/) software.
It provides tools to compute centralities, structural descriptors, mesoscale properties, percolation, and versatile visualizations—backed by [`graph-tool`](https://graph-tool.skewed.de/) and the scientific Python ecosystem.

---

## Features

- Construction of multilayer networks from CSV or metadata
- Aggregate and supra-adjacency matrix utilities
- Multiplex-aware centrality measures:
  - Eigenvector, Katz, Random Walk (PageRank/classical), Hub, Authority, K-Core
- Topological descriptors:
  - Largest connected components, shortest-path matrices, similarity
- Mesoscale analysis:
  - Modularity, blockmodels, assortativity
- Percolation analysis
- 3D multilayer network visualization with `matplotlib`

---

## Installation

This project uses a **conda + [uv](https://docs.astral.sh/uv/) hybrid** workflow: conda provides `graph-tool` (a C++ library not available on PyPI), and uv handles all Python dependencies.

### 1. Create conda environment with graph-tool

```bash
conda create -n muxvizpy python=3.12 graph-tool -c conda-forge
conda activate muxvizpy
```

### 2. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install the package

```bash
git clone https://github.com/your-username/MuxVizPy.git
cd MuxVizPy

# Core install
uv pip install -e .

# With development tools (pytest, ruff, mypy)
uv pip install -e ".[dev]"
```

> **Important**: Use `uv pip install` (not `uv sync`) so that uv operates inside the activated conda environment and can see the conda-installed `graph-tool`.

### Optional: PyTorch support

For tensor decomposition and sparse tensor operations:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
uv pip install -e ".[torch]"
```

---

## Basic Usage

For a basic usage script see the file `scripts/test.py`.

---

## Requirements

- Python >= 3.10
- graph-tool >= 2.45 (installed via conda, see above)
- numpy, scipy, pandas, polars, matplotlib, tqdm, tensorly, sparse, pyarrow

See `pyproject.toml` for the full list of core and optional dependencies.

---

## License

MIT License 2025

---

## Acknowledgements

- Inspired by the original [MuxViz](https://github.com/manlius/muxViz) platform.
- Built with `graph-tool`, a performant graph analysis library.

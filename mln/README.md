# Multilayers networks studies with MuxVizPy

**MuxVizPy** is a Python package for multilayer and multiplex network analysis, inspired by the original [MuxViz](http://muxviz.net/) software.
It provides tools to compute centralities, structural descriptors, mesoscale properties, percolation, and versatile visualizations—backed by [`graph-tool`](https://graph-tool.skewed.de/) and the scientific Python ecosystem.

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

# mln install, install the extra libraries
uv pip install -r mln/requirements-mln.txt

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

Goal: clustering the 285 samples/patients based on the 423 genes.

Starting from 2 different files (DNA methylation and RNA expression) different pipelines can be easily executed to find out the best clustering.

### Clustering directly on each input matrix alone

input: data/

### Clustering on pagerank results run on a single input matrix, for both matrices

input (secondario): output/pagerank/DNA
input (secondario): output/pagerank/RNA

### Clustering on pagerank run on the multiplex network with 2 layers

input (secondario): output/pagerank/mln

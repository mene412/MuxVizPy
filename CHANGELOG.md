# Changelog

All notable changes to MuxVizPy are documented in this file.

## [Unreleased] (since 540100d)

### New Features

- **Information theory module** — Von Neumann entropy and Jensen-Shannon divergence computation between networks, with density matrix builder. (`96f3422`, `d27776d`)

<!-- TODO: add mathematical details / pseudocode for entropy and JSD -->

- **Sparse CP decomposition** — New `decomposition.py` implementing sparse CP decomposition for 4D sparse tensors, with backend support for Numpy (CPU) and RAPIDS/cuML (GPU). (`47f2965`, `5f37e51`, `a0db3b5`, `ac00cd7`, `e681c6e`)

<!-- TODO: add mathematical details / pseudocode for CP decomposition -->

- **Global descriptors** — Average global overlap and average global clustering coefficient functions matching muxViz R reference. (`942d973`, `695a742`)

<!-- TODO: add mathematical details / pseudocode for global descriptors -->

- **Local clustering coefficient** — Added `compute_local_clustering_coefficient` to mesoscale module. (`c4e5ca9`)
- **Multidegree functions** — Added remaining multidegree functions. (`74869c8`)

### Refactoring

- **Utils reorganization** — Refactored `utils.py` into a `utils/` package with `io.py` (edgelist and supra-adjacency I/O), `parsing.py` (tensor/supra-adjacency conversions), and `misc.py` submodules. (`f5fc6b8`, `2501980`, `6561ee5`, `b52a1cf`)
- **Centrality unification** — Unified random walk centrality into single `compute_multi_rw_centrality` handling both classical and PageRank modes. (`5d030c5`, `a4fc5e5`)
- **Katz centrality** — Integrated dual computation pathway (exact + approximate) into single `compute_katz_centrality` function. (`876acc2`, `c961486`)
- **Eigenvector centrality** — Deprecated separate public API; unified into single entry point. (`baa5979`)
- **Power iteration** — Moved eigenvalue helpers to `leading_eigenv_approx` module, then deprecated the standalone script; relocated approximation logic into utils. (`a901f83`, `0a1b832`, `8522e63`)
- **Visualization** — Merged `plotMux.py` and `visualization.py` into a single module. (`149d2ef`)
- **versatility metrics integration** — Integrated multiple node-based metrics into `versatility.py`. (`c2b3d00`)
- **Renamed `global.py`** to `global_descriptors.py` and fixed `agov` halving bug. (`edd3e64`)
- **Backward-compat wrappers** — Added wrappers for old function names after renames. (`1acca5d`)

### Bug Fixes

- Fix hub/authority approximate eigenvector centrality: use `alpha=1.0, cval=eps` instead of PageRank constants, which produced incorrect results. Also, it now computes the eigenvector decomposition on the right matrices. (`4ee64d6`)
- Fix closeness reference values against muxViz expected output. @todo This still require attention. (`1acf591`)
- Fix import bug, categorical coupling, and Laplacian computation in `parsing.py`. (`28c4356`)
- Fix typos in missing parameters and import statements. (`c532490`)
- Skip local clustering computation for large networks to avoid excessive runtime. (`e0c309b`)
- Fix bug in get_multi_degree which was binarizing the aggregated network removing the molteplicity of edges. Now computes accordingly to the definition in the paper.

### Deprecations

- Deprecated non-negative constraint in optimization (caused convergence to local minima). (`110feb3`)
- Deprecated standalone muxvizpy power iteration eigensolver. (`2eb8384`)
- Deprecated some overlapping apis to compute the same metrics. (`a4fc5e5`, `baa5979`)
- Deprecated functions now defined in `utils/parsing`. (`2d146ca`)

"""
MuxVizPy: A Python implementation of MuxViz for multilayer network analysis.

This package provides tools to build, analyze, and visualize multilayer (edge-colored) networks,
including multilayer centrality measures, supra-adjacency matrix construction, percolation analysis,
and mesoscale community detection using stochastic block models.

Modules
-------
- core:           VirusMultiplex and VirusMultiplex_from_dirlist constructors
- versatility:    Multilayer centrality functions
- global_descriptors: Global properties of multilayer networks
- topology:       Connected components, LCC, LIC, LVC
- percolation:    Vertex percolation analysis
- mesoscale:      SBM modularity and inter-layer correlations
- utils:          IO utilities and node-name access
- visualization:  3D multiplex plotting and centrality-based edge-colored layouts
"""

from .utils.approx_utils import leading_eigenv_approx
import importlib

_LAZY_MODULES = [
    "versatility", "topology", "mesoscale", "percolation",
    "global_descriptors", "utils", "visualization",
]

__all__ = [
    "leading_eigenv_approx",
    *_LAZY_MODULES,
]

def __getattr__(name):
    if name in _LAZY_MODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""General-purpose utilities: IO, config, logging, misc helpers."""

import importlib

from .misc import writeComponent, readComponent, get_names

_LAZY_SUBMODULES = ["io", "misc", "parsing", "approx_utils", "katz_utils"]

__all__ = [
    *_LAZY_SUBMODULES,
    "writeComponent",
    "readComponent",
    "get_names",
]

def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
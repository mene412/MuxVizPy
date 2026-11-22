"""Plot individually selected clusterings from result directories.

Edit PLOTS below, then run:

    python mln/plot_all_clusterings.py

Each selected clustering gets PCA, UMAP, and t-SNE figures via
``plot_clusterings.py``. For Louvain, the final field can be ``none`` to plot
its result regardless of size, or an integer to select the result only when
Louvain detected that many clusters.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# User configuration: choose methods separately for each result folder.
# ---------------------------------------------------------------------

PLOTS = [
    {
        "base_dir": Path("output/string_net/clustering/baseline/RNA_k2-25"),
        "clusterings": ["euclidean,hierarchical,2"],
    },
    {
        "base_dir": Path("output/string_net/clustering/baseline/DNA_k2-25"),
        "clusterings": ["cosine,louvain"],
    },
    # Add more result folders and selections as needed, for example:
    # {
    #     "base_dir": Path("output/string_net/clustering/pagerank/RNA_k2-25"),
    #     "clusterings": ["euclidean,kmeans,5", "correlation,hierarchical,10"],
    # },
]


def main() -> None:
    script = Path(__file__).with_name("plot_clusterings.py")
    missing = [
        str(item["base_dir"])
        for item in PLOTS
        if not item["base_dir"].is_dir()
    ]
    if missing:
        print("Skipping missing result directories:")
        for directory in missing:
            print(f"  {directory}")

    for item in PLOTS:
        directory = item["base_dir"]
        if not directory.is_dir():
            continue

        clusterings = item["clusterings"]
        print(f"\nPlotting {directory}: {', '.join(clusterings)}")
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--base-dir",
                str(directory),
                "--clusterings",
                *clusterings,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

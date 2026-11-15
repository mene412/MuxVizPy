"""Plot clustering metrics from aggregate ``ALL_clustering_metrics.csv`` files.

Usage::

    python -m mln.plot_metrics output/string_net
    python mln/plot_metrics.py output/pathway_in_cancer

Plots are saved in a ``metric_plots`` directory under the supplied path.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = ("silhouette", "ari_group", "ari_cms")


def _safe_name(value: str) -> str:
    """Convert a relative path to a filename-friendly label."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _plot_file(csv_path: Path, root: Path, output_dir: Path) -> int:
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError) as exc:
        print(f"Skipping unreadable file {csv_path}: {exc}")
        return 0

    if frame.empty:
        print(f"Skipping empty file: {csv_path}")
        return 0

    missing = [column for column in METRICS if column not in frame.columns]
    if missing:
        print(f"Skipping {csv_path}: missing metric columns {', '.join(missing)}")
        return 0

    # Use k when available; Louvain rows without k are positioned by their
    # detected cluster count so they can appear on the same x axis.
    x_column = "k" if "k" in frame.columns else "n_clusters"
    if x_column not in frame.columns:
        print(f"Skipping {csv_path}: neither 'k' nor 'n_clusters' is present")
        return 0
    if x_column == "k" and "n_clusters" in frame.columns:
        frame[x_column] = frame[x_column].fillna(frame["n_clusters"])

    group_columns = [name for name in ("algorithm", "distance") if name in frame]
    if not group_columns:
        frame["method"] = "all results"
        group_columns = ["method"]

    label_column = " / ".join(group_columns)
    if len(group_columns) == 1:
        frame[label_column] = frame[group_columns[0]].fillna("unknown").astype(str)
    else:
        frame[label_column] = (
            frame[group_columns[0]].fillna("unknown").astype(str)
            + " / "
            + frame[group_columns[1]].fillna("unknown").astype(str)
        )

    relative = csv_path.parent.relative_to(root)
    prefix = _safe_name(str(relative)) or "root"
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for metric in METRICS:
        data = frame[[x_column, label_column, metric]].copy()
        data[x_column] = pd.to_numeric(data[x_column], errors="coerce")
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data = data.dropna(subset=[x_column, metric])
        if data.empty:
            print(f"No numeric {metric} values in {csv_path}")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for label, series in data.groupby(label_column, sort=True):
            series = series.groupby(x_column, as_index=False)[metric].mean()
            series = series.sort_values(x_column)
            ax.plot(series[x_column], series[metric], marker="o", label=label)

        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} — {relative or Path('.')}")
        ax.grid(True, alpha=0.3)
        ax.legend(title=label_column, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        destination = output_dir / f"{prefix}_{metric}.png"
        fig.savefig(destination, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {destination}")
        written += 1

    return written


def plot_metrics(input_path: Path) -> int:
    """Find aggregate metric files below *input_path* and plot their metrics."""
    root = input_path.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    files = sorted(root.rglob("ALL_clustering_metrics.csv"))
    if not files:
        print(f"No ALL_clustering_metrics.csv files found under {root}")
        return 0

    output_dir = root / "clustering" / "metric_plots"
    count = sum(_plot_file(path, root, output_dir) for path in files)
    print(f"Created {count} plot(s) from {len(files)} aggregate file(s).")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot silhouette and ARI metrics from clustering output files."
    )
    parser.add_argument("path", type=Path, help="Output directory to search recursively")
    args = parser.parse_args()
    plot_metrics(args.path)


if __name__ == "__main__":
    main()

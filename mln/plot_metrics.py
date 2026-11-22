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
METRIC_LABELS = {
    "silhouette": "Silhouette",
    "ari_group": "ARI: tumor vs normal",
    "ari_cms": "ARI: CMS subtypes",
}


def _safe_name(value: str) -> str:
    """Convert a relative path to a filename-friendly label."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _set_cluster_ticks(ax: plt.Axes) -> None:
    """Show every cluster count through five, preserving the wider spacing after."""
    existing = ax.get_xticks()
    ticks = sorted({2, 3, 4, 5, *(tick for tick in existing if tick > 5)})
    ax.set_xticks(ticks)


def _read_metrics(csv_path: Path, root: Path) -> tuple[str, pd.DataFrame] | None:
    """Read one aggregate file and return its display label and data."""
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError) as exc:
        print(f"Skipping unreadable file {csv_path}: {exc}")
        return None

    if frame.empty:
        print(f"Skipping empty file: {csv_path}")
        return None

    missing = [column for column in METRICS if column not in frame.columns]
    if missing:
        print(f"Skipping {csv_path}: missing metric columns {', '.join(missing)}")
        return None

    # Use k when available; Louvain rows without k are positioned by their
    # detected cluster count so they can appear on the same x axis.
    x_column = "k" if "k" in frame.columns else "n_clusters"
    if x_column not in frame.columns:
        print(f"Skipping {csv_path}: neither 'k' nor 'n_clusters' is present")
        return None
    if x_column == "k" and "n_clusters" in frame.columns:
        frame[x_column] = frame[x_column].fillna(frame["n_clusters"])

    group_columns = [name for name in ("algorithm", "distance") if name in frame]
    if not group_columns:
        frame["method"] = "all results"
        group_columns = ["method"]

    label_column = "method"
    if len(group_columns) == 1:
        frame[label_column] = frame[group_columns[0]].fillna("unknown").astype(str)
    else:
        frame[label_column] = (
            frame[group_columns[0]].fillna("unknown").astype(str)
            + " / "
            + frame[group_columns[1]].fillna("unknown").astype(str)
        )

    relative = csv_path.parent.relative_to(root)
    setup = str(relative) if str(relative) != "." else root.name
    frame["_x"] = pd.to_numeric(frame[x_column], errors="coerce")
    frame["_method"] = frame[label_column].astype(str)
    return setup, frame


def _plot_all(
    entries: list[tuple[str, pd.DataFrame]], output_dir: Path, root: Path
) -> Path:
    """Create a shared-scale metric-by-setup panel figure."""
    methods = sorted({
        method
        for _, frame in entries
        for method in frame["_method"].dropna().unique()
    })
    palette = plt.get_cmap("tab10")
    colors = {method: palette(i % 10) for i, method in enumerate(methods)}

    rows, columns = len(entries), len(METRICS)
    fig, axes = plt.subplots(
        rows, columns,
        figsize=(4.2 * columns, 4.5 * rows),
        sharey="col", squeeze=False,
    )
    for row, (setup, frame) in enumerate(entries):
        for column, metric in enumerate(METRICS):
            ax = axes[row, column]
            data = frame[["_x", "_method", metric]].copy()
            data[metric] = pd.to_numeric(data[metric], errors="coerce")
            data = data.dropna(subset=["_x", metric])
            for method, series in data.groupby("_method", sort=True):
                series = series.groupby("_x", as_index=False)[metric].mean()
                series = series.sort_values("_x")
                ax.plot(
                    series["_x"], series[metric], marker="o", linewidth=1.8,
                    markersize=4, color=colors[method], label=method,
                )
            ax.set_ylim(-0.1, 1.05)
            _set_cluster_ticks(ax)
            ax.set_xlim(right=26)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Number of clusters (k)")
            if row == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(f"{setup.replace('/', ' / ')}\n{METRIC_LABELS[metric]}", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    bottom = 0.18 if handles else 0.02
    if handles:
        fig.legend(
            handles, labels, title="Algorithm / distance",
            loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=min(5, len(labels)),
        )
    fig.suptitle(f"Clustering metrics — {root.name}", y=0.995)
    fig.tight_layout(rect=(0, bottom, 1, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "clustering_metrics.png"
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return destination


def _plot_individual(
    entries: list[tuple[str, pd.DataFrame]], output_dir: Path
) -> int:
    """Save one full-size figure per setup and metric for easier inspection."""
    written = 0
    for setup, frame in entries:
        prefix = _safe_name(setup) or "root"
        for metric in METRICS:
            data = frame[["_x", "_method", metric]].copy()
            data[metric] = pd.to_numeric(data[metric], errors="coerce")
            data = data.dropna(subset=["_x", metric])
            if data.empty:
                continue

            fig, ax = plt.subplots(figsize=(9, 6))
            for method, series in data.groupby("_method", sort=True):
                series = series.groupby("_x", as_index=False)[metric].mean()
                series = series.sort_values("_x")
                ax.plot(
                    series["_x"], series[metric], marker="o", linewidth=2,
                    markersize=6, label=method,
                )
            ax.set_ylim(-0.1, 1.05)
            _set_cluster_ticks(ax)
            ax.set_xlim(right=26)
            ax.set_xlabel("Number of clusters (k)")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(f"{METRIC_LABELS[metric]} — {setup}")
            ax.grid(True, alpha=0.3)
            ax.legend(title="Algorithm / distance", loc="best")
            fig.tight_layout()
            destination = output_dir / f"{prefix}_{metric}.png"
            fig.savefig(destination, dpi=220, bbox_inches="tight")
            plt.close(fig)
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
    entries = [entry for path in files if (entry := _read_metrics(path, root)) is not None]
    if not entries:
        print("No usable aggregate metric files found.")
        return 0

    destination = _plot_all(entries, output_dir, root)
    print(f"Saved combined metric figure: {destination}")
    individual_count = _plot_individual(entries, output_dir)
    print(f"Saved {individual_count} individual setup/metric plot(s).")
    print(f"Included {len(entries)} setup(s) from {len(files)} aggregate file(s).")
    return individual_count + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot silhouette and ARI metrics from clustering output files."
    )
    parser.add_argument("path", type=Path, help="Output directory to search recursively")
    args = parser.parse_args()
    plot_metrics(args.path)


if __name__ == "__main__":
    main()

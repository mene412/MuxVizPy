from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_DIR = Path("output/clustering/pagerank_comprehensive")
OUTPUT_DIR = INPUT_DIR / "silhouette_vs_k"

K_VALUES = range(2, 6)

SAVE_PDF = False
SAVE_PNG = True
DPI = 300

# ---------------------------------------------------------------------
# Plot style & utilities
# ---------------------------------------------------------------------


def configure_plotting() -> None:
    """Configure matplotlib for clean publication-style figures."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig, path: Path) -> None:
    make_dir(path.parent)

    if SAVE_PNG:
        fig.savefig(
            path.with_suffix(".png"),
            dpi=DPI,
            bbox_inches="tight",
        )

    if SAVE_PDF:
        fig.savefig(
            path.with_suffix(".pdf"),
            bbox_inches="tight",
        )

    plt.close(fig)


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[!] Missing: {path}")
        return None

    try:
        df = pd.read_csv(path)
        print(f"[+] Loaded: {path}")
        return df
    except Exception as exc:
        print(f"[!] Could not read {path}: {exc}")
        return None


# ---------------------------------------------------------------------
# Load data & assignments
# ---------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_path = INPUT_DIR / "aggregated_pagerank_matrix.csv"
    master_metrics_path = INPUT_DIR / "ALL_clustering_metrics.csv"

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist:\n{INPUT_DIR}")

    matrix = read_csv_safe(matrix_path)
    if matrix is None:
        raise FileNotFoundError(f"Could not load Pagerank matrix:\n{matrix_path}")

    metrics = read_csv_safe(master_metrics_path)
    if metrics is None:
        files = sorted(INPUT_DIR.rglob("*_clustering_metrics.csv"))
        if not files:
            raise FileNotFoundError("No clustering metrics CSV files found.")
        metrics = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        metrics.to_csv(master_metrics_path, index=False)

    matrix = matrix.set_index(matrix.columns[0])
    matrix.index = matrix.index.astype(str)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return matrix, metrics


def make_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    if metric in {"euclidean", "cosine", "correlation"}:
        return squareform(pdist(X, metric=metric))
    raise ValueError(f"Unsupported distance metric: {metric}")


def find_assignment_file(algorithm: str, distance: str, k: int | None) -> Path | None:
    if algorithm == "louvain":
        candidates = sorted(
            INPUT_DIR.glob("all_samples/louvain/*_louvain_assignments.csv")
        )
        return candidates[0] if candidates else None

    if k is None:
        return None

    if algorithm == "hierarchical":
        path = (
            INPUT_DIR
            / "all_samples"
            / "clusters_by_k"
            / distance
            / f"k{k}"
            / f"all_samples_hierarchical_{distance}_k{k}_assignments.csv"
        )
        return path if path.exists() else None

    if algorithm == "kmeans":
        path = (
            INPUT_DIR
            / "all_samples"
            / "clusters_by_k"
            / "euclidean"
            / f"k{k}"
            / f"all_samples_kmeans_k{k}_assignments.csv"
        )
        return path if path.exists() else None

    return None


def load_assignments(path: Path) -> pd.DataFrame:
    df = read_csv_safe(path)
    if df is None:
        raise FileNotFoundError(path)

    required = {"sample_id", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df[["sample_id", "cluster"]].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
    df = df.dropna(subset=["cluster"])
    df["cluster"] = df["cluster"].astype(int)
    return df.drop_duplicates(subset="sample_id")


# ---------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------


def plot_single_curve(df_subset: pd.DataFrame, title: str, output_path: Path) -> None:
    """Plot silhouette score vs k for one algorithm/distance combination."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        df_subset["k"],
        df_subset["silhouette"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="#1f77b4",
    )

    # Highlight best k
    best_idx = df_subset["silhouette"].idxmax()
    best_k = int(df_subset.loc[best_idx, "k"])
    best_val = df_subset.loc[best_idx, "silhouette"]

    ax.scatter(
        best_k,
        best_val,
        color="crimson",
        s=90,
        zorder=5,
        label=f"Peak: k={best_k} ({best_val:.3f})",
    )

    ax.set_title(title)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Mean Silhouette Score")
    ax.set_xticks(list(df_subset["k"]))
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_combined_curves(
    results_df: pd.DataFrame,
    louvain_score: float | None,
    louvain_k: int | None,
    output_path: Path,
) -> None:
    """Plot all algorithms and distance metrics in a single comparison plot."""
    fig, ax = plt.subplots(figsize=(9, 6))

    color_map = {
        ("hierarchical", "euclidean"): ("#1f77b4", "-"),
        ("hierarchical", "cosine"): ("#2ca02c", "-"),
        ("hierarchical", "correlation"): ("#ff7f0e", "-"),
        ("kmeans", "euclidean"): ("#9467bd", "--"),
    }

    grouped = results_df.dropna(subset=["k"]).groupby(["algorithm", "distance"])

    for (alg, dist), group in grouped:
        group_sorted = group.sort_values("k")
        color, style = color_map.get((alg, dist), (None, "-"))
        label = f"{alg.title()} ({dist})"

        ax.plot(
            group_sorted["k"],
            group_sorted["silhouette"],
            label=label,
            color=color,
            linestyle=style,
            marker="o",
            linewidth=2,
            markersize=5,
        )

    # Add Louvain as a reference marker
    if louvain_score is not None and not np.isnan(louvain_score):
        lbl = f"Louvain (cosine, k={louvain_k}: {louvain_score:.3f})"
        ax.axhline(
            y=louvain_score,
            color="grey",
            linestyle=":",
            linewidth=1.5,
            label=lbl,
        )

    ax.set_title("Silhouette Comparison Across Clustering Configurations")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Mean Silhouette Score")
    ax.set_xticks(list(K_VALUES))
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, output_path)


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------


def run_silhouette_analysis() -> None:
    configure_plotting()
    make_dir(OUTPUT_DIR)

    matrix, _ = load_data()
    print(f"\nLoaded matrix: {matrix.shape[0]} samples x {matrix.shape[1]} genes")

    results = []

    # 1. Evaluate Hierarchical + K-Means across k
    configs = [
        ("hierarchical", "euclidean"),
        ("hierarchical", "cosine"),
        ("hierarchical", "correlation"),
        ("kmeans", "euclidean"),
    ]

    for algorithm, distance in configs:
        curve_data = []

        for k in K_VALUES:
            assignment_path = find_assignment_file(algorithm, distance, k)
            if assignment_path is None:
                continue

            assignments = load_assignments(assignment_path)
            common_ids = matrix.index.intersection(assignments["sample_id"])

            if len(common_ids) < 3:
                continue

            X = matrix.loc[common_ids].values
            labels = assignments.set_index("sample_id").loc[common_ids, "cluster"].values

            n_clusters = len(np.unique(labels))
            if n_clusters < 2:
                continue

            dist_matrix = make_distance_matrix(X, metric=distance)
            score = silhouette_score(dist_matrix, labels, metric="precomputed")

            record = {
                "algorithm": algorithm,
                "distance": distance,
                "k": k,
                "n_clusters": n_clusters,
                "silhouette": score,
            }
            results.append(record)
            curve_data.append(record)

        # Plot curve for this specific algorithm/distance
        if curve_data:
            df_curve = pd.DataFrame(curve_data).sort_values("k")
            out_file = OUTPUT_DIR / f"{algorithm}_{distance}_silhouette_vs_k"
            plot_single_curve(
                df_curve,
                title=f"{algorithm.title()} ({distance}) - Silhouette vs. k",
                output_path=out_file,
            )

    # 2. Evaluate Louvain
    louvain_score = None
    louvain_k = None
    assignment_path = find_assignment_file("louvain", "cosine", k=None)

    if assignment_path is not None:
        assignments = load_assignments(assignment_path)
        common_ids = matrix.index.intersection(assignments["sample_id"])

        if len(common_ids) >= 3:
            X = matrix.loc[common_ids].values
            labels = assignments.set_index("sample_id").loc[common_ids, "cluster"].values
            n_clusters = len(np.unique(labels))

            if n_clusters >= 2:
                dist_matrix = make_distance_matrix(X, metric="cosine")
                louvain_score = silhouette_score(dist_matrix, labels, metric="precomputed")
                louvain_k = n_clusters

                results.append({
                    "algorithm": "louvain",
                    "distance": "cosine",
                    "k": np.nan,
                    "n_clusters": n_clusters,
                    "silhouette": louvain_score,
                })

    # 3. Save Summary Table & Combined Plot
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "silhouette_summary.csv", index=False)

    plot_combined_curves(
        results_df=results_df,
        louvain_score=louvain_score,
        louvain_k=louvain_k,
        output_path=OUTPUT_DIR / "all_methods_silhouette_vs_k",
    )

    print(f"\nSilhouette curve plots generated successfully.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_silhouette_analysis()
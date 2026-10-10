from pathlib import Path
import shutil
import warnings
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


RANDOM_STATE = 42
TSNE_PERPLEXITY = 30

sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_data(matrix_file, metadata_file, metrics_file):
    matrix = pd.read_csv(matrix_file, index_col=0)
    metadata = pd.read_csv(metadata_file)
    metrics = pd.read_csv(metrics_file)

    return matrix, metadata, metrics


# ---------------------------------------------------------------------
# Cluster assignment discovery
# ---------------------------------------------------------------------

def find_cluster_file(base_dir, distance, algorithm, k):
    """
    Search recursively for the assignment file matching a clustering.

    Both filenames and directory names are checked because naming
    conventions can differ between clustering algorithms.
    """

    search_root = base_dir / "all_samples"

    if not search_root.exists():
        search_root = base_dir

    candidates = []

    for path in search_root.rglob("*.csv"):
        name = path.name.lower()
        full_path = str(path.relative_to(base_dir)).lower()

        if not any(
            token in name
            for token in ("assignment", "cluster_label", "cluster_labels")
        ):
            continue

        if algorithm not in full_path:
            continue

        if algorithm != "louvain" and distance not in full_path:
            continue

        if k is not None:
            k_tokens = (
                f"k{k}",
                f"k_{k}",
                f"k-{k}",
            )

            if not any(token in full_path for token in k_tokens):
                continue

        candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        exact_candidates = [
            path for path in candidates
            if "assignments" in path.name.lower()
        ]

        if len(exact_candidates) == 1:
            return exact_candidates[0]

        print(
            f"  Multiple assignment files match "
            f"{algorithm}, {distance}, k={k}:"
        )

        for path in candidates:
            print(f"    {path}")

        print("  Skipping this clustering to avoid choosing incorrectly.")
        return None

    print(
        f"  No assignment file found for "
        f"{algorithm}, {distance}, k={k}."
    )

    return None


def load_cluster_labels(path):
    """Load a sample-to-cluster assignment table."""

    df = pd.read_csv(path)

    if df.empty or len(df.columns) < 2:
        raise ValueError(f"Invalid assignment file: {path}")

    sample_candidates = [
        "sample_id",
        "sample",
        "participant_tissue",
        "patient_id",
        "patient",
    ]

    cluster_candidates = [
        "cluster",
        "cluster_label",
        "cluster_labels",
        "label",
        "labels",
        "community",
    ]

    sample_col = next(
        (col for col in sample_candidates if col in df.columns),
        None,
    )

    cluster_col = next(
        (col for col in cluster_candidates if col in df.columns),
        None,
    )

    if sample_col is None:
        unnamed = [
            col for col in df.columns
            if str(col).lower().startswith("unnamed")
        ]

        sample_col = unnamed[0] if unnamed else df.columns[0]

    if cluster_col is None:
        remaining = [
            col for col in df.columns
            if col != sample_col
        ]

        if len(remaining) != 1:
            raise ValueError(
                f"Could not identify the cluster column in {path}. "
                f"Columns: {list(df.columns)}"
            )

        cluster_col = remaining[0]

    labels = df[[sample_col, cluster_col]].copy()
    labels.columns = ["sample_id", "cluster"]

    labels["sample_id"] = labels["sample_id"].astype(str).str.strip()

    return labels.drop_duplicates("sample_id").set_index("sample_id")


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------

def calculate_embeddings(matrix):
    """Calculate all three embeddings once for reuse."""

    n_samples = len(matrix)

    if n_samples < 4:
        raise ValueError("At least four samples are required for these plots.")

    print("Calculating PCA...")
    pca_values = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    ).fit_transform(matrix)

    print("Calculating UMAP...")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="n_jobs value .* overridden .* random_state",
        )

        umap_values = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, n_samples - 1),
            min_dist=0.1,
            metric="euclidean",
            random_state=RANDOM_STATE,
        ).fit_transform(matrix)

    print("Calculating t-SNE...")
    tsne_values = TSNE(
        n_components=2,
        perplexity=min(TSNE_PERPLEXITY, n_samples - 1),
        random_state=RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    ).fit_transform(matrix)

    return {
        "PCA": pd.DataFrame(
            pca_values,
            index=matrix.index,
            columns=["Dimension 1", "Dimension 2"],
        ),
        "UMAP": pd.DataFrame(
            umap_values,
            index=matrix.index,
            columns=["Dimension 1", "Dimension 2"],
        ),
        "t-SNE": pd.DataFrame(
            tsne_values,
            index=matrix.index,
            columns=["Dimension 1", "Dimension 2"],
        ),
    }


# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------

def plot_panel(ax, embedding, labels, title, show_legend=False):
    """Draw one embedding panel with categorical labels."""

    plot_df = embedding.copy()
    plot_df["label"] = labels.reindex(plot_df.index)
    plot_df = plot_df.dropna(subset=["label"])

    if plot_df.empty:
        ax.set_title(f"{title}\nNo valid labels")
        return

    plot_df["label"] = plot_df["label"].astype(str)

    categories = sorted(plot_df["label"].unique())
    palette = dict(
        zip(
            categories,
            sns.color_palette("tab10", n_colors=len(categories)),
        )
    )

    sns.scatterplot(
        data=plot_df,
        x="Dimension 1",
        y="Dimension 2",
        hue="label",
        hue_order=categories,
        palette=palette,
        s=35,
        alpha=0.85,
        linewidth=0,
        legend=show_legend,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(embedding.columns[0])
    ax.set_ylabel(embedding.columns[1])

    if show_legend:
        ax.legend(
            title=None,
            loc="best",
            fontsize=8,
            frameon=True,
        )


def metric_text(row):
    """Format the clustering metrics for the figure title."""

    def fmt(column):
        value = row.get(column, np.nan)

        if pd.isna(value):
            return "NA"

        return f"{value:.3f}"

    return (
        f"Stability ARI: {fmt('stability_ARI')}  |  "
        f"Silhouette: {fmt('silhouette')}  |  "
        f"ARI group: {fmt('ari_group')}  |  "
        f"ARI CMS: {fmt('ari_cms')}"
    )


def plot_clusters_and_group(
    embeddings,
    cluster_labels,
    group_labels,
    method_name,
    row,
    output_path,
):
    """Create a single 2-by-3 figure for clusters and Tumor/Normal."""

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10),
    )

    embedding_names = list(embeddings.keys())

    for col, embedding_name in enumerate(embedding_names):
        embedding = embeddings[embedding_name]

        plot_panel(
            axes[0, col],
            embedding,
            cluster_labels,
            f"{embedding_name}: clusters",
            show_legend=(col == 0),
        )

        plot_panel(
            axes[1, col],
            embedding,
            group_labels,
            f"{embedding_name}: Tumor / Normal",
            show_legend=(col == 0),
        )

    fig.suptitle(
        f"{method_name}\n{metric_text(row)}",
        fontsize=13,
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_cms(
    embeddings,
    cms_labels,
    cluster_labels,
    method_name,
    row,
    output_path,
):
    """Plot tumor-only cluster assignments and CMS labels."""

    valid_cms = cms_labels.dropna().index

    if len(valid_cms) == 0:
        print("  No valid CMS labels; skipping CMS figure.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    for col, (embedding_name, embedding) in enumerate(
        embeddings.items()
    ):
        tumor_embedding = embedding.loc[
            embedding.index.intersection(valid_cms)
        ]

        plot_panel(
            axes[0, col],
            tumor_embedding,
            cluster_labels,
            f"{embedding_name}: clusters (Tumor only)",
            show_legend=(col == 0),
        )

        plot_panel(
            axes[1, col],
            tumor_embedding,
            cms_labels,
            f"{embedding_name}: CMS",
            show_legend=(col == 0),
        )

    fig.suptitle(
        f"{method_name} — Tumor samples only\n{metric_text(row)}",
        fontsize=12,
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot selected clustering results."
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/clustering/DNA_original_pathway"),
        help="Directory containing the clustering results.",
    )

    parser.add_argument(
        "--clusterings",
        nargs="+",
        default=[
            "cosine,hierarchical,2",
            "euclidean,kmeans,2",
            "correlation,hierarchical,3",
            "euclidean,kmeans,4",
            "cosine,louvain,none",
        ],
        metavar="DISTANCE,ALGORITHM,K",
        help=(
            "Selected clusterings. Specify each as "
            "'distance,algorithm,k'. Use 'none' for k in Louvain."
        ),
    )

    args = parser.parse_args()

    selected_clusterings = []

    for item in args.clusterings:
        parts = item.split(",")

        if len(parts) != 3:
            parser.error(
                f"Invalid clustering '{item}'. "
                "Expected DISTANCE,ALGORITHM,K."
            )

        distance, algorithm, k_string = (
            part.strip().lower() for part in parts
        )

        if distance not in {
            "euclidean",
            "cosine",
            "correlation",
        }:
            parser.error(f"Unsupported distance: {distance}")

        if algorithm not in {
            "hierarchical",
            "kmeans",
            "louvain",
        }:
            parser.error(f"Unsupported algorithm: {algorithm}")

        if k_string == "none":
            k = None
        else:
            try:
                k = int(k_string)
            except ValueError:
                parser.error(f"Invalid k value: {k_string}")

            if k < 2:
                parser.error("k must be at least 2.")

        if algorithm == "louvain" and k is not None:
            parser.error("Louvain requires k='none'.")

        if algorithm != "louvain" and k is None:
            parser.error(
                f"{algorithm} requires an integer k."
            )

        selected_clusterings.append(
            (distance, algorithm, k)
        )

    args.clusterings = selected_clusterings

    return args

# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    base_dir = args.base_dir
    output_dir = base_dir / "plots"
    metrics_file = (
        base_dir / "all_samples" / "all_samples_clustering_metrics.csv"
    )
    matrix_file = base_dir / "aggregated_matrix.csv"
    metadata_file = base_dir / "sample_metadata_used_for_analysis.csv"

    matrix, metadata, metrics = load_data(
        matrix_file,
        metadata_file,
        metrics_file,
    )

    matrix.index = matrix.index.astype(str).str.strip()
    metadata["participant_tissue"] = (
        metadata["participant_tissue"].astype(str).str.strip()
    )

    metadata = metadata.set_index("participant_tissue")
    metadata = metadata.reindex(matrix.index)

    selected_clusterings = args.clusterings

    group_col = "group" if "group" in metadata.columns else None
    cms_col = "CMScaller" if "CMScaller" in metadata.columns else None

    if group_col is None:
        raise ValueError(
            "The clinical metadata must contain a 'group' column."
        )

    embeddings = calculate_embeddings(matrix)

    # Restrict CMS reference labels to tumor samples.
    if cms_col is not None:
        tumor_mask = (
            metadata[group_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("tumor")
        )

        cms_labels = metadata[cms_col].where(tumor_mask)
    else:
        cms_labels = pd.Series(
            index=metadata.index,
            dtype=object,
        )

    summary_rows = []

    for distance, algorithm, k in selected_clusterings:
        print(
            f"Processing {algorithm}, {distance}, k={k}"
        )

        matching = metrics[
            (metrics["algorithm"] == algorithm)
            & (metrics["distance"] == distance)
        ].copy()

        if k is None:
            matching = matching[matching["k"].isna()]
        else:
            matching = matching[matching["k"] == float(k)]

        if matching.empty:
            print("  No matching metrics row; skipping.")
            continue

        row = matching.iloc[0]

        cluster_file = find_cluster_file(
            base_dir,
            distance,
            algorithm,
            k,
        )

        if cluster_file is None:
            continue

        labels = load_cluster_labels(cluster_file)
        cluster_labels = labels["cluster"].reindex(matrix.index)

        if cluster_labels.notna().sum() == 0:
            print("  No sample IDs matched the molecular matrix; skipping.")
            continue

        method_name = (
            f"{algorithm}_{distance}"
            if algorithm != "louvain"
            else "louvain"
        )

        if k is not None:
            method_name += f"_k{k}"

        combined_path = output_dir / f"{method_name}.png"
        cms_path = output_dir / "cms" / f"{method_name}_cms.png"

        cms_path.parent.mkdir(parents=True, exist_ok=True)

        plot_clusters_and_group(
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            group_labels=metadata[group_col],
            method_name=method_name,
            row=row,
            output_path=combined_path,
        )

        if cms_col is not None:
            plot_cms(
                embeddings=embeddings,
                cms_labels=cms_labels,
                cluster_labels=cluster_labels,
                method_name=method_name,
                row=row,
                output_path=cms_path,
            )

        summary_rows.append(
            {
                "distance": distance,
                "algorithm": algorithm,
                "k": k,
                "assignment_file": str(cluster_file),
                "n_clusters": row.get("n_clusters", np.nan),
                "stability_ARI": row.get("stability_ARI", np.nan),
                "silhouette": row.get("silhouette", np.nan),
                "ari_group": row.get("ari_group", np.nan),
                "ari_cms": row.get("ari_cms", np.nan),
            }
        )

        print(f"  Saved {combined_path}")
        if cms_col is not None:
            print(f"  Saved {cms_path}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "selected_clusterings_summary.csv",
            index=False,
        )

    print(f"\nFinished. Compact figures are in: {output_dir}")


if __name__ == "__main__":
    main()
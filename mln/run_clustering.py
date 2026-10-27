from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from clustering_metrics import (
    cluster_stability,
    evaluate_labels,
    evaluate_reference_labels,
    gap_statistic,
    louvain_stability,
)


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# ---------------------------------------------------------------------
# Loading & Metadata
# ---------------------------------------------------------------------

def load_pagerank_matrix(
    input_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load PageRank files and return a sample x gene matrix.

    Supports both:
        - single-layer PageRank: 'pagerank'
        - multilayer PageRank: 'multilayer_pagerank'
    """
    logger.info(
        "Loading PageRank matrices from %s...",
        input_dir,
    )

    files = sorted(
        input_dir.glob("*_pagerank.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No PageRank files found in {input_dir}"
        )

    matrices = []

    for fpath in files:

        df = pd.read_csv(fpath)

        if "gene" not in df.columns:
            raise ValueError(
                f"{fpath} missing required column: "
                "'gene'"
            )

        # Support both single-layer and multilayer PageRank.
        if "pagerank" in df.columns:
            pagerank_column = "pagerank"

        elif "multilayer_pagerank" in df.columns:
            pagerank_column = "multilayer_pagerank"

        else:
            raise ValueError(
                f"{fpath} must contain either "
                "'pagerank' or "
                "'multilayer_pagerank'. "
                f"Found columns: {list(df.columns)}"
            )

        sample_id = (
            fpath.stem
            .replace("_pagerank", "")
        )

        sample_matrix = df[
            ["gene", pagerank_column]
        ].copy()

        sample_matrix = sample_matrix.dropna(
            subset=["gene"]
        )

        sample_matrix["gene"] = (
            sample_matrix["gene"]
            .astype(str)
        )

        # Prevent duplicated genes from creating
        # ambiguous matrix columns.
        if sample_matrix["gene"].duplicated().any():

            sample_matrix = (
                sample_matrix
                .groupby(
                    "gene",
                    as_index=False,
                )[pagerank_column]
                .mean()
            )

        sample_series = (
            sample_matrix
            .set_index("gene")[
                pagerank_column
            ]
        )

        sample_series.name = sample_id

        matrices.append(
            sample_series
        )

    matrix = pd.DataFrame(
        matrices
    )

    matrix.index.name = "sample"

    # Genes absent from an individual PageRank file
    # are treated as zero.
    matrix = matrix.fillna(0.0)

    logger.info(
        "Loaded PageRank matrix: "
        "%d samples x %d genes",
        matrix.shape[0],
        matrix.shape[1],
    )

    return matrix.astype(float)


def load_clustering_input(
    input_path: Path,
    input_type: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load clustering input and return a sample x feature matrix.
    """

    if input_type == "original":

        matrix = pd.read_csv(
            input_path,
            index_col=0,
        )

        # Original molecular matrices are genes x samples.
        matrix = matrix.T

    elif input_type == "original_pathway":

        matrix = pd.read_csv(
            input_path,
            index_col=0,
        )

        # Pathway-node matrices are already samples x nodes.
        matrix = matrix

    elif input_type == "pagerank":

        matrix = load_pagerank_matrix(
            input_path,
            logger,
        )

    else:
        raise ValueError(
            f"Unknown input type: {input_type}. "
            "Expected 'original', 'original_pathway', or 'pagerank'."
        )

    matrix = matrix.astype(float)

    if matrix.isna().any().any():

        n_missing = int(
            matrix.isna().sum().sum()
        )

        logger.warning(
            "Input matrix contains %d missing values. "
            "Imputing missing values using feature-wise medians.",
            n_missing,
        )

        feature_medians = matrix.median(axis=0)
        feature_medians = feature_medians.fillna(0.0)

        matrix = matrix.fillna(feature_medians)

    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError(
            "Input matrix contains NaN or infinite values "
            "after missing-value handling."
        )

    logger.info(
        "Loaded %s input: %d samples x %d features",
        input_type,
        matrix.shape[0],
        matrix.shape[1],
    )

    return matrix


def merge_clinical_metadata(
    sample_ids: pd.Index,
    clinical_file: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load clinical metadata and align it to sample IDs.
    """
    sample_ids = sample_ids.astype(str)

    metadata = pd.DataFrame(
        index=sample_ids
    )

    if not clinical_file.exists():

        logger.warning(
            "Clinical file %s not found. "
            "Proceeding without metadata.",
            clinical_file,
        )

        return metadata

    target_cols = [
        "participant_tissue",
        "group",
        "CMScaller",
    ]

    header = pd.read_csv(
        clinical_file,
        nrows=0,
    ).columns.tolist()

    usecols = [
        column
        for column in target_cols
        if column in header
    ]

    clinical = pd.read_csv(
        clinical_file,
        usecols=usecols,
    )

    clinical.columns = (
        clinical.columns.astype(str)
    )

    if "participant_tissue" not in clinical.columns:

        logger.warning(
            "'participant_tissue' column not found "
            "in %s.",
            clinical_file,
        )

        return metadata

    clinical["_sample_id"] = (
        clinical["participant_tissue"]
        .astype(str)
    )

    duplicate_ids = (
        clinical["_sample_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids > 0:

        logger.warning(
            "Found %d duplicated participant_tissue "
            "values. Keeping the first occurrence.",
            duplicate_ids,
        )

    clinical = (
        clinical
        .drop_duplicates("_sample_id")
        .set_index("_sample_id")
    )

    metadata = clinical.reindex(
        sample_ids
    )

    metadata.index = sample_ids

    matched = metadata["group"].notna().sum() \
        if "group" in metadata.columns \
        else 0

    logger.info(
        "Clinical metadata matched %d/%d samples.",
        matched,
        len(sample_ids),
    )

    return metadata


def infer_group(
    sample_ids: pd.Index,
    metadata: pd.DataFrame,
) -> pd.Series:

    if "group" not in metadata.columns:

        return pd.Series(
            index=sample_ids,
            dtype=object,
        )

    return metadata[
        "group"
    ].reindex(sample_ids)


def summarize_pagerank_matrix(
    matrix: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:

    summary = pd.DataFrame(
        index=matrix.index
    )

    summary["sum"] = matrix.sum(axis=1)
    summary["mean"] = matrix.mean(axis=1)
    summary["std"] = matrix.std(axis=1)
    summary["min"] = matrix.min(axis=1)
    summary["max"] = matrix.max(axis=1)

    summary["nonzero_genes"] = (
        matrix != 0
    ).sum(axis=1)

    summary["zero_genes"] = (
        matrix == 0
    ).sum(axis=1)

    summary.to_csv(
        output_dir
        / "pagerank_patient_qc.csv"
    )

    return summary


# ---------------------------------------------------------------------
# Clustering Algorithms
# ---------------------------------------------------------------------

def make_representation(
    matrix: pd.DataFrame,
) -> np.ndarray:

    return (
        matrix
        .astype(float)
        .values
        .copy()
    )


def hierarchical_labels(
    X: np.ndarray,
    k: int,
    distance: str,
) -> tuple[np.ndarray, np.ndarray]:

    if distance == "euclidean":

        Z = linkage(
            X,
            method="ward",
            metric="euclidean",
        )

    elif distance in {
        "cosine",
        "correlation",
    }:

        D = pdist(
            X,
            metric=distance,
        )

        Z = linkage(
            D,
            method="average",
        )

    else:

        raise ValueError(
            f"Unsupported distance metric: "
            f"{distance}"
        )

    labels = fcluster(
        Z,
        t=k,
        criterion="maxclust",
    )

    return labels, Z


def kmeans_labels(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:

    return (
        KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
        )
        .fit_predict(X)
        + 1
    )


def clustering_function_for_stability(
    X: np.ndarray,
    k: int,
    distance: str,
    random_state: int,
) -> np.ndarray:

    labels, _ = hierarchical_labels(
        X,
        k,
        distance,
    )

    return labels


# ---------------------------------------------------------------------
# Louvain
# ---------------------------------------------------------------------

def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "cosine",
    mutual: bool = True,
) -> nx.Graph:

    n_samples = X.shape[0]

    if n_samples < 3:
        raise ValueError(
            "At least 3 samples are required "
            "to build a graph."
        )

    n_neighbors = min(
        n_neighbors,
        n_samples - 1,
    )

    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=metric,
    )

    nn.fit(X)

    distances, indices = (
        nn.kneighbors(X)
    )

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    graph = nx.Graph()

    graph.add_nodes_from(
        range(n_samples)
    )

    knn_sets = {
        i: set(indices[i])
        for i in range(n_samples)
    }

    for i in range(n_samples):

        for j, distance in zip(
            indices[i],
            distances[i],
        ):

            if metric == "cosine":

                similarity = max(
                    0.0,
                    1.0 - distance,
                )

            else:

                similarity = (
                    1.0 / (1.0 + distance)
                )

            if similarity <= 0:
                continue

            if (
                mutual
                and i not in knn_sets[j]
            ):
                continue

            if graph.has_edge(i, j):

                graph[i][j]["weight"] = max(
                    graph[i][j]["weight"],
                    float(similarity),
                )

            else:

                graph.add_edge(
                    i,
                    j,
                    weight=float(similarity),
                )

    return graph


def louvain_labels(
    X: np.ndarray,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    metric: str = "cosine",
    mutual: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, nx.Graph]:

    graph = build_knn_graph(
        X,
        n_neighbors=n_neighbors,
        metric=metric,
        mutual=mutual,
    )

    communities = (
        nx.community
        .louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=random_state,
        )
    )

    communities = sorted(
        communities,
        key=lambda c: min(c),
    )

    labels = np.empty(
        X.shape[0],
        dtype=int,
    )

    for cluster_id, community in enumerate(
        communities,
        start=1,
    ):

        for sample_idx in community:
            labels[sample_idx] = cluster_id

    return labels, graph


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------

def export_embeddings(
    X: np.ndarray,
    sample_ids: pd.Index,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    random_state: int = 42,
) -> None:

    # -------------------------------------------------------------
    # PCA
    # -------------------------------------------------------------

    pca = PCA(
        n_components=2,
        random_state=random_state,
    )

    coords_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame({
        "sample_id": sample_ids,
        "PC1": coords_pca[:, 0],
        "PC2": coords_pca[:, 1],
        "cluster": labels,
    })

    for column in [
        "group",
        "CMScaller",
    ]:

        if column in metadata.columns:
            df_pca[column] = (
                metadata[column].values
            )

    df_pca.to_csv(
        out_dir
        / f"{prefix}_pca.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # t-SNE
    # -------------------------------------------------------------

    if len(sample_ids) >= 10:

        from sklearn.manifold import TSNE

        perplexity = min(
            30,
            max(
                5,
                (len(sample_ids) - 1) // 3,
            ),
        )

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )

        coords_tsne = (
            tsne.fit_transform(X)
        )

        pd.DataFrame({
            "sample_id": sample_ids,
            "dim1": coords_tsne[:, 0],
            "dim2": coords_tsne[:, 1],
            "cluster": labels,
        }).to_csv(
            out_dir
            / f"{prefix}_tsne_coordinates.csv",
            index=False,
        )

    # -------------------------------------------------------------
    # UMAP
    # -------------------------------------------------------------

    try:

        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(
                15,
                max(
                    2,
                    len(sample_ids) - 1,
                ),
            ),
            random_state=random_state,
        )

        coords_umap = (
            reducer.fit_transform(X)
        )

        pd.DataFrame({
            "sample_id": sample_ids,
            "dim1": coords_umap[:, 0],
            "dim2": coords_umap[:, 1],
            "cluster": labels,
        }).to_csv(
            out_dir
            / f"{prefix}_umap_coordinates.csv",
            index=False,
        )

    except ImportError:
        pass


# ---------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------

def export_assignments(
    sample_ids: pd.Index,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:

    assignments = pd.DataFrame({
        "sample_id": sample_ids,
        "cluster": labels,
    })

    if "group" in metadata.columns:
        assignments["group"] = (
            metadata["group"].values
        )

    if "CMScaller" in metadata.columns:
        assignments["CMScaller"] = (
            metadata["CMScaller"].values
        )

    assignments.to_csv(
        output_path,
        index=False,
    )

    return assignments


def export_cms_contingency(
    assignments: pd.DataFrame,
    output_path: Path,
) -> None:

    if "CMScaller" not in assignments.columns:
        return

    tumor_assignments = assignments[
        assignments["group"].eq("Tumor")
        & assignments["CMScaller"].notna()
    ]

    if tumor_assignments.empty:
        return

    contingency = pd.crosstab(
        tumor_assignments["cluster"],
        tumor_assignments["CMScaller"],
        margins=True,
    )

    contingency.to_csv(
        output_path
    )


# ---------------------------------------------------------------------
# Main Analysis Routine
# ---------------------------------------------------------------------

def run_analysis(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    analysis_name: str,
    output_dir: Path,
    k_values: Sequence[int],
    distances: List[str],
    louvain_k: int,
    louvain_res: float,
    louvain_metric: str,
    louvain_mutual: bool,
    n_repeats: int,
    stability_fraction: float,
    random_state: int,
    logger: logging.Logger,
) -> pd.DataFrame:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    logger.info(
        "ANALYSIS: %s (%d samples x %d genes)",
        analysis_name,
        matrix.shape[0],
        matrix.shape[1],
    )

    clusters_base_dir = (
        output_dir / "clusters_by_k"
    )

    clusters_base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X = make_representation(
        matrix
    )

    # -------------------------------------------------------------
    # Gap statistic
    # -------------------------------------------------------------

    valid_k = [
        k
        for k in k_values
        if 1 <= k < X.shape[0]
    ]

    if valid_k:

        for method in (
            "kmeans",
            "hierarchical",
        ):

            gap_df = gap_statistic(
                X,
                k_values=valid_k,
                method=method,
                n_refs=50,
                random_state=random_state,
            )

            gap_df.to_csv(
                output_dir
                / f"{analysis_name}_{method}_gap_statistic.csv",
                index=False,
            )

            selected = gap_df.loc[
                gap_df["selected_k"],
                "k",
            ].tolist()

            logger.info(
                "Gap statistic (%s): selected k = %s",
                method,
                selected
                if selected
                else "none",
            )

    # -------------------------------------------------------------
    # Hierarchical clustering + K-means
    # -------------------------------------------------------------

    for distance in distances:

        for k in k_values:

            if k >= len(matrix):
                continue

            k_dir = (
                clusters_base_dir
                / distance
                / f"k{k}"
            )

            k_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # =====================================================
            # Hierarchical clustering
            # =====================================================

            labels_h, _ = (
                hierarchical_labels(
                    X,
                    k,
                    distance,
                )
            )

            metrics_h = evaluate_labels(X, labels_h, metric=distance)

            reference_h = (
                evaluate_reference_labels(
                    labels_h,
                    metadata,
                )
            )

            stability_h = (
                cluster_stability(
                    X,
                    k,
                    "hierarchical",
                    distance,
                    clustering_function=(
                        clustering_function_for_stability
                    ),
                    n_repeats=n_repeats,
                    fraction=stability_fraction,
                    random_state=random_state,
                )
            )

            all_results.append({
                "analysis": analysis_name,
                "distance": distance,
                "algorithm": "hierarchical",
                "n_genes": matrix.shape[1],
                "k": k,
                "n_clusters": len(
                    np.unique(labels_h)
                ),
                "stability_ARI": stability_h,
                **metrics_h,
                **reference_h,
            })

            assignment_path = (
                k_dir
                / f"{analysis_name}_hierarchical_"
                f"{distance}_k{k}_assignments.csv"
            )

            assign_h = export_assignments(
                matrix.index,
                labels_h,
                metadata,
                assignment_path,
            )

            export_embeddings(
                X,
                matrix.index,
                labels_h,
                metadata,
                k_dir,
                prefix=(
                    f"{analysis_name}_hierarchical_"
                    f"{distance}_k{k}"
                ),
                random_state=random_state,
            )

            export_cms_contingency(
                assign_h,
                k_dir
                / f"{analysis_name}_hierarchical_"
                f"{distance}_k{k}_CMS_contingency.csv",
            )

            # =====================================================
            # K-means
            # =====================================================

            if distance == "euclidean":

                labels_k = kmeans_labels(
                    X,
                    k,
                    random_state=random_state,
                )

                metrics_k = evaluate_labels(
                    X,
                    labels_k,
                )

                reference_k = (
                    evaluate_reference_labels(
                        labels_k,
                        metadata,
                    )
                )

                stability_k = (
                    cluster_stability(
                        X,
                        k,
                        "kmeans",
                        distance,
                        clustering_function=(
                            lambda data, kk, dd, seed:
                            kmeans_labels(
                                data,
                                kk,
                                random_state=seed,
                            )
                        ),
                        n_repeats=n_repeats,
                        fraction=stability_fraction,
                        random_state=random_state,
                    )
                )

                all_results.append({
                    "analysis": analysis_name,
                    "distance": distance,
                    "algorithm": "kmeans",
                    "n_genes": matrix.shape[1],
                    "k": k,
                    "n_clusters": len(
                        np.unique(labels_k)
                    ),
                    "stability_ARI": stability_k,
                    **metrics_k,
                    **reference_k,
                })

                assignment_path = (
                    k_dir
                    / f"{analysis_name}_kmeans_"
                    f"k{k}_assignments.csv"
                )

                assign_k = export_assignments(
                    matrix.index,
                    labels_k,
                    metadata,
                    assignment_path,
                )

                export_embeddings(
                    X,
                    matrix.index,
                    labels_k,
                    metadata,
                    k_dir,
                    prefix=(
                        f"{analysis_name}_kmeans_k{k}"
                    ),
                    random_state=random_state,
                )

                export_cms_contingency(
                    assign_k,
                    k_dir
                    / f"{analysis_name}_kmeans_"
                    f"k{k}_CMS_contingency.csv",
                )

    # -------------------------------------------------------------
    # Louvain community detection
    # -------------------------------------------------------------

    logger.info(
        "Running Louvain "
        "(n_neighbors=%d, resolution=%.2f, "
        "metric=%s, mutual=%s)",
        louvain_k,
        louvain_res,
        louvain_metric,
        louvain_mutual,
    )

    labels_louvain, graph = (
        louvain_labels(
            X,
            n_neighbors=louvain_k,
            resolution=louvain_res,
            metric=louvain_metric,
            mutual=louvain_mutual,
            random_state=random_state,
        )
    )

    n_louvain_clusters = len(
        np.unique(labels_louvain)
    )

    metrics_louvain = evaluate_labels(
        X,
        labels_louvain,
    )

    reference_louvain = (
        evaluate_reference_labels(
            labels_louvain,
            metadata,
        )
    )

    stability_louvain = (
        louvain_stability(
            X,
            louvain_function=louvain_labels,
            n_neighbors=louvain_k,
            resolution=louvain_res,
            metric=louvain_metric,
            mutual=louvain_mutual,
            n_repeats=n_repeats,
            fraction=stability_fraction,
            random_state=random_state,
        )
    )

    logger.info(
        "Louvain detected %d communities; "
        "stability ARI = %.3f; "
        "ARI group = %.3f; "
        "ARI CMS = %.3f",
        n_louvain_clusters,
        stability_louvain,
        reference_louvain["ari_group"],
        reference_louvain["ari_cms"],
    )

    all_results.append({
        "analysis": analysis_name,
        "distance": louvain_metric,
        "algorithm": "louvain",
        "n_genes": matrix.shape[1],
        "k": np.nan,
        "n_clusters": n_louvain_clusters,
        "stability_ARI": stability_louvain,
        "graph_n_neighbors": louvain_k,
        "graph_resolution": louvain_res,
        "graph_mutual": louvain_mutual,
        **metrics_louvain,
        **reference_louvain,
    })

    # -------------------------------------------------------------
    # Louvain outputs
    # -------------------------------------------------------------

    louvain_dir = (
        output_dir / "louvain"
    )

    louvain_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    assign_louvain = export_assignments(
        matrix.index,
        labels_louvain,
        metadata,
        louvain_dir
        / f"{analysis_name}_louvain_assignments.csv",
    )

    export_embeddings(
        X,
        matrix.index,
        labels_louvain,
        metadata,
        louvain_dir,
        prefix=f"{analysis_name}_louvain",
        random_state=random_state,
    )

    export_cms_contingency(
        assign_louvain,
        louvain_dir
        / f"{analysis_name}_louvain_CMS_contingency.csv",
    )

    edge_df = pd.DataFrame([
        {
            "sample_i": matrix.index[i],
            "sample_j": matrix.index[j],
            "weight": data["weight"],
        }
        for i, j, data
        in graph.edges(data=True)
    ])

    edge_df.to_csv(
        louvain_dir
        / f"{analysis_name}_louvain_graph_edges.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Final metrics table
    # -------------------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    results.to_csv(
        output_dir
        / f"{analysis_name}_clustering_metrics.csv",
        index=False,
    )

    return results


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run sample clustering using "
            "Hierarchical clustering, K-means, "
            "and Louvain."
        )
    )

    # -------------------------------------------------------------
    # Input / output
    # -------------------------------------------------------------

    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help=(
            "Path to the original molecular matrix "
            "or PageRank directory."
        ),
    )

    parser.add_argument(
        "--input-type",
        choices=[
            "original",
            "pagerank",
            "original_pathway"
        ],
        default="pagerank",
        help=(
            "Type of input data: original, original_pathway, "
            "or pagerank."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/clustering/"
            "pagerank_comprehensive"
        ),
        help=(
            "Root output directory."
        ),
    )

    parser.add_argument(
        "--clinical-file",
        type=Path,
        default=Path(
            "mln/data/Clinical_features.csv"
        ),
        help=(
            "Clinical annotations CSV."
        ),
    )

    # -------------------------------------------------------------
    # Clustering parameters
    # -------------------------------------------------------------

    parser.add_argument(
        "--k-min",
        type=int,
        default=2,
        help=(
            "Minimum number of clusters "
            "(default: 2)."
        ),
    )

    parser.add_argument(
        "--k-max",
        type=int,
        default=5,
        help=(
            "Maximum number of clusters "
            "(default: 5)."
        ),
    )

    parser.add_argument(
        "--distances",
        nargs="+",
        default=[
            "euclidean",
            "cosine",
            "correlation",
        ],
        help=(
            "Distances for hierarchical clustering."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    # -------------------------------------------------------------
    # Stability
    # -------------------------------------------------------------

    parser.add_argument(
        "--n-repeats-stability",
        type=int,
        default=100,
        help=(
            "Number of subsampling iterations "
            "for stability."
        ),
    )

    parser.add_argument(
        "--stability-fraction",
        type=float,
        default=0.80,
        help=(
            "Fraction of samples retained in "
            "each stability iteration."
        ),
    )

    # -------------------------------------------------------------
    # Louvain
    # -------------------------------------------------------------

    parser.add_argument(
        "--louvain-n-neighbors",
        type=int,
        default=20,
        help=(
            "Number of k-NN neighbors "
            "for Louvain."
        ),
    )

    parser.add_argument(
        "--louvain-resolution",
        type=float,
        default=0.5,
        help=(
            "Louvain resolution parameter."
        ),
    )

    parser.add_argument(
        "--louvain-metric",
        type=str,
        default="cosine",
        choices=[
            "cosine",
            "euclidean",
            "correlation",
        ],
        help=(
            "Metric for Louvain graph construction."
        ),
    )

    parser.add_argument(
        "--louvain-mutual",
        action="store_true",
        help=(
            "Use mutual nearest neighbors "
            "for the Louvain graph."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] %(message)s"
        ),
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger(
        "clustering_pipeline"
    )

    args = parse_args()

    # -------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------

    if not args.input_path.exists():

        logger.error(
            "Input path '%s' does not exist.",
            args.input_path,
        )

        raise SystemExit(1)

    if (
        args.input_type == "pagerank"
        and not args.input_path.is_dir()
    ):

        logger.error(
            "PageRank input must be a directory: %s",
            args.input_path,
        )

        raise SystemExit(1)

    if (
        args.input_type == "original"
        and not args.input_path.is_file()
    ):

        logger.error(
            "Original input must be a file: %s",
            args.input_path,
        )

        raise SystemExit(1)

    if args.k_min < 2:

        logger.error(
            "--k-min must be >= 2."
        )

        raise SystemExit(1)

    if args.k_max < args.k_min:

        logger.error(
            "--k-max (%d) cannot be smaller "
            "than --k-min (%d).",
            args.k_max,
            args.k_min,
        )

        raise SystemExit(1)

    if not (
        0.0
        < args.stability_fraction
        <= 1.0
    ):

        logger.error(
            "--stability-fraction must be "
            "in (0.0, 1.0]."
        )

        raise SystemExit(1)

    if args.n_repeats_stability < 1:

        logger.error(
            "--n-repeats-stability must be >= 1."
        )

        raise SystemExit(1)

    k_values = range(
        args.k_min,
        args.k_max + 1,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Load matrix
    # -------------------------------------------------------------

    logger.info(
        "Loading %s input from %s...",
        args.input_type,
        args.input_path,
    )

    try:

        matrix = load_clustering_input(
            args.input_path,
            args.input_type,
            logger,
        )

    except Exception as exc:

        logger.exception(
            "Failed to load input: %s",
            exc,
        )

        raise SystemExit(1) from exc

    if matrix.empty:

        logger.error(
            "Input matrix is empty."
        )

        raise SystemExit(1)

    if not np.isfinite(
        matrix.to_numpy()
    ).all():

        logger.error(
            "Input matrix contains NaN or infinite values."
        )

        raise SystemExit(1)

    matrix.to_csv(
        args.output_dir
        / "aggregated_matrix.csv"
    )

    # Keep the previous filename for compatibility
    # with PageRank analyses.
    matrix.to_csv(
        args.output_dir
        / "aggregated_pagerank_matrix.csv"
    )

    summarize_pagerank_matrix(
        matrix,
        args.output_dir,
    )

    # -------------------------------------------------------------
    # Load clinical metadata
    # -------------------------------------------------------------

    logger.info(
        "Merging clinical annotations from %s...",
        args.clinical_file,
    )

    metadata = merge_clinical_metadata(
        matrix.index,
        args.clinical_file,
        logger,
    )

    metadata["group"] = infer_group(
        matrix.index,
        metadata,
    )

    metadata.to_csv(
        args.output_dir
        / "sample_metadata_used_for_analysis.csv"
    )

    # -------------------------------------------------------------
    # Run clustering
    # -------------------------------------------------------------

    run_analysis(
        matrix=matrix,
        metadata=metadata,
        analysis_name="all_samples",
        output_dir=(
            args.output_dir
            / "all_samples"
        ),
        k_values=k_values,
        distances=args.distances,
        louvain_k=args.louvain_n_neighbors,
        louvain_res=args.louvain_resolution,
        louvain_metric=args.louvain_metric,
        louvain_mutual=args.louvain_mutual,
        n_repeats=args.n_repeats_stability,
        stability_fraction=args.stability_fraction,
        random_state=args.random_state,
        logger=logger,
    )

    # -------------------------------------------------------------
    # Aggregate metrics
    # -------------------------------------------------------------

    summary_files = list(
        args.output_dir.rglob(
            "*_clustering_metrics.csv"
        )
    )

    summary_files = [
        f
        for f in summary_files
        if f.name
        != "ALL_clustering_metrics.csv"
    ]

    if summary_files:

        dfs = [
            pd.read_csv(f)
            for f in summary_files
        ]

        if dfs:

            pd.concat(
                dfs,
                ignore_index=True,
            ).drop_duplicates().to_csv(
                args.output_dir
                / "ALL_clustering_metrics.csv",
                index=False,
            )

    logger.info(
        "Clustering analysis completed successfully. "
        "Results written to %s",
        args.output_dir,
    )


if __name__ == "__main__":
    main()

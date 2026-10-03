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
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# Loading & Metadata
# ---------------------------------------------------------------------

def load_pagerank_matrix(
    pagerank_dir: Path, logger: logging.Logger
) -> pd.DataFrame:
    csv_files = sorted(pagerank_dir.glob("*_pagerank.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_pagerank.csv files found in {pagerank_dir}")

    patient_records = {}
    for i, fpath in enumerate(csv_files, start=1):
        sample_id = fpath.name.replace("_pagerank.csv", "")
        df = pd.read_csv(fpath)
        required = {"gene", "multilayer_pagerank"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{fpath} missing required columns: {sorted(missing)}")

        if df["gene"].duplicated().any():
            series = df.groupby("gene")["multilayer_pagerank"].sum()
        else:
            series = df.set_index("gene")["multilayer_pagerank"]

        patient_records[sample_id] = series.astype(float)
        logger.info("[%d/%d] Loaded %s", i, len(csv_files), sample_id)

    matrix = pd.DataFrame.from_dict(patient_records, orient="index")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return matrix.sort_index().sort_index(axis=1)


def merge_clinical_metadata(
    sample_ids: pd.Index, clinical_file: Path, logger: logging.Logger
) -> pd.DataFrame:
    metadata = pd.DataFrame(index=sample_ids.astype(str))
    if not clinical_file.exists():
        logger.warning("Clinical file %s not found. Proceeding without metadata.", clinical_file)
        return metadata

    target_cols = ["participant_tissue", "group", "CMScaller"]
    header = pd.read_csv(clinical_file, nrows=0).columns.tolist()
    usecols = [c for c in target_cols if c in header]

    clinical = pd.read_csv(clinical_file, usecols=usecols)
    clinical.columns = clinical.columns.astype(str)

    id_col = "participant_tissue" if "participant_tissue" in clinical.columns else None
    if id_col is None:
        logger.warning("'participant_tissue' column not found in %s.", clinical_file)
        return metadata

    clinical["_sample_id"] = clinical[id_col].astype(str)
    clinical = clinical.drop_duplicates("_sample_id").set_index("_sample_id")

    metadata = clinical.reindex(sample_ids.astype(str))
    metadata.index = sample_ids
    return metadata


def infer_group(sample_ids: pd.Index, metadata: pd.DataFrame) -> pd.Series:
    if "group" not in metadata.columns:
        return pd.Series(index=sample_ids, dtype=object)
    return metadata["group"].reindex(sample_ids)


def summarize_pagerank_matrix(matrix: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = pd.DataFrame(index=matrix.index)
    summary["sum"] = matrix.sum(axis=1)
    summary["mean"] = matrix.mean(axis=1)
    summary["std"] = matrix.std(axis=1)
    summary["min"] = matrix.min(axis=1)
    summary["max"] = matrix.max(axis=1)
    summary["nonzero_genes"] = (matrix != 0).sum(axis=1)
    summary["zero_genes"] = (matrix == 0).sum(axis=1)
    summary.to_csv(output_dir / "pagerank_patient_qc.csv")
    return summary

# ---------------------------------------------------------------------
# Representations & Clustering Engines
# ---------------------------------------------------------------------

def make_representation(matrix: pd.DataFrame) -> np.ndarray:
    return matrix.astype(float).values.copy()


def hierarchical_labels(
    X: np.ndarray, k: int, distance: str
) -> tuple[np.ndarray, np.ndarray]:
    if distance == "euclidean":
        Z = linkage(X, method="ward", metric="euclidean")
    elif distance in {"cosine", "correlation"}:
        D = pdist(X, metric=distance)
        Z = linkage(D, method="average")
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")
    return fcluster(Z, t=k, criterion="maxclust"), Z


def kmeans_labels(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=random_state, n_init=20).fit_predict(X) + 1


def safe_metric(metric_func, *args) -> float:
    try:
        return float(metric_func(*args))
    except Exception:
        return np.nan


def evaluate_labels(X: np.ndarray, labels: np.ndarray) -> dict:
    result = {"silhouette": np.nan}
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return result
    result["silhouette"] = safe_metric(silhouette_score, X, labels)
    return result


def cluster_stability(
    X: np.ndarray,
    k: int,
    method: str,
    distance: str,
    n_repeats: int = 100,
    fraction: float = 0.80,
    random_state: int = 42,
) -> float:
    n = X.shape[0]
    if n < max(10, k * 3):
        return np.nan

    if method == "hierarchical":
        full_labels, _ = hierarchical_labels(X, k, distance)
    elif method == "kmeans":
        full_labels = kmeans_labels(X, k, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    rng = np.random.default_rng(random_state)
    subset_size = max(k * 2, int(np.floor(n * fraction)))
    scores = []

    for _ in range(n_repeats):
        idx = rng.choice(n, size=subset_size, replace=False)
        X_sub = X[idx]
        if method == "hierarchical":
            sub_labels, _ = hierarchical_labels(X_sub, k, distance)
        elif method == "kmeans":
            sub_labels = kmeans_labels(X_sub, k, random_state=random_state)

        scores.append(adjusted_rand_score(full_labels[idx], sub_labels))

    return float(np.mean(scores))


def export_embeddings(
    X: np.ndarray,
    sample_ids: pd.Index,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    random_state: int = 42,
) -> None:
    # PCA
    pca = PCA(n_components=2, random_state=random_state)
    coords_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame({
        "sample_id": sample_ids,
        "PC1": coords_pca[:, 0],
        "PC2": coords_pca[:, 1],
        "cluster": labels,
    })
    for col in ["group", "CMScaller", "CMS"]:
        if col in metadata.columns:
            df_pca[col] = metadata[col].values
    df_pca.to_csv(out_dir / f"{prefix}_pca.csv", index=False)

    # t-SNE
    if len(sample_ids) >= 10:
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, (len(sample_ids) - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        coords_tsne = tsne.fit_transform(X)
        pd.DataFrame({
            "sample_id": sample_ids,
            "dim1": coords_tsne[:, 0],
            "dim2": coords_tsne[:, 1],
            "cluster": labels,
        }).to_csv(out_dir / f"{prefix}_tsne_coordinates.csv", index=False)

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, len(sample_ids) - 1)),
            random_state=random_state,
        )
        coords_umap = reducer.fit_transform(X)
        pd.DataFrame({
            "sample_id": sample_ids,
            "dim1": coords_umap[:, 0],
            "dim2": coords_umap[:, 1],
            "cluster": labels,
        }).to_csv(out_dir / f"{prefix}_umap_coordinates.csv", index=False)
    except ImportError:
        pass


def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "cosine",
    mutual: bool = True,
) -> nx.Graph:
    n_samples = X.shape[0]
    if n_samples < 3:
        raise ValueError("At least 3 samples are required to build a graph.")

    n_neighbors = min(n_neighbors, n_samples - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nn.fit(X)

    distances, indices = nn.kneighbors(X)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    graph = nx.Graph()
    graph.add_nodes_from(range(n_samples))

    knn_sets = {i: set(indices[i]) for i in range(n_samples)}

    for i in range(n_samples):
        for j, distance in zip(indices[i], distances[i]):
            similarity = max(0.0, 1.0 - distance) if metric == "cosine" else 1.0 / (1.0 + distance)

            if similarity <= 0:
                continue
            if mutual and i not in knn_sets[j]:
                continue

            if graph.has_edge(i, j):
                graph[i][j]["weight"] = max(graph[i][j]["weight"], float(similarity))
            else:
                graph.add_edge(i, j, weight=float(similarity))

    return graph


def louvain_labels(
    X: np.ndarray,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    metric: str = "cosine",
    mutual: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, nx.Graph]:
    graph = build_knn_graph(X, n_neighbors=n_neighbors, metric=metric, mutual=mutual)

    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=resolution,
        seed=random_state,
    )

    communities = sorted(communities, key=lambda c: min(c))
    labels = np.empty(X.shape[0], dtype=int)

    for cluster_id, community in enumerate(communities, start=1):
        for sample_idx in community:
            labels[sample_idx] = cluster_id

    return labels, graph


def louvain_stability(
    X: np.ndarray,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    metric: str = "cosine",
    mutual: bool = True,
    n_repeats: int = 100,
    fraction: float = 0.80,
    random_state: int = 42,
) -> float:
    n = X.shape[0]
    if n < max(10, n_neighbors + 2):
        return np.nan

    full_labels, _ = louvain_labels(
        X,
        n_neighbors=n_neighbors,
        resolution=resolution,
        metric=metric,
        mutual=mutual,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    subset_size = max(n_neighbors + 2, int(np.floor(n * fraction)))
    subset_size = min(subset_size, n)

    scores = []
    for _ in range(n_repeats):
        idx = rng.choice(n, size=subset_size, replace=False)
        X_sub = X[idx]

        sub_labels, _ = louvain_labels(
            X_sub,
            n_neighbors=min(n_neighbors, subset_size - 1),
            resolution=resolution,
            metric=metric,
            mutual=mutual,
            random_state=random_state,
        )

        scores.append(adjusted_rand_score(full_labels[idx], sub_labels))

    return float(np.mean(scores))

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
    output_dir.mkdir(parents=True, exist_ok=True)
    cms_col = "CMScaller" if "CMScaller" in metadata.columns else None

    all_results = []
    logger.info(
        "ANALYSIS: %s (%d samples x %d genes)",
        analysis_name, matrix.shape[0], matrix.shape[1],
    )
    clusters_base_dir = output_dir / "clusters_by_k"
    clusters_base_dir.mkdir(parents=True, exist_ok=True)
    X = make_representation(matrix)

    # 1. Hierarchical clustering + K-means
    for distance in distances:
        for k in k_values:
            if k >= len(matrix):
                continue

            k_dir = clusters_base_dir / distance / f"k{k}"
            k_dir.mkdir(parents=True, exist_ok=True)

            # Hierarchical
            labels_h, _ = hierarchical_labels(X, k, distance)
            metrics_h = evaluate_labels(X, labels_h)
            stability_h = cluster_stability(
                X, k, "hierarchical", distance,
                n_repeats=n_repeats, fraction=stability_fraction, random_state=random_state,
            )

            all_results.append({
                "analysis": analysis_name,
                "distance": distance,
                "algorithm": "hierarchical",
                "n_genes": matrix.shape[1],
                "k": k,
                "n_clusters": len(np.unique(labels_h)),
                "stability_ARI": stability_h,
                **metrics_h,
            })

            assign_h = pd.DataFrame({"sample_id": matrix.index, "cluster": labels_h})
            if "group" in metadata.columns:
                assign_h["group"] = metadata["group"].values
            if cms_col is not None:
                assign_h["CMS"] = metadata[cms_col].values

            assign_h.to_csv(
                k_dir / f"{analysis_name}_hierarchical_{distance}_k{k}_assignments.csv",
                index=False,
            )

            export_embeddings(
                X, matrix.index, labels_h, metadata, k_dir,
                prefix=f"{analysis_name}_hierarchical_{distance}_k{k}",
                random_state=random_state,
            )

            if cms_col is not None:
                pd.crosstab(assign_h["cluster"], assign_h["CMS"], margins=True).to_csv(
                    k_dir / f"{analysis_name}_hierarchical_{distance}_k{k}_CMS_contingency.csv"
                )

            # K-means (Euclidean only)
            if distance == "euclidean":
                labels_k = kmeans_labels(X, k, random_state=random_state)
                metrics_k = evaluate_labels(X, labels_k)
                stability_k = cluster_stability(
                    X, k, "kmeans", distance,
                    n_repeats=n_repeats, fraction=stability_fraction, random_state=random_state,
                )

                all_results.append({
                    "analysis": analysis_name,
                    "distance": distance,
                    "algorithm": "kmeans",
                    "n_genes": matrix.shape[1],
                    "k": k,
                    "n_clusters": len(np.unique(labels_k)),
                    "stability_ARI": stability_k,
                    **metrics_k,
                })

                assign_k = pd.DataFrame({"sample_id": matrix.index, "cluster": labels_k})
                if "group" in metadata.columns:
                    assign_k["group"] = metadata["group"].values
                if cms_col is not None:
                    assign_k["CMS"] = metadata[cms_col].values

                assign_k.to_csv(
                    k_dir / f"{analysis_name}_kmeans_k{k}_assignments.csv",
                    index=False,
                )

                export_embeddings(
                    X, matrix.index, labels_k, metadata, k_dir,
                    prefix=f"{analysis_name}_kmeans_k{k}",
                    random_state=random_state,
                )

                if cms_col is not None:
                    pd.crosstab(assign_k["cluster"], assign_k["CMS"], margins=True).to_csv(
                        k_dir / f"{analysis_name}_kmeans_k{k}_CMS_contingency.csv"
                    )

    # 2. Louvain community detection
    logger.info(
        "Running Louvain (n_neighbors=%d, resolution=%.2f, metric=%s, mutual=%s)",
        louvain_k, louvain_res, louvain_metric, louvain_mutual,
    )

    labels_louvain, graph = louvain_labels(
        X,
        n_neighbors=louvain_k,
        resolution=louvain_res,
        metric=louvain_metric,
        mutual=louvain_mutual,
        random_state=random_state,
    )

    n_louvain_clusters = len(np.unique(labels_louvain))
    metrics_louvain = evaluate_labels(X, labels_louvain)
    stability_louvain = louvain_stability(
        X,
        n_neighbors=louvain_k,
        resolution=louvain_res,
        metric=louvain_metric,
        mutual=louvain_mutual,
        n_repeats=n_repeats,
        fraction=stability_fraction,
        random_state=random_state,
    )

    logger.info(
        "Louvain detected %d communities; stability ARI = %.3f",
        n_louvain_clusters, stability_louvain,
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
    })

    louvain_dir = output_dir / "louvain"
    louvain_dir.mkdir(parents=True, exist_ok=True)

    assign_louvain = pd.DataFrame({"sample_id": matrix.index, "cluster": labels_louvain})
    if "group" in metadata.columns:
        assign_louvain["group"] = metadata["group"].values
    if cms_col is not None:
        assign_louvain["CMS"] = metadata[cms_col].values

    assign_louvain.to_csv(louvain_dir / f"{analysis_name}_louvain_assignments.csv", index=False)
    export_embeddings(
        X, matrix.index, labels_louvain, metadata, louvain_dir,
        prefix=f"{analysis_name}_louvain", random_state=random_state,
    )

    if cms_col is not None:
        pd.crosstab(assign_louvain["cluster"], assign_louvain["CMS"], margins=True).to_csv(
            louvain_dir / f"{analysis_name}_louvain_CMS_contingency.csv"
        )

    edge_df = pd.DataFrame([
        {
            "sample_i": matrix.index[i],
            "sample_j": matrix.index[j],
            "weight": data["weight"],
        }
        for i, j, data in graph.edges(data=True)
    ])
    edge_df.to_csv(louvain_dir / f"{analysis_name}_louvain_graph_edges.csv", index=False)

    results = pd.DataFrame(all_results)
    results.to_csv(output_dir / f"{analysis_name}_clustering_metrics.csv", index=False)
    return results

# ---------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-method sample clustering (Hierarchical, K-means, Louvain) on multilayer PageRank matrices."
    )

    # File paths
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/centralities/pagerank"),
        help="Directory containing *_pagerank.csv files (default: output/centralities/pagerank).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/clustering/pagerank_comprehensive"),
        help="Root output directory (default: output/clustering/pagerank_comprehensive).",
    )
    parser.add_argument(
        "--clinical-file",
        type=Path,
        default=Path("mln/data/Clinical_features.csv"),
        help="Path to clinical annotations CSV (default: mln/data/Clinical_features.csv).",
    )

    # General clustering parameters
    parser.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="Minimum number of clusters k to test (inclusive, default: 2).",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=5,
        help="Maximum number of clusters k to test (inclusive, default: 5).",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        default=["euclidean", "cosine", "correlation"],
        help="Distance metrics for hierarchical clustering (default: euclidean cosine correlation).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    # Stability parameters
    parser.add_argument(
        "--n-repeats-stability",
        type=int,
        default=100,
        help="Subsampling iterations for ARI stability assessment (default: 100).",
    )
    parser.add_argument(
        "--stability-fraction",
        type=float,
        default=0.80,
        help="Fraction of samples retained per bootstrap iteration (default: 0.80).",
    )

    # Louvain parameters
    parser.add_argument(
        "--louvain-n-neighbors",
        type=int,
        default=20,
        help="k-NN neighbors for sample similarity graph (default: 20).",
    )
    parser.add_argument(
        "--louvain-resolution",
        type=float,
        default=0.5,
        help="Louvain modularity resolution parameter (default: 0.5).",
    )
    parser.add_argument(
        "--louvain-metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "correlation"],
        help="Metric for graph construction (default: cosine).",
    )
    parser.add_argument(
        "--louvain-mutual",
        action="store_true",
        help="Enforce mutual nearest neighbors when constructing the graph (default: False).",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("clustering_pipeline")

    args = parse_args()

    # Input validations
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        logger.error("Input directory '%s' does not exist or is not a directory.", args.input_dir)
        return

    if args.k_min < 2:
        logger.error("--k-min must be >= 2.")
        return

    if args.k_max < args.k_min:
        logger.error("--k-max (%d) cannot be smaller than --k-min (%d).", args.k_max, args.k_min)
        return

    if not (0.0 < args.stability_fraction <= 1.0):
        logger.error("--stability-fraction must be in (0.0, 1.0].")
        return

    k_values = range(args.k_min, args.k_max + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info("Loading PageRank matrices from %s...", args.input_dir)
    try:
        matrix = load_pagerank_matrix(args.input_dir, logger)
    except Exception as exc:
        logger.exception("Failed to load matrices: %s", exc)
        return

    matrix.to_csv(args.output_dir / "aggregated_pagerank_matrix.csv")
    summarize_pagerank_matrix(matrix, args.output_dir)

    # 2. Metadata
    logger.info("Merging clinical annotations from %s...", args.clinical_file)
    metadata = merge_clinical_metadata(matrix.index, args.clinical_file, logger)
    metadata["group"] = infer_group(matrix.index, metadata)
    metadata.to_csv(args.output_dir / "sample_metadata_used_for_analysis.csv")

    # 3. Run clustering analysis
    run_analysis(
        matrix=matrix,
        metadata=metadata,
        analysis_name="all_samples",
        output_dir=args.output_dir / "all_samples",
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

    # 4. Aggregate metrics across sub-analyses
    summary_files = list(args.output_dir.rglob("*_clustering_metrics.csv"))
    if summary_files:
        dfs = [pd.read_csv(f) for f in summary_files if f.name != "ALL_clustering_metrics.csv"]
        if dfs:
            pd.concat(dfs, ignore_index=True).drop_duplicates().to_csv(
                args.output_dir / "ALL_clustering_metrics.csv", index=False
            )

    logger.info("Clustering analysis completed successfully. Results written to %s", args.output_dir)


if __name__ == "__main__":
    main()
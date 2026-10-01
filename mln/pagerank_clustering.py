from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import networkx as nx

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score


warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_DIR = Path("output/centralities/pagerank")
OUTPUT_DIR = Path("output/clustering/pagerank_comprehensive")
CLINICAL_FILE = Path("mln/data/Clinical_features.csv")

K_VALUES = range(2, 6)
RANDOM_STATE = 42
N_REPEATS_STABILITY = 100
STABILITY_FRACTION = 0.80

DISTANCES = ["euclidean", "cosine", "correlation"]

# ---------------------------------------------------------------------
# Loading & Metadata
# ---------------------------------------------------------------------

def load_pagerank_matrix(pagerank_dir: Path) -> pd.DataFrame:
    csv_files = sorted(pagerank_dir.glob("*_pagerank.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_pagerank.csv files found in {pagerank_dir}")

    patient_records = {}
    for i, fpath in enumerate(csv_files, start=1):
        sample_id = fpath.name.replace("_pagerank.csv", "")
        df = pd.read_csv(fpath)
        required = {"gene", "multilayer_pagerank"}
        if required - set(df.columns):
            raise ValueError(f"{fpath} missing required columns: {sorted(required - set(df.columns))}")

        if df["gene"].duplicated().any():
            series = df.groupby("gene")["multilayer_pagerank"].sum()
        else:
            series = df.set_index("gene")["multilayer_pagerank"]

        patient_records[sample_id] = series.astype(float)
        print(f"[{i}/{len(csv_files)}] Loaded {sample_id}")

    matrix = pd.DataFrame.from_dict(patient_records, orient="index")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return matrix.sort_index().sort_index(axis=1)

def merge_clinical_metadata(sample_ids: pd.Index, clinical_file: Path) -> pd.DataFrame:
    metadata = pd.DataFrame(index=sample_ids.astype(str))
    if not clinical_file.exists():
        return metadata

    target_cols = ["participant_tissue", "group", "CMScaller"]
    
    header = pd.read_csv(clinical_file, nrows=0).columns.tolist()
    usecols = [c for c in target_cols if c in header]
    
    clinical = pd.read_csv(clinical_file, usecols=usecols)
    clinical.columns = clinical.columns.astype(str)

    id_col = "participant_tissue" if "participant_tissue" in clinical.columns else None
    if id_col is None:
        return metadata

    clinical["_sample_id"] = clinical[id_col].astype(str)
    clinical = clinical.drop_duplicates("_sample_id").set_index("_sample_id")
    
    metadata = clinical.reindex(sample_ids.astype(str))
    metadata.index = sample_ids
    return metadata

def infer_group(sample_ids: pd.Index, metadata: pd.DataFrame) -> pd.Series:
    if "group" not in metadata.columns:
        raise KeyError("Column 'group' not found in metadata DataFrame.")
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
    X = matrix.astype(float).values.copy()
    return X

def hierarchical_labels(X: np.ndarray, k: int, distance: str) -> tuple[np.ndarray, np.ndarray]:
    if distance == "euclidean":
        Z = linkage(X, method="ward", metric="euclidean")
    elif distance in {"cosine", "correlation"}:
        D = pdist(X, metric=distance)
        Z = linkage(D, method="average")
    else:
        raise ValueError(distance)
    return fcluster(Z, t=k, criterion="maxclust"), Z

def kmeans_labels(X: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(X) + 1

def safe_metric(metric_func, *args) -> float:
    try:
        return float(metric_func(*args))
    except Exception:
        return np.nan

def evaluate_labels(X: np.ndarray, labels: np.ndarray) -> dict:
    result = {
        "silhouette": np.nan,
    }
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
    n_repeats: int = N_REPEATS_STABILITY,
    fraction: float = STABILITY_FRACTION,
) -> float:
    n = X.shape[0]
    if n < max(10, k * 3):
        return np.nan

    if method == "hierarchical":
        full_labels, _ = hierarchical_labels(X, k, distance)
    elif method == "kmeans":
        full_labels = kmeans_labels(X, k)

    rng = np.random.default_rng(RANDOM_STATE)
    subset_size = max(k * 2, int(np.floor(n * fraction)))
    scores = []

    for _ in range(n_repeats):
        idx = rng.choice(n, size=subset_size, replace=False)
        X_sub = X[idx]
        if method == "hierarchical":
            sub_labels, _ = hierarchical_labels(X_sub, k, distance)
        elif method == "kmeans":
            sub_labels = kmeans_labels(X_sub, k)

        scores.append(adjusted_rand_score(full_labels[idx], sub_labels))

    return float(np.mean(scores))

def export_embeddings(X: np.ndarray, sample_ids: pd.Index, labels: np.ndarray, metadata: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    # PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame({"sample_id": sample_ids, "PC1": coords_pca[:, 0], "PC2": coords_pca[:, 1], "cluster": labels})
    for col in ["group", "CMScaller", "CMS"]:
        if col in metadata.columns:
            df_pca[col] = metadata[col].values
    df_pca.to_csv(out_dir / f"{prefix}_pca.csv", index=False)

    # t-SNE
    if len(sample_ids) >= 10:
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, (len(sample_ids) - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE, init="pca", learning_rate="auto")
        coords_tsne = tsne.fit_transform(X)
        pd.DataFrame({"sample_id": sample_ids, "dim1": coords_tsne[:, 0], "dim2": coords_tsne[:, 1], "cluster": labels}).to_csv(
            out_dir / f"{prefix}_tsne_coordinates.csv", index=False
        )

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, max(2, len(sample_ids) - 1)), random_state=RANDOM_STATE)
        coords_umap = reducer.fit_transform(X)
        pd.DataFrame({"sample_id": sample_ids, "dim1": coords_umap[:, 0], "dim2": coords_umap[:, 1], "cluster": labels}).to_csv(
            out_dir / f"{prefix}_umap_coordinates.csv", index=False
        )
    except ImportError:
        pass

# ---------------------------------------------------------------------
# Main Analysis Engine
# ---------------------------------------------------------------------

def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "cosine",
    mutual: bool = True,
) -> nx.Graph:
    """
    Build a weighted sample-similarity kNN graph.

    Nodes:
        samples

    Edges:
        connect similar samples.

    Edge weight:
        similarity = 1 - distance

    Parameters
    ----------
    X : np.ndarray
        Samples x features representation.

    n_neighbors : int
        Number of nearest neighbours considered for each sample.

    metric : str
        Distance metric accepted by sklearn NearestNeighbors.

    mutual : bool
        If True, retain an edge only when the two samples are
        mutual nearest neighbours.
    """
    n_samples = X.shape[0]

    if n_samples < 3:
        raise ValueError("At least 3 samples are required to build a graph.")

    n_neighbors = min(n_neighbors, n_samples - 1)

    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=metric
    )
    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    # Remove self-neighbour at position 0.
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    graph = nx.Graph()
    graph.add_nodes_from(range(n_samples))

    # First collect directed kNN relationships.
    knn_sets = {
        i: set(indices[i])
        for i in range(n_samples)
    }

    for i in range(n_samples):
        for j, distance in zip(indices[i], distances[i]):

            # Convert distance to similarity.
            if metric == "cosine":
                similarity = max(0.0, 1.0 - distance)
            else:
                # For non-cosine metrics, transform distance into
                # a positive similarity.
                similarity = 1.0 / (1.0 + distance)

            if similarity <= 0:
                continue

            if mutual and i not in knn_sets[j]:
                continue

            # Avoid duplicate edges in the undirected graph.
            if graph.has_edge(i, j):
                # Keep the stronger similarity.
                graph[i][j]["weight"] = max(
                    graph[i][j]["weight"],
                    float(similarity)
                )
            else:
                graph.add_edge(
                    i,
                    j,
                    weight=float(similarity)
                )

    return graph

def louvain_labels(
    X: np.ndarray,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    metric: str = "cosine",
    mutual: bool = True,
) -> tuple[np.ndarray, nx.Graph]:
    """
    Perform Louvain community detection on a sample-similarity graph.

    The number of clusters is NOT specified.

    Returns
    -------
    labels : np.ndarray
        Cluster label for each sample.

    graph : nx.Graph
        Weighted sample-similarity graph.
    """
    graph = build_knn_graph(
        X,
        n_neighbors=n_neighbors,
        metric=metric,
        mutual=mutual,
    )

    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=resolution,
        seed=RANDOM_STATE,
    )

    # Convert communities into deterministic integer labels.
    # Sorting by smallest sample index makes labels reproducible.
    communities = sorted(
        communities,
        key=lambda c: min(c)
    )

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
    n_repeats: int = N_REPEATS_STABILITY,
    fraction: float = STABILITY_FRACTION,
) -> float:
    """
    Estimate Louvain stability using repeated subsampling.

    Louvain is rerun on each subset, with the graph reconstructed
    from that subset.

    Stability is measured with Adjusted Rand Index (ARI) between
    the full-data clustering and the clustering obtained on each
    subset.
    """
    n = X.shape[0]

    if n < max(10, n_neighbors + 2):
        return np.nan

    full_labels, _ = louvain_labels(
        X,
        n_neighbors=n_neighbors,
        resolution=resolution,
        metric=metric,
        mutual=mutual,
    )

    rng = np.random.default_rng(RANDOM_STATE)

    subset_size = max(
        n_neighbors + 2,
        int(np.floor(n * fraction))
    )

    subset_size = min(subset_size, n)

    scores = []

    for _ in range(n_repeats):

        idx = rng.choice(
            n,
            size=subset_size,
            replace=False
        )

        X_sub = X[idx]

        sub_labels, _ = louvain_labels(
            X_sub,
            n_neighbors=min(n_neighbors, subset_size - 1),
            resolution=resolution,
            metric=metric,
            mutual=mutual,
        )

        # Compare labels for the same samples.
        scores.append(
            adjusted_rand_score(
                full_labels[idx],
                sub_labels
            )
        )

    return float(np.mean(scores))

def run_analysis(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    analysis_name: str,
    output_dir: Path,
) -> pd.DataFrame:

    output_dir.mkdir(parents=True, exist_ok=True)
    cms_col = "CMScaller" if "CMScaller" in metadata.columns else None
    cms = None
    if cms_col is not None:
        cms = (
            metadata[cms_col]
            .astype(str)
            .str.upper()
            .str.strip()
        )
        cms = cms.where(
            cms.str.match(r"^CMS[1-4]$"),
            np.nan
        )

    all_results = []
    print(
        f"\nANALYSIS: {analysis_name} "
        f"({matrix.shape[0]} samples x {matrix.shape[1]} genes)"
    )
    clusters_base_dir = output_dir / "clusters_by_k"
    clusters_base_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    X = make_representation(matrix)

    # ============================================================
    # 1. Hierarchical clustering + K-means
    # ============================================================

    for distance in DISTANCES:
        for k in K_VALUES:

            if k >= len(matrix):
                continue

            k_dir = (
                clusters_base_dir
                / f"{distance}"
                / f"k{k}"
            )

            k_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # ----------------------------------------------------
            # Hierarchical clustering
            # ----------------------------------------------------

            labels_h, _ = hierarchical_labels(
                X,
                k,
                distance
            )

            metrics_h = evaluate_labels(
                X,
                labels_h
            )

            stability_h = cluster_stability(
                X,
                k,
                "hierarchical",
                distance
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

            # Save hierarchical assignments for EVERY distance
            assign_h = pd.DataFrame({
                "sample_id": matrix.index,
                "cluster": labels_h,
            })

            if "group" in metadata.columns:
                assign_h["group"] = (
                    metadata["group"].values
                )

            if cms_col is not None:
                assign_h["CMS"] = (
                    metadata[cms_col].values
                )

            assign_h.to_csv(
                k_dir
                / f"{analysis_name}_hierarchical_"
                f"{distance}_k{k}_assignments.csv",
                index=False,
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
            )

            if cms_col is not None:
                contingency = pd.crosstab(
                    assign_h["cluster"],
                    assign_h["CMS"],
                    margins=True,
                )

                contingency.to_csv(
                    k_dir
                    / f"{analysis_name}_hierarchical_"
                    f"{distance}_k{k}_CMS_contingency.csv"
                )

            # ----------------------------------------------------
            # K-means
            # ----------------------------------------------------

            if distance == "euclidean":

                labels_k = kmeans_labels(
                    X,
                    k
                )

                metrics_k = evaluate_labels(
                    X,
                    labels_k
                )

                stability_k = cluster_stability(
                    X,
                    k,
                    "kmeans",
                    distance
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
                })

                # Save K-means assignments
                assign_k = pd.DataFrame({
                    "sample_id": matrix.index,
                    "cluster": labels_k,
                })

                if "group" in metadata.columns:
                    assign_k["group"] = (
                        metadata["group"].values
                    )

                if cms_col is not None:
                    assign_k["CMS"] = (
                        metadata[cms_col].values
                    )

                assign_k.to_csv(
                    k_dir
                    / f"{analysis_name}_kmeans_k{k}_assignments.csv",
                    index=False,
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
                )

                if cms_col is not None:
                    contingency_k = pd.crosstab(
                        assign_k["cluster"],
                        assign_k["CMS"],
                        margins=True,
                    )

                    contingency_k.to_csv(
                        k_dir
                        / f"{analysis_name}_kmeans_k{k}_CMS_contingency.csv"
                    )

    # ----------------------------------------------------
    # LOUIVAIN
    # ----------------------------------------------------

    LOUVAIN_N_NEIGHBORS = 20
    LOUVAIN_RESOLUTION = 0.5
    LOUVAIN_METRIC = "cosine"
    LOUVAIN_MUTUAL = False

    print(
        "\nRunning Louvain community detection "
        f"(n_neighbors={LOUVAIN_N_NEIGHBORS}, "
        f"resolution={LOUVAIN_RESOLUTION}, "
        f"metric={LOUVAIN_METRIC}, "
        f"mutual={LOUVAIN_MUTUAL})"
    )

    labels_louvain, graph = louvain_labels(
        X,
        n_neighbors=LOUVAIN_N_NEIGHBORS,
        resolution=LOUVAIN_RESOLUTION,
        metric=LOUVAIN_METRIC,
        mutual=LOUVAIN_MUTUAL,
    )

    n_louvain_clusters = len(
        np.unique(labels_louvain)
    )

    metrics_louvain = evaluate_labels(
        X,
        labels_louvain
    )

    stability_louvain = louvain_stability(
        X,
        n_neighbors=LOUVAIN_N_NEIGHBORS,
        resolution=LOUVAIN_RESOLUTION,
        metric=LOUVAIN_METRIC,
        mutual=LOUVAIN_MUTUAL,
    )

    print(
        f"Louvain detected "
        f"{n_louvain_clusters} communities; "
        f"stability ARI = {stability_louvain:.3f}"
    )

    all_results.append({
        "analysis": analysis_name,
        "distance": LOUVAIN_METRIC,
        "algorithm": "louvain",
        "n_genes": matrix.shape[1],
        # Louvain does not require a predefined k.
        "k": np.nan,
        "n_clusters": n_louvain_clusters,
        "stability_ARI": stability_louvain,
        # Record graph parameters for reproducibility.
        "graph_n_neighbors": LOUVAIN_N_NEIGHBORS,
        "graph_resolution": LOUVAIN_RESOLUTION,
        "graph_mutual": LOUVAIN_MUTUAL,

        **metrics_louvain,
    })

    louvain_dir = (
        output_dir
        / "louvain"
    )

    louvain_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    assign_louvain = pd.DataFrame({
        "sample_id": matrix.index,
        "cluster": labels_louvain,
    })

    if "group" in metadata.columns:
        assign_louvain["group"] = (
            metadata["group"].values
        )

    if cms_col is not None:
        assign_louvain["CMS"] = (
            metadata[cms_col].values
        )

    assign_louvain.to_csv(
        louvain_dir
        / f"{analysis_name}_louvain_assignments.csv",
        index=False,
    )

    export_embeddings(
        X,
        matrix.index,
        labels_louvain,
        metadata,
        louvain_dir,
        prefix=f"{analysis_name}_louvain",
    )

    if cms_col is not None:
        louvain_contingency = pd.crosstab(
            assign_louvain["cluster"],
            assign_louvain["CMS"],
            margins=True,
        )
        louvain_contingency.to_csv(
            louvain_dir
            / f"{analysis_name}_louvain_CMS_contingency.csv"
        )

    edge_df = pd.DataFrame(
        [
            {
                "sample_i": matrix.index[i],
                "sample_j": matrix.index[j],
                "weight": data["weight"],
            }
            for i, j, data in graph.edges(data=True)
        ]
    )

    edge_df.to_csv(
        louvain_dir
        / f"{analysis_name}_louvain_graph_edges.csv",
        index=False,
    )

    results = pd.DataFrame(all_results)

    results.to_csv(
        output_dir
        / f"{analysis_name}_clustering_metrics.csv",
        index=False,
    )

    return results

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = load_pagerank_matrix(INPUT_DIR)
    matrix.to_csv(OUTPUT_DIR / "aggregated_pagerank_matrix.csv")
    summarize_pagerank_matrix(matrix, OUTPUT_DIR)

    metadata = merge_clinical_metadata(matrix.index, CLINICAL_FILE)
    metadata["group"] = infer_group(matrix.index, metadata)
    metadata.to_csv(OUTPUT_DIR / "sample_metadata_used_for_analysis.csv")

    run_analysis(matrix, metadata, "all_samples", OUTPUT_DIR / "all_samples")

    summary_files = list(OUTPUT_DIR.rglob("*_clustering_metrics.csv"))
    if summary_files:
        pd.concat([pd.read_csv(f) for f in summary_files if f.name != "ALL_clustering_metrics.csv"], ignore_index=True).drop_duplicates().to_csv(
            OUTPUT_DIR / "ALL_clustering_metrics.csv", index=False
        )
    print("\nClustering analysis completed successfully across all k values.")

if __name__ == "__main__":
    main()
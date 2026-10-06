from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


# ---------------------------------------------------------------------
# Internal clustering metrics
# ---------------------------------------------------------------------

def safe_metric(metric_func: Callable, *args) -> float:
    """Return a metric value or NaN if the metric cannot be computed."""
    try:
        return float(metric_func(*args))
    except Exception:
        return np.nan


def evaluate_labels(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Calculate internal clustering metrics.

    Metrics:
        - silhouette
        - within-cluster SSE
        - within-cluster variance
    """
    result = {
        "silhouette": np.nan,
        "within_cluster_sse": np.nan,
        "within_cluster_variance": np.nan,
    }

    unique = np.unique(labels)
    n = len(labels)
    k = len(unique)

    within_ss = 0.0

    for cluster_id in unique:
        X_cluster = X[labels == cluster_id]

        if len(X_cluster) == 0:
            continue

        center = X_cluster.mean(axis=0)
        within_ss += np.sum((X_cluster - center) ** 2)

    result["within_cluster_sse"] = float(within_ss)

    if n > k:
        result["within_cluster_variance"] = float(
            within_ss / (n - k)
        )

    if 1 < k < n:
        result["silhouette"] = safe_metric(
            silhouette_score,
            X,
            labels,
        )

    return result


# ---------------------------------------------------------------------
# Reference / biological validation metrics
# ---------------------------------------------------------------------

def evaluate_reference_labels(
    predicted_labels: np.ndarray,
    metadata: pd.DataFrame,
) -> dict:
    """
    Compare inferred clusters with clinical/reference labels.

    Two ARI values are calculated:

        ari_group:
            Agreement with Tumor vs Normal.

        ari_cms:
            Agreement with CMScaller (CMS1-CMS4), using
            Tumor samples only.
    """
    result = {
        "ari_group": np.nan,
        "ari_cms": np.nan,
    }

    # -------------------------------------------------------------
    # ARI: Tumor vs Normal
    # -------------------------------------------------------------

    if "group" in metadata.columns:
        valid_group = metadata["group"].notna()

        if valid_group.sum() >= 2:
            reference_group = metadata.loc[
                valid_group, "group"
            ]

            predicted_group = predicted_labels[
                valid_group.to_numpy()
            ]

            if reference_group.nunique() >= 2:
                result["ari_group"] = float(
                    adjusted_rand_score(
                        reference_group,
                        predicted_group,
                    )
                )

    # -------------------------------------------------------------
    # ARI: CMS1-CMS4
    # Tumor samples only
    # -------------------------------------------------------------

    if {
        "group",
        "CMScaller",
    }.issubset(metadata.columns):

        tumor_mask = (
            metadata["group"].eq("Tumor")
            & metadata["CMScaller"].notna()
        )

        if tumor_mask.sum() >= 2:
            reference_cms = metadata.loc[
                tumor_mask, "CMScaller"
            ]

            predicted_cms = predicted_labels[
                tumor_mask.to_numpy()
            ]

            if reference_cms.nunique() >= 2:
                result["ari_cms"] = float(
                    adjusted_rand_score(
                        reference_cms,
                        predicted_cms,
                    )
                )

    return result


# ---------------------------------------------------------------------
# Gap statistic
# ---------------------------------------------------------------------

def _hierarchical_euclidean_labels(
    X: np.ndarray,
    k: int,
) -> np.ndarray:
    """Hierarchical clustering using Ward linkage."""
    Z = linkage(
        X,
        method="ward",
        metric="euclidean",
    )

    return fcluster(
        Z,
        t=k,
        criterion="maxclust",
    )


def gap_statistic(
    X: np.ndarray,
    k_values: Sequence[int],
    method: str = "kmeans",
    n_refs: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate the Gap statistic.

    Reference datasets are generated independently for each feature
    using uniform distributions bounded by the observed minimum and
    maximum of that feature.

    Supported methods:
        - kmeans
        - hierarchical
    """
    rng = np.random.default_rng(random_state)

    n, p = X.shape

    lower = X.min(axis=0)
    upper = X.max(axis=0)

    def fit_labels(
        data: np.ndarray,
        k: int,
        seed: int,
    ) -> np.ndarray:

        if method == "kmeans":
            return KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=20,
            ).fit_predict(data)

        if method == "hierarchical":
            return _hierarchical_euclidean_labels(
                data,
                k,
            )

        raise ValueError(
            f"Unsupported Gap statistic method: {method}"
        )

    def calculate_sse(
        data: np.ndarray,
        labels: np.ndarray,
    ) -> float:

        total = 0.0

        for cluster_id in np.unique(labels):
            cluster = data[labels == cluster_id]

            if len(cluster) == 0:
                continue

            center = cluster.mean(axis=0)

            total += np.sum(
                (cluster - center) ** 2
            )

        return max(
            float(total),
            np.finfo(float).tiny,
        )

    rows = []

    for k in k_values:

        if k < 1 or k >= n:
            continue

        # Observed data
        labels = fit_labels(
            X,
            k,
            random_state,
        )

        log_w_observed = np.log(
            calculate_sse(X, labels)
        )

        # Reference datasets
        log_w_reference = []

        for b in range(n_refs):

            X_ref = rng.uniform(
                low=lower,
                high=upper,
                size=(n, p),
            )

            labels_ref = fit_labels(
                X_ref,
                k,
                random_state + b,
            )

            log_w_reference.append(
                np.log(
                    calculate_sse(
                        X_ref,
                        labels_ref,
                    )
                )
            )

        log_w_reference = np.asarray(
            log_w_reference
        )

        gap = float(
            log_w_reference.mean()
            - log_w_observed
        )

        if n_refs > 1:
            sd = float(
                log_w_reference.std(ddof=1)
            )
        else:
            sd = 0.0

        gap_se = sd * np.sqrt(
            1.0 + 1.0 / n_refs
        )

        rows.append({
            "algorithm": method,
            "k": k,
            "log_w_observed": log_w_observed,
            "gap": gap,
            "gap_se": gap_se,
        })

    result = pd.DataFrame(rows)

    result["selected_k"] = False

    # Tibshirani one-standard-error rule.
    if len(result) >= 2:

        for i in range(len(result) - 1):

            current = result.iloc[i]
            following = result.iloc[i + 1]

            if current["k"] + 1 != following["k"]:
                continue

            if current["gap"] >= (
                following["gap"]
                - following["gap_se"]
            ):
                result.loc[
                    result.index[i],
                    "selected_k",
                ] = True

                break

    return result


# ---------------------------------------------------------------------
# Clustering stability
# ---------------------------------------------------------------------

def cluster_stability(
    X: np.ndarray,
    k: int,
    method: str,
    distance: str,
    clustering_function: Callable,
    n_repeats: int = 100,
    fraction: float = 0.80,
    random_state: int = 42,
) -> float:
    """
    Estimate clustering stability using subsampling and ARI.

    The clustering_function must have the signature:

        clustering_function(X, k, distance, random_state)

    and return a one-dimensional label array.
    """
    n = X.shape[0]

    if n < max(10, k * 3):
        return np.nan

    full_labels = clustering_function(
        X,
        k,
        distance,
        random_state,
    )

    rng = np.random.default_rng(
        random_state
    )

    subset_size = max(
        k * 2,
        int(np.floor(n * fraction)),
    )

    subset_size = min(
        subset_size,
        n,
    )

    scores = []

    for repeat in range(n_repeats):

        idx = rng.choice(
            n,
            size=subset_size,
            replace=False,
        )

        X_sub = X[idx]

        sub_labels = clustering_function(
            X_sub,
            k,
            distance,
            random_state + repeat + 1,
        )

        score = adjusted_rand_score(
            full_labels[idx],
            sub_labels,
        )

        scores.append(score)

    if not scores:
        return np.nan

    return float(np.mean(scores))


def louvain_stability(
    X: np.ndarray,
    louvain_function: Callable,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    metric: str = "cosine",
    mutual: bool = True,
    n_repeats: int = 100,
    fraction: float = 0.80,
    random_state: int = 42,
) -> float:
    """
    Estimate Louvain stability using subsampling and ARI.
    """
    n = X.shape[0]

    if n < max(
        10,
        n_neighbors + 2,
    ):
        return np.nan

    full_labels, _ = louvain_function(
        X,
        n_neighbors=n_neighbors,
        resolution=resolution,
        metric=metric,
        mutual=mutual,
        random_state=random_state,
    )

    rng = np.random.default_rng(
        random_state
    )

    subset_size = max(
        n_neighbors + 2,
        int(np.floor(n * fraction)),
    )

    subset_size = min(
        subset_size,
        n,
    )

    scores = []

    for repeat in range(n_repeats):

        idx = rng.choice(
            n,
            size=subset_size,
            replace=False,
        )

        X_sub = X[idx]

        sub_labels, _ = louvain_function(
            X_sub,
            n_neighbors=min(
                n_neighbors,
                subset_size - 1,
            ),
            resolution=resolution,
            metric=metric,
            mutual=mutual,
            random_state=random_state + repeat + 1,
        )

        score = adjusted_rand_score(
            full_labels[idx],
            sub_labels,
        )

        scores.append(score)

    if not scores:
        return np.nan

    return float(np.mean(scores))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def compute_vn_entropy(density: sp.spmatrix) -> float:
    """
    Compute the Von Neumann entropy of a density matrix.

        H(rho) = -sum_i lambda_i * log(lambda_i)   (over positive eigenvalues)

    Uses sparse eigendecomposition (eigsh) with k = N-1 eigenvalues and
    recovers the last eigenvalue from the trace condition tr(rho) = 1:
        lambda_last = 1 - sum(other eigenvalues)

    Parameters
    ----------
    density : scipy.sparse matrix
        Symmetric, positive semi-definite density matrix with trace 1.

    Returns
    -------
    float
        Von Neumann entropy (nats, base-e logarithm).
    """
    N = density.shape[0]
    if N == 1:
        return 0.0

    # eigsh requires k < N; we get the N-1 largest eigenvalues and
    # recover the last one from the trace condition.
    k = N - 1
    eigenvalues = eigsh(density, k=k, which="LM", return_eigenvectors=False)
    last = 1.0 - eigenvalues.sum()
    all_eigenvalues = np.append(eigenvalues, last)

    pos = all_eigenvalues[all_eigenvalues > 0]
    return float(-np.sum(pos * np.log(pos)))


def compute_js_divergence(
    adj1: sp.spmatrix,
    adj2: sp.spmatrix,
    vn1: float,
    vn2: float,
) -> float:
    """
    Compute the Jensen-Shannon divergence between two networks.

    Given adjacency matrices A1, A2 and their pre-computed Von Neumann
    entropies H1, H2:

        JSD(rho || sigma) = H(M) - 0.5 * (H1 + H2)

    where M = (rho + sigma) / 2 and rho, sigma are the BGS density matrices
    built from A1, A2 respectively.

    Passing the entropies explicitly lets callers cache them when computing
    pairwise JSD across many layer pairs.

    Parameters
    ----------
    adj1 : scipy.sparse matrix
        Adjacency matrix of the first network.
    adj2 : scipy.sparse matrix
        Adjacency matrix of the second network.
    vn1 : float
        Von Neumann entropy of the first network (pre-computed).
    vn2 : float
        Von Neumann entropy of the second network (pre-computed).

    Returns
    -------
    float
        Jensen-Shannon divergence (non-negative).

    References
    ----------
    De Domenico et al. (2015) "Structural reducibility of multilayer networks",
    Nature Communications, 6, 6864.
    """
    from .utils.parsing import build_density_bgs_from_adjacency_matrix

    rho = build_density_bgs_from_adjacency_matrix(adj1)
    sigma = build_density_bgs_from_adjacency_matrix(adj2)
    M = (rho + sigma) * 0.5

    entropy_M = compute_vn_entropy(M)
    return float(entropy_M - 0.5 * (vn1 + vn2))

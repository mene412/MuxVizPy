import numpy as np
import scipy as sp
import scipy.sparse as sps
import pandas as pd
from scipy.sparse import find, identity, coo_matrix
import graph_tool as gt
from graph_tool import centrality #, inference
import graph_tool.correlations as gtcorr
import graph_tool.clustering as gtclust
import logging
from typing import Optional

from MuxVizPy.utils import approx_utils
from MuxVizPy.utils.approx_utils import get_largest_magnitude_eigenvalue, get_largest_real_eigenvalue, approximate_largest_eigenvalue
from MuxVizPy.utils import parsing as parsing_utils
from MuxVizPy.utils.katz_utils import _katz_neumann, _katz_krylov, _VALID_SOLVERS


# ---------------------------------------------------------------------------
# Block accumulation and aggregation helpers (integrated from hornet/node_based)
# ---------------------------------------------------------------------------

def is_in_diagonal_block(i: int, j: int, n: int, l: int) -> bool:
    """Check if nodes i and j are in the same diagonal block (same layer)."""
    return (i // n) == (j // n)


def _accumulate_on_diagonal_blocks(
    adj: sps.csr_matrix, n: int, l: int,
    is_out_of_diagonal: bool, weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Vectorized accumulation of weights on diagonal (or off-diagonal) blocks.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Supra-adjacency matrix of shape (n*l, n*l).
    n : int
        Number of nodes.
    l : int
        Number of layers.
    is_out_of_diagonal : bool
        If True, accumulate off-diagonal (inter-layer) blocks.
    weights : np.ndarray or None
        Custom weights; if None, uses matrix data.

    Returns
    -------
    np.ndarray
        Shape (n, l) array of accumulated values.
    """
    NL = n * l
    if adj.shape != (NL, NL):
        raise ValueError(f"Adjacency matrix shape {adj.shape} does not match expected shape {(NL, NL)}")

    coo = adj.tocoo(copy=False)
    row_layers = coo.row // n
    col_layers = coo.col // n
    diagonal_mask = row_layers == col_layers
    if is_out_of_diagonal:
        diagonal_mask = ~diagonal_mask

    tgt_nodes = coo.col[diagonal_mask] % n
    tgt_layers = col_layers[diagonal_mask]
    flat_idx = tgt_layers * n + tgt_nodes

    if weights is None:
        w = coo.data[diagonal_mask]
    else:
        w = weights[diagonal_mask]

    accum = np.bincount(flat_idx, weights=w, minlength=NL)
    return accum.reshape(l, n).T  # shape (n, l)


def aggregate_metrics_over_layers(metrics: np.ndarray, method: str = "sum") -> np.ndarray:
    """
    Aggregate per-layer metrics to per-node metrics.

    Parameters
    ----------
    metrics : np.ndarray
        Array of shape (n, l).
    method : str
        One of "mean", "sum", "max", "min".

    Returns
    -------
    np.ndarray
        Array of shape (n,).
    """
    if method == "mean":
        return metrics.mean(axis=1)
    elif method == "sum":
        return metrics.sum(axis=1)
    elif method == "max":
        return metrics.max(axis=1)
    elif method == "min":
        return metrics.min(axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ---------------------------------------------------------------------------
# Per-layer degree / strength metrics (integrated from hornet/node_based)
# ---------------------------------------------------------------------------

def compute_indegree(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute indegree of each node stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = np.ones_like(adj.data, dtype=np.float64)
    indegree = _accumulate_on_diagonal_blocks(adj, n, l, is_out_of_diagonal=False, weights=weights)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Indegree shape: {indegree.shape}, mean: {indegree.mean()}, max: {indegree.max()}, min: {indegree.min()}")
    return indegree


def compute_aggregated_indegree(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated indegree over layers. Returns shape (n,)."""
    indegree = compute_indegree(adj, n, l, logger=logger)
    return aggregate_metrics_over_layers(indegree, method=method)


def compute_instrength(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute instrength of each node stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = adj.data.astype(np.float64)
    instrength = _accumulate_on_diagonal_blocks(adj, n, l, is_out_of_diagonal=False, weights=weights)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Instrength shape: {instrength.shape}, mean: {instrength.mean()}, max: {instrength.max()}, min: {instrength.min()}")
    return instrength


def compute_aggregated_instrength(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", *, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated instrength over layers. Returns shape (n,)."""
    instrength = compute_instrength(adj, n, l, logger=logger)
    return aggregate_metrics_over_layers(instrength, method=method)


def compute_outdegree(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute outdegree of each node stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = np.ones_like(adj.data, dtype=np.float64)
    outdegree = _accumulate_on_diagonal_blocks(adj.T, n, l, is_out_of_diagonal=False, weights=weights)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Outdegree shape: {outdegree.shape}, mean: {outdegree.mean()}, max: {outdegree.max()}, min: {outdegree.min()}")
    return outdegree


def compute_aggregated_outdegree(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", *, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated outdegree over layers. Returns shape (n,)."""
    outdegree = compute_outdegree(adj, n, l, logger=logger)
    return aggregate_metrics_over_layers(outdegree, method=method)


def compute_outstrength(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute outstrength of each node stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = adj.data.astype(np.float64)
    outstrength = _accumulate_on_diagonal_blocks(adj.T, n, l, is_out_of_diagonal=False, weights=weights)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Outstrength shape: {outstrength.shape}, mean: {outstrength.mean()}, max: {outstrength.max()}, min: {outstrength.min()}")
    return outstrength


def compute_aggregated_outstrength(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated outstrength over layers. Returns shape (n,)."""
    outstrength = compute_outstrength(adj, n, l, logger=logger)
    return aggregate_metrics_over_layers(outstrength, method=method)


def compute_multiindegree(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute multi-indegree (inter-layer blocks) stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = np.ones_like(adj.data, dtype=np.float64)
    return _accumulate_on_diagonal_blocks(adj, n, l, is_out_of_diagonal=True, weights=weights)


def compute_aggregated_multiindegree(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated multi-indegree over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_multiindegree(adj, n, l, logger=logger), method=method)


def compute_multiinstrength(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute multi-instrength (inter-layer blocks) stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = adj.data.astype(np.float64)
    return _accumulate_on_diagonal_blocks(adj, n, l, is_out_of_diagonal=True, weights=weights)


def compute_aggregated_multiinstrength(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", *, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated multi-instrength over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_multiinstrength(adj, n, l, logger=logger), method=method)


def compute_multioutdegree(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute multi-outdegree (inter-layer blocks) stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = np.ones_like(adj.data, dtype=np.float64)
    return _accumulate_on_diagonal_blocks(adj.T, n, l, is_out_of_diagonal=True, weights=weights)


def compute_aggregated_multioutdegree(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated multi-outdegree over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_multioutdegree(adj, n, l, logger=logger), method=method)


def compute_multioutstrength(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute multi-outstrength (inter-layer blocks) stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = adj.data.astype(np.float64)
    return _accumulate_on_diagonal_blocks(adj.T, n, l, is_out_of_diagonal=True, weights=weights)


def compute_aggregated_multioutstrength(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated multi-outstrength over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_multioutstrength(adj, n, l, logger=logger), method=method)


def compute_total_indegree(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute total indegree (intra + inter layer) stratified by layer. Returns shape (n, l)."""
    return compute_indegree(adj, n, l, logger=logger) + compute_multiindegree(adj, n, l, logger=logger)


def compute_aggregated_total_indegree(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated total indegree over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_total_indegree(adj, n, l, logger=logger), method=method)


def compute_total_instrength(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute total instrength (intra + inter layer) stratified by layer. Returns shape (n, l)."""
    return compute_instrength(adj, n, l, logger=logger) + compute_multiinstrength(adj, n, l, logger=logger)


def compute_aggregated_total_instrength(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated total instrength over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_total_instrength(adj, n, l, logger=logger), method=method)


def compute_total_outdegree(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute total outdegree (intra + inter layer) stratified by layer. Returns shape (n, l)."""
    return compute_outdegree(adj, n, l, logger=logger) + compute_multioutdegree(adj, n, l, logger=logger)


def compute_aggregated_total_outdegree(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated total outdegree over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_total_outdegree(adj, n, l, logger=logger), method=method)


def compute_total_outstrength(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute total outstrength (intra + inter layer) stratified by layer. Returns shape (n, l)."""
    return compute_outstrength(adj, n, l, logger=logger) + compute_multioutstrength(adj, n, l, logger=logger)


def compute_aggregated_total_outstrength(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated total outstrength over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_total_outstrength(adj, n, l, logger=logger), method=method)


def compute_multidegree(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute multidegree (total indegree + total outdegree) stratified by layer. Returns shape (n, l)."""
    return compute_total_indegree(adj, n, l, logger=logger) + compute_total_outdegree(adj, n, l, logger=logger)


def compute_aggregated_multidegree(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated multidegree over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_multidegree(adj, n, l, logger=logger), method=method)


def compute_multistrength(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute multistrength (total instrength + total outstrength) stratified by layer. Returns shape (n, l)."""
    return compute_total_instrength(adj, n, l, logger=logger) + compute_total_outstrength(adj, n, l, logger=logger)


def compute_aggregated_multistrength(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated multistrength over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_multistrength(adj, n, l, logger=logger), method=method)


def compute_multi_degree(
    adj: sps.csr_matrix, n: int, l: int,
    is_directed: bool = True,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute multi-layer degree following muxViz R semantics.

    Binarized in-degree + out-degree on intra-layer (diagonal) blocks,
    aggregated across layers.  For undirected networks the sum is halved
    to avoid double-counting.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix of shape (n*l, n*l).
    n : int
        Number of nodes.
    l : int
        Number of layers.
    is_directed : bool
        If True, returns in + out. If False, returns (in + out) / 2.
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Multi-degree per physical node, shape (n,).
    """
    indeg = compute_aggregated_indegree(adj, n, l, method="sum", logger=logger)
    outdeg = compute_aggregated_outdegree(adj, n, l, method="sum", logger=logger)
    if is_directed:
        return indeg + outdeg
    else:
        return (indeg + outdeg) / 2.0


# ---------------------------------------------------------------------------
# Centrality implementations (integrated from hornet/node_based)
# ---------------------------------------------------------------------------

def compute_eigenvector_centrality(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Compute multi-layer eigenvector centrality using the dominant eigenvector of A^T.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix of shape (n*l, n*l).
    n : int
        Number of nodes.
    l : int
        Number of layers.
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized eigenvector centrality per physical node, shape (n,).
    """
    NL = n * l
    if not sps.isspmatrix(adj):
        raise TypeError("adj must be a SciPy sparse matrix")
    if adj.shape != (NL, NL):
        raise ValueError(f"Adjacency matrix shape {adj.shape} != ({NL}, {NL})")

    AT = adj.transpose().tocsc()
    lam, lvec = get_largest_magnitude_eigenvalue(AT, logger=logger)

    X = np.reshape(lvec, (n, l), order="F")
    eig_centrality = X.sum(axis=1)

    maxv = eig_centrality.max() if eig_centrality.size else 0.0
    ec = np.zeros_like(eig_centrality, dtype=np.float32) if maxv == 0 else (eig_centrality / maxv).astype(np.float32)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Eigenvector: lambda_max=%.6g, min/mean/max=%.4g/%.4g/%.4g",
                     lam, ec.min(), ec.mean(), ec.max())
    return ec


def compute_katz_centrality(
    adj: sps.csr_matrix, n: int, l: int,
    *,
    solver: str = "direct",
    maxiter: int = 10000,
    tol: float = 1e-6,
    return_eigenvalue: bool = False,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute multi-layer Katz centrality.

    Katz centrality is defined as the solution x of

        (I - alpha * A) x = 1,

    with ``alpha = (1 - EPS) / rho(A)``, where ``rho(A)`` is the spectral
    radius of the supra-adjacency matrix ``A``. The vector x is then reshaped
    to ``(n, l)`` in column-major order, summed across layers to aggregate
    replicas, and max-normalized so that the largest score equals 1.

    Four solvers are available via the ``solver`` keyword:

    - ``"direct"`` *(default, exact)*: sparse LU factorization of
      ``Aop = I - alpha * A`` via ``scipy.sparse.linalg.spsolve``.
      Most accurate but does not exploit the spectral-radius bound and has
      the highest memory cost.
    - ``"neumann"`` *(stationary iterative)*: Richardson / Neumann-series
      fixed-point iteration ``x_{k+1} = alpha * A @ x_k + b`` which unrolls
      to ``sum_k (alpha*A)^k b``. Needs only matvecs with ``adj`` and never
      materializes ``Aop``. Converges unconditionally because Katz forces
      ``rho(alpha*A) = 1 - EPS < 1`` by construction, although convergence
      can be slow when ``EPS`` is small.
    - ``"gmres"``, ``"bicgstab"`` *(Krylov iterative)*: Krylov subspace
      solvers applied to ``Aop``, preconditioned by an incomplete-LU
      factorization (``spilu``). Typically faster than the Neumann branch
      when ``Aop`` is ill-conditioned.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix of shape ``(n*l, n*l)``.
    n : int
        Number of physical nodes.
    l : int
        Number of layers.
    solver : {"direct", "neumann", "gmres", "bicgstab"}, default ``"direct"``
        Linear solver used to invert ``(I - alpha * A)``. See the function
        description for the trade-offs of each branch.
    maxiter : int, default 1000
        Maximum iterations for iterative solvers. Ignored when
        ``solver="direct"``.
    tol : float, default 1e-6
        Convergence tolerance for iterative solvers (passed as ``rtol`` to
        the scipy Krylov solvers and as the L2 step size for Neumann).
        Ignored when ``solver="direct"``.
    return_eigenvalue : bool, default False
        If True, also return the Rayleigh quotient ``x^T A x / x^T x`` as
        an estimate of the dominant eigenvalue associated with the solution.
    logger : logging.Logger, optional
        If provided, logs diagnostics at DEBUG level (eigenvalue, alpha,
        iteration count for the Neumann branch).

    Returns
    -------
    katz : np.ndarray, shape (n,)
        Max-normalized Katz centrality per physical node, ``float32``.
    eigenvalue : float, optional
        Rayleigh quotient estimate. Returned only when
        ``return_eigenvalue=True``.

    Raises
    ------
    TypeError
        If ``adj`` is not a SciPy sparse matrix.
    ValueError
        If ``adj.shape != (n*l, n*l)`` or if ``solver`` is not one of the
        supported values.

    Notes
    -----
    The ``"neumann"`` branch initializes its iterate from
    ``np.random.randn``; seed the global NumPy RNG before calling for
    reproducibility.
    """
    EPS = 1e-5
    NL = n * l
    if not sps.isspmatrix(adj):
        raise TypeError("adj must be a SciPy sparse matrix")
    if adj.shape != (NL, NL):
        raise ValueError(f"Adjacency matrix shape {adj.shape} != ({NL}, {NL})")
    if solver not in _VALID_SOLVERS:
        raise ValueError(
            f"Unknown solver {solver!r}. Valid options: {_VALID_SOLVERS}"
        )

    lam, _ = get_largest_magnitude_eigenvalue(adj, logger=logger)

    spectral_radius = float(np.abs(lam))
    alpha = (1 - EPS) / spectral_radius
    # IMPORTANT - we choose this alpha because of MUXVIZ. EVEN if it has some issues.

    b = np.ones(NL, dtype=np.float64)

    if solver == "neumann":
        # Neumann power iteration
        x = _katz_neumann(adj, alpha, b, maxiter=maxiter, tol=tol, logger=logger)
    else:
        # Direct and Krylov branches both need the explicit operator Aop.
        I = sps.eye(NL, format="csc", dtype=np.float64)    
        Aop = I - alpha * adj.tocsc()
        if solver == "direct":
            x = sps.linalg.spsolve(Aop, b)
        else:  # "gmres" or "bicgstab"
            x = _katz_krylov(Aop, b, method=solver, maxiter=maxiter, tol=tol)

    eigenvalue = (x.T @ adj @ x) / (x.T @ x) # Rayleigh quotient for the Katz operator

    X = np.reshape(x, (n, l), order="F")
    katz_centrality = X.sum(axis=1)
    maxv = katz_centrality.max() if katz_centrality.size else 0.0
    katz = np.zeros_like(katz_centrality, dtype=np.float32) if maxv == 0 else (katz_centrality / maxv).astype(np.float32)

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Katz: alpha=%.6g, lambda_max=%.6g, min/mean/max=%.4g/%.4g/%.4g",
                     alpha, lam, katz.min(), katz.mean(), katz.max())
        
    if return_eigenvalue:
        return katz, eigenvalue
    else:
        return katz


def compute_multi_rw_centrality(
    adj: sps.csr_matrix, n: int, l: int, kind: str,
    *,
    alpha: float = 0.85, tol: float = 1e-12, max_iter: int = 10000,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute multi-layer random walk centrality.

    Two modes:
        - ``kind="classical"``: stationary distribution via the dominant
          eigenvector of the row-stochastic transition matrix T^T.
        - ``kind="pagerank"``: PageRank power iteration with teleportation
          and dangling-node handling.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix of shape (n*l, n*l).
    n : int
        Number of nodes.
    l : int
        Number of layers.
    kind : str
        ``"classical"`` or ``"pagerank"``.
    alpha : float
        Damping factor (only used when ``kind="pagerank"``). Default 0.85.
    tol : float
        Convergence tolerance (only used when ``kind="pagerank"``). Default 1e-12.
    max_iter : int
        Maximum iterations (only used when ``kind="pagerank"``). Default 10000.
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized RW centrality per physical node, shape (n,).
    """
    NL = n * l
    kind = kind.lower().strip()

    if kind == "classical":
        tran_matrix = parsing_utils.build_transition_matrix_from_adjacency_matrix(
            adj, n, l, kind="classical", logger=logger,
        )
        eigvals, eigvecs = sps.linalg.eigs(tran_matrix.T, k=1, which="LM", return_eigenvectors=True)
        lam = float(np.real_if_close(eigvals[0]))
        vec = np.real_if_close(eigvecs[:, 0])

        x = vec / vec.sum()
        x = np.reshape(x, (n, l), order="F")
        x = x.sum(axis=1)

        maxv = x.max() if x.size else 0.0
        rc = np.zeros_like(x, dtype=np.float32) if maxv == 0 else (x / maxv).astype(np.float32)
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Random Walk (classical): lambda_max=%.6g, min/mean/max=%.4g/%.4g/%.4g",
                         lam, rc.min(), rc.mean(), rc.max())
        return rc

    elif kind == "pagerank":
        P = parsing_utils.build_transition_matrix_from_adjacency_matrix(
            adj, n, l, kind="classical", logger=logger,
        ).tocsr()

        row_sums = np.asarray(P.sum(axis=1)).ravel()
        dangling_mask = row_sums == 0.0
        n_dangling = dangling_mask.sum()

        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug("PageRank: %d/%d dangling nodes (%.1f%%)", n_dangling, NL, 100 * n_dangling / NL)

        v = np.full(NL, 1.0 / NL, dtype=np.float64)
        x = np.full(NL, 1.0 / NL, dtype=np.float64)
        PT = P.T.tocsr()

        err = np.inf
        it = 0
        for it in range(max_iter):
            dangling_mass = x[dangling_mask].sum()
            x_next = alpha * (PT @ x) + (alpha * dangling_mass + (1.0 - alpha)) * v
            x_sum = x_next.sum()
            if x_sum > 0:
                x_next /= x_sum
            err = np.abs(x_next - x).sum()
            x = x_next
            if err < tol:
                break

        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug("PageRank converged in %d iterations with error %.3e", it + 1, err)

        X = np.reshape(x, (n, l), order="F")
        mpc = X.sum(axis=1)
        maxv = mpc.max() if mpc.size else 0.0
        mpc_norm = np.zeros_like(mpc, dtype=np.float32) if maxv == 0 else (mpc / maxv).astype(np.float32)

        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug("PageRank: alpha=%.3f, iters=%d, err=%.3e, min/mean/max=%.4g/%.4g/%.4g",
                         alpha, it + 1, err, mpc_norm.min(), mpc_norm.mean(), mpc_norm.max())
        return mpc_norm

    else:
        raise ValueError(f"Unknown RW kind: {kind!r}. Expected 'classical' or 'pagerank'.")


def compute_multipagerank_centrality(
    adj: sps.csr_matrix, n: int, l: int,
    alpha: float = 0.85, tol: float = 1e-12, max_iter: int = 10000,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute multi-layer PageRank centrality.

    Thin wrapper around :func:`compute_multi_rw_centrality` with
    ``kind="pagerank"``.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix of shape (n*l, n*l).
    n : int
        Number of nodes.
    l : int
        Number of layers.
    alpha : float
        Damping factor. Default 0.85.
    tol : float
        Convergence tolerance. Default 1e-12.
    max_iter : int
        Maximum iterations. Default 10000.
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized PageRank per physical node, shape (n,).
    """
    return compute_multi_rw_centrality(
        adj, n, l, kind="pagerank",
        alpha=alpha, tol=tol, max_iter=max_iter, logger=logger,
    )


def compute_multi_hub_centrality(
    adj: sps.csr_matrix, n: int, l: int,
    eps: float = 1e-16, max_attempts: int = 10,
    approx: bool = False, approx_args: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute multi-layer hub centrality using the dominant eigenvector of A * A^T.

    N.B. The retry loop (max_attempts) exists because scipy.sparse.linalg.eigs
    (ARPACK) uses random initialization and can return degenerate eigenvectors
    on small or very sparse supra-matrices. The approximate path (power
    iteration with eps perturbation) is deterministic and does not need retries.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix.
    n : int
        Number of nodes.
    l : int
        Number of layers.
    eps : float
        Small value added for Perron-Frobenius uniqueness.
    max_attempts : int
        Max eigenvalue search retries (relevant for exact path only).
    approx : bool
        Use approximate eigenvalue computation.
    approx_args : dict, optional
        Arguments for approximate computation.
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized hub centrality per physical node, shape (n,).
    """
    AA = adj.dot(adj.T)
    if approx and approx_args is None:
        approx_args = {"maxiter": 1000, "tol": 1e-6}
    eigen_search_failed = True
    count_eigen_attempts = 0
    while eigen_search_failed and count_eigen_attempts < max_attempts:
        if approx:
            if logger:
                logger.debug("Using approximate largest eigenvalue computation for hub centrality")
            eigenval, eigenvec = approximate_largest_eigenvalue(
                AA, alpha=1.0, cval=approx_args.get("cval", eps),
                maxiter=approx_args.get("maxiter", 1000),
                tol=approx_args.get("tol", 1e-6))
        else:
            eigenval, eigenvec = get_largest_real_eigenvalue(AA, logger=logger)
        hc = eigenvec.reshape((l, n)).sum(axis=0)
        hc = hc / hc.max()
        maxv = hc.max() if hc.size else 0.0
        count_eigen_attempts += 1
        if np.mean(hc) > 0 and maxv > 0:
            eigen_search_failed = False

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Hub with attempts %d: lambda_max=%.6g, and maxv=%.4g min/mean/max=%.4g/%.4g/%.4g",
                     count_eigen_attempts, eigenval, maxv, hc.min(), hc.mean(), hc.max())
    return hc


def compute_multi_authority_centrality(
    adj: sps.csr_matrix, n: int, l: int,
    eps: float = 1e-16, max_attempts: int = 10,
    approx: bool = False, approx_args: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute multi-layer authority centrality using the dominant eigenvector of A^T * A.

    Solves by scipy.sparse.linalg.eigs (exact) or by power iteration if approx=True.

    N.B. The retry loop (max_attempts) exists because scipy.sparse.linalg.eigs
    (ARPACK) uses random initialization and can return degenerate eigenvectors
    on small or very sparse supra-matrices. The approximate path (power
    iteration with eps perturbation) is deterministic and does not need retries.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix.
    n : int
        Number of nodes.
    l : int
        Number of layers.
    eps : float
        Small value added for Perron-Frobenius uniqueness.
    max_attempts : int
        Max eigenvalue search retries (relevant for exact path only).
    approx : bool
        Use approximate eigenvalue computation via power iteration.
    approx_args : dict, optional
        Arguments for approximate computation (keys: cval, maxiter, tol).
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized authority centrality per physical node, shape (n,).
    """
    AAT = adj.T.dot(adj).astype(np.float64)
    AAT.data += eps
    if approx and approx_args is None:
        approx_args = {"maxiter": 1000, "tol": 1e-6}

    eigen_search_failed = True
    count_eigen_attempts = 0
    while eigen_search_failed and count_eigen_attempts < max_attempts:
        if approx:
            if logger:
                logger.debug("Using approximate largest eigenvalue computation for authority centrality")
            eigenval, eigenvec = approximate_largest_eigenvalue(
                AAT, alpha=1.0, cval=approx_args.get("cval", eps),
                maxiter=approx_args.get("maxiter", 1000),
                tol=approx_args.get("tol", 1e-6))
        else:
            eigenval, eigenvec = get_largest_real_eigenvalue(AAT, logger=logger)
        X = np.reshape(eigenvec, (n, l), order="F")
        authority_centrality = X.sum(axis=1)
        maxv = authority_centrality.max() if authority_centrality.size else 0.0
        ac = np.zeros_like(authority_centrality, dtype=np.float32) if maxv == 0 else (authority_centrality / maxv).astype(np.float32)
        count_eigen_attempts += 1
        if np.mean(ac) > 0 and maxv > 0:
            eigen_search_failed = False

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Authority with attempts %d: lambda_max=%.6g, min/mean/max=%.4g/%.4g/%.4g",
                     count_eigen_attempts, eigenval, ac.min(), ac.mean(), ac.max())
    return ac


# ---------------------------------------------------------------------------
# Public API with backend dispatch
# ---------------------------------------------------------------------------

def get_multi_degree(
    supra: sps.spmatrix, layers: int, nodes: int,
    is_directed: bool = True, backend: str = "muxvizpy",
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Computes the degree of each physical node by aggregating the supra-adjacency matrix.

    Parameters
    ----------
    supra : scipy.sparse.spmatrix
        Supra-adjacency matrix of the multilayer network.
    layers : int
        Number of layers.
    nodes : int
        Number of physical nodes.
    is_directed : bool
        If True, returns in + out degree. If False, returns (in + out) / 2.
    backend : str
        "muxvizpy" (aggregate network column sum) or "hornet" (in + out on
        intra-layer blocks, matching muxViz R semantics).
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Degree vector for physical nodes (aggregated across layers).
    """
    if backend == "hornet":
        return compute_multi_degree(supra, nodes, layers, is_directed=is_directed, logger=logger)
    tensor = parsing_utils.build_edge_colored_matrices_from_supra_adjacency_matrix(supra, layers)
    agg_mat = parsing_utils.get_aggregate_network(tensor, return_mat=True)
    centrality_vector = np.array(agg_mat.sum(axis=0)).ravel()
    return centrality_vector

def get_multi_RW_centrality_edge_colored(node_tensor: list[sps.spmatrix], cval: float = 0.15):
    """
    Computes multilayer RW centrality over edge-colored supra-adjacency without interlayer links.

    Parameters
    ----------
    node_tensor : list of scipy.sparse matrices
        Adjacency matrices per layer.
    cval : float, optional
        Value used in leading eigenvalue approximation (default: 0.15).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ["phy nodes", "vers"] where "vers" is the normalized score.
    """
    nodes = node_tensor[0].shape[0]
    layers = len(node_tensor)
    #create a supra adjacency matrix without interlayer connections
    supra = parsing_utils.build_supra_adjacency_matrix_from_edge_colored_matrices(
                                                                    intra_networks=node_tensor,
                                                                    layer_coupling_matrix=sps.csr_matrix((layers,layers)),
                                                                    num_nodes=nodes)
    #compute the degree for each replica node
    supra_strength = supra.sum(axis=1).flatten()
    #take the inverse to normalize the probabilities
    supra_strength[0,np.array(supra_strength>0)[0]] = 1. / supra_strength[0,np.array(supra_strength>0)[0]]
    #create a diagonal matrix to be able to multiply such a vector in a matrix multiplication fashion
    supra_strength = sps.diags(np.array(supra_strength)[0])
    #create super transition matrix
    supra_transition = supra_strength.dot(supra)
    #check witch replica nodes have degree > 0
    nonzero_idx = np.where(np.logical_not(supra_transition.sum(axis=0)==0))[1]
    #remove the corresponding zero rows and columns from the matrix
    supra_transition = supra_transition[nonzero_idx]
    supra_transition = supra_transition[:,nonzero_idx]
    #compute the leading eigenvector with the approximation methos
    eig,pr_v = approx_utils.leading_eigenv_approx(supra_transition.T, max_iter=10000, tol=1e-8, cval=0.15)
    #aggregate by summing together probabilities corresponding to the same physical node to have the final result
    res_df = pd.DataFrame({"phy nodes": nonzero_idx-((nonzero_idx//nodes)*nodes), "vers": pr_v/max(pr_v)})

    return res_df.groupby("phy nodes").aggregate(sum).reset_index()

def get_multi_Kcore_centrality(supra: sps.spmatrix, layers: int, nodes: int):
    """
    Computes multilayer k-core centrality as the minimum core index across all layers.

    Parameters
    ----------
    supra : scipy.sparse matrix
        Supra-adjacency matrix.
    layers : int
        Number of layers.
    nodes : int
        Number of physical nodes.

    Returns
    -------
    np.ndarray
        Minimum k-core index per node across all layers.
    """
    #calculate centrality in each layer separately and then get the max per node
    kcore_table = np.zeros([nodes,layers])
    nodes_tensor = parsing_utils.build_edge_colored_matrices_from_supra_adjacency_matrix(supra, layers)

    for l in range(layers):
        g_tmp = gt.Graph(directed=False)
        g_tmp.add_edge_list(np.transpose(nodes_tensor[l].nonzero()))
        kcore_table[:,l] = gt.topology.kcore_decomposition(g_tmp).get_array()

    centrality_vector = np.min(kcore_table, axis=1)
    return centrality_vector
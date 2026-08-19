import scipy.sparse as sp
import numpy as np
import logging
import typing

def compute_average_global_clustering_coefficient(adj: sp.csr_matrix, n:int, l: int, logger: typing.Optional[logging.Logger] = None) -> float:
    """
    Compute the average global clustering coefficient for a multilayer network.
    This implementation matches the R version from the reference, using the formula:
        C = tr(A^2*A) / (max(A) * tr(A*F*A))

    Where F is the matrix with 1s everywhere except the diagonal.
    Args:
        adj: scipy.sparse.csr_matrix - the adjacency matrix of the multilayer network (shape: (n*l, n*l))
        n: int - number of nodes per layer
        l: int - number of layers
        logger: Optional[logging.Logger] - logger for debugging (default: None)

    Returns:
        float - the average global clustering coefficient

    Note:
    F = J-I where J = 1 1^T and I is the identity matrix.
    tr(A F A) = tr(A J A) - tr(A^2)
    which means that we compute one at a time and find semplifications
    tr(A J A) = 1^T (A^T A) 1 = v^T*u where u = A 1 and v = A^T 1
    tr(A^2) = sum of squares of all elements in A = A A^T

    Reference:
        De Domenico et al. (2013) "Mathematical Formulation of Multilayer Networks", Physical review X, 3(4), p.041022.
    """

    NL = n*l
    # compute numerator
    num = float((adj.dot(adj).dot(adj)).diagonal().sum())
    # compute denominator
    AJA_trace = float((adj.sum(axis=0).dot(adj.sum(axis=1))).diagonal().sum()) # this is the same as 1^T (A^T A) 1
    AIA_trace = float((adj.dot(adj)).diagonal().sum()) # this is the same as sum of squares of all elements in A
    denom = AJA_trace - AIA_trace
    denom = denom * adj.data.max()

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Numerator (tr(A^2 A)): {num}")
        logger.debug(f"Denominator (max(A) * (tr(A J A) - tr(A^2))): {denom}")
    return num / denom if denom != 0 else 0.0


def _sparse_pmin(A: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
    """Element-wise minimum of two non-negative sparse matrices.

    The result is nonzero only where both inputs are nonzero (since for
    non-negative values min(0, x) = 0).  Any explicit zeros introduced by
    scipy are removed with eliminate_zeros().
    """
    result = A.minimum(B)
    result.eliminate_zeros()
    return result



def compute_average_global_overlap(
    adj: sp.csr_matrix,
    n: int,
    l: int,
    weighted: bool = False,
    logger: typing.Optional[logging.Logger] = None,
) -> float:
    """
    Compute the average global edge overlap for a multilayer network.

    For each edge (i,j), the overlap is the minimum weight across all layers
    (or 1 if the edge exists in all layers, for the unweighted case).  The
    scalar result is:

        AvGlobOverl = L * sum(O) / NormTotal

    where O_{ij} = min_α A^α_{ij} and NormTotal = Σ_α sum(A^α).
    For undirected networks (symmetric supra-adjacency) the result is halved.

    Args:
        adj: scipy.sparse.csr_matrix - supra-adjacency matrix (n*l, n*l)
        n: int - number of nodes per layer
        l: int - number of layers
        weighted: bool - if True use actual edge weights; if False binarize first (default: False)
        logger: Optional[logging.Logger] - logger for debugging (default: None)

    Returns:
        float - average global edge overlap

    Reference:
        De Domenico et al. (2015) "Structural reducibility of multilayer networks",
        Nature Communications, 6, 6864.
    """
    if l < 2:
        raise ValueError("At least two layers are required.")

    layers = []
    for alpha in range(l):
        A = adj[alpha * n:(alpha + 1) * n, alpha * n:(alpha + 1) * n].astype(float)
        if not weighted:
            A = (A > 0).astype(float)
        layers.append(A)

    # Running element-wise minimum: nonzero only where ALL layers share the edge
    O = _sparse_pmin(layers[0], layers[1])
    norm_total = float(layers[0].sum())

    if l > 2:
        for alpha in range(1, l):          # mirrors R: for (l in 2:Layers)
            O = _sparse_pmin(O, layers[alpha])
            norm_total += float(layers[alpha].sum())

    sum_O = float(O.sum())
    avg = l * sum_O / norm_total if norm_total != 0 else 0.0

    # R checks `sum(A - A^T) == 0` before halving.  However sum(A) == sum(A^T)
    # for any matrix (transposing preserves the total), so R's condition is
    # always True and R always halves.  We replicate that behaviour here.
    avg /= 2

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"sum(O): {sum_O}, NormTotal: {norm_total}, L: {l}")

    return avg


def compute_average_global_overlap_matrix(
    adj: sp.csr_matrix,
    n: int,
    l: int,
    weighted: bool = False,
    logger: typing.Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute the pairwise average global edge overlap matrix (L×L).

    For each pair of layers (α, β):

        M[α, β] = 2 * sum(pmin(A^α, A^β)) / (sum(A^α) + sum(A^β))

    This is a Dice-like similarity coefficient in [0, 1].  The diagonal is 1.
    Args:
        adj: scipy.sparse.csr_matrix - supra-adjacency matrix (n*l, n*l)
        n: int - number of nodes per layer
        l: int - number of layers
        weighted: bool - if True use actual edge weights; if False binarize first (default: False)
        logger: Optional[logging.Logger] - logger for debugging (default: None)

    Returns:
        numpy.ndarray of shape (l, l) - symmetric overlap matrix with 1s on diagonal

    Reference:
        De Domenico et al. (2015) "Structural reducibility of multilayer networks",
        Nature Communications, 6, 6864.
    """
    if l < 2:
        raise ValueError("At least two layers are required.")

    layers = []
    layer_sums = []
    for alpha in range(l):
        A = adj[alpha * n:(alpha + 1) * n, alpha * n:(alpha + 1) * n].astype(float)
        if not weighted:
            A = (A > 0).astype(float)
        layers.append(A)
        layer_sums.append(float(A.sum()))

    M = np.eye(l, dtype=float)
    for l1 in range(l - 1):
        for l2 in range(l1 + 1, l):
            O = _sparse_pmin(layers[l1], layers[l2])
            denom = layer_sums[l1] + layer_sums[l2]
            val = 2.0 * float(O.sum()) / denom if denom != 0 else 0.0
            M[l1, l2] = val
            M[l2, l1] = val

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Overlap matrix:\n{M}")

    return M


def compute_average_global_node_overlap_matrix(
    adj: sp.csr_matrix,
    n: int,
    l: int,
    logger: typing.Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute the pairwise average global node overlap matrix (L×L).

    A node is considered active in a layer if it has non-zero in-degree OR
    out-degree in that layer.  For each pair of layers (α, β):

        M[α, β] = |active_nodes(α) ∩ active_nodes(β)| / N

    The diagonal is 1 (all active nodes overlap with themselves).

    Args:
        adj: scipy.sparse.csr_matrix - supra-adjacency matrix (n*l, n*l)
        n: int - number of nodes per layer
        l: int - number of layers
        logger: Optional[logging.Logger] - logger for debugging (default: None)

    Returns:
        numpy.ndarray of shape (l, l) - symmetric overlap matrix with 1s on diagonal

    Reference:
        De Domenico et al. (2015) "Structural reducibility of multilayer networks",
        Nature Communications, 6, 6864.
    """
    if l < 2:
        raise ValueError("At least two layers are required.")

    active = []
    for alpha in range(l):
        A = adj[alpha * n:(alpha + 1) * n, alpha * n:(alpha + 1) * n]
        out_deg = np.asarray(A.sum(axis=1)).ravel()   # row sums
        in_deg  = np.asarray(A.sum(axis=0)).ravel()   # col sums
        active.append(set(np.where((out_deg > 0) | (in_deg > 0))[0]))

    M = np.eye(l, dtype=float)
    for l1 in range(l - 1):
        for l2 in range(l1 + 1, l):
            val = len(active[l1] & active[l2]) / n
            M[l1, l2] = val
            M[l2, l1] = val

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Node overlap matrix:\n{M}")

    return M

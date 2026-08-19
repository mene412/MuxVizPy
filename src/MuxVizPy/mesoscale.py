import logging
import typing
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from tqdm import tqdm
import graph_tool as gt

def get_mod(
    g_multi,
    n_iter,
    return_state: bool = False
) -> list:
    """
    Fit a layered SBM repeatedly and collect module counts and modularity scores.

    Parameters
    ----------
    g_multi : graph_tool.Graph
        Multilayer graph with an edge property map ``weight`` used as the
        layer indicator (``ec``).
    n_iter : int
        Number of independent SBM fits to run.
    return_state : bool, optional
        If ``True``, the final ``BlockState`` object is appended as a third
        element of the returned list. Default is ``False``.

    Returns
    -------
    list
        ``[modules_list, modularity_list]`` where each element is a list of
        length ``n_iter``.  If ``return_state=True``, a third element
        containing the last fitted ``BlockState`` is appended.
    """
    from graph_tool import inference as gti  # <- Moved here

    modules_list = []
    modularity_list = []

    for It_Com in tqdm(range(n_iter)):
        state_multi = gti.minimize_blockmodel_dl(g_multi,
            state_args=dict(
                base_type=gti.LayeredBlockState,
                state_args=dict(ec=g_multi.ep.weight, layers=True)
            )
        )

        modules_list.append(state_multi.get_nonempty_B())
        modularity_list.append(gti.modularity(g_multi, state_multi.get_blocks()))

    if return_state:
        return [modules_list, modularity_list, state_multi] 
    else:
        return [modules_list, modularity_list]

def inter_layer_assortativity(
    g_list: list[gt.Graph], layers: int
) -> dict[str, np.ndarray]:
    """
    Compute inter-layer degree assortativity (Pearson and Spearman) for all layer pairs.

    Parameters
    ----------
    g_list : list of graph_tool.Graph
        One graph per layer of the multilayer network.  All graphs must share
        the same set of physical nodes.
    layers : int
        Number of layers (must equal ``len(g_list)``).

    Returns
    -------
    dict
        Dictionary with two keys:

        - ``"Pearson"``: ``(layers, layers)`` array of Pearson correlation
          coefficients between layer degree sequences.
        - ``"Spearman"``: ``(layers, layers)`` array of Spearman correlation
          coefficients between layer degree sequences.
    """
    degrees = [g_list[i].get_total_degrees(np.arange(g_list[i].num_vertices())) for i in range(layers)]
    pearson_ass = np.zeros([layers, layers])
    spearman_ass = np.zeros([layers, layers])
    
    for l1 in range(layers):
        for l2 in range(layers):
            pearson_ass[l1,l2] = sp.stats.pearsonr(degrees[l1], degrees[l2]).statistic
            spearman_ass[l1,l2] = sp.stats.spearmanr(degrees[l1], degrees[l2]).statistic
    return {"Pearson": pearson_ass, "Spearman": spearman_ass}


def compute_local_clustering_coefficient(
    adj: sp.sparse.spmatrix,
    n: int,
    l: int,
    logger: typing.Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute the multilayer local clustering coefficient for each physical node.

    Implements the formulation from De Domenico et al. (2013, PhysRevX):
    for each physical node *p*, the coefficient is the ratio of closed
    walks of length 3 to possible triangles, summed over all layer pairs.

    Parameters
    ----------
    adj : scipy.sparse matrix, shape (N·L, N·L)
        Binary supra-adjacency matrix with supra-node indexing
        ``row = layer * n + node``.
    n : int
        Number of physical nodes.
    l : int
        Number of layers.
    logger : optional
        Unused; kept for API consistency with other descriptor functions.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Local clustering coefficient in ``[0, 1]`` for each physical node.
        Nodes with no possible triangles (isolated or degree-1 nodes) are
        assigned 0.

    References
    ----------
    De Domenico, M., et al. (2013). Mathematical formulation of multilayer
    networks. *Physical Review X*, 3(4), 041022.
    https://doi.org/10.1103/PhysRevX.3.041022
    """
    NL = n * l
    A = adj.astype(float)

    # Fold matrix P of shape (NL x n): P[i*n+p, p] = 1 for all layers i.
    # P.T @ M @ P sums all (l x l) blocks of M into a single (n x n) matrix.
    idx = np.arange(NL)
    P = csr_matrix((np.ones(NL), (idx, idx % n)), shape=(NL, n))

    A2 = A @ A
    A3 = A2 @ A

    # Numerator: block-sum of A³
    B_num = (P.T @ A3 @ P).toarray()

    # Denominator: block-sum of A·(J − I)·A = A·J·A − A²
    # A·J·A[i,j] = row_sums[i] * col_sums[j], so after block-summing:
    #   diag(B_den)[p] = agg_row[p] * agg_col[p] − diag(B_A2)[p]
    B_A2 = (P.T @ A2 @ P).toarray()
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    col_sums = np.asarray(A.sum(axis=0)).ravel()
    agg_row = np.asarray(P.T @ row_sums).ravel()
    agg_col = np.asarray(col_sums @ P).ravel()

    diag_num = np.diag(B_num)
    diag_den = agg_row * agg_col - np.diag(B_A2)

    with np.errstate(divide="ignore", invalid="ignore"):
        clus = np.where(diag_den > 0, diag_num / diag_den, 0.0)

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Local clustering: min={clus.min():.4f}, max={clus.max():.4f}, mean={clus.mean():.4f}")

    if np.any((clus > 1 + 1e-10) | (clus < -1e-10)):
        raise ValueError(
            "compute_local_clustering_coefficient: impossible values outside [0, 1]."
        )

    return clus

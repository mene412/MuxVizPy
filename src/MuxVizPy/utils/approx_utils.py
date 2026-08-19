import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.sparse import find, identity, coo_matrix
import graph_tool as gt
#from graph_tool import centrality, inference
import graph_tool.correlations as gtcorr
import graph_tool.clustering as gtclust

import logging
from typing import Optional, Union, List


def get_largest_magnitude_eigenvalue(adj: sps.spmatrix, logger: Optional[logging.Logger] = None) -> tuple[float, np.ndarray]:
    """
    Compute the largest eigenvalue and corresponding eigenvector of a sparse matrix.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Adjacency matrix.
    logger : logging.Logger, optional
        Logger for debug information.

    Returns
    -------
    tuple
        (largest eigenvalue, corresponding eigenvector)
    """
    eigvals, eigvecs = sps.linalg.eigs(adj, k=1, which="LM", return_eigenvectors=True)
    real_mask = np.abs(np.imag(eigvals)) < 1e-6

    if not np.any(real_mask):
        if logger:
            logger.warning("Warning! Complex numbers in the leading eigenvalue.")
        lam = float(np.real(eigvals[0]))
        vec = np.real(eigvecs[:, 0])
    else:
        lam = float(np.real(eigvals[0]))
        vec = np.real(eigvecs[:, 0])

    if not np.allclose(np.imag(vec), 0):
        if logger:
            logger.warning("Warning! Complex numbers in the leading eigenvector.")

    vec[(vec > -1e-12) & (vec < 1e-12)] = 0.0

    non_zero_vec = vec[vec != 0]
    if len(non_zero_vec) > 0 and np.all(non_zero_vec < 0):
        vec = -vec

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Largest eigenvalue: {lam}, eigenvector min: {vec.min()}, max: {vec.max()}, mean: {vec.mean()}")

    return lam, vec

def get_largest_real_eigenvalue(adj: sps.spmatrix, logger: Optional[logging.Logger] = None) -> tuple[float, np.ndarray]:
    """
    Compute the largest eigenvalue and corresponding eigenvector of a sparse matrix.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Adjacency matrix.
    logger : logging.Logger, optional
        Logger for debug information.

    Returns
    -------
    tuple
        (largest eigenvalue, corresponding eigenvector)
    """
    eigvals, eigvecs = sps.linalg.eigs(adj, k=1, which="LR", return_eigenvectors=True)
    real_mask = np.abs(np.imag(eigvals)) < 1e-6

    if not np.any(real_mask):
        if logger:
            logger.warning("Warning! Complex numbers in the leading eigenvalue.")
        lam = float(np.real(eigvals[0]))
        vec = np.real(eigvecs[:, 0])
    else:
        lam = float(np.real(eigvals[0]))
        vec = np.real(eigvecs[:, 0])

    if not np.allclose(np.imag(vec), 0):
        if logger:
            logger.warning("Warning! Complex numbers in the leading eigenvector.")

    vec[(vec > -1e-12) & (vec < 1e-12)] = 0.0

    non_zero_vec = vec[vec != 0]
    if len(non_zero_vec) > 0 and np.all(non_zero_vec < 0):
        vec = -vec

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Largest eigenvalue: {lam}, eigenvector min: {vec.min()}, max: {vec.max()}, mean: {vec.mean()}")

    return lam, vec


def approximate_largest_eigenvalue(
    adj: sps.spmatrix, maxiter: int = 1000, tol: float = 1e-6,
    cval: Optional[float] = None, alpha: float = 0.85,
) -> tuple[float, np.ndarray]:
    """
    Approximate the largest eigenvalue using the power method with PageRank-style shift.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Adjacency matrix.
    maxiter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    cval : float or None
        Constant shift value (default: (1 - alpha) / n).
    alpha : float
        Damping factor.

    Returns
    -------
    tuple
        (approximate eigenvalue, eigenvector)
    """
    n = adj.shape[0]
    x = np.random.rand(n)
    if cval is None:
        cval = (1 - alpha) / n

    for counter in range(maxiter):
        y = alpha * (adj @ x) + cval * np.ones_like(x) * x.sum()
        x_new = y / y.sum()
        delta = np.linalg.norm(x_new - x)
        if delta < tol:
            x = x_new
            break
        x = x_new

    num = alpha * (x @ (adj @ x)) + cval * (x.sum() ** 2)
    den = x @ x
    lam = num / den
    return lam, x


def leading_eigenv_approx(
    A: sps.spmatrix, max_iter: int = 1000, tol: float = 1e-6, cval: Optional[float] = None, alpha: float = 0.85
) -> List[Union[float, np.ndarray]]:
    """
    Approximates the leading eigenvalue and eigenvector of a modified matrix C = A + B,
    where A is a sparse matrix and B is a constant matrix with entries cval.

    Parameters
    ----------
    A : scipy.sparse matrix
        Sparse matrix for which the leading eigenpair is computed. Automatically scaled by 0.85.
    max_iter : int, optional
        Maximum number of power iterations. Default is 1000.
    tol : float, optional
        Tolerance for convergence (Euclidean distance between iterates). Default is 1e-6.
    cval : float or None, optional
        Constant value to use for dense matrix B. If None, uses (1 - 0.85) / n.
    alpha : float, optional
        Damping factor for the power iteration. Default is 0.85.

    Returns
    -------
    eigenvalue : float
        Approximate dominant eigenvalue of A + B.
    eigenvector : np.ndarray
        Corresponding normalized eigenvector.
    """
    # Initialize the starting vector x as a random vector
    n = A.shape[0]
    x = np.random.rand(n)
    if cval==None:
        cval = (1 - alpha) / n

    # Iterate until convergence
    for i in range(max_iter):
        # Compute y = (alpha * A + cval) x
        y = alpha * (A @ x) + cval * np.ones(n) * x.sum()

        # Normalize y to obtain the new vector x
        # x_new = y / np.linalg.norm(y) # L2 norm
        x_new = y / y.sum() # L1 norm

        # Compute the change in x and check for convergence
        delta_x = np.linalg.norm(x_new - x)
        if delta_x < tol:
            #print("Reached convergence")
            x = x_new
            break

        # Update x for the next iteration
        x = x_new

    # Compute the dominant eigenvalue and eigenvector
    # eigenvalue = x.T@A@x + (np.ones(x.shape)*x.sum()*cval).T @ x
    num = alpha * (x @ (A @ x)) + cval * (x.sum() ** 2)
    den = x @ x
    eigenvalue = num / den # Rayleigh quotient
    eigenvector = x

    return [eigenvalue, eigenvector]

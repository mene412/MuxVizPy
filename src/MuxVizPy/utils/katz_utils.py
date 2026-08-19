import warnings
import logging
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from typing import Optional

_VALID_SOLVERS = ("direct", "neumann", "gmres", "bicgstab")
_KRYLOV_SOLVERS = ("gmres", "bicgstab")


def _katz_neumann(
    adj: sps.spmatrix,
    alpha: float,
    b: np.ndarray,
    *,
    maxiter: int = 1000,
    tol: float = 1e-6,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Solve (I - alpha*A) x = b by Richardson / Neumann-series iteration.

    Uses the fixed-point form

        x_{k+1} = alpha * A @ x_k + b,

    which unrolls to the Neumann series  sum_{k>=0} (alpha*A)^k b  and
    converges to (I - alpha*A)^{-1} b whenever rho(alpha*A) < 1. For Katz
    centrality that condition is guaranteed by construction because alpha
    is chosen strictly below 1/rho(A).

    This branch never materializes the operator (I - alpha*A); it only
    performs sparse matvecs with ``adj``. The trade-off is that convergence
    can be slow when alpha is close to 1/rho(A).

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix A.
    alpha : float
        Katz attenuation factor; must satisfy alpha * rho(A) < 1 for
        convergence.
    b : np.ndarray
        Right-hand side vector, shape (NL,).
    maxiter : int, default 1000
        Maximum number of fixed-point iterations.
    tol : float, default 1e-6
        Convergence tolerance on ``||x_{k+1} - x_k||_2``.
    logger : logging.Logger, optional
        If provided, logs convergence at DEBUG level.

    Returns
    -------
    np.ndarray
        Approximate solution x, shape (NL,). Issues a ``UserWarning`` if
        the iteration did not converge within ``maxiter``.

    Notes
    -----
    The starting vector is drawn from ``np.random.randn``; seed the global
    NumPy RNG before calling if you need reproducibility.
    """
    NL = b.shape[0]
    x = np.random.randn(NL)
    delta_x = float("inf")
    for i in range(maxiter):
        x_new = alpha * adj @ x + b
        # diff_norm = np.linalg.norm(x_new - x)
        # x_norm = np.linalg.norm(x_new)
        # if x_norm > 0:
        #     delta_x = diff_norm / x_norm
        # else:
        #     delta_x = diff_norm

        max_change = np.max(np.abs(x_new - x))
        max_val = np.max(np.abs(x_new))
        delta_x = max_change / max_val if max_val > 0 else max_change
        x = x_new

        if delta_x < tol:
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Katz Neumann iteration converged after %d iterations, delta=%.3e",
                    i + 1, delta_x,
                )
            return x
    warnings.warn(
        f"Katz Neumann iteration did not converge within {maxiter} iterations "
        f"(final delta={delta_x:.3e}). Result may be inaccurate. "
        "Consider solver='gmres' or solver='bicgstab'.",
        UserWarning,
        stacklevel=3,
    )
    return x


def _katz_krylov(
    Aop: sps.csc_matrix,
    b: np.ndarray,
    *,
    method: str,
    maxiter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Solve Aop x = b with a preconditioned Krylov-subspace method.

    Builds an incomplete-LU preconditioner M ~ Aop via ``scipy.sparse.linalg.spilu``
    and hands the preconditioned system to GMRES or BiCGSTAB. Unlike the Neumann
    branch, this path *does* need Aop explicitly (both for the matvec and for
    the ILU factorization).

    Parameters
    ----------
    Aop : scipy.sparse.csc_matrix
        The Katz operator (I - alpha * A) in CSC format.
    b : np.ndarray
        Right-hand side vector, shape (NL,).
    method : {"gmres", "bicgstab"}
        Krylov solver to dispatch to.
    maxiter : int, default 1000
        Maximum iterations forwarded to the scipy solver.
    tol : float, default 1e-6
        Relative tolerance forwarded as ``rtol`` to the scipy solver.

    Returns
    -------
    np.ndarray
        Approximate solution x, shape (NL,). Issues a ``UserWarning`` if
        the solver reports non-convergence or a breakdown.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported Krylov methods.
    """
    if method not in _KRYLOV_SOLVERS:
        raise ValueError(
            f"_katz_krylov: method must be one of {_KRYLOV_SOLVERS}, got {method!r}"
        )
    ilu = spla.spilu(Aop)
    M = spla.LinearOperator(Aop.shape, ilu.solve)
    solver_fn = spla.gmres if method == "gmres" else spla.bicgstab
    x, info = solver_fn(Aop, b, M=M, rtol=tol, maxiter=maxiter)
    if info != 0:
        if info > 0:
            detail = f"did not converge within {maxiter} iterations"
        else:
            detail = f"encountered a breakdown (info={info})"
        warnings.warn(
            f"Katz {method} solver {detail}. Result may be inaccurate.",
            UserWarning,
            stacklevel=3,
        )
    return x

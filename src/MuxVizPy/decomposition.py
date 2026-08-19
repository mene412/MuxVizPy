from __future__ import annotations
import logging
import warnings
from typing import Literal

import numpy as np
import torch

from MuxVizPy.utils.decomposition_utils import get_backend, CPBackend

# ---- helper functions -----

def _prepare_tensor(
    tensor: "torch.Tensor",
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Extract coordinates and values from PyTorch sparse tensor.

    Args:
        tensor: PyTorch sparse COO tensor.

    Returns:
        Tuple of (coords, values, shape) where:
        - coords: int64 array of shape (n_modes, nnz)
        - values: float64 array of shape (nnz,)
        - shape: tuple of dimension sizes
    """
    if not tensor.is_sparse:
        raise ValueError("Input tensor must be sparse")

    tensor = tensor.coalesce()
    coords = tensor.indices().numpy().astype(np.int64)
    values = tensor.values().numpy().astype(np.float64)
    shape = tuple(tensor.shape)

    return coords, values, shape

def _initialize_factors(
        shape: tuple[int, ...],
        rank: int,
        method: str,
        random_state: int | None,
        backend: CPBackend,
        coords: np.ndarray | None = None,
        values: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    initiaize factor matrices.

    Args:
        shape: Tensor dimensions.
        rank: number of components.
        method: 'random' or 'hosvd'
        random_state: random seed
        backend: computation backend
        coords: tensor coordinates (for HOSVD)
        values: tensor values (for HOSVD)

    Returns:
        list[np.ndarray]: List of initialized factor matrices
    """
    shapes = [(dim, rank) for dim in shape]

    if method == "random":
        return backend.random_init(shapes, random_state)
    
    elif method == "hosvd":
        # truncated svd init for sparse tensors via unfolding.
        factors = []

        for mode, dim, in enumerate(shape):
            # create sparse unfolding matrix
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import svds

            # build csr matrix for mode unfolding
            row_indices = coords[mode]
            col_indices = _compute_multi_index(coords, shape, exclude_mode=mode)
            n_cols = np.prod([s for i, s in enumerate(shape) if i!=mode])
            unfold = csr_matrix(
                (values, (row_indices, col_indices)),
                shape = (dim, int(n_cols)),
            )

            # truncated svd
            k = min(rank, min(unfold.shape)-1)
            if k>0:
                try:
                    u, _, _ = svds(unfold, k=k)
                    # pad if needed
                    if u.shape[1] < rank:
                        rng = np.random.default_rng(random_state)
                        extra = rng.standard_normal((dim, rank-u.shape[1]))
                        u = np.hstack([u, extra])
                    factors.append(u[:, :rank])
                except Exception as e:
                    # fallback to random if SVD fails
                    warnings.warn(
                        f"SVD failed, fall back to random init"
                        f"Reason: {e}"
                    )
                    rng = np.random.default_rng(random_state)
                    factors.append(rng.standard_normal((dim,rank)))
            else:
                rng = np.random.default_rng(random_state)
                factors.append(rng.standard_normal((dim, rank)))
        return factors
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
def _compute_multi_index(
        coords: np.ndarray,
        shape: tuple[int, ...],
        exclude_mode: int,
) -> np.ndarray:
    """
    compute flattened column indices for mode unfolding.
    """
    other_modes = [i for i in range(len(shape)) if i!=exclude_mode]
    other_dims = [shape[i] for i in other_modes]

    # compute strides
    strides = np.ones(len(other_dims), dtype=np.int64)
    for i in range(len(other_dims)-2, -1, -1):
        strides[i] = strides[i+1] * other_dims[i+1]

    # compute multiindex
    result = np.zeros(coords.shape[1], dtype=np.int64)
    for idx, mode in enumerate(other_modes):
        result += coords[mode]*strides[idx]
    return result

def _compute_recon_error(
        mode_indices: list[np.ndarray],
        values: np.ndarray,
        factors: list[np.ndarray],
        weights: np.ndarray,
        backend: CPBackend,
) -> float:
    """
    compute reconstruction error on backend.

    Args:
        mode_indices: list of coordinate arrays for each mode (n_modes x nnz)
        values: array of non-zero values (nnz,)
        factors: list of factor matrices (each of shape (dim, rank))
        weights: array of component weights (rank,)

    Returns:
        reconstruction error (float)
    """
    n_values = int(values.shape[0] if hasattr(values, "shape") else len(values))
    if n_values == 0:
        return 0.0
    
    # compute reconstructed values at non-zero coordinates
    n_modes = len(factors)
    factor_vals = [factors[m][mode_indices[m],:] for m in range(n_modes)] # gather factor values: (nnz, rank) for each mode
    product = factor_vals[0].copy()
    for fv in factor_vals[1:]: # element wise product across modes
        product *= fv
    recon = product @ weights # apply weights: (nnz,) and perform reconstruction

    # compute error
    diff  = values - recon
    diff_norm = backend.compute_norm(diff)
    values_norm = backend.compute_norm(values)
    return diff_norm / values_norm if values_norm > 0 else 0.0

# ---- core functions ----

def _sparse_mttkrp_no_weights(
        mode: int, 
        mode_indices: list[np.ndarray],
        values: np.ndarray,
        factors: list[np.ndarray],
        shape: tuple[int, ...],
        backend: CPBackend,
) -> np.ndarray:
    """
    compute sparse mttkrp without separate weights (weights in factors)
    for mode m, computes:
        MTTKRP[i,r] = Σ_{j,k,l: X[i,j,k,l]≠0} X[i,j,k,l] * B[j,r] * C[k,r] * D[l,r]

    Args:
        mode: mode to compute MTTKRP for (0,1,2,3)
        mode_indices: list of coordinate arrays for each mode (n_modes x nnz)
        values: array of non-zero values (nnz,)
        factors: list of factor matrices (each of shape (dim, rank))
        shape: shape of the original tensor
        backend: computation backend
    Returns:
        MTTKRP result for the specified mode (shape: (dim, rank))
    """
    rank = factors[0].shape[1]
    dim = shape[mode]

    # init output on backend
    result = backend.zeros((dim, rank), dtype=np.float64)

    # compute khatri-rao contribution using backend
    kr_contrib = backend.multiply_gather(values, mode_indices, factors, mode)

    # scatter-add contributions to result
    target_indices = mode_indices[mode]
    backend.scatter_add(result, target_indices, kr_contrib)
    return result

def _compute_combined_gram_no_weights(
        factors: list[np.ndarray],
        exclude_mode: int,
        backend: CPBackend,
) -> np.ndarray:
    """
    compute hadamard product of gram matrices without separate weights
    
    the combined gram matrix is:
    G = = (A'A) * (B'B) * (C'C) * (D'D)  (excluding the target mode)

    Args:
        factors: list of factor matrices (each of shape (dim, rank))
        exclude_mode: mode to exclude from the product
        backend: computation backend
    Returns:
        combined gram matrix (shape: (rank, rank))
    """
    other_modes = [m for m in range(len(factors)) if m!=exclude_mode]

    # compute individual gram matrices
    grams = []
    for m in other_modes:
        g = backend.gram_matrix(factors[m])
        grams.append(g)

    # compute hadamard product
    return backend.hadamard_gram(grams)

def sparse_cp_decomposition(
        tensor: torch.Tensor,
        rank: int,
        init: Literal["random", "hosvd"] = "random",
        max_iter: int = 100,
        tol: float = 1e-6,
        #non_negative: bool = False,
        regularization: float = 1e-12,
        random_state: int | None = None,
        logger: logging.Logger | None = None,
        backend: Literal["numpy", "rapids", "auto"] = "auto",
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Perform sparse CP/PARAFAC decomposition on a multi-layer network tensor.

    This implementation operates directly on non-zero entries, achieving O(nnz x rank) complexity per iteration instead of O(N^2 x L^2).

    CP decomposition approximates the tensor X as:
        X[i,j,k,l] ≈ Σ_r λ_r · A[i,r] · B[j,r] · C[k,r] · D[l,r]

    For multi-layer networks, the 4 modes correspond to:
        - Mode 0: source nodes
        - Mode 1: source layers
        - Mode 2: target nodes
        - Mode 3: target layers

    Args:
        tensor (torch.Tensor): Pytorch sparse COO tensor with shape (N, L, N, L). 
        rank (int): The number of components for the CP decomposition.
        init (str): Initialization method for factor matrices ("random" or "hosvd").
            - random: Initialize factor matrices with random values.
            - hosvd: Use Higher-Order SVD for initialization (slower but could converge faster).
        max_iter (int): Maximum number of iterations for the ALS algorithm.
        tol (float): Convergence tolerance for the ALS algorithm. Stops when the relative change in reconstruction error is below this threshold.
        DEPRECATED non_negative (bool): If True, enforce non-negativity constraints on factor matrices using projected ALS (values are clipped to >= 0 after each update).
        regularization (float): Tikhonov regularization for least-squares solvers. Adds a small ridge penalty to improve numerical stability, especially for sparse data.
        random_state (int | None): Seed for random initialization.
        logger (logging.Logger | None): Optional logger for debugging and progress tracking.
        backend (str): Computational backend to use ("numpy", "rapids", or "auto").

    Returns:
        A (torch.Tensor): Factor matrix for source nodes (shape: N x rank).
        B (torch.Tensor): Factor matrix for source layers (shape: L x rank).
        C (torch.Tensor): Factor matrix for target nodes (shape: N x rank).
        D (torch.Tensor): Factor matrix for target layers (shape: L x rank).
        lambdas (torch.Tensor): Weights of the components (shape: rank).

    Example:
        >>> import torch
        >>> from MuxVizPy.decomposition import sparse_cp_decomposition
        >>> # Create a random sparse tensor with shape (N, L, N, L)
        >>> N, L = 1000, 10
        >>> indices = torch.randint(0, N, (4, 10000))  # 10k non-zero entries
        >>> values = torch.rand(10000)
        >>> tensor = torch.sparse_coo_tensor(indices, values, size=(N, L, N, L))
        >>> rank = 5
        >>> A, B, C, D, lambdas = sparse_cp_decomposition(tensor, rank)
    """
    # Initialize the backend
    be = get_backend(backend)

    # prepare tensor data
    coords, values, shape = _prepare_tensor(tensor)
    n_modes = len(shape)
    nnz = len(values)

    # transfer data to backend device
    values = be.to_backend(values)
    mode_indices_backend = [be.to_backend(coords[m]) for m in range(n_modes)]

    # Initialize factors
    factors = _initialize_factors(
        shape, rank, init, random_state, be, coords, values
    )
    weights = np.ones(rank)

    # use pre-transferred mode indices already on backend
    mode_indices = mode_indices_backend
    
    # start loop for ALS iterations
    prev_error = np.inf
    prev_time = 0.0
    convergence_history: dict[str, list[float]] = {"reconstruction_error": [], "elapsed_time": [], "elapsed_time_per_mode": []}
    iIter = 0
    notConverged = True
    while iIter < max_iter and notConverged:
        # at start, absorb weights into first factor so all factors carry the scale during updates
        factors[0] = factors[0] * weights
        weights = np.ones(rank)

        time = be.get_time()
        # update each factor in turn
        elapsed_time_for_modes = []
        for mode in range(n_modes):
            mode_time = be.get_time()
            # compute mttkrp for current mode
            mttkrp = _sparse_mttkrp_no_weights(
                mode, mode_indices, values, factors, shape, be
            )

            # compute combined gram matrix (hadamard of other factors' gramians)
            gram_matrix = _compute_combined_gram_no_weights(factors, mode, be)

            # solve least squares problem to update factor
            factors[mode] = be.solve_least_squares(gram_matrix, mttkrp, regularization)

            # enforce non-negativity if needed
            # TODO: Note that this can introduce stuck in local minima. Known issue for projected ALS for nncp
            # if non_negative:
            #     factors[mode] = be.maximum(factors[mode], 0.0)

            elapsed_time_for_modes.append(be.get_time() - mode_time)

        # normalize all factors and extract weights
        factors, weights = be.normalize_factors(factors, weights=None)

        # compute reconstruction error on backend
        error = _compute_recon_error(
            mode_indices, values, factors, weights, be
        )
        elapsed_time = be.get_time() - time

        convergence_history["reconstruction_error"].append(error)
        convergence_history["elapsed_time"].append(elapsed_time)
        convergence_history["elapsed_time_per_mode"].append(elapsed_time_for_modes)

        # check convergence (skip first iteration where prev_error is inf)
        if np.isfinite(prev_error):
            rel_change = abs(prev_error - error) / max(prev_error, 1e-10)
            if rel_change < tol:
                notConverged = False
        prev_error = error
        iIter += 1

    factors = [be.to_numpy(f) for f in factors]
    weights = be.to_numpy(weights)

    return (factors[0], factors[1], factors[2], factors[3]), weights, convergence_history
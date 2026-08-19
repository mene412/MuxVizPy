# ------ backend abstraction for CP decomposition ------
"""
Provides CPU (numpy/scipy) and optional GPU (RAPIDS/cuML) bacends for the sparse PARAFAC algorithm.
"""

from __future__ import annotations
import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import NDArray

_CUPY_INSTALL_MSG = """
RAPIDS/CuPy requries NVIDIA GPU with CUDA drivers support.
To install CuPy, choose the package matching your CUDA version:
  - CUDA 12.x: pip install cupy-cuda12x
  - CUDA 11.x: pip install cupy-cuda11x
  - Auto-detect: pip install cupy-wheel

Or with hornet extras:
  - uv sync --extra cuda      (CUDA 12.x)
  - uv sync --extra cuda11    (CUDA 11.x)

Verify CUDA is available: nvidia-smi
"""

class CPBackend(ABC):
    """
    Abstract backend base class for CP decomposition. Defines the interface for tensor operations and solvers.
    Each backend must implement the core operations needed for sparse ALS:
    - scatter_add: Accumulate valuest @ indices (for MTTKRP updates)
    - solve_least_squares: Solve regularized least-squares problems for factor updates
    - gram_matrix: Compute A^T @ A efficiently
    - hadamard_gram: Elementwise product of multiple Gram matrices
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend (e.g., 'numpy', 'rapids')"""
        pass

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: np.dtype | None = None) -> NDArray:
        """Create an array of zeros with the given shape and dtype."""
        pass

    @abstractmethod
    def scatter_add(self, target: NDArray, indices: NDArray, values: NDArray) -> None:
        """
        Accumulate values at the specified indices in the target array.
        Equivalent to: target[indices] += values, but must handle duplicate indices correctly (e.g., using np.add.at).

        Args:
            target (NDArray): The array to be updated in-place. (dim, rank)
            indices (NDArray): The indices where values should be added. (nvalues, )
            values (NDArray): The values to add at the specified indices. (nvalues, rank)
        """
        pass

    @abstractmethod
    def gram_matrix(self, factor: NDArray) -> NDArray:
        """
        Compute the Gram matrix A^T @ A efficiently.
        
        Args:
            factor (NDArray): Factor matrix (dim, rank)
        Returns:
            NDArray: Gram matrix (rank, rank)
        """
        pass

    @abstractmethod
    def hadamard_gram(self, gramians: list[NDArray]) -> NDArray:
        """
        Compute the elementwise product of multiple Gram matrices.
        
        Args:
            gramians (list[NDArray]): List of Gram matrices to be multiplied (each of shape (rank, rank))
        Returns:
            NDArray: The resulting Hadamard product (rank, rank)
        """
        pass

    @abstractmethod
    def solve_least_squares(self, grams: NDArray, rhs: NDArray, regularization: float) -> NDArray:
        """
        Solve the regularized least-squares problem for factor updates.
        Solves (G + λI) @ x = rhs, where G is the Hadamard product of Grams and λ is the regularization parameter.

        Args:
            grams (NDArray): The Hadamard product of Gram matrices (rank, rank)
            rhs (NDArray): The right-hand side vector (MTTKRP result) (dim, rank)
            regularization (float): The regularization parameter to ensure numerical stability
        Returns:
            NDArray: The solution vector (dim, rank)
        """
        pass

    @abstractmethod
    def normalize_factors(self, factors: list[NDArray], weights: NDArray) -> tuple[list[NDArray], NDArray]:
        """
        Normalize the factor matrices and adjust the component weights accordingly.
        Normalize each column of each factor to unit norm, accumulating the norms into a weight vector.

        Args:
            factors (list[NDArray]): List of factor matrices to be normalized (each of shape (dim, rank))
            weights (NDArray): The component weights to be updated based on the norms (shape: rank) 
        
        Return:
            tuple[list[NDArray], NDArray]: A tuple containing the list of normalized factor matrices and the updated weights.
        """
        pass

    @abstractmethod
    def random_init(self, shapes: list[tuple[int,int]], random_state: np.random.Generator | int | None = None) -> list[NDArray]:
        """
        Initialize factor matrices with random values.

        Args:
            shapes (list[tuple[int,int]]): List of shapes for each factor matrix (e.g., [(N, rank), (L, rank), ...])
            random_state (np.random.Generator | int | None): Random state for reproducibility. Can be a numpy Generator, an integer seed, or None for default randomness.
        
        Returns:
            list[NDArray]: A list of randomly initialized factor matrices corresponding to the provided shapes.
        """
        pass

    def to_backend(self, array: NDArray) -> NDArray:
        """
        Transfer array to backend device.
        Default returns the array unchanges (CPU backend)

        Args:
            array (NDArray): The array to be transferred to the backend device.
        Returns:
            NDArray: The array transferred to the backend device (e.g., GPU array for RAPIDS).
        """
        return array
    
    def to_numpy(self, arr: NDArray) -> np.ndarray:
        """Transfer array from backend device to NumPy.

        Default implementation returns the array unchanged (CPU backends).

        Args:
            arr: Array on this backend's device.

        Returns:
            NumPy array on CPU.
        """
        return np.asarray(arr)
    
    def get_time(self) -> float:
        """
        Get the current time for performance measurement. For GPU backends, this should synchronize and return accurate timing.

        Returns:
            float: Current time in seconds.
        """
        import time
        return time.time()
    
    @abstractmethod
    def multiply_gather(
            self, values: NDArray, indices: NDArray, factors: list[NDArray], exclude_mode: int
    ) -> NDArray:
        """
        Compute the Kathri-Rao product contribution for MTTKRP.
        For each non-zero entry, computes the product of factor values at the corresponding indices. (excluding one mode).

        Args:
            values (NDArray): Non-zero values of the tensor (shape: nvalues)
            indices (NDArray): Corresponding indices for each non-zero value (shape: nvalues x 4)
            factors (list[NDArray]): List of factor matrices (each of shape (dim, rank))
            exclude_mode (int): The mode to exclude from the product (0, 1, 2, or 3)

        Returns:
            NDArray: The resulting MTTKRP contribution for the specified mode (shape: nvalues x rank)
        """
        pass

    @abstractmethod
    def compute_norm(self, array: NDArray) -> float:
        """
        Compute the Frobenius norm of an array by the backend. This is used for convergence checks and error computation.
        Args:
            array (NDArray): The array for which to compute the norm.
        Returns:
            float: The computed Frobenius norm of the array.
        """
        pass

class NumPyBackend(CPBackend):
    """CPU backend implementation using NumPy and SciPy."""
    @property
    def name(self) -> str:
        return "numpy"

    def zeros(self, shape: tuple[int, ...], dtype: np.dtype | None = None) -> NDArray:
        if dtype is None:
            dtype = np.float64
        return np.zeros(shape, dtype=dtype)

    def scatter_add(self, target: NDArray, indices: NDArray, values: NDArray) -> None:
        np.add.at(target, indices, values)

    def gram_matrix(self, factor: NDArray) -> NDArray:
        return factor.T @ factor

    def hadamard_gram(self, gramians: list[NDArray]) -> NDArray:
        result = gramians[0].copy()
        for g in gramians[1:]:
            result *= g
        return result

    def solve_least_squares(self, grams: NDArray, rhs: NDArray, regularization: float) -> NDArray:
        """
        Solve via Cholesky factorization with regularization for numerical stability. Solves (G + λI) @ x = rhs.
        """
        from scipy import linalg
        rank = grams.shape[0]
        reg_grams = grams + regularization * np.eye(rank)

        try:
            # cholesky is more stable and faster for positive definite
            cho = linalg.cho_factor(reg_grams)
            return linalg.cho_solve(cho, rhs.T).T
        except linalg.LinAlgError:
            # Fallback to general solver if cholesky fails (e.g., if reg_grams is not positive definite)
            warnings.warn("Cholesky factorization failed, falling back to general least-squares solver. Consider increasing regularization.")
            return linalg.solve(reg_grams, rhs.T, assume_a="pos").T # still assume that the matrix is positive definite, but solve with a more robust method that can handle near-singularity better than cho_solve.

    def normalize_factors(self, factors: list[NDArray], weights: NDArray) -> tuple[list[NDArray], NDArray]:
        rank = factors[0].shape[1]
        if weights is None:
            weights = np.ones(rank)
        else:
            weights = weights.copy()

        normalized = []
        for factor in factors:
            norms = np.linalg.norm(factor, axis=0)
            # avoid division by zero
            norms = np.where(norms > 0, norms, 1.0)
            normalized.append(factor / norms)
            weights *= norms
        return normalized, weights
    
    def multiply_gather(self, values, indices, factors, exclude_mode):
        rank = factors[0].shape[1]
        n_modes = len(factors)

        # Start with values broadcast to (nnz, rank)
        kr_contrib = np.broadcast_to(
            values[:, np.newaxis], (len(values), rank)
        ).copy()

        # Multiply by factor values at each non-zero's coordinates
        for m in range(n_modes):
            if m != exclude_mode:
                factor_vals = factors[m][indices[m], :]  # (nnz, rank)
                kr_contrib *= factor_vals

        return kr_contrib
    
    def maximum(self, array: NDArray, value: float) -> NDArray:
        return np.maximum(array, value)

    def compute_norm(self, array: NDArray) -> float:
        return np.linalg.norm(array)

    def random_init(self, shapes: list[tuple[int,int]], random_state: np.random.Generator | int | None = None) -> list[NDArray]:
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        return [rng.standard_normal(shape) for shape in shapes]
    
class RAPIDSBackend(CPBackend):
    """
    GPU backend using RAPIDS cuPy.
    Requires: cupy (NVIDIA GPU with CUDA)

    This backend accelerates the core operations using GPU:
    - scatter_add: Uses cupy.scatter_add for efficient accumulation on GPU.
    - solve Cholesky decomposition (cuPy linalg)
    - gram_matrix via cuPy matmul

    Memory considerations for large tensors:
    - Gram matrices are small: (rank, rank), so they fit in GPU memory even for large tensors.
    - Factor matrices scale with dimensions and rank, but typically rank is small (e.g., 10-100), so they are manageable on GPU.
    - MTTKRP scale with nnz: O(nnz x rank)
    - All heavy operations stay on GPU to avoid transfer overhead
    """

    def __init__(self) -> None:
        try:
            import cupy as cp
            import cupyx

            self._cp = cp
            self._cupyx = cupyx
        except ImportError as e:
            raise ImportError(
                f"RAPIDS backend requires CuPy. {str(e)}\n{_CUPY_INSTALL_MSG}"
            ) from e
        
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                raise RuntimeError("No CUDA-compatible GPU detected. RAPIDS backend cannot be used.")
        except cp.cuda.runtime.CUDARuntimeError as e:
            raise RuntimeError(f"CUDA runtime error: {str(e)}. Ensure that NVIDIA drivers and CUDA toolkit are properly installed.") from e
        
    @property
    def name(self) -> str:
        return "rapids"
    
    def zeros(self, shape: tuple[int, ...], dtype: np.dtype | None = None) -> NDArray:
        if dtype is None:
            dtype = np.float64
        return self._cp.zeros(shape, dtype=dtype)
    
    def scatter_add(self, target: NDArray, indices: NDArray, values: NDArray) -> None:
        if not isinstance(target, self._cp.ndarray):
            raise TypeError("Target array must be a CuPy array for RAPIDS backend.")
        
        # ensure arrays are on GPU
        indices_gpu = self._cp.asarray(indices) if not isinstance(indices, self._cp.ndarray) else indices
        values_gpu = self._cp.asarray(values) if not isinstance(values, self._cp.ndarray) else values

        # cupyx.scatter_add expects slices as a tuple for multi-dimensional indexing
        # we are indexing rows (first axis) and adding to all columns
        self._cupyx.scatter_add(target, (indices_gpu, slice(None)), values_gpu)

    def gram_matrix(self, factor: NDArray) -> NDArray:
        return factor.T @ factor
    
    def hadamard_gram(self, gramians: list[NDArray]) -> NDArray:
        result = gramians[0].copy()
        for g in gramians[1:]:
            result *= g
        return result
    
    def solve_least_squares(self, grams: NDArray, rhs: NDArray, regularization: float) -> NDArray:
        cp = self._cp
        rank = grams.shape[0]
        reg_grams = grams + regularization * cp.eye(rank, grams.dtype)

        try:
            # cholesky decomposition regularized
            L = cp.linalg.cholesky(reg_grams)

            # solve L @ z = rhs.T (forward substitution) -> solve L^T @ X^T = z (backward substitution) which combined is:
            # X^T = solve(L^T, solve(L, rhs.T))
            z = cp.linalg.solve_triangular(L, rhs.T, lower=True)
            # solve L^T @ X^T = z (backward substitution)
            X = cp.linalg.solve_triangular(L.T, z, lower=False)
            return X.T
        except cp.linalg.LinAlgError as e:
            # Fallback to general solver if Cholesky fails
            warnings.warn(f"Cholesky factorization failed on GPU: {str(e)}. Falling back to general least-squares solver. Consider increasing regularization.")
            return cp.linalg.solve(reg_grams, rhs.T).T

    def normalize_factors(self, factors: list[NDArray], weights: NDArray | None = None) -> tuple[list[NDArray], NDArray]:
        cp = self._cp
        rank = factors[0].shape[1]
        if weights is None:
            weights = cp.ones(rank, dtype=factors[0].dtype)
        else:
            weights = weights.copy()

        normalized = []
        for factor in factors:
            norms = cp.linalg.norm(factor, axis=0)
            # avoid division by zero
            norms = cp.where(norms > 0, norms, 1.0)
            normalized.append(factor / norms)
            weights *= norms
        return normalized, weights

    def random_init(self, shapes: list[tuple[int, int]], random_state: int | None = None) -> list[NDArray]:
        cp = self._cp
        # Use isolated RandomState to avoid affecting global state
        if random_state is not None:
            seed = random_state if isinstance(random_state, int) else 42
            rs = cp.random.RandomState(seed)
            return [rs.standard_normal(shape).astype(cp.float64) for shape in shapes]

        return [cp.random.standard_normal(shape).astype(cp.float64) for shape in shapes]

    def to_backend(self, array: NDArray) -> NDArray:
        return self._cp.asarray(array)

    def to_numpy(self, arr: NDArray) -> np.ndarray:
        if isinstance(arr, self._cp.ndarray):
            return arr.get()
        return np.asarray(arr)
    
    def get_time(self) -> float:
        cp = self._cp
        # synchronize to ensure all GPU operations are complete before timing
        cp.cuda.Stream.null.synchronize()
        import time
        return time.time()

    def multiply_gather(
            self, values: NDArray, indices: NDArray, factors: list[NDArray], exclude_mode: int
    ) -> NDArray:
        cp = self._cp
        rank = factors[0].shape[1]
        n_modes = len(factors)

        # Ensure values is on GPU and broadcast to (nvalues, rank)
        if not isinstance(values, cp.ndarray):
            values = cp.asarray(values)
        
        kr_contrib = cp.broadcast_to(
            values[:, cp.newaxis], (len(values), rank)
        ).copy() # (nvalues, rank)

        # Multiply by factor values at each non-zero's coordinates
        for m in range(n_modes): # 0-->3
            if m != exclude_mode:
                idx = indices[m]
                if not isinstance(idx, cp.ndarray):
                    idx = cp.asarray(idx)
                factor_vals = factors[m][indices[m], :] # (nvalues, rank)
                kr_contrib *= factor_vals
        return kr_contrib
    
    def compute_norm(self, array: NDArray) -> float:
        return float(self._cp.linalg.norm(array))
    
# Backend registry
_BACKENDS: dict[str, type[CPBackend]] = {
    "numpy": NumPyBackend,
    "rapids": RAPIDSBackend
}

def get_backend(name: str = "auto", warn_fallback: bool = True) -> CPBackend:
    """
    Get a backend instance by name

    Args:
        name: Backend name ('numpy', 'rapids') or 'auto' for auto-detection
        warn_fallback: If True, emit warning when auto falls back to numpy.

    Returns:
        Initialized backend instance.

    Raises:
        ValueError: if backend name is unknown.
        ImportError: if requested backend dependencies are not available.
    """

    if name == "auto":
        # try rapids then fall back eventually
        try:
            return RAPIDSBackend()
        except (ImportError, RuntimeError) as e:
            if warn_fallback:
                warnings.warn(
                    f"GPU backend not available, fallback to numpy cpu. "
                    f"Reason {e}", 
                    UserWarning,
                    stacklevel=2,
                )
            return NumPyBackend()
        
    if name not in _BACKENDS:
        available = list(_BACKENDS.keys())
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    
    try:
        return _BACKENDS[name]()
    except (ImportError, RuntimeError) as e:
        if name == "rapids":
            raise ImportError(
                f"Requested 'rapids' backend but it is not available. \n"
                f"Error: {e}\n{_CUPY_INSTALL_MSG}"
            ) from e
        raise

def list_backends() -> list[str]:
    return list(_BACKENDS.keys())

def available_backends() -> list[str]:
    available = []
    for name in _BACKENDS:
        if is_backend_available(name):
            available.append(name)
    return available

def is_backend_available(name: str) -> bool:
    """
    Check if a backend is available (dependencies are installed)

    Args: 
        name: backend name to check
    Returns:
        True if backend can be initialized, False otherwise
    """
    if name not in _BACKENDS:
        return False
    
    try:
        get_backend(name, warn_fallback=False)
        return True
    except (ImportError, RuntimeError):
        return False
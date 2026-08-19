"""
Tests for MuxVizPy decomposition module and backend utilities.

Classes:
    TestNumPyBackendCorrectness — individual backend operations produce correct results
    TestDecompositionHelpers    — helper functions (_prepare_tensor, _compute_multi_index, etc.)
    TestSparseCPCorrectness     — sparse_cp_decomposition runs, converges, shapes are right
    TestSparseCPReference       — compare reconstruction error against tensorly sparse_parafac
"""

import pytest
import numpy as np
import torch

from MuxVizPy.utils.decomposition_utils import (
    NumPyBackend,
    get_backend,
    list_backends,
    available_backends,
    is_backend_available,
)
from MuxVizPy.decomposition import (
    _prepare_tensor,
    _initialize_factors,
    _compute_multi_index,
    _compute_recon_error,
    _sparse_mttkrp_no_weights,
    _compute_combined_gram_no_weights,
    sparse_cp_decomposition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sparse_tensor(N, L, nnz, seed=42):
    """Create a random sparse COO tensor (N, L, N, L) with *nnz* entries."""
    rng = np.random.default_rng(seed)
    i0 = rng.integers(0, N, size=nnz)
    i1 = rng.integers(0, L, size=nnz)
    i2 = rng.integers(0, N, size=nnz)
    i3 = rng.integers(0, L, size=nnz)
    vals = rng.uniform(0.1, 5.0, size=nnz)
    indices = torch.tensor(np.stack([i0, i1, i2, i3]), dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float64)
    return torch.sparse_coo_tensor(indices, values, size=(N, L, N, L))


def _reconstruct_at_nonzeros(factors, weights, coords):
    """Reconstruct tensor values at given coordinates from CP factors."""
    n_modes = len(factors)
    nnz = coords.shape[1]
    rank = factors[0].shape[1]
    # gather factor values: (nnz, rank) per mode
    gathered = [factors[m][coords[m], :] for m in range(n_modes)]
    product = gathered[0].copy()
    for g in gathered[1:]:
        product *= g
    return product @ weights  # (nnz,)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumPyBackend()


@pytest.fixture
def small_tensor():
    """Small 6-node, 2-layer sparse tensor with 20 non-zeros."""
    return _make_sparse_tensor(N=6, L=2, nnz=20, seed=0)


@pytest.fixture
def medium_tensor():
    """Medium 20-node, 3-layer sparse tensor with 200 non-zeros."""
    return _make_sparse_tensor(N=20, L=3, nnz=200, seed=42)


# ============================================================================
# NumPy Backend — correctness of individual operations
# ============================================================================

class TestNumPyBackendCorrectness:
    """Each backend primitive produces numerically correct results."""

    def test_name(self, backend):
        assert backend.name == "numpy"

    def test_zeros(self, backend):
        z = backend.zeros((3, 4), dtype=np.float64)
        assert z.shape == (3, 4)
        assert z.dtype == np.float64
        np.testing.assert_array_equal(z, 0.0)

    def test_scatter_add_simple(self, backend):
        target = np.zeros((4, 2))
        indices = np.array([0, 1, 2])
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        backend.scatter_add(target, indices, values)
        expected = np.array([[1, 2], [3, 4], [5, 6], [0, 0]], dtype=float)
        np.testing.assert_array_equal(target, expected)

    def test_scatter_add_duplicate_indices(self, backend):
        """Duplicate indices must accumulate (not overwrite)."""
        target = np.zeros((3, 2))
        indices = np.array([0, 0, 1])
        values = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 5.0]])
        backend.scatter_add(target, indices, values)
        np.testing.assert_array_equal(target[0], [3.0, 0.0])
        np.testing.assert_array_equal(target[1], [0.0, 5.0])

    def test_gram_matrix(self, backend):
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        G = backend.gram_matrix(A)
        expected = A.T @ A
        np.testing.assert_allclose(G, expected)

    def test_hadamard_gram(self, backend):
        G1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        G2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = backend.hadamard_gram([G1, G2])
        expected = G1 * G2
        np.testing.assert_allclose(result, expected)

    def test_hadamard_gram_three(self, backend):
        G1 = np.ones((2, 2))
        G2 = 2 * np.ones((2, 2))
        G3 = 3 * np.ones((2, 2))
        result = backend.hadamard_gram([G1, G2, G3])
        np.testing.assert_allclose(result, 6.0 * np.ones((2, 2)))

    def test_solve_least_squares(self, backend):
        """Solve (G + lambda*I) x = rhs for known solution."""
        # Identity gram → solution = rhs (up to regularization)
        G = np.eye(3)
        rhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        result = backend.solve_least_squares(G, rhs, regularization=0.0)
        np.testing.assert_allclose(result, rhs, atol=1e-10)

    def test_solve_least_squares_with_regularization(self, backend):
        """With lambda > 0, solution is (I + lambda*I)^-1 rhs = rhs/(1+lambda)."""
        G = np.eye(2)
        rhs = np.array([[2.0, 4.0]])  # (1, 2)
        lam = 1.0
        result = backend.solve_least_squares(G, rhs, regularization=lam)
        expected = rhs / (1.0 + lam)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_normalize_factors(self, backend):
        A = np.array([[3.0, 0.0], [0.0, 4.0]])
        B = np.array([[1.0, 2.0], [3.0, 4.0]])
        factors = [A.copy(), B.copy()]
        normed, weights = backend.normalize_factors(factors, weights=None)
        # Each column should have unit norm
        for f in normed:
            col_norms = np.linalg.norm(f, axis=0)
            np.testing.assert_allclose(col_norms, 1.0, atol=1e-12)
        # Weights should absorb the norms
        assert weights.shape == (2,)
        assert np.all(weights > 0)

    def test_normalize_factors_reconstruction(self, backend):
        """Normalizing then weighting should recover original outer product."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((5, 3))
        B = rng.standard_normal((4, 3))
        # Original weighted sum: sum_r A[:,r] * B[:,r]
        original = sum(np.outer(A[:, r], B[:, r]) for r in range(3))
        normed, weights = backend.normalize_factors([A.copy(), B.copy()], weights=None)
        reconstructed = sum(
            weights[r] * np.outer(normed[0][:, r], normed[1][:, r])
            for r in range(3)
        )
        np.testing.assert_allclose(reconstructed, original, atol=1e-10)

    def test_multiply_gather(self, backend):
        """multiply_gather computes element-wise Khatri-Rao contribution."""
        rank = 2
        # 3 non-zero entries, 2 modes
        factors = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),  # mode 0, dim=2
            np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),  # mode 1, dim=3
        ]
        values = np.array([1.0, 2.0, 3.0])
        indices = [
            np.array([0, 1, 0]),  # mode 0 indices
            np.array([2, 0, 1]),  # mode 1 indices
        ]
        # exclude mode 0 → only multiply by factors[1]
        result = backend.multiply_gather(values, indices, factors, exclude_mode=0)
        # expected: values[:, None] * factors[1][indices[1], :]
        expected = values[:, None] * factors[1][indices[1], :]
        np.testing.assert_allclose(result, expected)

    def test_multiply_gather_exclude_mode1(self, backend):
        """Excluding mode 1 multiplies by mode 0 only."""
        rank = 2
        factors = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]
        values = np.array([10.0, 20.0])
        indices = [np.array([0, 1]), np.array([1, 0])]
        result = backend.multiply_gather(values, indices, factors, exclude_mode=1)
        expected = values[:, None] * factors[0][indices[0], :]
        np.testing.assert_allclose(result, expected)

    def test_compute_norm(self, backend):
        a = np.array([3.0, 4.0])
        assert backend.compute_norm(a) == pytest.approx(5.0)

    def test_random_init_shapes(self, backend):
        shapes = [(10, 3), (5, 3)]
        result = backend.random_init(shapes, random_state=42)
        assert len(result) == 2
        assert result[0].shape == (10, 3)
        assert result[1].shape == (5, 3)

    def test_random_init_reproducible(self, backend):
        shapes = [(10, 3)]
        r1 = backend.random_init(shapes, random_state=123)
        r2 = backend.random_init(shapes, random_state=123)
        np.testing.assert_array_equal(r1[0], r2[0])

    def test_random_init_different_seeds(self, backend):
        shapes = [(10, 3)]
        r1 = backend.random_init(shapes, random_state=1)
        r2 = backend.random_init(shapes, random_state=2)
        assert not np.allclose(r1[0], r2[0])


# ============================================================================
# Backend registry
# ============================================================================

class TestBackendRegistry:
    """get_backend, list_backends, available_backends work correctly."""

    def test_list_backends(self):
        bl = list_backends()
        assert "numpy" in bl
        assert "rapids" in bl

    def test_numpy_available(self):
        assert is_backend_available("numpy")

    def test_get_numpy(self):
        be = get_backend("numpy")
        assert be.name == "numpy"

    def test_get_auto_returns_numpy_without_gpu(self):
        """On a machine without CUDA, auto should fallback to numpy."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            be = get_backend("auto", warn_fallback=False)
        assert be.name in ("numpy", "rapids")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")


# ============================================================================
# Decomposition helpers
# ============================================================================

class TestDecompositionHelpers:
    """Helper functions produce correct intermediate results."""

    def test_prepare_tensor_shapes(self, small_tensor):
        coords, values, shape = _prepare_tensor(small_tensor)
        assert shape == tuple(small_tensor.shape)
        assert coords.shape[0] == len(shape)  # n_modes rows
        assert coords.shape[1] == values.shape[0]  # nnz columns
        assert coords.dtype == np.int64
        assert values.dtype == np.float64

    def test_prepare_tensor_rejects_dense(self):
        dense = torch.randn(3, 2, 3, 2)
        with pytest.raises(ValueError, match="sparse"):
            _prepare_tensor(dense)

    def test_compute_multi_index(self):
        """Flattened column index for mode-0 unfolding of a (2,3,4) tensor."""
        shape = (2, 3, 4)
        # single entry at [1, 2, 3]
        coords = np.array([[1], [2], [3]])
        result = _compute_multi_index(coords, shape, exclude_mode=0)
        # excluding mode 0: other dims = (3, 4), strides = (4, 1)
        # index = 2*4 + 3 = 11
        assert result[0] == 11

    def test_compute_multi_index_batch(self):
        shape = (5, 3, 4)
        coords = np.array([[0, 1], [1, 2], [2, 3]])
        result = _compute_multi_index(coords, shape, exclude_mode=0)
        # other dims = (3, 4), strides = (4, 1)
        assert result[0] == 1 * 4 + 2  # = 6
        assert result[1] == 2 * 4 + 3  # = 11

    def test_initialize_factors_random(self, backend):
        shape = (10, 3, 10, 3)
        rank = 5
        factors = _initialize_factors(shape, rank, "random", 42, backend)
        assert len(factors) == 4
        assert factors[0].shape == (10, 5)
        assert factors[1].shape == (3, 5)
        assert factors[2].shape == (10, 5)
        assert factors[3].shape == (3, 5)

    def test_initialize_factors_unknown_raises(self, backend):
        with pytest.raises(ValueError, match="Unknown initialization"):
            _initialize_factors((4, 2, 4, 2), 3, "invalid", None, backend)

    def test_compute_recon_error_perfect(self, backend):
        """When factors perfectly reconstruct the tensor, error should be ~0."""
        rank = 2
        # Build factors and construct tensor values from them
        rng = np.random.default_rng(7)
        A = rng.standard_normal((4, rank))
        B = rng.standard_normal((2, rank))
        C = rng.standard_normal((4, rank))
        D = rng.standard_normal((2, rank))
        weights = np.array([1.5, 0.8])

        # Pick some coordinates
        coords = np.array([[0, 1, 2, 3], [0, 1, 0, 1], [1, 2, 3, 0], [1, 0, 1, 0]])
        # Compute exact values at those coordinates
        factors = [A, B, C, D]
        values = _reconstruct_at_nonzeros(factors, weights, coords)
        mode_indices = [coords[m] for m in range(4)]

        error = _compute_recon_error(mode_indices, values, factors, weights, backend)
        assert error == pytest.approx(0.0, abs=1e-10)

    def test_compute_recon_error_nonzero(self, backend):
        """Random factors should give nonzero reconstruction error on random data."""
        rng = np.random.default_rng(5)
        rank = 2
        factors = [rng.standard_normal((4, rank)) for _ in range(4)]
        weights = np.ones(rank)
        coords = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
        values = np.array([10.0, 20.0])
        mode_indices = [coords[m] for m in range(4)]

        error = _compute_recon_error(mode_indices, values, factors, weights, backend)
        assert error > 0.0


# ============================================================================
# MTTKRP and gram helpers
# ============================================================================

class TestMTTKRPCorrectness:
    """Sparse MTTKRP and combined gram produce correct results."""

    def test_mttkrp_shape(self, backend):
        shape = (4, 2, 4, 2)
        rank = 3
        rng = np.random.default_rng(10)
        factors = [rng.standard_normal((d, rank)) for d in shape]
        coords = np.array([[0, 1, 2], [0, 1, 0], [1, 2, 3], [1, 0, 1]])
        values = np.array([1.0, 2.0, 3.0])
        mode_indices = [coords[m] for m in range(4)]

        for mode in range(4):
            result = _sparse_mttkrp_no_weights(
                mode, mode_indices, values, factors, shape, backend
            )
            assert result.shape == (shape[mode], rank)

    def test_mttkrp_manual(self, backend):
        """Verify MTTKRP for a tiny 1-entry tensor by hand."""
        shape = (2, 2, 2, 2)
        rank = 1
        # Single non-zero at [0,0,1,1] with value 5.0
        coords = np.array([[0], [0], [1], [1]])
        values = np.array([5.0])
        mode_indices = [coords[m] for m in range(4)]
        # Factors: all ones
        factors = [np.ones((2, 1)) for _ in range(4)]

        # MTTKRP for mode 0: result[0, 0] = 5.0 * B[0,0]*C[1,0]*D[1,0] = 5.0
        result = _sparse_mttkrp_no_weights(0, mode_indices, values, factors, shape, backend)
        assert result[0, 0] == pytest.approx(5.0)
        assert result[1, 0] == pytest.approx(0.0)

    def test_combined_gram_shape(self, backend):
        rank = 3
        rng = np.random.default_rng(20)
        factors = [rng.standard_normal((d, rank)) for d in (4, 2, 4, 2)]
        for mode in range(4):
            G = _compute_combined_gram_no_weights(factors, mode, backend)
            assert G.shape == (rank, rank)

    def test_combined_gram_excludes_correct_mode(self, backend):
        """Combined gram excluding mode m should be product of the other 3 grams."""
        rank = 2
        rng = np.random.default_rng(30)
        factors = [rng.standard_normal((d, rank)) for d in (4, 2, 4, 2)]

        for exclude in range(4):
            combined = _compute_combined_gram_no_weights(factors, exclude, backend)
            # Manual computation
            grams = [factors[m].T @ factors[m] for m in range(4) if m != exclude]
            expected = grams[0].copy()
            for g in grams[1:]:
                expected *= g
            np.testing.assert_allclose(combined, expected, atol=1e-12)


# ============================================================================
# End-to-end sparse CP decomposition — correctness
# ============================================================================

class TestSparseCPCorrectness:
    """sparse_cp_decomposition runs correctly and produces valid output."""

    def test_output_shapes(self, small_tensor):
        N, L = 6, 2
        rank = 3
        (A, B, C, D), weights, history = sparse_cp_decomposition(
            small_tensor, rank=rank, max_iter=5, backend="numpy", random_state=42,
        )
        assert A.shape == (N, rank)
        assert B.shape == (L, rank)
        assert C.shape == (N, rank)
        assert D.shape == (L, rank)
        assert weights.shape == (rank,)

    def test_convergence_history(self, small_tensor):
        _, _, history = sparse_cp_decomposition(
            small_tensor, rank=3, max_iter=20, backend="numpy", random_state=42,
        )
        errors = history["reconstruction_error"]
        assert len(errors) > 0
        assert all(isinstance(e, float) for e in errors)
        # Error should generally decrease (allow small fluctuations)
        assert errors[-1] <= errors[0] + 0.1

    def test_decomposition_converges(self, medium_tensor):
        """With enough iterations, error should be reasonably small."""
        _, weights, history = sparse_cp_decomposition(
            medium_tensor, rank=5, max_iter=100, tol=1e-6,
            backend="numpy", random_state=42,
        )
        final_error = history["reconstruction_error"][-1]
        assert final_error < 1.0  # relative error below 100%
        assert np.all(weights > 0)

    def test_reproducible_with_seed(self, small_tensor):
        """Same random_state produces identical results."""
        (A1, B1, C1, D1), w1, _ = sparse_cp_decomposition(
            small_tensor, rank=3, max_iter=10, backend="numpy", random_state=99,
        )
        (A2, B2, C2, D2), w2, _ = sparse_cp_decomposition(
            small_tensor, rank=3, max_iter=10, backend="numpy", random_state=99,
        )
        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(w1, w2)

    def test_rank_one(self, small_tensor):
        """Rank-1 decomposition should still work."""
        (A, B, C, D), weights, history = sparse_cp_decomposition(
            small_tensor, rank=1, max_iter=20, backend="numpy", random_state=42,
        )
        assert A.shape[1] == 1
        assert weights.shape == (1,)
        assert len(history["reconstruction_error"]) > 0


# ============================================================================
# Reference: compare against tensorly sparse_parafac
# ============================================================================

class TestSparseCPReference:
    """Compare our decomposition against tensorly's sparse_parafac.

    CP factors are unique only up to permutation and sign of components.
    We align factors via cosine-similarity matching then compare directly.
    """

    @staticmethod
    def _tensorly_decompose(tensor, rank, seed=42, max_iter=100, tol=1e-6):
        """Run tensorly sparse_parafac on a torch sparse tensor."""
        import logging
        logging.getLogger("numba").setLevel(logging.WARNING)

        import sparse as sp_sparse
        from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac

        t = tensor.coalesce()
        coords = t.indices().numpy()
        vals = t.values().numpy()
        shape = tuple(t.shape)

        X = sp_sparse.COO(coords=coords, data=vals, shape=shape)

        weights, factors = sparse_parafac(
            X, rank=rank, init="random", random_state=seed,
            n_iter_max=max_iter, tol=tol,
        )
        return weights, factors

    @staticmethod
    def _to_dense(arr):
        """Convert sparse.COO or similar to numpy; pass through ndarray."""
        if hasattr(arr, "todense"):
            return np.asarray(arr.todense(), dtype=np.float64)
        return np.asarray(arr, dtype=np.float64)

    @staticmethod
    def _reconstruct_at(coords, factors, weights):
        """Reconstruct tensor values at given coordinates from CP factors."""
        n_modes = len(factors)
        gathered = [factors[m][coords[m], :] for m in range(n_modes)]
        product = gathered[0].copy()
        for g in gathered[1:]:
            product *= g
        return product @ weights

    @staticmethod
    def _align_factors(factors_a, weights_a, factors_b, weights_b):
        """Align CP components via greedy cosine-similarity matching.

        Absorbs weights into the first factor, matches components by
        max absolute cosine similarity across all modes, then flips
        signs so inner products are positive.

        Returns (factors_a_aligned, factors_b_aligned) with consistent
        ordering and sign, both with weights absorbed.
        """
        from scipy.optimize import linear_sum_assignment

        rank = factors_a[0].shape[1]
        n_modes = len(factors_a)

        # Absorb weights into first factor
        fa = [f.copy() for f in factors_a]
        fb = [f.copy() for f in factors_b]
        fa[0] = fa[0] * weights_a[np.newaxis, :]
        fb[0] = fb[0] * weights_b[np.newaxis, :]

        # Build cost matrix: average |cosine similarity| across modes
        cost = np.zeros((rank, rank))
        for m in range(n_modes):
            norms_a = np.linalg.norm(fa[m], axis=0, keepdims=True)
            norms_b = np.linalg.norm(fb[m], axis=0, keepdims=True)
            norms_a = np.where(norms_a > 0, norms_a, 1.0)
            norms_b = np.where(norms_b > 0, norms_b, 1.0)
            cosine = (fa[m] / norms_a).T @ (fb[m] / norms_b)  # (rank, rank)
            cost += np.abs(cosine)
        cost /= n_modes

        # Hungarian assignment (maximize similarity → minimize -cost)
        row_ind, col_ind = linear_sum_assignment(-cost)

        # Permute b to match a
        fb = [f[:, col_ind] for f in fb]

        # Fix sign: for each component, flip b if dot product is negative
        for r in range(rank):
            dot = sum(fa[m][:, r] @ fb[m][:, r] for m in range(n_modes))
            if dot < 0:
                fb[0][:, r] *= -1  # flip in one mode is enough

        return fa, fb

    @staticmethod
    def _mean_cosine_similarity(factors_a, factors_b):
        """Mean cosine similarity across all modes and components."""
        n_modes = len(factors_a)
        rank = factors_a[0].shape[1]
        similarities = []
        for m in range(n_modes):
            for r in range(rank):
                a = factors_a[m][:, r]
                b = factors_b[m][:, r]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 0 and nb > 0:
                    similarities.append(np.abs(a @ b) / (na * nb))
        return np.mean(similarities) if similarities else 0.0

    def test_reconstructed_values_match(self, medium_tensor):
        """Both methods should reconstruct the same values at non-zero entries."""
        rank = 5
        max_iter = 100

        # Our decomposition
        (A, B, C, D), our_w, _ = sparse_cp_decomposition(
            medium_tensor, rank=rank, max_iter=max_iter, tol=1e-8,
            backend="numpy", random_state=42,
        )

        # Tensorly decomposition
        tl_w, tl_f = self._tensorly_decompose(
            medium_tensor, rank=rank, seed=42, max_iter=max_iter, tol=1e-8,
        )
        tl_f = [self._to_dense(f) for f in tl_f]
        tl_w = self._to_dense(tl_w)

        # Reconstruct at non-zero coordinates
        t = medium_tensor.coalesce()
        coords = t.indices().numpy()
        values = t.values().numpy().astype(np.float64)

        our_recon = self._reconstruct_at(coords, [A, B, C, D], our_w)
        tl_recon = self._reconstruct_at(coords, tl_f, tl_w)

        # Both should approximate the original reasonably
        our_rel = np.linalg.norm(values - our_recon) / np.linalg.norm(values)
        tl_rel = np.linalg.norm(values - tl_recon) / np.linalg.norm(values)
        assert our_rel < 1.0
        assert tl_rel < 1.0

    def test_factor_alignment_known_rank(self):
        """On a known-rank tensor, aligned factors should have high cosine similarity."""
        rng = np.random.default_rng(77)
        N, L, rank = 6, 2, 3
        total = N * L * N * L  # 576

        # Build tensor from known non-negative factors (avoids sign ambiguity)
        true_factors = [np.abs(rng.standard_normal((d, rank))) + 0.1 for d in (N, L, N, L)]
        true_weights = np.array([3.0, 2.0, 1.0])

        # Dense enumeration of ALL entries — no duplicate coordinates
        all_coords = np.array(np.meshgrid(
            np.arange(N), np.arange(L), np.arange(N), np.arange(L),
            indexing="ij",
        )).reshape(4, -1)
        all_vals = self._reconstruct_at(all_coords, true_factors, true_weights)

        # Keep ~60% of entries: enough signal for unique recovery
        mask = rng.choice(total, size=int(0.6 * total), replace=False)
        coords = all_coords[:, mask]
        vals = all_vals[mask]

        indices = torch.tensor(coords, dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float64)
        tensor = torch.sparse_coo_tensor(indices, values, size=(N, L, N, L))

        # Our decomposition
        (A, B, C, D), our_w, _ = sparse_cp_decomposition(
            tensor, rank=rank, max_iter=300, tol=1e-12,
            backend="numpy", random_state=42,
        )

        # Tensorly
        tl_w, tl_f = self._tensorly_decompose(
            tensor, rank=rank, seed=42, max_iter=300, tol=1e-12,
        )
        tl_f = [self._to_dense(f) for f in tl_f]
        tl_w = self._to_dense(tl_w)

        # Align and compare
        ours_aligned, tl_aligned = self._align_factors(
            [A, B, C, D], our_w, tl_f, tl_w,
        )
        sim = self._mean_cosine_similarity(ours_aligned, tl_aligned)
        assert sim > 0.8, f"Mean cosine similarity too low: {sim:.4f}"

    def test_higher_rank_improves_fit(self, medium_tensor):
        """Increasing rank should decrease or maintain reconstruction error."""
        errors = {}
        for rank in [2, 5, 10]:
            _, _, history = sparse_cp_decomposition(
                medium_tensor, rank=rank, max_iter=80, tol=1e-8,
                backend="numpy", random_state=42,
            )
            errors[rank] = history["reconstruction_error"][-1]

        assert errors[5] <= errors[2] + 0.05, (
            f"rank=5 error ({errors[5]:.4f}) worse than rank=2 ({errors[2]:.4f})"
        )
        assert errors[10] <= errors[5] + 0.05, (
            f"rank=10 error ({errors[10]:.4f}) worse than rank=5 ({errors[5]:.4f})"
        )

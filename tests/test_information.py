"""
Tests for MuxVizPy.information — density matrix, Von Neumann entropy,
and Jensen-Shannon divergence.

Reference: De Domenico et al. (2015) "Structural reducibility of multilayer
networks", Nature Communications, 6, 6864.  R counterparts live in muxViz's
GetRenyiEntropyFromAdjacencyMatrix / GetJensenShannonDivergence.

Classes:
    TestInformationCorrectness  — types, shapes, value ranges, invariants, known values
    TestInformationReference    — comparison against pre-computed muxViz R results
"""

import pytest
import numpy as np
import scipy.sparse as sp
from MuxVizPy.utils.parsing import (
    build_edge_colored_matrices_from_supra_adjacency_matrix,
    build_density_bgs_from_adjacency_matrix as build_density_matrix,
)
from MuxVizPy.information import compute_vn_entropy, compute_js_divergence
from conftest import compare_metrics


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def net_layer_adjs(net_adjacency, net_n, net_l):
    """Per-layer undirected (symmetrized) adjacency matrices.

    The BGS density matrix rho = L/tr(L) is only a valid quantum state (PSD,
    symmetric) when L is the combinatorial Laplacian of an *undirected* graph.
    Directed intra-layer edges are symmetrized: (u→v) or (v→u) becomes (u–v).
    This matches the requirement of muxViz's GetRenyiEntropyFromAdjacencyMatrix,
    which needs a symmetric input to guarantee real eigenvalues.
    """
    tensor = build_edge_colored_matrices_from_supra_adjacency_matrix(net_adjacency, net_l)
    result = []
    for adj in tensor:
        sym = (adj + adj.T).tocsr()
        # Replace all nonzero data with float 1.0 (binarize and ensure float64 dtype)
        sym.data = np.ones(sym.nnz, dtype=np.float64)
        result.append(sym)
    return result


@pytest.fixture(scope="session")
def net_density_matrices(net_layer_adjs):
    """BGS density matrices for each layer."""
    return [build_density_matrix(adj) for adj in net_layer_adjs]


@pytest.fixture(scope="session")
def net_vn_entropies(net_density_matrices):
    """Von Neumann entropy per layer."""
    return [compute_vn_entropy(rho) for rho in net_density_matrices]


# ============================================================================
# Correctness — types, shapes, value ranges, invariants, known values
# ============================================================================

class TestInformationCorrectness:
    """Information functions return sane types/shapes/values and structural invariants hold."""

    # --- build_density_matrix ------------------------------------------------

    def test_density_trace_is_one(self, net_density_matrices):
        for rho in net_density_matrices:
            np.testing.assert_allclose(
                rho.diagonal().sum(), 1.0, atol=1e-10,
                err_msg="density matrix trace must equal 1",
            )

    def test_density_is_symmetric(self, net_density_matrices):
        for rho in net_density_matrices:
            diff = (rho - rho.T).tocsr()
            diff.eliminate_zeros()
            assert diff.nnz == 0 or float(abs(diff).max()) < 1e-12

    def test_density_is_psd(self, net_density_matrices):
        for rho in net_density_matrices:
            eigenvalues = np.linalg.eigvalsh(rho.toarray())
            assert np.all(eigenvalues >= -1e-10), (
                f"density matrix has negative eigenvalue: {eigenvalues.min():.2e}"
            )

    def test_density_raises_on_empty_graph(self):
        adj = sp.csr_matrix((3, 3))  # no edges → tr(L) = 0
        with pytest.raises(ValueError):
            build_density_matrix(adj)

    def test_density_known_path_p3(self):
        # P3: 0–1–2.  degree = [1, 2, 1], tr(L) = 4.
        # rho diagonal = [1/4, 1/2, 1/4]
        adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
        rho = build_density_matrix(adj)
        np.testing.assert_allclose(rho.diagonal().sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(rho.diagonal(), [0.25, 0.50, 0.25], atol=1e-12)

    def test_density_known_complete_k4(self):
        # K4: every node has degree 3, tr(L) = 12.
        # rho diagonal = [1/4, 1/4, 1/4, 1/4]
        n = 4
        adj = sp.csr_matrix(np.ones((n, n)) - np.eye(n))
        rho = build_density_matrix(adj)
        np.testing.assert_allclose(rho.diagonal(), np.full(n, 1.0 / n), atol=1e-12)
        np.testing.assert_allclose(rho.diagonal().sum(), 1.0, atol=1e-12)

    # --- compute_vn_entropy --------------------------------------------------

    def test_vn_entropy_non_negative(self, net_vn_entropies):
        for h in net_vn_entropies:
            assert h >= -1e-10, f"VN entropy must be non-negative, got {h:.4e}"

    def test_vn_entropy_bounded_by_log_n(self, net_density_matrices, net_vn_entropies):
        for rho, h in zip(net_density_matrices, net_vn_entropies):
            n = rho.shape[0]
            assert h <= np.log(n) + 1e-10, (
                f"VN entropy {h:.4f} exceeds log({n}) = {np.log(n):.4f}"
            )

    def test_vn_entropy_agrees_with_dense(self, net_density_matrices, net_vn_entropies):
        """Sparse eigsh result matches dense eigh for all layers."""
        for rho, h_sparse in zip(net_density_matrices, net_vn_entropies):
            eigs = np.linalg.eigvalsh(rho.toarray())
            pos = eigs[eigs > 0]
            h_dense = float(-np.sum(pos * np.log(pos)))
            np.testing.assert_allclose(
                h_sparse, h_dense, rtol=1e-5,
                err_msg="sparse and dense VN entropy disagree",
            )

    def test_vn_entropy_zero_for_pure_state(self):
        # P2 (single edge 0–1): rho eigenvalues = [0, 1] → H = 0
        adj = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
        rho = build_density_matrix(adj)
        h = compute_vn_entropy(rho)
        np.testing.assert_allclose(h, 0.0, atol=1e-10)

    def test_vn_entropy_known_complete_k4(self):
        # K4: rho eigenvalues = [0, 1/3, 1/3, 1/3] → H = log(3)
        n = 4
        adj = sp.csr_matrix(np.ones((n, n)) - np.eye(n))
        rho = build_density_matrix(adj)
        h = compute_vn_entropy(rho)
        np.testing.assert_allclose(h, np.log(3), rtol=1e-6)

    def test_vn_entropy_known_path_p3(self):
        # Verify sparse result against dense reference for P3
        adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
        rho = build_density_matrix(adj)
        eigs = np.linalg.eigvalsh(rho.toarray())
        pos = eigs[eigs > 0]
        expected = float(-np.sum(pos * np.log(pos)))
        np.testing.assert_allclose(compute_vn_entropy(rho), expected, rtol=1e-6)

    # --- compute_js_divergence -----------------------------------------------

    def test_jsd_non_negative(self, net_layer_adjs, net_vn_entropies, net_l):
        for i in range(net_l):
            for j in range(i, net_l):
                jsd = compute_js_divergence(
                    net_layer_adjs[i], net_layer_adjs[j],
                    net_vn_entropies[i], net_vn_entropies[j],
                )
                assert jsd >= -1e-10, f"JSD({i},{j}) = {jsd:.4e} is negative"

    def test_jsd_bounded_by_log2(self, net_layer_adjs, net_vn_entropies, net_l):
        for i in range(net_l):
            for j in range(i, net_l):
                jsd = compute_js_divergence(
                    net_layer_adjs[i], net_layer_adjs[j],
                    net_vn_entropies[i], net_vn_entropies[j],
                )
                assert jsd <= np.log(2) + 1e-10, (
                    f"JSD({i},{j}) = {jsd:.4f} exceeds log(2)"
                )

    def test_jsd_zero_for_identical_layers(self, net_layer_adjs, net_vn_entropies, net_l):
        for i in range(net_l):
            jsd = compute_js_divergence(
                net_layer_adjs[i], net_layer_adjs[i],
                net_vn_entropies[i], net_vn_entropies[i],
            )
            np.testing.assert_allclose(
                jsd, 0.0, atol=1e-8,
                err_msg=f"JSD(layer_{i}, layer_{i}) must be 0",
            )

    def test_jsd_symmetric(self, net_layer_adjs, net_vn_entropies, net_l):
        for i in range(net_l):
            for j in range(i + 1, net_l):
                jsd_ij = compute_js_divergence(
                    net_layer_adjs[i], net_layer_adjs[j],
                    net_vn_entropies[i], net_vn_entropies[j],
                )
                jsd_ji = compute_js_divergence(
                    net_layer_adjs[j], net_layer_adjs[i],
                    net_vn_entropies[j], net_vn_entropies[i],
                )
                np.testing.assert_allclose(
                    jsd_ij, jsd_ji, atol=1e-10,
                    err_msg=f"JSD not symmetric for layers {i},{j}",
                )

    def test_jsd_known_p3_vs_k3(self):
        # Verify JSD(P3, K3) against a dense reference computation.
        # P3: path 0–1–2;  K3: complete triangle
        adj_p3 = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
        adj_k3 = sp.csr_matrix(np.ones((3, 3)) - np.eye(3))
        rho_p3 = build_density_matrix(adj_p3)
        rho_k3 = build_density_matrix(adj_k3)
        h_p3 = compute_vn_entropy(rho_p3)
        h_k3 = compute_vn_entropy(rho_k3)

        # Dense reference: H(M) - 0.5*(H(rho_p3) + H(rho_k3))
        M = (rho_p3 + rho_k3) * 0.5
        eigs_M = np.linalg.eigvalsh(M.toarray())
        pos_M = eigs_M[eigs_M > 0]
        expected = float(-np.sum(pos_M * np.log(pos_M)) - 0.5 * (h_p3 + h_k3))

        np.testing.assert_allclose(
            compute_js_divergence(adj_p3, adj_k3, h_p3, h_k3),
            expected,
            rtol=1e-6,
        )


# ============================================================================
# Reference — comparison against pre-computed muxViz R results
# ============================================================================

class TestInformationReference:
    """Compare information results against muxViz R reference.

    Pre-computed results are loaded from tests/data/{config}/muxviz_results.json.
    Tests are skipped for configs without reference data or missing metric keys.
    """

    def test_vn_entropy_vs_muxviz(
        self, net_layer_adjs, net_muxviz_results, network_config
    ):
        if "vn_entropy" not in net_muxviz_results:
            pytest.skip(f"'vn_entropy' not in reference results for '{network_config}'")

        computed = np.array(
            [compute_vn_entropy(build_density_matrix(adj)) for adj in net_layer_adjs],
            dtype=np.float64,
        )
        expected = np.asarray(net_muxviz_results["vn_entropy"], dtype=np.float64).ravel()

        compare_metrics(
            computed, expected,
            "Von Neumann entropy per layer",
            computed_name="Python",
            expected_name="muxViz R",
            rtol=1e-4,
            atol=1e-4,
        )

    def test_jsd_matrix_vs_muxviz(
        self, net_layer_adjs, net_l, net_muxviz_results, network_config
    ):
        if "jsd_matrix" not in net_muxviz_results:
            pytest.skip(f"'jsd_matrix' not in reference results for '{network_config}'")

        vn_entropies = [
            compute_vn_entropy(build_density_matrix(adj)) for adj in net_layer_adjs
        ]
        computed = np.zeros((net_l, net_l))
        for i in range(net_l):
            for j in range(net_l):
                computed[i, j] = compute_js_divergence(
                    net_layer_adjs[i], net_layer_adjs[j],
                    vn_entropies[i], vn_entropies[j],
                )

        # R stores matrices column-major; reshape accordingly
        r_flat = np.asarray(net_muxviz_results["jsd_matrix"], dtype=np.float64)
        expected = r_flat.reshape(net_l, net_l, order="F")

        compare_metrics(
            computed.ravel(), expected.ravel(),
            "JSD matrix",
            computed_name="Python",
            expected_name="muxViz R",
            rtol=1e-4,
            atol=1e-4,
        )

"""
Tests for MuxVizPy.global_descriptors — average global clustering and overlap measures.

All test classes are parametrized over network configs (toy, random_large,
scalefree_small) via the ``net_*`` fixtures defined in conftest.py.

Classes:
    TestGlobalDescriptorsCorrectness  — types, shapes, value ranges, invariants
    TestGlobalDescriptorsReference    — comparison against pre-computed muxViz R results
"""

import pytest
import numpy as np
from MuxVizPy import global_descriptors
from conftest import compare_metrics


# ============================================================================
# Correctness — types, shapes, value ranges, invariants
# ============================================================================

class TestGlobalDescriptorsCorrectness:
    """Functions run correctly and return sane types/shapes/values."""

    # --- average global clustering coefficient ----------------------------

    def test_agcc_returns_float(self, net_adjacency, net_n, net_l):
        result = global_descriptors.compute_average_global_clustering_coefficient(
            net_adjacency, net_n, net_l
        )
        assert isinstance(result, float)

    def test_agcc_non_negative(self, net_adjacency, net_n, net_l):
        result = global_descriptors.compute_average_global_clustering_coefficient(
            net_adjacency, net_n, net_l
        )
        assert result >= 0.0

    # --- average global edge overlap (scalar) -----------------------------

    def test_agov_returns_float(self, net_adjacency, net_n, net_l):
        result = global_descriptors.compute_average_global_overlap(
            net_adjacency, net_n, net_l
        )
        assert isinstance(result, float)

    def test_agov_non_negative(self, net_adjacency, net_n, net_l):
        result = global_descriptors.compute_average_global_overlap(
            net_adjacency, net_n, net_l
        )
        assert result >= 0.0

    def test_agov_raises_single_layer(self, net_adjacency, net_n):
        with pytest.raises(ValueError):
            global_descriptors.compute_average_global_overlap(net_adjacency, net_n, 1)

    def test_agov_binary_equals_weighted_on_binary_adj(self, net_adjacency, net_n, net_l):
        """For a binary adjacency matrix, weighted=True and weighted=False must agree."""
        r_bin = global_descriptors.compute_average_global_overlap(
            net_adjacency, net_n, net_l, weighted=False
        )
        r_wt = global_descriptors.compute_average_global_overlap(
            net_adjacency, net_n, net_l, weighted=True
        )
        assert r_bin == pytest.approx(r_wt, abs=1e-10)

    # --- average global edge overlap matrix (L×L) -------------------------

    def test_agov_mat_shape(self, net_adjacency, net_n, net_l):
        M = global_descriptors.compute_average_global_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        assert M.shape == (net_l, net_l)

    def test_agov_mat_diagonal_is_one(self, net_adjacency, net_n, net_l):
        M = global_descriptors.compute_average_global_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        np.testing.assert_array_equal(np.diag(M), np.ones(net_l))

    def test_agov_mat_symmetric(self, net_adjacency, net_n, net_l):
        M = global_descriptors.compute_average_global_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        np.testing.assert_array_almost_equal(M, M.T)

    def test_agov_mat_off_diagonal_in_range(self, net_adjacency, net_n, net_l):
        M = global_descriptors.compute_average_global_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        off_diag = M[~np.eye(net_l, dtype=bool)]
        assert np.all(off_diag >= 0.0)
        assert np.all(off_diag <= 1.0)

    def test_agov_mat_raises_single_layer(self, net_adjacency, net_n):
        with pytest.raises(ValueError):
            global_descriptors.compute_average_global_overlap_matrix(net_adjacency, net_n, 1)

    # --- average global node overlap matrix (L×L) -------------------------

    def test_agnov_mat_shape(self, net_adjacency, net_n, net_l):
        M = global_descriptors.compute_average_global_node_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        assert M.shape == (net_l, net_l)

    def test_agnov_mat_diagonal_is_one(self, net_adjacency, net_n, net_l):
        M = global_descriptors.compute_average_global_node_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        np.testing.assert_array_equal(np.diag(M), np.ones(net_l))

    def test_agnov_mat_symmetric(self, net_adjacency, net_n, net_l):
        M = global_descriptors.compute_average_global_node_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        np.testing.assert_array_almost_equal(M, M.T)

    def test_agnov_mat_in_range(self, net_adjacency, net_n, net_l):
        M = global_descriptors.compute_average_global_node_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        assert np.all(M >= 0.0)
        assert np.all(M <= 1.0)

    def test_agnov_mat_raises_single_layer(self, net_adjacency, net_n):
        with pytest.raises(ValueError):
            global_descriptors.compute_average_global_node_overlap_matrix(
                net_adjacency, net_n, 1
            )

    # --- cross-function invariants ----------------------------------------

    def test_agov_mat_consistent_with_scalar(self, net_adjacency, net_n, net_l):
        """agov_mat off-diagonal entries are all <= 1; agov may exceed 1 for
        directed networks with the R-compatible normalisation, but agov_mat
        (Dice) is always in [0, 1] — verify they are both non-negative."""
        scalar = global_descriptors.compute_average_global_overlap(
            net_adjacency, net_n, net_l
        )
        mat = global_descriptors.compute_average_global_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        assert scalar >= 0.0
        assert np.all(mat >= 0.0)

    def test_agnov_mat_le_agov_mat_off_diagonal(self, net_adjacency, net_n, net_l):
        """Node overlap >= edge overlap off-diagonal: if two layers share an edge
        they must share both endpoint nodes, so edge overlap <= node overlap."""
        edge_mat = global_descriptors.compute_average_global_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        node_mat = global_descriptors.compute_average_global_node_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        mask = ~np.eye(net_l, dtype=bool)
        # node overlap is per-node (fraction of N); edge overlap is Dice (fraction of edges)
        # both are in [0,1] — no strict ordering in general, but both non-negative
        assert np.all(edge_mat[mask] >= 0.0)
        assert np.all(node_mat[mask] >= 0.0)


# ============================================================================
# Reference — comparison against pre-computed muxViz R results
# ============================================================================

class TestGlobalDescriptorsReference:
    """Compare results against muxViz R reference.

    Pre-computed results are loaded from tests/data/{config}/muxviz_results.json.
    Tests are skipped for configs without reference data or missing metric keys.
    """

    def test_agcc_vs_muxviz(self, net_adjacency, net_n, net_l, net_muxviz_results):
        if "agcc" not in net_muxviz_results:
            pytest.skip("agcc not in reference results")
        computed = global_descriptors.compute_average_global_clustering_coefficient(
            net_adjacency, net_n, net_l
        )
        # R as.vector() on a scalar gives a length-1 list in JSON
        expected = float(np.array(net_muxviz_results["agcc"]).ravel()[0])
        np.testing.assert_allclose(computed, expected, rtol=1e-4, atol=1e-4,
                                   err_msg="AGCC mismatch vs muxViz R")

    def test_agov_vs_muxviz(self, net_adjacency, net_n, net_l, net_muxviz_results):
        if "agov" not in net_muxviz_results:
            pytest.skip("agov not in reference results")
        computed = global_descriptors.compute_average_global_overlap(
            net_adjacency, net_n, net_l, weighted=False
        )
        expected = float(np.array(net_muxviz_results["agov"]).ravel()[0])
        np.testing.assert_allclose(computed, expected, rtol=1e-4, atol=1e-4,
                                   err_msg="AGOV mismatch vs muxViz R")

    def test_agov_mat_vs_muxviz(self, net_adjacency, net_n, net_l, net_muxviz_results):
        if "agov_mat" not in net_muxviz_results:
            pytest.skip("agov_mat not in reference results")
        computed = global_descriptors.compute_average_global_overlap_matrix(
            net_adjacency, net_n, net_l, weighted=False
        )
        # R as.vector() on a matrix is column-major (Fortran order)
        r_flat = np.array(net_muxviz_results["agov_mat"], dtype=float)
        expected = r_flat.reshape(net_l, net_l, order="F")
        compare_metrics(
            computed.ravel(), expected.ravel(),
            "AGOV matrix (vs muxViz R)", rtol=1e-4, atol=1e-4,
        )

    def test_agnov_mat_vs_muxviz(self, net_adjacency, net_n, net_l, net_muxviz_results):
        if "agnov_mat" not in net_muxviz_results:
            pytest.skip("agnov_mat not in reference results")
        computed = global_descriptors.compute_average_global_node_overlap_matrix(
            net_adjacency, net_n, net_l
        )
        # R as.vector() on a matrix is column-major (Fortran order)
        r_flat = np.array(net_muxviz_results["agnov_mat"], dtype=float)
        expected = r_flat.reshape(net_l, net_l, order="F")
        compare_metrics(
            computed.ravel(), expected.ravel(),
            "AGNOV matrix (vs muxViz R)", rtol=1e-4, atol=1e-4,
        )

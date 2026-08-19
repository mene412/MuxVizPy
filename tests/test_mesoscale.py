"""
Tests for MuxVizPy.mesoscale — compute_local_clustering_coefficient.

Classes:
    TestLocalClusteringCorrectness  — types, shapes, value ranges, edge cases
    TestLocalClusteringReference    — comparison against pre-computed muxViz R results
"""

import pytest
import numpy as np
import scipy.sparse as sp
from MuxVizPy import mesoscale
from conftest import compare_metrics


# ============================================================================
# Correctness — types, shapes, value ranges, edge cases
# ============================================================================

class TestLocalClusteringCorrectness:
    """compute_local_clustering_coefficient returns sane types/shapes/values."""

    def test_returns_ndarray(self, net_adjacency, net_n, net_l):
        result = mesoscale.compute_local_clustering_coefficient(net_adjacency, net_n, net_l)
        assert isinstance(result, np.ndarray)

    def test_shape(self, net_adjacency, net_n, net_l):
        result = mesoscale.compute_local_clustering_coefficient(net_adjacency, net_n, net_l)
        assert result.shape == (net_n,)

    def test_range_non_negative(self, net_adjacency, net_n, net_l):
        result = mesoscale.compute_local_clustering_coefficient(net_adjacency, net_n, net_l)
        assert np.all(result >= 0.0), f"Negative values found: {result[result < 0]}"

    def test_range_at_most_one(self, net_adjacency, net_n, net_l):
        result = mesoscale.compute_local_clustering_coefficient(net_adjacency, net_n, net_l)
        assert np.all(result <= 1.0), f"Values > 1 found: {result[result > 1]}"

    def test_zero_adjacency_gives_zero_clustering(self):
        n, l = 5, 3
        zero_adj = sp.csr_matrix((n * l, n * l))
        result = mesoscale.compute_local_clustering_coefficient(zero_adj, n, l)
        assert np.allclose(result, 0.0)

    def test_sample_network(self, sample_adjacency, n_nodes, n_layers):
        result = mesoscale.compute_local_clustering_coefficient(
            sample_adjacency, n_nodes, n_layers
        )
        assert result.shape == (n_nodes,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ============================================================================
# Reference — comparison against muxViz R
# ============================================================================

class TestLocalClusteringReference:
    """compare_local_clustering_coefficient against pre-computed muxViz R results."""

    def test_local_clus_vs_muxviz(
        self, net_adjacency, net_n, net_l, net_muxviz_results, network_config
    ):
        if network_config == "random_large":
            pytest.skip("GetLocalClustering scales as O(N²L²) and is too slow for random_large")
        if "local_clus" not in net_muxviz_results:
            pytest.skip(f"'local_clus' not in reference results for '{network_config}'")

        computed = mesoscale.compute_local_clustering_coefficient(net_adjacency, net_n, net_l)
        expected = np.asarray(net_muxviz_results["local_clus"], dtype=np.float64).ravel()

        compare_metrics(
            computed, expected,
            "local_clustering_coefficient",
            computed_name="Python",
            expected_name="muxViz R",
            rtol=1e-4,
            atol=1e-4,
        )

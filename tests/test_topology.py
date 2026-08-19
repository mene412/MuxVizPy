"""
Tests for MuxVizPy.topology — connected components, LCC/LIC/LVC,
path statistics, and layer similarity.

All parametrized test classes use the ``net_*`` fixtures defined in
conftest.py (toy, random_large, scalefree_small configurations).

Classes:
    TestTopologyCorrectness  — types, shapes, value ranges, invariants, edge cases
    TestTopologyReference    — comparison against pre-computed muxViz R results
"""

import pytest
import numpy as np
import scipy.sparse as sp
import graph_tool as gt
from MuxVizPy import topology
from MuxVizPy.utils.parsing import (
    supra_adjacency_to_network_list,
    build_edge_colored_matrices_from_supra_adjacency_matrix,
)
from conftest import compare_metrics


# ---------------------------------------------------------------------------
# Local fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def net_glist(net_adjacency, net_n, net_l):
    """Per-layer gt.Graph list (intra-layer edges only) for the current network config."""
    return supra_adjacency_to_network_list(net_adjacency, net_l, net_n)


# ============================================================================
# Correctness — types, shapes, value ranges, invariants, edge cases
# ============================================================================

class TestTopologyCorrectness:
    """Topology functions return sane types/shapes/values and satisfy structural invariants."""

    # --- get_multi_LCC -------------------------------------------------------

    def test_lcc_returns_ndarray(self, net_adjacency, net_n, net_l, net_glist):
        result = topology.get_multi_LCC(net_glist, obj_type="glist")
        assert isinstance(result, np.ndarray)

    def test_lcc_indices_in_range(self, net_adjacency, net_n, net_l, net_glist):
        result = topology.get_multi_LCC(net_glist, obj_type="glist")
        assert np.all(result >= 0) and np.all(result < net_n)

    def test_lcc_length_at_most_n(self, net_adjacency, net_n, net_l, net_glist):
        result = topology.get_multi_LCC(net_glist, obj_type="glist")
        assert len(result) <= net_n

    def test_lcc_glist_tensor_modes_agree(self, net_adjacency, net_n, net_l, net_glist):
        tensor = build_edge_colored_matrices_from_supra_adjacency_matrix(net_adjacency, net_l)
        r_glist = topology.get_multi_LCC(net_glist, obj_type="glist")
        r_tensor = topology.get_multi_LCC(tensor, obj_type="tensor")
        np.testing.assert_array_equal(np.sort(r_glist), np.sort(r_tensor))

    # --- get_multi_LIC -------------------------------------------------------

    def test_lic_returns_ndarray(self, net_adjacency, net_n, net_l, net_glist):
        result = topology.get_multi_LIC(net_glist, obj_type="glist")
        assert isinstance(result, np.ndarray)

    def test_lic_indices_in_range(self, net_adjacency, net_n, net_l, net_glist):
        result = topology.get_multi_LIC(net_glist, obj_type="glist")
        assert np.all(result >= 0) and np.all(result < net_n)

    def test_lic_subset_of_per_layer_lccs(self, net_adjacency, net_n, net_l, net_glist):
        lic = set(topology.get_multi_LIC(net_glist, obj_type="glist"))
        for g in net_glist:
            layer_lcc = set(gt.topology.extract_largest_component(g).get_vertices())
            assert lic <= layer_lcc

    def test_lic_disconnected_layer_shrinks_result(self):
        # 3 nodes, 2 layers: layer 0 has only 0–1; layer 1 has 0–1 and 1–2
        # per-layer LCCs: {0,1} and {0,1,2} → intersection = {0,1}
        g0 = gt.Graph(directed=False)
        g0.add_vertex(3)
        g0.add_edge(g0.vertex(0), g0.vertex(1))
        g1 = gt.Graph(directed=False)
        g1.add_vertex(3)
        g1.add_edge(g1.vertex(0), g1.vertex(1))
        g1.add_edge(g1.vertex(1), g1.vertex(2))
        lic = topology.get_multi_LIC([g0, g1], obj_type="glist")
        assert set(lic) == {0, 1}

    # --- get_multi_LVC -------------------------------------------------------

    def test_lvc_returns_array_or_list(self, net_adjacency, net_n, net_l, net_glist):
        result = topology.get_multi_LVC(net_glist, printt=False)
        assert isinstance(result, (np.ndarray, list))

    def test_lvc_subset_of_lic(self, net_adjacency, net_n, net_l, net_glist):
        lic = set(topology.get_multi_LIC(net_glist, obj_type="glist"))
        lvc = topology.get_multi_LVC(net_glist, printt=False)
        assert set(lvc) <= lic

    def test_lvc_bridge_node_only(self):
        # 3 nodes, 2 layers: layer 0 has 0–1; layer 1 has 1–2
        # per-layer LCCs: {0,1} and {1,2} → LIC = {1} → LVC = {1}
        g0 = gt.Graph(directed=False)
        g0.add_vertex(3)
        g0.add_edge(g0.vertex(0), g0.vertex(1))
        g1 = gt.Graph(directed=False)
        g1.add_vertex(3)
        g1.add_edge(g1.vertex(1), g1.vertex(2))
        lvc = topology.get_multi_LVC([g0, g1], printt=False)
        assert len(lvc) == 1 and 1 in lvc

    # --- get_connected_components --------------------------------------------

    def test_cc_returns_ndarray(self, net_adjacency, net_n, net_l):
        result = topology.get_connected_components(net_adjacency, net_l, net_n)
        assert isinstance(result, np.ndarray)

    def test_cc_shape(self, net_adjacency, net_n, net_l):
        result = topology.get_connected_components(net_adjacency, net_l, net_n)
        assert result.shape == (net_n,)

    def test_cc_labels_non_negative(self, net_adjacency, net_n, net_l):
        result = topology.get_connected_components(net_adjacency, net_l, net_n)
        assert np.all(result >= 0)

    def test_cc_two_components(self):
        # 4 nodes, 2 layers (NL=8); supra-index = layer * 4 + node
        # Component A: physical nodes {0,1} — intra-layer 0 edge 0–1, plus inter-layer
        #              connections 0↔0 and 1↔1 across layers
        # Component B: physical nodes {2,3} — intra-layer 1 edge 2–3 (supra 6–7), plus
        #              inter-layer connections 2↔2 and 3↔3 across layers
        # Inter-layer edges are required so that replicas of the same physical node land
        # in the same supra-component, satisfying the function's consistency check.
        rows = [0, 1, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]
        cols = [1, 0, 7, 6, 4, 0, 5, 1, 6, 2, 7, 3]
        supra = sp.csr_matrix((np.ones(12), (rows, cols)), shape=(8, 8))
        result = topology.get_connected_components(supra, layers=2, nodes=4)
        assert len(np.unique(result)) == 2

    def test_cc_raises_on_ambiguous_components(self):
        # node 0 in layer 0 (supra 0) is connected to node 1 (supra 1): non-trivial component
        # node 0 in layer 1 (supra 4) is connected to node 2 (supra 6): different non-trivial
        rows = [0, 1, 4, 6]
        cols = [1, 0, 6, 4]
        supra = sp.csr_matrix((np.ones(4), (rows, cols)), shape=(8, 8))
        with pytest.raises(ValueError):
            topology.get_connected_components(supra, layers=2, nodes=4)

    # --- get_multi_path_statistics -------------------------------------------

    def test_path_stats_returns_dict(self, net_adjacency, net_n, net_l, network_config):
        if network_config == "random_large":
            pytest.skip("path statistics too slow for random_large")
        result = topology.get_multi_path_statistics(net_adjacency, net_l, net_n)
        assert isinstance(result, dict)

    def test_path_stats_keys(self, net_adjacency, net_n, net_l, network_config):
        if network_config == "random_large":
            pytest.skip("path statistics too slow for random_large")
        result = topology.get_multi_path_statistics(net_adjacency, net_l, net_n)
        assert set(result.keys()) == {"distance_matrix", "avg_path_length", "closeness"}

    def test_path_stats_distance_matrix_shape(self, net_adjacency, net_n, net_l, network_config):
        if network_config == "random_large":
            pytest.skip("path statistics too slow for random_large")
        result = topology.get_multi_path_statistics(net_adjacency, net_l, net_n)
        assert result["distance_matrix"].shape == (net_n, net_n)

    def test_path_stats_diagonal_zero(self, net_adjacency, net_n, net_l, network_config):
        if network_config == "random_large":
            pytest.skip("path statistics too slow for random_large")
        result = topology.get_multi_path_statistics(net_adjacency, net_l, net_n)
        np.testing.assert_allclose(np.diag(result["distance_matrix"]), 0.0)

    def test_path_stats_distances_non_negative(self, net_adjacency, net_n, net_l, network_config):
        if network_config == "random_large":
            pytest.skip("path statistics too slow for random_large")
        result = topology.get_multi_path_statistics(net_adjacency, net_l, net_n)
        assert np.all(result["distance_matrix"] >= 0)

    def test_path_stats_avg_path_length_positive(self, net_adjacency, net_n, net_l, network_config):
        if network_config == "random_large":
            pytest.skip("path statistics too slow for random_large")
        result = topology.get_multi_path_statistics(net_adjacency, net_l, net_n)
        assert result["avg_path_length"] > 0

    def test_path_stats_closeness_length(self, net_adjacency, net_n, net_l, network_config):
        if network_config == "random_large":
            pytest.skip("path statistics too slow for random_large")
        result = topology.get_multi_path_statistics(net_adjacency, net_l, net_n)
        assert len(result["closeness"]) == net_n

    def test_path_stats_closeness_non_negative(self, net_adjacency, net_n, net_l, network_config):
        if network_config == "random_large":
            pytest.skip("path statistics too slow for random_large")
        result = topology.get_multi_path_statistics(net_adjacency, net_l, net_n)
        assert np.all(np.array(result["closeness"]) >= 0)

    # --- get_SP_similarity_matrix --------------------------------------------

    def test_sp_sim_returns_ndarray(self, net_adjacency, net_n, net_l):
        result = topology.get_SP_similarity_matrix(net_adjacency, net_l, net_n)
        assert isinstance(result, np.ndarray)

    def test_sp_sim_shape(self, net_adjacency, net_n, net_l):
        result = topology.get_SP_similarity_matrix(net_adjacency, net_l, net_n)
        assert result.shape == (net_l, net_l)

    def test_sp_sim_diagonal_is_one(self, net_adjacency, net_n, net_l):
        result = topology.get_SP_similarity_matrix(net_adjacency, net_l, net_n)
        np.testing.assert_allclose(np.diag(result), 1.0)

    def test_sp_sim_symmetric(self, net_adjacency, net_n, net_l):
        result = topology.get_SP_similarity_matrix(net_adjacency, net_l, net_n)
        np.testing.assert_array_almost_equal(result, result.T)

    def test_sp_sim_range_in_01(self, net_adjacency, net_n, net_l):
        result = topology.get_SP_similarity_matrix(net_adjacency, net_l, net_n)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)


# ============================================================================
# Reference — comparison against pre-computed muxViz R results
# ============================================================================

class TestTopologyReference:
    """Compare topology results against muxViz R reference.

    Pre-computed results are loaded from tests/data/{config}/muxviz_results.json.
    Tests are skipped for configs without reference data or missing metric keys.
    """

    def test_closeness_vs_muxviz(
        self, net_adjacency, net_n, net_l, net_muxviz_results, network_config
    ):
        if network_config == "random_large":
            pytest.skip("path statistics too slow for random_large")
        if "closeness" not in net_muxviz_results:
            pytest.skip(f"'closeness' not in reference results for '{network_config}'")

        stats = topology.get_multi_path_statistics(net_adjacency, net_l, net_n)
        computed = np.array(stats["closeness"], dtype=np.float64)
        # GetMultiClosenessCentrality returns a named list in R; older JSON entries
        # may have been serialized as {"closeness": [...]} instead of a plain array.
        closeness_raw = net_muxviz_results["closeness"]
        if isinstance(closeness_raw, dict):
            closeness_raw = next(iter(closeness_raw.values()))
        expected = np.asarray(closeness_raw, dtype=np.float64).ravel()

        compare_metrics(
            computed, expected,
            "closeness centrality",
            computed_name="Python",
            expected_name="muxViz R",
            rtol=1e-4,
            atol=1e-4,
        )

    def test_sp_similarity_vs_muxviz(
        self, net_adjacency, net_n, net_l, net_muxviz_results, network_config
    ):
        if "sp_similarity" not in net_muxviz_results:
            pytest.skip(f"'sp_similarity' not in reference results for '{network_config}'")

        computed = topology.get_SP_similarity_matrix(net_adjacency, net_l, net_n)
        r_flat = np.array(net_muxviz_results["sp_similarity"], dtype=np.float64)
        expected = r_flat.reshape(net_l, net_l, order="F")

        compare_metrics(
            computed.ravel(), expected.ravel(),
            "SP similarity matrix",
            computed_name="Python",
            expected_name="muxViz R",
            rtol=1e-4,
            atol=1e-4,
        )

"""
Tests for MuxVizPy.versatility — centrality measures and node-based metrics.

All test classes are parametrized over network configs (toy, random_large,
scalefree_small) via the ``net_*`` fixtures defined in conftest.py.

Classes:
    TestVersatilityCorrectness      — shapes, value ranges, basic invariants
    TestKatzCentralityAgreement     — approx vs exact and legacy comparison
    TestVersatilityBackendComparison — muxvizpy vs hornet backend agreement
    TestVersatilityReference        — comparison against pre-computed muxViz R results
"""

import pytest
import numpy as np
import scipy.sparse as sp
from MuxVizPy import versatility
from conftest import compare_metrics


# ============================================================================
# Correctness — shapes, value ranges, basic invariants
# ============================================================================

class TestVersatilityCorrectness:
    """Functions run correctly and return sane shapes/values."""

    # --- eigenvalue helpers -----------------------------------------------

    def test_get_largest_eigenvalue_returns_tuple(self, net_adjacency, net_nl):
        lam, vec = versatility.get_largest_eigenvalue(net_adjacency)
        assert isinstance(lam, float)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (net_nl,)

    def test_get_largest_eigenvalue_positive(self, net_adjacency):
        lam, _ = versatility.get_largest_eigenvalue(net_adjacency)
        assert lam > 0

    def test_approximate_largest_eigenvalue_returns_tuple(self, net_adjacency, net_nl):
        lam, vec = versatility.approximate_largest_eigenvalue(net_adjacency)
        assert isinstance(lam, float)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (net_nl,)

    # --- block accumulation helpers ---------------------------------------

    def test_is_in_diagonal_block(self):
        assert versatility.is_in_diagonal_block(0, 1, n=10, l=3) is True
        assert versatility.is_in_diagonal_block(0, 10, n=10, l=3) is False
        assert versatility.is_in_diagonal_block(15, 19, n=10, l=3) is True

    def test_accumulate_diagonal_blocks_shape(self, net_adjacency, net_n, net_l):
        result = versatility._accumulate_on_diagonal_blocks(
            net_adjacency, net_n, net_l, is_out_of_diagonal=False,
        )
        assert result.shape == (net_n, net_l)

    def test_accumulate_off_diagonal_blocks_shape(self, net_adjacency, net_n, net_l):
        result = versatility._accumulate_on_diagonal_blocks(
            net_adjacency, net_n, net_l, is_out_of_diagonal=True,
        )
        assert result.shape == (net_n, net_l)

    def test_accumulate_wrong_shape_raises(self, net_n, net_l):
        wrong = sp.eye(5, format="csr")
        with pytest.raises(ValueError, match="does not match"):
            versatility._accumulate_on_diagonal_blocks(wrong, net_n, net_l, is_out_of_diagonal=False)

    # --- aggregate_metrics_over_layers ------------------------------------

    def test_aggregate_sum(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(versatility.aggregate_metrics_over_layers(m, "sum"), [3.0, 7.0])

    def test_aggregate_mean(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(versatility.aggregate_metrics_over_layers(m, "mean"), [1.5, 3.5])

    def test_aggregate_max(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(versatility.aggregate_metrics_over_layers(m, "max"), [2.0, 4.0])

    def test_aggregate_min(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(versatility.aggregate_metrics_over_layers(m, "min"), [1.0, 3.0])

    def test_aggregate_unknown_raises(self):
        m = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="Unknown"):
            versatility.aggregate_metrics_over_layers(m, "median")

    # --- per-layer degree / strength shapes -------------------------------

    def test_compute_indegree_shape(self, net_adjacency, net_n, net_l):
        result = versatility.compute_indegree(net_adjacency, net_n, net_l)
        assert result.shape == (net_n, net_l)

    def test_compute_aggregated_indegree_shape(self, net_adjacency, net_n, net_l):
        result = versatility.compute_aggregated_indegree(net_adjacency, net_n, net_l)
        assert result.shape == (net_n,)

    def test_compute_outdegree_shape(self, net_adjacency, net_n, net_l):
        result = versatility.compute_outdegree(net_adjacency, net_n, net_l)
        assert result.shape == (net_n, net_l)

    def test_compute_aggregated_outdegree_shape(self, net_adjacency, net_n, net_l):
        result = versatility.compute_aggregated_outdegree(net_adjacency, net_n, net_l)
        assert result.shape == (net_n,)

    def test_compute_instrength_shape(self, net_interaction, net_n, net_l):
        result = versatility.compute_instrength(net_interaction, net_n, net_l)
        assert result.shape == (net_n, net_l)

    def test_compute_outstrength_shape(self, net_interaction, net_n, net_l):
        result = versatility.compute_outstrength(net_interaction, net_n, net_l)
        assert result.shape == (net_n, net_l)

    def test_compute_multiindegree_shape(self, net_adjacency, net_n, net_l):
        result = versatility.compute_multiindegree(net_adjacency, net_n, net_l)
        assert result.shape == (net_n, net_l)

    def test_compute_multioutdegree_shape(self, net_adjacency, net_n, net_l):
        result = versatility.compute_multioutdegree(net_adjacency, net_n, net_l)
        assert result.shape == (net_n, net_l)

    def test_compute_multiinstrength_shape(self, net_interaction, net_n, net_l):
        result = versatility.compute_multiinstrength(net_interaction, net_n, net_l)
        assert result.shape == (net_n, net_l)

    def test_compute_multioutstrength_shape(self, net_interaction, net_n, net_l):
        result = versatility.compute_multioutstrength(net_interaction, net_n, net_l)
        assert result.shape == (net_n, net_l)

    # --- degree consistency: aggregated indegreesum = indegree + multiindegree

    def test_indegree_plus_multiindegree_equals_total(self, net_adjacency, net_n, net_l):
        intra = versatility.compute_aggregated_indegree(net_adjacency, net_n, net_l)
        inter = versatility.compute_aggregated_multiindegree(net_adjacency, net_n, net_l)
        total = np.asarray(net_adjacency.sum(axis=0)).ravel()
        total_per_node = total.reshape(net_l, net_n).sum(axis=0)
        np.testing.assert_allclose(intra + inter, total_per_node, atol=1e-10)

    def test_outdegree_plus_multioutdegree_equals_total(self, net_adjacency, net_n, net_l):
        intra = versatility.compute_aggregated_outdegree(net_adjacency, net_n, net_l)
        inter = versatility.compute_aggregated_multioutdegree(net_adjacency, net_n, net_l)
        total = np.asarray(net_adjacency.sum(axis=1)).ravel()
        total_per_node = total.reshape(net_l, net_n).sum(axis=0)
        np.testing.assert_allclose(intra + inter, total_per_node, atol=1e-10)

    # --- centrality shapes and ranges -------------------------------------

    def test_compute_eigenvector_centrality_shape(self, net_adjacency, net_n, net_l):
        ec = versatility.compute_eigenvector_centrality(net_adjacency, net_n, net_l)
        assert ec.shape == (net_n,)
        assert ec.max() == pytest.approx(1.0, abs=1e-6)

    def test_compute_katz_centrality_exact_shape(self, net_interaction, net_n, net_l):
        katz, eigenvalue = versatility.compute_katz_centrality(
            net_interaction, net_n, net_l, solver="direct", return_eigenvalue=True,
        )
        assert katz.shape == (net_n,)
        assert katz.max() == pytest.approx(1.0, abs=1e-6)
        assert isinstance(eigenvalue, float)

    def test_compute_katz_centrality_neumann_shape(self, net_interaction, net_n, net_l):
        np.random.seed(42)
        katz, eigenvalue = versatility.compute_katz_centrality(
            net_interaction, net_n, net_l, solver="neumann",
            maxiter=100000, tol=1e-4,
            return_eigenvalue=True,
        )
        assert katz.shape == (net_n,)
        assert katz.max() == pytest.approx(1.0, abs=1e-6)
        assert isinstance(eigenvalue, float)

    def test_compute_multi_rw_centrality_classical_shape(self, net_adjacency, net_n, net_l):
        rc = versatility.compute_multi_rw_centrality(net_adjacency, net_n, net_l, kind="classical")
        assert rc.shape == (net_n,)
        assert rc.max() == pytest.approx(1.0, abs=1e-6)

    def test_compute_multi_rw_centrality_pagerank_shape(self, net_adjacency, net_n, net_l):
        pr = versatility.compute_multi_rw_centrality(net_adjacency, net_n, net_l, kind="pagerank")
        assert pr.shape == (net_n,)
        assert pr.max() == pytest.approx(1.0, abs=1e-6)

    def test_compute_multipagerank_centrality_shape(self, net_adjacency, net_n, net_l):
        pr = versatility.compute_multipagerank_centrality(net_adjacency, net_n, net_l)
        assert pr.shape == (net_n,)
        assert pr.max() == pytest.approx(1.0, abs=1e-6)

    def test_multipagerank_wrapper_matches_direct(self, net_adjacency, net_n, net_l):
        """compute_multipagerank_centrality is a wrapper and must match compute_multi_rw_centrality(kind='pagerank')."""
        direct = versatility.compute_multi_rw_centrality(net_adjacency, net_n, net_l, kind="pagerank")
        wrapper = versatility.compute_multipagerank_centrality(net_adjacency, net_n, net_l)
        np.testing.assert_array_equal(direct, wrapper)

    def test_compute_multi_rw_centrality_unknown_kind_raises(self, net_adjacency, net_n, net_l):
        with pytest.raises(ValueError, match="Unknown RW kind"):
            versatility.compute_multi_rw_centrality(net_adjacency, net_n, net_l, kind="bogus")

    def test_compute_multi_hub_centrality_shape(self, net_adjacency, net_n, net_l):
        hc = versatility.compute_multi_hub_centrality(net_adjacency, net_n, net_l)
        assert hc.shape == (net_n,)

    def test_compute_multi_authority_centrality_shape(self, net_interaction, net_n, net_l):
        ac = versatility.compute_multi_authority_centrality(net_interaction, net_n, net_l)
        assert ac.shape == (net_n,)

    # --- public API shapes ------------------------------------------------

    def test_get_multi_degree_shape(self, net_adjacency, net_n, net_l):
        deg = versatility.get_multi_degree(net_adjacency, net_l, net_n)
        assert deg.shape == (net_n,)

    def test_compute_eigenvector_centrality_public_shape(self, net_adjacency, net_n, net_l):
        ec = versatility.compute_eigenvector_centrality(net_adjacency, net_n, net_l)
        assert ec.shape == (net_n,)


# ============================================================================
# Katz centrality agreement — approx vs exact and method dispatch
# ============================================================================

class TestKatzCentralityAgreement:
    """Approx vs exact agreement and method-dispatch guard for Katz centrality."""

    def test_katz_invalid_solver_raises(self, net_interaction, net_n, net_l):
        with pytest.raises(ValueError, match="Unknown solver"):
            versatility.compute_katz_centrality(
                net_interaction, net_n, net_l, solver="invalid_solver",
            )

    def test_katz_neumann_warns_when_not_converged(self, net_interaction, net_n, net_l):
        with pytest.warns(UserWarning, match="did not converge"):
            versatility.compute_katz_centrality(
                net_interaction, net_n, net_l, solver="neumann",
                maxiter=1, tol=1e-10,
            )

    def test_katz_gmres_matches_direct(self, net_interaction, net_n, net_l):
        exact = versatility.compute_katz_centrality(
            net_interaction, net_n, net_l, solver="direct",
        )
        gmres_result = versatility.compute_katz_centrality(
            net_interaction, net_n, net_l, solver="gmres",
            maxiter=1000, tol=1e-8,
        )
        np.testing.assert_allclose(gmres_result, exact, atol=1e-3)

    def test_katz_bicgstab_matches_direct(self, net_interaction, net_n, net_l):
        exact = versatility.compute_katz_centrality(
            net_interaction, net_n, net_l, solver="direct",
        )
        bicgstab_result = versatility.compute_katz_centrality(
            net_interaction, net_n, net_l, solver="bicgstab",
            maxiter=1000, tol=1e-8,
        )
        np.testing.assert_allclose(bicgstab_result, exact, atol=5e-4)


# ============================================================================
# Backend comparison — muxvizpy vs hornet
# ============================================================================

class TestVersatilityBackendComparison:
    """Both backends produce consistent results on the same input.

    Known semantic differences between backends:
    - **degree**: muxvizpy aggregates via the aggregate network (all edges
      including inter-layer), hornet uses intra-layer out-degree only.
    """

    def test_degree_backends_both_run(self, net_adjacency, net_n, net_l):
        """Degree backends have different semantics: muxvizpy collapses into
        an aggregate network (parallel edges merged), hornet sums per-layer
        out-degree (keeps parallel edges). Verify both run and produce valid
        shapes and non-negative values."""
        mv = versatility.get_multi_degree(net_adjacency, net_l, net_n, backend="muxvizpy")
        hn = versatility.get_multi_degree(net_adjacency, net_l, net_n, backend="hornet")
        assert mv.shape == (net_n,)
        assert hn.shape == (net_n,)
        assert np.all(mv >= 0)
        assert np.all(hn >= 0)

# ============================================================================
# Reference — comparison against pre-computed muxViz R results
# ============================================================================

class TestVersatilityReference:
    """Compare results against muxViz R reference.

    Pre-computed results are loaded from tests/data/{config}/muxviz_results.json.
    Tests are skipped for configs without reference data.
    """

    # --- centrality reference tests ---------------------------------------

    def test_katz_exact_vs_muxviz(self, net_interaction, net_n, net_l, net_muxviz_results):
        computed = versatility.compute_katz_centrality(
            net_interaction, net_n, net_l, solver="direct", return_eigenvalue=False,
        )
        # muxviz reference is rounded to 4 dp → need atol >= 5e-4
        compare_metrics(computed, net_muxviz_results["katz"], "Katz exact (vs muxViz R)",
                        rtol=5e-4, atol=5e-4)

    def test_katz_neumann_vs_muxviz(self, net_interaction, net_n, net_l, net_muxviz_results):
        np.random.seed(42)
        computed = versatility.compute_katz_centrality(
            net_interaction, net_n, net_l, solver="neumann",
            maxiter=100000, tol=1e-4,
            return_eigenvalue=False,
        )
        compare_metrics(computed, net_muxviz_results["katz"], "Katz Neumann (vs muxViz R)",
                         rtol=0.05, atol=0.05)

    def test_pagerank_vs_muxviz(self, net_interaction, net_n, net_l, net_muxviz_results):
        computed = versatility.compute_multipagerank_centrality(net_interaction, net_n, net_l)
        compare_metrics(computed, net_muxviz_results["pagerank"], "PageRank (vs muxViz R)")

    def test_hub_exact_vs_muxviz(self, net_adjacency, net_n, net_l, net_muxviz_results):
        computed = versatility.compute_multi_hub_centrality(net_adjacency, net_n, net_l)
        compare_metrics(computed, net_muxviz_results["hub"], "Hub (vs muxViz R)")

    def test_hub_approx_vs_muxviz(self, net_adjacency, net_n, net_l, net_muxviz_results):
        np.random.seed(42)
        computed = versatility.compute_multi_hub_centrality(
            net_adjacency, net_n, net_l, approx=True,
            approx_args={"maxiter": 10000, "tol": 1e-10},
        )
        compare_metrics(computed, net_muxviz_results["hub"], "Hub approx (vs muxViz R)",
                        rtol=0.05, atol=0.05)

    def test_auth_exact_vs_muxviz(self, net_interaction, net_n, net_l, net_muxviz_results):
        computed = versatility.compute_multi_authority_centrality(net_interaction, net_n, net_l)
        compare_metrics(computed, net_muxviz_results["auth"], "Authority (vs muxViz R)")

    def test_auth_approx_vs_muxviz(self, net_interaction, net_n, net_l, net_muxviz_results):
        np.random.seed(42)
        computed = versatility.compute_multi_authority_centrality(
            net_interaction, net_n, net_l, approx=True,
            approx_args={"maxiter": 10000, "tol": 1e-10},
        )
        compare_metrics(computed, net_muxviz_results["auth"], "Authority approx (vs muxViz R)",
                        rtol=0.05, atol=0.05)

    def test_eigenvector_vs_muxviz(self, net_interaction, net_n, net_l, net_muxviz_results):
        computed = versatility.compute_eigenvector_centrality(net_interaction, net_n, net_l)
        compare_metrics(computed, net_muxviz_results["eigenvector"], "Eigenvector (vs muxViz R)",
                        rtol=5e-4, atol=5e-4)

    # --- degree / strength reference tests --------------------------------

    def test_indegree_sum_vs_muxviz(self, net_adjacency, net_n, net_l, net_muxviz_results):
        computed = versatility.compute_aggregated_indegree(net_adjacency, net_n, net_l)
        compare_metrics(computed, net_muxviz_results["indegree"], "Indegree (vs muxViz R)")

    def test_outdegree_sum_vs_muxviz(self, net_adjacency, net_n, net_l, net_muxviz_results):
        computed = versatility.compute_aggregated_outdegree(net_adjacency, net_n, net_l)
        compare_metrics(computed, net_muxviz_results["outdegree"], "Outdegree (vs muxViz R)")

    def test_instrength_sum_vs_muxviz(self, net_interaction, net_n, net_l, net_muxviz_results):
        computed = versatility.compute_aggregated_instrength(net_interaction, net_n, net_l)
        compare_metrics(computed, net_muxviz_results["instrength"], "Instrength (vs muxViz R)")

    def test_outstrength_sum_vs_muxviz(self, net_interaction, net_n, net_l, net_muxviz_results):
        computed = versatility.compute_aggregated_outstrength(net_interaction, net_n, net_l)
        compare_metrics(computed, net_muxviz_results["outstrength"], "Outstrength (vs muxViz R)")

    # --- multi-degree derived tests (indegreesum - indegree = multiindegree)

    def test_multiindegree_vs_muxviz_derived(self, net_adjacency, net_n, net_l, net_muxviz_results):
        if "indegreesum" not in net_muxviz_results:
            pytest.skip("indegreesum not in reference results")
        computed = versatility.compute_aggregated_multiindegree(net_adjacency, net_n, net_l)
        expected = np.array(net_muxviz_results["indegreesum"]) - np.array(net_muxviz_results["indegree"])
        compare_metrics(computed, expected, "MultiIndegree (vs muxViz derived)")

    def test_multioutdegree_vs_muxviz_derived(self, net_adjacency, net_n, net_l, net_muxviz_results):
        if "outdegreesum" not in net_muxviz_results:
            pytest.skip("outdegreesum not in reference results")
        computed = versatility.compute_aggregated_multioutdegree(net_adjacency, net_n, net_l)
        expected = np.array(net_muxviz_results["outdegreesum"]) - np.array(net_muxviz_results["outdegree"])
        compare_metrics(computed, expected, "MultiOutdegree (vs muxViz derived)")

    def test_multiinstrength_vs_muxviz_derived(self, net_interaction, net_n, net_l, net_muxviz_results):
        if "instrengthsum" not in net_muxviz_results:
            pytest.skip("instrengthsum not in reference results")
        computed = versatility.compute_aggregated_multiinstrength(net_interaction, net_n, net_l)
        expected = np.array(net_muxviz_results["instrengthsum"]) - np.array(net_muxviz_results["instrength"])
        compare_metrics(computed, expected, "MultiInstrength (vs muxViz derived)")

    def test_multioutstrength_vs_muxviz_derived(self, net_interaction, net_n, net_l, net_muxviz_results):
        if "outstrengthsum" not in net_muxviz_results:
            pytest.skip("outstrengthsum not in reference results")
        computed = versatility.compute_aggregated_multioutstrength(net_interaction, net_n, net_l)
        expected = np.array(net_muxviz_results["outstrengthsum"]) - np.array(net_muxviz_results["outstrength"])
        compare_metrics(computed, expected, "MultiOutstrength (vs muxViz derived)")

    # --- multi degree reference tests (R GetMultiDegree = indegree + outdegree)

    def test_multi_degree_vs_muxviz(self, net_adjacency, net_n, net_l, net_muxviz_results):
        """R GetMultiDegree = GetMultiInDegree + GetMultiOutDegree (directed)."""
        expected = np.array(net_muxviz_results["indegree"]) + np.array(net_muxviz_results["outdegree"])
        computed = versatility.compute_multi_degree(net_adjacency, net_n, net_l, is_directed=True)
        compare_metrics(computed, expected, "MultiDegree (vs muxViz R derived)")

    def test_multi_degree_hornet_vs_muxviz(self, net_adjacency, net_n, net_l, net_muxviz_results):
        """get_multi_degree hornet backend matches R GetMultiDegree."""
        expected = np.array(net_muxviz_results["indegree"]) + np.array(net_muxviz_results["outdegree"])
        computed = versatility.get_multi_degree(net_adjacency, net_l, net_n, backend="hornet")
        compare_metrics(computed, expected, "MultiDegree hornet (vs muxViz R derived)")
"""
Tests for MuxVizPy.utils.parsing — functions that will replace build.py.

Classes:
    TestInterlayerCouplingMatrix    — scipy interlayer coupling (ordered/categorical/temporal)
    TestInterlayerCouplingTensor    — torch 4D interlayer coupling
    TestSupraFromEdgeColored        — supra-adjacency from edge-colored matrices
    TestEdgeColoredFromSupra        — extract diagonal blocks from supra-adjacency
    TestTensorGraphToolRoundtrip    — tensor ↔ graph-tool graph list
    TestAggregateNetwork            — aggregate network (sum/max/min)
    TestLaplacianFromGraphList      — per-layer Laplacians from graph-tool
    TestLaplacianFromTensor         — 4D torch Laplacian
"""

import pytest
import numpy as np
import scipy.sparse as sp
import torch
import graph_tool as gt
import graph_tool.spectral

from MuxVizPy.utils import parsing
from conftest import SAMPLE_EDGES, SAMPLE_N_NODES, SAMPLE_N_LAYERS

N = SAMPLE_N_NODES
L = SAMPLE_N_LAYERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_intra_networks():
    """Build per-layer scipy adjacency matrices from SAMPLE_EDGES."""
    mats = [sp.lil_matrix((N, N)) for _ in range(L)]
    for nf, lf, nt, lt, w in SAMPLE_EDGES:
        if lf == lt:  # intra-layer only
            mats[lf][nf, nt] = w
    return [m.tocsr() for m in mats]


def _make_sample_graph_list():
    """Build per-layer graph-tool graphs from SAMPLE_EDGES (intra only)."""
    mats = _make_sample_intra_networks()
    graphs = []
    for adj in mats:
        g = gt.Graph(directed=True)
        g.add_vertex(N)
        coo = sp.coo_matrix(adj)
        weight = g.new_edge_property("double")
        for i, j, v in zip(coo.row, coo.col, coo.data):
            e = g.add_edge(i, j)
            weight[e] = v
        g.ep["weight"] = weight
        graphs.append(g)
    return graphs


# ============================================================================
# Interlayer coupling matrix (scipy)
# ============================================================================

class TestInterlayerCouplingMatrix:

    @pytest.mark.parametrize("kind", ["ordered", "categorical", "temporal"])
    def test_shape(self, kind):
        mat = parsing.build_interlayer_coupling_matrix(L, omega=1.0, kind=kind)
        assert mat.shape == (L, L)

    @pytest.mark.parametrize("kind", ["ordered", "categorical", "temporal"])
    def test_zero_diagonal(self, kind):
        """No self-layer coupling."""
        mat = parsing.build_interlayer_coupling_matrix(L, omega=1.0, kind=kind)
        np.testing.assert_array_equal(mat.diagonal(), 0.0)

    def test_ordered_is_symmetric(self):
        mat = parsing.build_interlayer_coupling_matrix(4, omega=1.5, kind="ordered")
        diff = (mat - mat.T).toarray()
        np.testing.assert_allclose(diff, 0.0, atol=1e-15)

    def test_ordered_tridiagonal(self):
        """Ordered coupling should only have entries on super/sub-diagonals."""
        mat = parsing.build_interlayer_coupling_matrix(4, omega=2.0, kind="ordered").toarray()
        for i in range(4):
            for j in range(4):
                if abs(i - j) == 1:
                    assert mat[i, j] == 2.0
                else:
                    assert mat[i, j] == 0.0

    def test_categorical_is_symmetric(self):
        mat = parsing.build_interlayer_coupling_matrix(3, omega=1.0, kind="categorical")
        diff = (mat - mat.T).toarray()
        np.testing.assert_allclose(diff, 0.0, atol=1e-15)

    def test_categorical_all_to_all(self):
        """Every off-diagonal entry should equal omega."""
        omega = 0.7
        mat = parsing.build_interlayer_coupling_matrix(3, omega=omega, kind="categorical").toarray()
        expected = (np.ones((3, 3)) - np.eye(3)) * omega
        np.testing.assert_allclose(mat, expected)

    def test_temporal_is_directed(self):
        """Temporal coupling: only i → i+1, not i+1 → i."""
        mat = parsing.build_interlayer_coupling_matrix(3, omega=1.0, kind="temporal").toarray()
        # Upper diagonal should be 1
        assert mat[0, 1] == 1.0
        assert mat[1, 2] == 1.0
        # Lower diagonal should be 0
        assert mat[1, 0] == 0.0
        assert mat[2, 1] == 0.0

    def test_omega_scaling(self):
        omega = 3.14
        mat = parsing.build_interlayer_coupling_matrix(3, omega=omega, kind="ordered").toarray()
        assert mat[0, 1] == pytest.approx(omega)

    def test_single_layer_returns_empty(self):
        mat = parsing.build_interlayer_coupling_matrix(1, omega=1.0, kind="ordered")
        assert mat.shape == (1, 1)
        assert mat.nnz == 0

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            parsing.build_interlayer_coupling_matrix(3, omega=1.0, kind="bogus")


# ============================================================================
# Interlayer coupling tensor (torch)
# ============================================================================

class TestInterlayerCouplingTensor:

    def _make_empty_intra_tensor(self, n, l):
        """Create an empty sparse tensor with the right shape."""
        indices = torch.empty((4, 0), dtype=torch.long)
        values = torch.empty(0, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, size=(n, l, n, l))

    @pytest.mark.parametrize("kind", ["ordered", "categorical", "temporal"])
    def test_shape(self, kind):
        t = self._make_empty_intra_tensor(N, L)
        coupling = parsing.build_interlayer_coupling_from_tensor(t, omega=1.0, kind=kind)
        assert coupling.shape == (N, L, N, L)

    @pytest.mark.parametrize("kind", ["ordered", "categorical", "temporal"])
    def test_sparse(self, kind):
        t = self._make_empty_intra_tensor(N, L)
        coupling = parsing.build_interlayer_coupling_from_tensor(t, omega=1.0, kind=kind)
        assert coupling.is_sparse

    def test_ordered_connects_same_node_across_layers(self):
        """Ordered coupling: node j in layer i connects to node j in layer i±1."""
        n, l = 3, 4
        t = self._make_empty_intra_tensor(n, l)
        coupling = parsing.build_interlayer_coupling_from_tensor(t, omega=1.0, kind="ordered")
        dense = coupling.to_dense()
        # node 0, layer 0 → node 0, layer 1
        assert dense[0, 0, 0, 1] == 1.0
        # node 0, layer 1 → node 0, layer 0 (symmetric)
        assert dense[0, 1, 0, 0] == 1.0
        # no coupling across 2 layers
        assert dense[0, 0, 0, 2] == 0.0

    def test_categorical_all_layer_pairs(self):
        n, l = 2, 3
        t = self._make_empty_intra_tensor(n, l)
        coupling = parsing.build_interlayer_coupling_from_tensor(t, omega=1.0, kind="categorical")
        dense = coupling.to_dense()
        for node in range(n):
            for li in range(l):
                for lj in range(l):
                    if li != lj:
                        assert dense[node, li, node, lj] == 1.0
                    else:
                        assert dense[node, li, node, lj] == 0.0

    def test_temporal_directed(self):
        n, l = 2, 3
        t = self._make_empty_intra_tensor(n, l)
        coupling = parsing.build_interlayer_coupling_from_tensor(t, omega=1.0, kind="temporal")
        dense = coupling.to_dense()
        # forward: layer 0 → layer 1
        assert dense[0, 0, 0, 1] == 1.0
        # no backward
        assert dense[0, 1, 0, 0] == 0.0

    def test_unknown_kind_raises(self):
        t = self._make_empty_intra_tensor(N, L)
        with pytest.raises(NotImplementedError):
            parsing.build_interlayer_coupling_from_tensor(t, omega=1.0, kind="bogus")


# ============================================================================
# Supra-adjacency from edge-colored matrices
# ============================================================================

class TestSupraFromEdgeColored:

    def test_shape(self):
        intra = _make_sample_intra_networks()
        coupling = parsing.build_interlayer_coupling_matrix(L, omega=1.0, kind="categorical")
        supra = parsing.build_supra_adjacency_matrix_from_edge_colored_matrices(intra, coupling, N)
        assert supra.shape == (N * L, N * L)

    def test_diagonal_blocks_match_intra(self):
        """The block-diagonal part should equal the original intra-layer matrices."""
        intra = _make_sample_intra_networks()
        coupling = parsing.build_interlayer_coupling_matrix(L, omega=0.0, kind="categorical")
        supra = parsing.build_supra_adjacency_matrix_from_edge_colored_matrices(intra, coupling, N)
        for i in range(L):
            block = supra[i * N:(i + 1) * N, i * N:(i + 1) * N].toarray()
            np.testing.assert_allclose(block, intra[i].toarray())

    def test_interlayer_coupling_present(self):
        """With omega > 0, off-diagonal blocks should have nonzero entries."""
        intra = _make_sample_intra_networks()
        coupling = parsing.build_interlayer_coupling_matrix(L, omega=1.0, kind="categorical")
        supra = parsing.build_supra_adjacency_matrix_from_edge_colored_matrices(intra, coupling, N)
        # Off-diagonal block (0,1) should have identity * omega entries
        off_block = supra[0:N, N:2 * N].toarray()
        np.testing.assert_allclose(off_block, np.eye(N))

    def test_wrong_shape_raises(self):
        bad = [sp.eye(3, format="csr")]  # wrong size
        coupling = parsing.build_interlayer_coupling_matrix(1, omega=0.0, kind="ordered")
        with pytest.raises(AssertionError):
            parsing.build_supra_adjacency_matrix_from_edge_colored_matrices(bad, coupling, N)


# ============================================================================
# Edge-colored matrices from supra-adjacency
# ============================================================================

class TestEdgeColoredFromSupra:

    def test_roundtrip(self):
        """intra → supra → extract intra recovers original."""
        intra = _make_sample_intra_networks()
        coupling = parsing.build_interlayer_coupling_matrix(L, omega=0.0, kind="ordered")
        supra = parsing.build_supra_adjacency_matrix_from_edge_colored_matrices(intra, coupling, N)
        recovered = parsing.build_edge_colored_matrices_from_supra_adjacency_matrix(supra, L)
        assert len(recovered) == L
        for i in range(L):
            np.testing.assert_allclose(
                recovered[i].toarray(), intra[i].toarray(),
                err_msg=f"Layer {i} mismatch",
            )

    def test_count(self):
        intra = _make_sample_intra_networks()
        coupling = parsing.build_interlayer_coupling_matrix(L, omega=0.0, kind="ordered")
        supra = parsing.build_supra_adjacency_matrix_from_edge_colored_matrices(intra, coupling, N)
        recovered = parsing.build_edge_colored_matrices_from_supra_adjacency_matrix(supra, L)
        assert len(recovered) == L

    def test_shapes(self):
        intra = _make_sample_intra_networks()
        coupling = parsing.build_interlayer_coupling_matrix(L, omega=0.0, kind="ordered")
        supra = parsing.build_supra_adjacency_matrix_from_edge_colored_matrices(intra, coupling, N)
        recovered = parsing.build_edge_colored_matrices_from_supra_adjacency_matrix(supra, L)
        for mat in recovered:
            assert mat.shape == (N, N)


# ============================================================================
# Tensor ↔ graph-tool round-trip
# ============================================================================

class TestTensorGraphToolRoundtrip:

    def test_tensor_from_graphs_shape(self):
        graphs = _make_sample_graph_list()
        t = parsing.build_tensor_from_list_of_graphs(graphs)
        assert t.shape == (N, L, N, L)

    def test_tensor_from_graphs_sparse(self):
        graphs = _make_sample_graph_list()
        t = parsing.build_tensor_from_list_of_graphs(graphs)
        assert t.is_sparse

    def test_tensor_from_graphs_intra_only(self):
        """All edges should be intra-layer (layer_from == layer_to)."""
        graphs = _make_sample_graph_list()
        t = parsing.build_tensor_from_list_of_graphs(graphs)
        t = t.coalesce()
        indices = t.indices()
        # indices[1] = layer_from, indices[3] = layer_to
        np.testing.assert_array_equal(
            indices[1].numpy(), indices[3].numpy(),
            err_msg="Graph list should produce only intra-layer edges",
        )

    def test_graphs_from_tensor_count(self, sample_tensor):
        graphs = parsing.build_list_of_graphs_from_tensor(sample_tensor)
        assert len(graphs) == L

    def test_graphs_from_tensor_node_count(self, sample_tensor):
        graphs = parsing.build_list_of_graphs_from_tensor(sample_tensor)
        for g in graphs:
            assert g.num_vertices() == N

    def test_roundtrip_graphs_tensor_graphs(self):
        """graphs → tensor → graphs preserves adjacency structure."""
        graphs_orig = _make_sample_graph_list()
        t = parsing.build_tensor_from_list_of_graphs(graphs_orig)
        graphs_back = parsing.build_list_of_graphs_from_tensor(t)

        for i in range(L):
            adj_orig = gt.spectral.adjacency(graphs_orig[i]).toarray()
            adj_back = gt.spectral.adjacency(graphs_back[i]).toarray()
            np.testing.assert_allclose(
                adj_back, adj_orig, atol=1e-10,
                err_msg=f"Layer {i} adjacency mismatch after roundtrip",
            )

    def test_empty_graphs_raises(self):
        with pytest.raises(ValueError):
            parsing.build_tensor_from_list_of_graphs([])

    def test_mismatched_node_count_raises(self):
        g1 = gt.Graph(directed=True)
        g1.add_vertex(3)
        g2 = gt.Graph(directed=True)
        g2.add_vertex(5)
        with pytest.raises(ValueError):
            parsing.build_tensor_from_list_of_graphs([g1, g2])


# ============================================================================
# Aggregate network from tensor
# ============================================================================

class TestAggregateNetwork:

    def test_sum_shape(self, sample_tensor):
        agg = parsing.build_aggregate_network_from_tensor(sample_tensor, kind="sum")
        assert agg.shape == (N, N)

    def test_sum_values(self, sample_tensor):
        """Sum aggregation: edges present in both layers should have weight = sum of weights."""
        agg = parsing.build_aggregate_network_from_tensor(sample_tensor, kind="sum")
        dense = agg.toarray()
        # Edge 0→1 exists in layer 0 (w=1) and layer 1 (w=1) → sum = 2
        assert dense[0, 1] == pytest.approx(2.0)
        # Edge 2→3 exists in layer 0 (w=3) and layer 1 (w=2) → sum = 5
        assert dense[2, 3] == pytest.approx(5.0)

    def test_max_values(self, sample_tensor):
        agg = parsing.build_aggregate_network_from_tensor(sample_tensor, kind="max")
        dense = agg.toarray()
        # Edge 2→3: max(3, 2) = 3
        assert dense[2, 3] == pytest.approx(3.0)

    def test_min_values(self, sample_tensor):
        agg = parsing.build_aggregate_network_from_tensor(sample_tensor, kind="min")
        dense = agg.toarray()
        # Edge 2→3: min(3, 2) = 2
        assert dense[2, 3] == pytest.approx(2.0)

    def test_sum_includes_interlayer(self, sample_tensor):
        """Inter-layer edges (0,L0)→(0,L1) should appear in aggregate as node 0→0."""
        agg = parsing.build_aggregate_network_from_tensor(sample_tensor, kind="sum")
        dense = agg.toarray()
        # Inter-layer edge 0→0 (w=1) exists
        assert dense[0, 0] == pytest.approx(1.0)

    def test_unknown_kind_raises(self, sample_tensor):
        with pytest.raises(ValueError):
            parsing.build_aggregate_network_from_tensor(sample_tensor, kind="bogus")


# ============================================================================
# Laplacian from graph list
# ============================================================================

class TestLaplacianFromGraphList:

    def test_count(self):
        graphs = _make_sample_graph_list()
        laps = parsing.get_laplacian_from_list_of_graphs(graphs)
        assert len(laps) == L

    def test_shapes(self):
        graphs = _make_sample_graph_list()
        laps = parsing.get_laplacian_from_list_of_graphs(graphs)
        for lap in laps:
            assert lap.shape == (N, N)

    def test_row_sums_zero(self):
        """Each Laplacian should have rows summing to zero."""
        graphs = _make_sample_graph_list()
        laps = parsing.get_laplacian_from_list_of_graphs(graphs)
        for i, lap in enumerate(laps):
            row_sums = np.asarray(lap.sum(axis=1)).ravel()
            np.testing.assert_allclose(
                row_sums, 0.0, atol=1e-10,
                err_msg=f"Layer {i} Laplacian rows don't sum to 0",
            )

    def test_diagonal_equals_degree(self):
        """Diagonal of L should equal the degree from gt.spectral.adjacency."""
        graphs = _make_sample_graph_list()
        laps = parsing.get_laplacian_from_list_of_graphs(graphs)
        for i in range(L):
            adj = gt.spectral.adjacency(graphs[i])
            degree = np.asarray(adj.sum(axis=1)).ravel()
            np.testing.assert_allclose(
                laps[i].diagonal(), degree, atol=1e-10,
                err_msg=f"Layer {i} Laplacian diagonal != degree",
            )


# ============================================================================
# Laplacian from tensor (4D torch)
# ============================================================================

class TestLaplacianFromTensor:

    def test_shape(self, sample_tensor):
        lap = parsing.build_laplacian_from_tensor(sample_tensor)
        assert lap.shape == (N, L, N, L)

    def test_sparse(self, sample_tensor):
        lap = parsing.build_laplacian_from_tensor(sample_tensor)
        assert lap.is_sparse

    def test_row_sums_zero(self, sample_tensor):
        """For each (node, layer), sum over all (node', layer') should be 0."""
        lap = parsing.build_laplacian_from_tensor(sample_tensor)
        dense = lap.to_dense()
        # Sum over the last two dimensions (node_to, layer_to)
        row_sums = dense.sum(dim=(2, 3))
        np.testing.assert_allclose(
            row_sums.numpy(), 0.0, atol=1e-6,
            err_msg="Tensor Laplacian (node,layer) rows don't sum to 0",
        )

    def test_diagonal_nonnegative(self, sample_tensor):
        """Diagonal entries L[i,k,i,k] should be >= 0 (they are degrees)."""
        lap = parsing.build_laplacian_from_tensor(sample_tensor)
        dense = lap.to_dense()
        for i in range(N):
            for k in range(L):
                assert dense[i, k, i, k] >= 0.0

    def test_off_diagonal_nonpositive(self, sample_tensor):
        """Off-diagonal entries should be <= 0 (they are -weight)."""
        lap = parsing.build_laplacian_from_tensor(sample_tensor)
        dense = lap.to_dense()
        for i in range(N):
            for ki in range(L):
                for j in range(N):
                    for kj in range(L):
                        if i != j or ki != kj:
                            assert dense[i, ki, j, kj] <= 0.0, (
                                f"L[{i},{ki},{j},{kj}] = {dense[i,ki,j,kj]} should be <= 0"
                            )

    def test_non_sparse_raises(self):
        dense = torch.zeros(3, 2, 3, 2)
        with pytest.raises(NotImplementedError):
            parsing.build_laplacian_from_tensor(dense)

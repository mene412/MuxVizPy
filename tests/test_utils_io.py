"""
Tests for MuxVizPy.utils.io — edgelist reading and writing.

Classes:
    TestIoCorrectness  — each reader/writer works as documented
    TestIoRoundTrips   — write → read preserves data
"""

import pytest
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from MuxVizPy.utils import io as io_utils
from MuxVizPy.utils import parsing
from conftest import SAMPLE_EDGES, SAMPLE_N_NODES, SAMPLE_N_LAYERS


# ============================================================================
# Correctness
# ============================================================================

class TestIoCorrectness:
    """Each I/O function returns the right type, shape, and values."""

    def test_read_edgelist_as_dataframe(self, sample_edgelist_file):
        df = io_utils.read_edgelist_as_dataframe(sample_edgelist_file)

        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"node.from", "layer.from", "node.to", "layer.to", "weight"}
        assert len(df) == len(SAMPLE_EDGES)

    def test_read_edgelist_as_tensor(self, sample_edgelist_file):
        t = io_utils.read_edgelist_as_tensor(sample_edgelist_file)

        assert t.is_sparse
        assert t.shape == (SAMPLE_N_NODES, SAMPLE_N_LAYERS, SAMPLE_N_NODES, SAMPLE_N_LAYERS)
        assert t._nnz() == len(SAMPLE_EDGES)

    def test_read_single_layer_edgelist_as_tensor(self, sample_single_layer_file):
        t = io_utils.read_single_layer_edgelist_as_tensor(sample_single_layer_file)

        assert t.is_sparse
        assert t.ndim == 2
        # node ids are 0..3 → size should be (4, 4)
        assert t.shape[0] == t.shape[1]
        assert t._nnz() == 4  # 4 edges in single-layer sample

    def test_read_edgelist_as_supraadjacencymatrix(self, sample_edgelist_file):
        adj, n, l = io_utils.read_edgelist_as_supraadjacencymatrix(sample_edgelist_file)
        NL = n * l

        assert sp.issparse(adj)
        assert adj.shape == (NL, NL)
        assert n == SAMPLE_N_NODES
        assert l == SAMPLE_N_LAYERS
        # Binary: all nonzero values should be 1
        assert np.all(adj.data == 1.0)

    def test_read_edgelist_as_suprainteractionmatrix(self, sample_edgelist_file):
        inter, n, l = io_utils.read_edgelist_as_suprainteractionmatrix(sample_edgelist_file)
        NL = n * l

        assert sp.issparse(inter)
        assert inter.shape == (NL, NL)
        assert n == SAMPLE_N_NODES
        assert l == SAMPLE_N_LAYERS
        # Weighted: should contain values > 1
        assert inter.data.max() > 1.0

    def test_adjacency_vs_interaction_when_binary(self, tmp_path):
        """When all weights are 1, adjacency and interaction matrices should match."""
        binary_edges = [(n_f, l_f, n_t, l_t, 1.0) for n_f, l_f, n_t, l_t, _ in SAMPLE_EDGES]
        df = pl.DataFrame(
            binary_edges,
            schema=["node.from", "layer.from", "node.to", "layer.to", "weight"],
            orient="row",
        )
        path = tmp_path / "binary_edges.csv"
        df.write_csv(str(path))

        adj, _, _ = io_utils.read_edgelist_as_supraadjacencymatrix(path)
        inter, _, _ = io_utils.read_edgelist_as_suprainteractionmatrix(path)

        diff = (adj - inter).tocsr()
        assert diff.nnz == 0, "Adjacency and interaction should match for binary edges"

    @pytest.mark.parametrize("reader", [
        io_utils.read_edgelist_as_dataframe,
        io_utils.read_edgelist_as_tensor,
        io_utils.read_edgelist_as_supraadjacencymatrix,
        io_utils.read_edgelist_as_suprainteractionmatrix,
    ])
    def test_file_not_found(self, reader):
        with pytest.raises(FileNotFoundError):
            reader("/nonexistent/path.csv")

    def test_empty_dataframe_write_raises(self, tmp_path):
        empty_df = pl.DataFrame(
            schema=["node.from", "layer.from", "node.to", "layer.to", "weight"]
        )
        with pytest.raises(ValueError):
            io_utils.write_edgelist_from_dataframe(empty_df, tmp_path / "out.csv")


# ============================================================================
# Round-trips
# ============================================================================

class TestIoRoundTrips:
    """Write then read back — data must be preserved."""

    def test_dataframe_roundtrip(self, sample_dataframe, tmp_path):
        path = tmp_path / "rt.csv"
        io_utils.write_edgelist_from_dataframe(sample_dataframe, path)
        result = io_utils.read_edgelist_as_dataframe(path)

        assert result.shape == sample_dataframe.shape
        # Sort both for stable comparison
        orig = sample_dataframe.sort(by=sample_dataframe.columns)
        got = result.sort(by=result.columns)
        assert orig.equals(got)

    def test_tensor_roundtrip(self, sample_tensor, tmp_path):
        path = tmp_path / "rt_tensor.csv"
        io_utils.write_edgelist_from_tensor(sample_tensor, path)
        result = io_utils.read_edgelist_as_tensor(path)

        assert result.shape == sample_tensor.shape
        # Compare dense representations
        np.testing.assert_allclose(
            result.to_dense().numpy(),
            sample_tensor.to_dense().numpy(),
            atol=1e-6,
        )

    def test_read_consistency_tensor_vs_matrix(self, sample_edgelist_file):
        """Reading as tensor then converting should match reading as matrix directly."""
        tensor = io_utils.read_edgelist_as_tensor(sample_edgelist_file)
        adj_from_tensor = parsing.build_supra_adjacency_matrix_from_tensor(tensor)

        adj_direct, _, _ = io_utils.read_edgelist_as_supraadjacencymatrix(sample_edgelist_file)

        diff = (adj_from_tensor - adj_direct).tocsr()
        assert diff.nnz == 0, "Tensor-derived adjacency should match direct read"
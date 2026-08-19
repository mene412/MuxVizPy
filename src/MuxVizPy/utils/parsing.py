from __future__ import annotations

import polars as pl
import scipy.sparse as sp
import numpy as np
import typing
import logging
import pandas as pd
import graph_tool as gt
import graph_tool.spectral

try:
    import torch
except ImportError:
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError(
            "torch is required for this function. "
            "Install it with: uv pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
    
def build_interlayer_coupling_from_tensor(t: torch.Tensor, omega: float, kind: str) -> torch.Tensor:
    """
    Build the interlayer coupling from a tensor of only intralayer interactions.
    Args:
        t: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network. Only intralayer interactions should be non-zero.
        omega: The interlayer coupling strength.
        kind: The type of interlayer coupling to build. Options are:
            - "ordered": chain coupling (undirected)
            - "categorical": all-to-all undirected coupling
            - "temporal": directed chain coupling
    Returns:
        A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the interlayer coupling.
    """
    kind = kind.lower().strip()
    n, l = t.shape[0], t.shape[1]
    if len(t.shape) != 4 or t.shape[0] != n or t.shape[2] != n or t.shape[1] != l or t.shape[3] != l:
        raise ValueError("Input tensor must have shape (num_nodes, num_layers, num_nodes, num_layers).")
    if not t.is_sparse:
        raise NotImplementedError("Input tensor must be a sparse tensor.")
    
    if kind == "ordered":
        # Vectorized version for chain coupling (undirected)
        rows = np.arange(n)
        layers = np.arange(l - 1)

        # All (j, i, j, i+1) and (j, i+1, j, i) pairs
        from_indices = np.stack(np.meshgrid(rows, layers, indexing='ij'), axis=-1).reshape(-1, 2)
        to_indices = from_indices.copy()
        to_indices[:, 1] += 1

        # Stack both directions
        indices = np.concatenate([
            np.stack([from_indices[:, 0], from_indices[:, 1], to_indices[:, 0], to_indices[:, 1]], axis=1),
            np.stack([to_indices[:, 0], to_indices[:, 1], from_indices[:, 0], from_indices[:, 1]], axis=1)
        ], axis=0)

        indices = torch.tensor(indices, dtype=torch.long).t().contiguous()
        values = torch.full((indices.shape[1],), omega, dtype=torch.float32)

    elif kind == "categorical":
        # All-to-all undirected coupling: node j in layer i ↔ node j in layer k, for all i ≠ k
        assert n * l < 10000, (
            "Too many edges for categorical coupling; "
            "consider using a smaller network or a different coupling type. "
            "Consider using only a layer-by-layer coupling and handle it as layer couples."
        )

        # For each node, create all (layer_from, layer_to) pairs with layer_from != layer_to
        nodes = np.arange(n)
        layer_from, layer_to = np.meshgrid(np.arange(l), np.arange(l), indexing='ij')
        mask = layer_from != layer_to
        lf = layer_from[mask]
        lt = layer_to[mask]

        # Repeat for all nodes
        all_nodes = np.repeat(nodes, len(lf))
        all_lf = np.tile(lf, n)
        all_lt = np.tile(lt, n)

        indices = np.stack([all_nodes, all_lf, all_nodes, all_lt], axis=1)
        indices = torch.tensor(indices, dtype=torch.long).t().contiguous()
        values = torch.full((indices.shape[1],), omega, dtype=torch.float32)
    
    elif kind == "temporal":
        # Vectorized version for directed chain coupling
        rows = np.arange(n)
        layers = np.arange(l - 1)

        from_indices = np.stack(np.meshgrid(rows, layers, indexing='ij'), axis=-1).reshape(-1, 2)
        to_indices = from_indices.copy()
        to_indices[:, 1] += 1

        indices = np.stack([from_indices[:, 0], from_indices[:, 1], to_indices[:, 0], to_indices[:, 1]], axis=1)

        indices = torch.tensor(indices, dtype=torch.long).t().contiguous()
        values = torch.full((indices.shape[1],), omega, dtype=torch.float32)

    else:
        raise NotImplementedError(f"Unknown coupling type: {kind}")
    
    t_coupling = torch.sparse_coo_tensor(
        indices,
        values,
        size=(n, l, n, l),
        dtype=torch.float32,
    )
    t_coupling = t_coupling.coalesce()
    return t_coupling

def build_interlayer_coupling_matrix(l: int, omega: float, kind: str) -> sp.csr_matrix:
        """
        Build the interlayer coupling matrix (layers x layers) for a multilayer network.
        Args:
            l: Number of layers.
            omega: Interlayer coupling strength.
            kind: Type of coupling: "ordered", "categorical", or "temporal".
        Returns:
            scipy.sparse.csr_matrix of shape (l, l)
        """
        kind = kind.lower().strip()
        if l <= 1:
            # No interlayer coupling possible
            return sp.csr_matrix((l, l))

        if kind == "ordered":
            # Chain coupling (undirected)
            mat = sp.diags([np.ones(l-1), np.ones(l-1)], [1, -1]) * omega
        elif kind == "categorical":
            # All-to-all undirected coupling (excluding self-layer)
            mat = (np.ones((l, l)) - np.eye(l)) * omega
            mat = sp.csr_matrix(mat)
        elif kind == "temporal":
            # Directed chain coupling (i -> i+1)
            mat = sp.diags([np.ones(l-1)], [1]) * omega
        else:
            raise ValueError(f"Unknown coupling kind: {kind}")
        return mat if isinstance(mat, sp.csr_matrix) else mat.tocsr()

def build_supra_adjacency_matrix_from_edge_colored_matrices(intra_networks: list[sp.csr_matrix], layer_coupling_matrix: sp.csr_matrix, num_nodes: int) -> sp.csr_matrix:
    """
    Build a supra-adjacency matrix from a list of intralayer adjacency matrices and an interlayer coupling matrix.
    Args:
        intra_networks: List of sparse adjacency matrices for each layer (shape: num_nodes x num_nodes).
        layer_coupling_matrix: Sparse interlayer coupling matrix (shape: num_layers x num_layers).
        num_nodes: Number of nodes in each layer.
    Returns:
        scipy.sparse.csr_matrix: supra-adjacency matrix of shape (num_layers*num_nodes, num_layers*num_nodes).
    """
    # Assert all intra_networks have shape (num_nodes, num_nodes)
    for net in intra_networks:
        assert net.shape == (num_nodes, num_nodes), (
            f"All intra_networks must have shape ({num_nodes}, {num_nodes}), but got {net.shape}"
        )
    t = sp.block_diag(intra_networks, format="csr")
    identity = sp.eye(num_nodes, format="csr")
    t += sp.kron(layer_coupling_matrix, identity, format="csr")
    return t

def build_edge_colored_matrices_from_supra_adjacency_matrix(supra_adj: sp.csr_matrix, num_layers: int) -> list[sp.csr_matrix]:
    """
    Retrieve the block diagonal intralayer adjacency matrices and the interlayer coupling matrix from a supra-adjacency matrix.
    num_layers: number of layers in the multi-layer network.
    Args:
        supra_adj: sparse supra-adjacency matrix of shape (num_layers*num_nodes, num_layers*num_nodes).
        num_layers: number of layers in the multi-layer network.
    Returns:
        intra_networks: List of sparse adjacency matrices for each layer (shape: num_nodes x num_nodes).
    """
    num_nodes = supra_adj.shape[0] // num_layers
    intra_networks = []
    for i in range(num_layers):
        start = i * num_nodes
        end = (i + 1) * num_nodes
        intra_networks.append(supra_adj[start:end, start:end].tocsr())
    return intra_networks

def get_node_tensor_from_network_list(glist: list[gt.Graph]) -> list[sp.spmatrix]:
    """
    Convert a list of graph-tool graphs to their scipy sparse adjacency matrices.
    Args:
        glist: List of graph-tool Graph objects, one per layer.
    Returns:
        List of scipy sparse adjacency matrices, one per layer.
    """
    return [gt.spectral.adjacency(g) for g in glist]

def supra_adjacency_to_network_list(supra: sp.spmatrix, num_layers: int, num_nodes: int) -> list[gt.Graph]:
    """
    Convert a supra-adjacency matrix into a list of graph-tool graphs (one per layer).
    Args:
        supra: Supra-adjacency matrix of shape (num_layers*num_nodes, num_layers*num_nodes).
        num_layers: Number of layers.
        num_nodes: Number of nodes per layer.
    Returns:
        List of graph-tool Graph objects, one per layer.
    """
    intra_networks = build_edge_colored_matrices_from_supra_adjacency_matrix(supra, num_layers)
    graphs = []
    for adj in intra_networks:
        g = gt.Graph(directed=False)
        g.add_edge_list(np.transpose(adj.nonzero()))
        graphs.append(g)
    return graphs

def build_tensor_from_list_of_graphs(glist: list[gt.Graph]) -> torch.Tensor:
    """
    Build a tensor from a list of Graphs representing the layers of a multi-layer network.
    Args:
        glist: List of Graphs representing the layers of the multi-layer network. Each graph should have the same number of nodes.
    Returns:
        A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers) representing the multi-layer network.
    """
    _require_torch()
    if not glist:
        raise ValueError("Input list of graphs is empty.")
    
    num_layers = len(glist)
    num_nodes = glist[0].num_vertices()
    for g in glist:
        if g.num_vertices() != num_nodes:
            raise ValueError("All graphs must have the same number of nodes.")

    indices_list = []
    values_list = []
    for layer_from, g in enumerate(glist):
        adj = gt.spectral.adjacency(g)
        adj = adj.tocoo()
        for node_from, node_to, value in zip(adj.row, adj.col, adj.data):
            indices_list.append([node_from, layer_from, node_to, layer_from])
            values_list.append(float(value))

    if not indices_list:
        # No edges in any layer; return an empty sparse tensor
        return torch.sparse_coo_tensor(
            torch.empty((4, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            size=(num_nodes, num_layers, num_nodes, num_layers),
            dtype=torch.float32,
        )

    indices = torch.tensor(indices_list, dtype=torch.long).t().contiguous()
    values = torch.tensor(values_list, dtype=torch.float32)

    t = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_layers, num_nodes, num_layers),
        dtype=torch.float32,
    )
    t = t.coalesce()
    return t

def build_aggregate_network_from_tensor(t: torch.Tensor, kind="sum") -> sp.csr_matrix:
    """
    Build an aggregate network from a tensor of sparse interactions.
    Args:
        t: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    Returns:
        A sparse matrix of shape (num_nodes, num_nodes) representing the aggregate network.
    """

    if kind == "sum":
        # Sum over layers: A_agg[i, j] = sum_{k, l} t[i, k, j, l]
        agg = t.coalesce()
        indices = agg.indices()
        values = agg.values()

        # Create a dictionary to accumulate sums for each (i, j) pair
        from collections import defaultdict
        edge_dict = defaultdict(float)
        for idx in range(indices.shape[1]):
            i, k, j, l = indices[:, idx]
            edge_dict[(i.item(), j.item())] += values[idx].item()

        # Convert the dictionary back to COO format
        row = []
        col = []
        data = []
        for (i, j), weight in edge_dict.items():
            row.append(i)
            col.append(j)
            data.append(weight)

        num_nodes = t.shape[0]
        return sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    elif kind == "max":
        # Max over layers: A_agg[i, j] = max_{k, l} t[i, k, j, l]
        agg = t.coalesce()
        indices = agg.indices()
        values = agg.values()

        # Create a dictionary to keep track of the max weight for each (i, j) pair
        edge_dict = {}
        for idx in range(indices.shape[1]):
            i, k, j, l = indices[:, idx]
            weight = values[idx].item()
            if (i.item(), j.item()) not in edge_dict or weight > edge_dict[(i.item(), j.item())]:
                edge_dict[(i.item(), j.item())] = weight

        # Convert the dictionary back to COO format
        row = []
        col = []
        data = []
        for (i, j), weight in edge_dict.items():
            row.append(i)
            col.append(j)
            data.append(weight)

        num_nodes = t.shape[0]
        return sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    elif kind == "min":
        # Min over layers: A_agg[i, j] = min_{k, l} t[i, k, j, l]
        agg = t.coalesce()
        indices = agg.indices()
        values = agg.values()

        # Create a dictionary to keep track of the min weight for each (i, j) pair
        edge_dict = {}
        for idx in range(indices.shape[1]):
            i, k, j, l = indices[:, idx]
            weight = values[idx].item()
            if (i.item(), j.item()) not in edge_dict or weight < edge_dict[(i.item(), j.item())]:
                edge_dict[(i.item(), j.item())] = weight

        # Convert the dictionary back to COO format
        row = []
        col = []
        data = []
        for (i, j), weight in edge_dict.items():
            row.append(i)
            col.append(j)
            data.append(weight)

        num_nodes = t.shape[0]
        return sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    else:
        raise ValueError(f"Unknown aggregation kind: {kind}")
    
def build_list_of_graphs_from_tensor(t: torch.Tensor) -> list[gt.Graph]:
    """
    Build a list of Graphs from a tensor of sparse interactions.
    Args:
        t: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    Returns:
        A list of Graphs representing the layers of the multi-layer network. Each graph will have the same number of nodes.
    """
    _require_torch()
    if not t.is_sparse:
        raise NotImplementedError("Input tensor must be a sparse tensor.")
    
    n, l = t.shape[0], t.shape[1]
    if len(t.shape) != 4 or t.shape[0] != n or t.shape[2] != n or t.shape[1] != l or t.shape[3] != l:
        raise ValueError("Input tensor must have shape (num_nodes, num_layers, num_nodes, num_layers).")

    t = t.coalesce()
    indices = t.indices()
    values = t.values()

    # Create a list of adjacency matrices for each layer
    adj_matrices = [sp.lil_matrix((n, n)) for _ in range(l)]
    for idx in range(indices.shape[1]):
        i, k, j, l_ = indices[:, idx]
        weight = values[idx].item()
        adj_matrices[k][i, j] = weight

    # Convert adjacency matrices to Graphs
    graphs = []
    for adj in adj_matrices:
        g = gt.Graph(adj.tocsr())
        graphs.append(g)

    return graphs

def build_laplacian_from_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Build the combinatorial Laplacian from a tensor of sparse interactions.
    Args:
        t: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    Returns:
        A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the combinatorial Laplacian of the multi-layer network.
    """
    _require_torch()
    if not t.is_sparse:
        raise NotImplementedError("Input tensor must be a sparse tensor.")
    
    n, l = t.shape[0], t.shape[1]
    if len(t.shape) != 4 or t.shape[0] != n or t.shape[2] != n or t.shape[1] != l or t.shape[3] != l:
        raise ValueError("Input tensor must have shape (num_nodes, num_layers, num_nodes, num_layers).")

    t = t.coalesce()
    indices = t.indices()
    values = t.values()

    # Create a dictionary to accumulate the degree for each node-layer pair
    degree_dict = {}
    for idx in range(indices.shape[1]):
        i, k, j, l_ = indices[:, idx]
        weight = values[idx].item()
        degree_dict[(i.item(), k.item())] = degree_dict.get((i.item(), k.item()), 0.0) + weight

    # Create the Laplacian entries
    laplacian_indices = []
    laplacian_values = []
    for idx in range(indices.shape[1]):
        i, k, j, l_ = indices[:, idx]
        weight = values[idx].item()
        laplacian_indices.append([i.item(), k.item(), j.item(), l_.item()])
        laplacian_values.append(-weight)

    for (i, k), degree in degree_dict.items():
        laplacian_indices.append([i, k, i, k])
        laplacian_values.append(degree)

    laplacian_indices = torch.tensor(laplacian_indices, dtype=torch.long).t().contiguous()
    laplacian_values = torch.tensor(laplacian_values, dtype=torch.float32)

    laplacian = torch.sparse_coo_tensor(
        laplacian_indices,
        laplacian_values,
        size=(n, l, n, l),
        dtype=torch.float32,
    )
    laplacian = laplacian.coalesce()
    return laplacian

def get_laplacian_from_list_of_graphs(glist: list[gt.Graph]) -> list[sp.csr_matrix]:
    """
    Compute the laplacian matrix of a graph-tool graph for each intra layer
    Args:
        glist: List of Graphs representing the layers of the multi-layer network. Each graph should have the same number of nodes.
    Returns:
        A list of sparse matrix of shape (num_nodes, num_nodes) representing the laplacian of each layer of the multi-layer network.
    """
    laplacians = []
    for g in glist:
        adj = gt.spectral.adjacency(g)
        degree = np.asarray(adj.sum(axis=1)).ravel()
        D = sp.diags(degree, format="csr")
        L = D - adj
        row_sums = np.asarray(L.sum(axis=1)).ravel()
        if np.abs(row_sums).sum() > 1e-8:
            raise ValueError("ERROR! The Laplacian matrix has rows that don't sum to 0. Aborting process.")
        L.eliminate_zeros()
        laplacians.append(L.tocsr())
    return laplacians

def build_tensor_from_dataframe(df):
    """
    Build a tensor from a DataFrame representing an edge list.
    Args:
        df: A DataFrame representing the edge list in the format
        (node.from, layer.from, node.to, layer.to, weight).
    Returns:
        A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
        representing the multi-layer network.
    """
    _require_torch()
    if df.is_empty():
        raise ValueError("Input DataFrame is empty.")

    n = max(df["node.from"].max(), df["node.to"].max()) + 1
    l = max(df["layer.from"].max(), df["layer.to"].max()) + 1

    indices = torch.tensor(
        np.array([
            df["node.from"].to_numpy(),
            df["layer.from"].to_numpy(),
            df["node.to"].to_numpy(),
            df["layer.to"].to_numpy(),
        ]),
        dtype=torch.long
    )
    values = torch.tensor(df["weight"].to_numpy(), dtype=torch.float32)

    t = torch.sparse_coo_tensor(
        indices,
        values,
        size=(n, l, n, l),
        dtype=torch.float32,
    )
    t = t.coalesce()
    return t

def build_edgelist_from_tensor(t):
    """
    Build an edge list from a tensor of sparse interactions.
    Args:
        t: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    Returns:
        pl.DataFrame: A DataFrame representing the edge list in the format
        (node.from, layer.from, node.to, layer.to, weight).
    """
    _require_torch()
    if not t.is_sparse:
        raise NotImplementedError("Input tensor must be a sparse tensor.")

    n, l = t.shape[0], t.shape[1]
    if len(t.shape) != 4 or t.shape[0] != n or t.shape[2] != n or t.shape[1] != l or t.shape[3] != l:
        raise ValueError("Input tensor must have shape (num_nodes, num_layers, num_nodes, num_layers).")
    t = t.coalesce()

    indices = t.indices()
    values = t.values()

    edge_list = pl.DataFrame(
        {
            "node.from": indices[0].to(torch.int32).numpy(),
            "layer.from": indices[1].to(torch.int32).numpy(),
            "node.to": indices[2].to(torch.int32).numpy(),
            "layer.to": indices[3].to(torch.int32).numpy(),
            "weight": values.numpy(),
        }
    )
    return edge_list

def build_supra_interaction_matrix_from_tensor(t):
    """
    Build a supra-interaction matrix (weighted) from a tensor of sparse interactions.
    Args:
        t: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    Returns:
        scipy.sparse.csr_matrix: supra-interaction matrix of shape (num_layers*num_nodes, num_layers*num_nodes).
    """
    _require_torch()
    n, l = t.shape[0], t.shape[1]
    if len(t.shape) != 4 or t.shape[0] != n or t.shape[2] != n or t.shape[1] != l or t.shape[3] != l:
        raise ValueError("Input tensor must have shape (num_nodes, num_layers, num_nodes, num_layers).")

    t = t.coalesce()
    indices = t.indices()
    row = (indices[1] * n + indices[0]).detach().cpu().numpy()
    col = (indices[3] * n + indices[2]).detach().cpu().numpy()
    data = t.values().detach().cpu().numpy()
    return sp.coo_matrix((data, (row, col)), shape=(l*n, l*n)).tocsr()

def build_supra_adjacency_matrix_from_tensor(t):
    """
    Build a supra-adjacency matrix (binary) from a tensor of sparse interactions.
    Args:
        t: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    Returns:
        scipy.sparse.csr_matrix: supra-adjacency matrix of shape (num_layers*num_nodes, num_layers*num_nodes).
    """
    _require_torch()
    n, l = t.shape[0], t.shape[1]
    if len(t.shape) != 4 or t.shape[0] != n or t.shape[2] != n or t.shape[1] != l or t.shape[3] != l:
        raise ValueError("Input tensor must have shape (num_nodes, num_layers, num_nodes, num_layers).")

    t = t.coalesce()
    indices = t.indices()
    row = (indices[1] * n + indices[0]).detach().cpu().numpy()
    col = (indices[3] * n + indices[2]).detach().cpu().numpy()
    return sp.coo_matrix((np.ones_like(row), (row, col)), shape=(l*n, l*n)).tocsr()


def build_tensor_from_supra_adjacency_matrix(a, num_layers, num_nodes):
    """
    Build a tensor of sparse interactions from a supra-adjacency matrix.
    Args:
        a: sparse supra-adjacency matrix of shape (num_layers*num_nodes, num_layers*num_nodes).
        num_layers: number of layers in the multi-layer network.
        num_nodes: number of nodes in each layer of the multi-layer network.
    Returns:
        A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    """
    _require_torch()
    if a.shape != (num_layers*num_nodes, num_layers*num_nodes):
        raise ValueError("Input matrix must have shape (num_layers*num_nodes, num_layers*num_nodes).")

    a = a.tocoo()
    row = a.row
    col = a.col
    data = a.data

    layer_from = row // num_nodes
    node_from = row % num_nodes
    layer_to = col // num_nodes
    node_to = col % num_nodes

    indices = torch.tensor(np.array([node_from, layer_from, node_to, layer_to]), dtype=torch.long)
    values = torch.tensor(data, dtype=torch.float32)

    t = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_layers, num_nodes, num_layers),
        dtype=torch.float32,
    )
    t = t.coalesce()
    return t

def build_transition_matrix_from_adjacency_matrix(
        adj: sp.csr_matrix,
        n: int,
        l: int,
        kind: str,
        *,
        alpha: float | None = None,
        logger: typing.Optional[logging.Logger] = None,
) -> sp.csr_matrix:
    """
    Build supra-transition matrix from a supra-adjacency matrix (NL x NL).

    Implemented kinds:
        - 'classical': row-stochastic D^{-1} A; rows with zero out-strength become uniform 1/NL.
        - 'pagerank': alpha * (row-stochastic) + (1-alpha)/NL; zero rows become uniform 1/NL.

    Notes:
        - Teleportation term makes the matrix dense; returned as CSR for consistency.
    """
    NL = n * l
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    if adj.shape != (NL, NL):
        raise ValueError(f"Incompatible shape {adj.shape} for n={n}, l={l} (expected {(NL, NL)})")

    kind = kind.lower().strip()

    # Common: row sums (out-strength)
    strength = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()

    if kind in ("classical", "pagerank"):
        # D^{-1} A with safe handling for zero rows
        inv_strength = np.zeros_like(strength, dtype=np.float64)
        nz = strength > 0.0
        inv_strength[nz] = 1.0 / strength[nz]
        Dinv = sp.diags(inv_strength, format="csr")
        P = (Dinv @ adj).tocsr()

        if kind == "classical":
            return P

        a = 0.85 if alpha is None else float(alpha)
        if not (0.0 < a <= 1.0):
            raise ValueError("alpha must be in (0, 1] for pagerank")

        P = P.multiply(a).tocsr()
        return P

    elif kind == "diffusive":
        raise NotImplementedError("type='diffusive' not implemented yet")
    elif kind == "maxent":
        raise NotImplementedError("type='maxent' not implemented yet")
    elif kind == "physical":
        raise NotImplementedError("type='physical' not implemented yet")
    elif kind == "relaxed-physical":
        raise NotImplementedError("type='relaxed-physical' not implemented yet")
    else:
        raise NotImplementedError(f"Unknown transition type: {kind}")

def build_laplacian_matrix_from_adjacency_matrix(adj: sp.csr_matrix) -> sp.csr_matrix:
    """
    Build the combinatorial Laplacian L = D - A from a supra-adjacency matrix.
    """
    NL = adj.shape[1]
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    if adj.shape != (NL, NL):
        raise ValueError(f"Incompatible shape {adj.shape} (expected {(NL, NL)})")

    strength = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()
    D = sp.diags(strength, format="csr")
    L = D - adj
    return L

def build_density_bgs_from_adjacency_matrix(
        adj: sp.csr_matrix,
) -> sp.csr_matrix:
    """
    Build supra-density matrix from a supra-adjacency matrix (NL x NL).

    Density matrix is defined as rho = L / Tr(L), where L is the Laplacian.
    """
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)

    den = build_laplacian_matrix_from_adjacency_matrix(adj)
    trace = den.diagonal().sum()
    if trace == 0:
        raise ValueError("Adjacency matrix has no edges; Laplacian trace is zero.")
    return den / trace


def get_aggregate_network(
    obj: list[sp.spmatrix] | list[gt.Graph],
    obj_type: str = "tensor",
    return_mat: bool = False,
    binarize: bool = True,
) -> sp.spmatrix | gt.Graph:
    """
    Aggregate multilayer networks into a single network.

    Parameters
    ----------
    obj : list
        List of networks, either as Graph objects or adjacency matrices.
    obj_type : str, optional
        Format of `obj`: "tensor" or "glist" (default: "tensor").
    return_mat : bool, optional
        If True, return adjacency matrix (default: False).
    binarize : bool, optional
        If True, edge weights are binarized.

    Returns
    -------
    graph_tool.Graph or scipy.sparse matrix
        Aggregate network.
    """
    if obj_type == "glist":
        obj = get_node_tensor_from_network_list(obj)

    agg_mat = sp.coo_matrix(obj[0].shape)
    for layer in obj:
        agg_mat += layer

    if binarize:
        agg_mat[agg_mat > 0] = 1

    if return_mat:
        return agg_mat

    g_agg = gt.Graph(directed=False)
    g_agg.add_edge_list(np.transpose(agg_mat.nonzero()))
    g_agg.add_vertex(agg_mat.shape[0] - g_agg.num_vertices())
    return g_agg


def supra_adjacency_to_block_tensor(
    supra: sp.spmatrix, layers: int, nodes: int
) -> list[list[sp.spmatrix]]:
    """
    Convert supra-adjacency matrix to block tensor format.

    Parameters
    ----------
    supra : scipy.sparse matrix
        Supra-adjacency matrix.
    layers : int
        Number of layers.
    nodes : int
        Number of nodes.

    Returns
    -------
    list of lists
        Block tensor [layers x layers] of adjacency submatrices.
    """
    return [
        [supra[i * nodes:(i + 1) * nodes, j * nodes:(j + 1) * nodes] for j in range(layers)]
        for i in range(layers)
    ]


def node_tensor_to_network_list(
    tensor: list[sp.spmatrix], layers: int, nodes: int
) -> list[gt.Graph]:
    """
    Convert node-layer adjacency matrices into graph_tool graphs.

    Parameters
    ----------
    tensor : list of scipy.sparse matrices
        List of per-layer adjacency matrices.
    layers : int
        Number of layers.
    nodes : int
        Number of nodes.

    Returns
    -------
    list of graph_tool.Graph
        One graph per layer.
    """
    g_list = []
    for i in range(layers):
        g = gt.Graph(directed=False)
        g.add_edge_list(np.transpose(tensor[i].nonzero()))
        g_list.append(g)
    return g_list


def build_supra_adjacency_matrix_from_extended_edgelist(
    dfEdges: pd.DataFrame, Layers: int, Nodes: int, isDirected: bool
) -> sp.spmatrix:
    """
    Build supra-adjacency matrix from extended edge list.

    Parameters
    ----------
    dfEdges : pandas.DataFrame
        DataFrame with columns ["NodeIN", "LayerIN", "NodeOUT", "LayerOUT"].
    Layers : int
        Number of layers.
    Nodes : int
        Number of nodes.
    isDirected : bool
        Whether the graph is directed.

    Returns
    -------
    scipy.sparse matrix
        Supra-adjacency matrix.
    """
    if len(pd.unique(pd.concat([dfEdges["LayerIN"], dfEdges["LayerOUT"]]))) != Layers:
        raise ValueError("Error: expected number of layers does not match the data. Aborting process.")

    def supra_idx(node, layer):
        return layer * Nodes + node

    frm = supra_idx(dfEdges["NodeIN"].to_numpy(), dfEdges["LayerIN"].to_numpy())
    to = supra_idx(dfEdges["NodeOUT"].to_numpy(), dfEdges["LayerOUT"].to_numpy())
    w = np.ones(len(dfEdges), dtype=float)

    order = Layers * Nodes
    M = sp.coo_matrix((w, (frm, to)), shape=(order, order)).tocsr()

    if not isDirected:
        M = M + M.T

    if abs((M - M.T).sum()) > 1e-12 and not isDirected:
        raise ValueError("WARNING: The input data is directed but isDirected=FALSE, I am symmetrizing by average.")

    return M


def create_supra_transition_matrix_virus(
    supra: sp.spmatrix,
    node_tensor: list[sp.spmatrix],
    nodes: int,
    layers: int,
    p_intra: float = 1,
) -> sp.spmatrix:
    """
    Construct a custom supra-transition matrix mixing intra- and inter-layer dynamics.

    Parameters
    ----------
    supra : scipy.sparse matrix
        Supra adjacency matrix.
    node_tensor : list of scipy.sparse matrices
        Adjacency matrices per layer.
    nodes : int
        Number of nodes.
    layers : int
        Number of layers.
    p_intra : float, optional
        Weight for intra-layer contributions (default=1).

    Returns
    -------
    scipy.sparse matrix
        Row-normalized supra transition matrix.
    """
    supra_sum = np.array(supra.sum(axis=0).tolist()[0])

    blocks = []
    for l in range(layers):
        block = sp.identity(nodes).multiply(supra_sum[l * nodes:(l + 1) * nodes] - (layers - 1))
        blocks.append(block)

    mat = []
    for la in range(layers):
        norm_fac = np.sum(supra_sum.reshape(layers, nodes)[np.delete(np.arange(layers), la)] - (layers - 1), axis=0)
        norm_fac = np.where(norm_fac != 0, 1 / norm_fac, 0)
        mat.append([blocks[i].multiply(norm_fac) for i in np.delete(np.arange(layers), la)])

    diag_blocks = []
    for la in range(layers):
        t0_sum = np.array(list(node_tensor[la].sum(axis=0)))[0][0]
        t0_sum = np.where(t0_sum != 0, 1 / t0_sum, 0)
        diag_blocks.append(node_tensor[la].dot(sp.diags(t0_sum)).T)

    for i in range(layers):
        mat[i].insert(i, diag_blocks[i].multiply(np.array([p_intra] * nodes)))

    comb_mat = sp.vstack([sp.hstack(mat[i]) for i in range(layers)])
    valss = np.where(comb_mat.sum(axis=1) != 0, 1 / comb_mat.sum(axis=1), 0).flatten()
    return comb_mat.T.multiply(valss).T
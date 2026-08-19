import polars as pl
from typing import Tuple
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
from pathlib import Path

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


def read_single_layer_edgelist_as_tensor(file_path: str | Path, delimiter: str = ","):
    """
    Read a single-layer edgelist from a file and convert it to a sparse tensor.
    Args:
        file_path: Path to the edgelist file.
        delimiter: Delimiter used in the edgelist file.
    Returns:
        torch.Tensor: A sparse tensor of shape (num_nodes, num_nodes) representing the single-layer network.
    """
    _require_torch()
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pl.read_csv(file_path, separator=delimiter,
        has_header=True,
        new_columns=["node.from", "node.to", "weight"])
    indices = torch.tensor(df.select(["node.from", "node.to"]).to_numpy().T, dtype=torch.long)
    values = torch.tensor(df["weight"].to_numpy(), dtype=torch.float32)
    n = df.select(pl.max("node.from")).item() + 1
    tensor = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(n, n),
        dtype=torch.float32
    ).coalesce()
    return tensor

def read_edgelist_as_tensor(file_path: str | Path, delimiter: str = ","):
    """
    Read an edgelist from a file and convert it to a sparse tensor.
    Args:
        file_path: Path to the edgelist file.
        delimiter: Delimiter used in the edgelist file.

    Returns:
        torch.Tensor: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    """
    _require_torch()
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pl.read_csv(file_path, separator=delimiter,
        has_header=True,
        new_columns=["node.from", "layer.from", "node.to", "layer.to", "weight"])
    indices = torch.tensor(df.select(["node.from", "layer.from", "node.to", "layer.to"]).to_numpy().T, dtype=torch.long)
    values = torch.tensor(df["weight"].to_numpy(), dtype=torch.float32)
    n = df.select(pl.max("node.from")).item() + 1
    l = df.select(pl.max("layer.from")).item() + 1
    tensor = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(n, l, n, l),
        dtype=torch.float32
    ).coalesce()
    return tensor

def read_edgelist_as_supraadjacencymatrix(file_path: str | Path, delimiter: str = ",") -> Tuple[sp.csr_matrix, int, int]:
    """
    Read an edgelist from a file and convert it to a supra-adjacency matrix (binary).
    Args:
        file_path: Path to the edgelist file.
        delimiter: Delimiter used in the edgelist file.

    Returns:
        scipy.sparse.csr_matrix: A supra-adjacency matrix of shape (num_nodes * num_layers, num_nodes * num_layers).
        int: The number of nodes in the network.
        int: The number of layers in the network.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pl.read_csv(file_path, separator=delimiter,
        has_header=True,
        new_columns=["node.from", "layer.from", "node.to", "layer.to", "weight"])
    num_nodes = df.select(pl.max("node.from")).item() + 1
    num_layers = df.select(pl.max("layer.from")).item() + 1
    return sp.coo_matrix(
        (
            np.ones_like(df["weight"].to_numpy(), dtype=np.int8),
            (
                df["node.from"].to_numpy() + df["layer.from"].to_numpy() * num_nodes,
                df["node.to"].to_numpy() + df["layer.to"].to_numpy() * num_nodes,
            ),
        ), shape=(num_nodes * num_layers, num_nodes * num_layers), dtype=np.float32
    ).tocsr(), num_nodes, num_layers

def read_edgelist_as_suprainteractionmatrix(file_path: str | Path, delimiter: str = ",") -> Tuple[sp.csr_matrix, int, int]:
    """
    Read an edgelist from a file and convert it to a supra-interaction matrix (weighted).
    Args:
        file_path: Path to the edgelist file.
        delimiter: Delimiter used in the edgelist file.

    Returns:
        scipy.sparse.csr_matrix: A supra-interaction matrix of shape (num_nodes * num_layers, num_nodes * num_layers).
        int: The number of nodes in the network.
        int: The number of layers in the network.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pl.read_csv(file_path, separator=delimiter,
        has_header=True,
        new_columns=["node.from", "layer.from", "node.to", "layer.to", "weight"])
    num_nodes = df.select(pl.max("node.from")).item() + 1
    num_layers = df.select(pl.max("layer.from")).item() + 1
    return sp.coo_matrix(
        (
            df["weight"].to_numpy(),
            (
                df["node.from"].to_numpy() + df["layer.from"].to_numpy() * num_nodes,
                df["node.to"].to_numpy() + df["layer.to"].to_numpy() * num_nodes,
            ),
        ), shape=(num_nodes * num_layers, num_nodes * num_layers), dtype=np.float32
    ).tocsr(), num_nodes, num_layers

def write_edgelist_from_tensor(
    tensor,
    output_file: str | Path
) -> None:
    """
    Write a sparse tensor's edge list to a file.
    Args:
        tensor: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers).
        output_file: Path to the output file.
    """
    _require_torch()
    from . import parsing

    if not tensor.is_sparse:
        raise NotImplementedError("Input tensor must be a sparse tensor.")

    n, l = tensor.shape[0], tensor.shape[1]
    if len(tensor.shape) != 4 or tensor.shape[0] != n or tensor.shape[2] != n or tensor.shape[1] != l or tensor.shape[3] != l:
        raise ValueError("Input tensor must have shape (num_nodes, num_layers, num_nodes, num_layers).")
    tensor = tensor.coalesce()
    df = parsing.build_edgelist_from_tensor(tensor)
    df.write_csv(output_file, separator=",")

def read_edgelist_as_dataframe(file_path: str | Path, delimiter: str = ",") -> pl.DataFrame:
    """
    Read an edgelist from a file and convert it to a Polars DataFrame.
    Args:
        file_path: Path to the edgelist file.
        delimiter: Delimiter used in the edgelist file.
    Returns:
        pl.DataFrame: A DataFrame with columns (node.from, layer.from, node.to, layer.to, weight).
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    df = pl.read_csv(file_path, separator=delimiter,
        has_header=True,
        new_columns=["node.from", "layer.from", "node.to", "layer.to", "weight"])
    return df

def write_edgelist_from_dataframe(
    df: pl.DataFrame,
    output_file: str | Path,
    delimiter: str = ","
) -> None:
    """
    Write a Polars DataFrame's edge list to a file.
    Args:
        df: A DataFrame with columns (node.from, layer.from, node.to, layer.to, weight).
        output_file: Path to the output file.
        delimiter: Delimiter to use in the output file.
    """
    if df.is_empty():
        raise ValueError("Input DataFrame is empty.")
    df.write_csv(output_file, separator=delimiter)
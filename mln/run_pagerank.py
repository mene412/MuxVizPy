import argparse
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import scipy.sparse as sps

from MuxVizPy.versatility import compute_multipagerank_centrality

def compute_multilayer_pagerank(
    supra_adj: sps.csr_matrix,
    n_nodes: int,
    n_layers: int,
    alpha: float = 0.85,
    tol: float = 1e-12,
    max_iter: int = 10000,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Computes Classic Edge-Weighted Multilayer PageRank using MuxVizPy's
    compute_multipagerank_centrality.

    Parameters
    ----------
    supra_adj : sps.csr_matrix
        Supra-adjacency matrix of shape (n*l, n*l).
    n_nodes : int
        Number of physical nodes (n).
    n_layers : int
        Number of layers (l).
    alpha : float, optional
        Damping / teleportation factor (default: 0.85).
    tol : float, optional
        Convergence tolerance (default: 1e-12).
    max_iter : int, optional
        Maximum power iteration steps (default: 10000).
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized PageRank centrality per physical node, shape (n,).
    """
    return compute_multipagerank_centrality(
        adj=supra_adj,
        n=n_nodes,
        l=n_layers,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
        logger=logger,
    )


def process_sample_network(
    sample_dir: Path,
    output_dir: Path,
    alpha: float = 0.85,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    """
    Loads network files for a sample, computes Multilayer PageRank using MuxVizPy,
    and writes results to CSV.

    Parameters
    ----------
    sample_dir : Path
        Path to the sample folder inside output/multiplex_networks/.
    output_dir : Path
        Directory where PageRank scores will be saved.
    alpha : float, optional
        Teleportation parameter (default: 0.85).
    logger : logging.Logger, optional

    Returns
    -------
    pd.DataFrame or None
    """
    sample_name = sample_dir.name
    supra_file = sample_dir / f"{sample_name}_supra_adj.npz"
    node_file = sample_dir / f"{sample_name}_node_attributes.csv"

    if not supra_file.exists():
        if logger:
            logger.warning("Skipping %s: %s not found.", sample_name, supra_file.name)
        return None

    # 1. Load the supra-adjacency matrix as CSR format
    supra_adj = sps.load_npz(supra_file).tocsr()
    total_dim = supra_adj.shape[0]

    # 2. Load gene/node names
    if node_file.exists():
        node_df = pd.read_csv(node_file)
        if "gene" in node_df.columns:
            gene_names = node_df["gene"].tolist()
        elif "node" in node_df.columns:
            gene_names = node_df["node"].tolist()
        else:
            gene_names = node_df.iloc[:, 0].tolist()
        n_nodes = len(gene_names)
    else:
        # Default fallback (2 layers: expression & methylation)
        n_nodes = total_dim // 2
        gene_names = [f"Gene_{i}" for i in range(n_nodes)]

    n_layers = total_dim // n_nodes

    # 3. Call compute_multilayer_pagerank
    node_pr = compute_multilayer_pagerank(
        supra_adj=supra_adj,
        n_nodes=n_nodes,
        n_layers=n_layers,
        alpha=alpha,
        logger=logger,
    )

    # 4. Construct ranking DataFrame
    results_df = pd.DataFrame({
        "gene": gene_names,
        "multilayer_pagerank": node_pr
    }).sort_values(by="multilayer_pagerank", ascending=False).reset_index(drop=True)

    # 5. Save output CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{sample_name}_pagerank.csv"
    results_df.to_csv(out_file, index=False)

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute multilayer PageRank for multiplex networks."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the sample network folders.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="output/centralities/pagerank",
        help="Directory where PageRank CSV files will be saved (default output/centralities/pagerank).",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="PageRank damping/teleportation factor (default: 0.85).",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Convergence tolerance (default: 1e-12).",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="Maximum number of PageRank iterations (default: 10000).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.exists():
        print(f"[!] Input directory '{args.input_dir}' not found.")
        return

    if not args.input_dir.is_dir():
        print(f"[!] Input path '{args.input_dir}' is not a directory.")
        return

    if not 0 < args.alpha < 1:
        print("[!] --alpha must be between 0 and 1.")
        return

    if args.tol <= 0:
        print("[!] --tol must be greater than 0.")
        return

    if args.max_iter <= 0:
        print("[!] --max-iter must be greater than 0.")
        return

    sample_dirs = sorted(
        d for d in args.input_dir.iterdir() if d.is_dir()
    )

    if not sample_dirs:
        print(f"[!] No sample folders found in '{args.input_dir}'.")
        return

    total_samples = len(sample_dirs)

    for idx, sample_dir in enumerate(sample_dirs, start=1):

        df_pr = process_sample_network(
            sample_dir=sample_dir,
            output_dir=args.output_dir,
            alpha=args.alpha,
            tol=args.tol,
            max_iter=args.max_iter,
        )

        if df_pr is not None:
            print(
                f"[{idx}/{total_samples}] "
                f"Computed multilayer PageRank for patient: {sample_dir.name}"
            )


if __name__ == "__main__":
    main()
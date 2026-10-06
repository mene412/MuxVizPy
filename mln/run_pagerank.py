import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sps

from MuxVizPy.versatility import compute_multipagerank_centrality


def setup_logger() -> logging.Logger:
    """Create and configure the application logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    return logging.getLogger(__name__)


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
    Compute multilayer PageRank using MuxVizPy.

    Parameters
    ----------
    supra_adj : sps.csr_matrix
        Supra-adjacency matrix of shape (n_nodes * n_layers,
        n_nodes * n_layers).
    n_nodes : int
        Number of physical nodes.
    n_layers : int
        Number of network layers.
    alpha : float
        PageRank damping/teleportation factor.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    np.ndarray
        PageRank centrality for each physical node.
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
    tol: float = 1e-12,
    max_iter: int = 10000,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    """
    Load one multiplex network, compute multilayer PageRank,
    and save the results.

    The sample directory may have a name such as:

        TCGA-3L-AA1B_Tumor_omega_0.5_agg_func_mean

    while the files inside are named using the patient ID:

        TCGA-3L-AA1B_Tumor_supra_adj.npz
        TCGA-3L-AA1B_Tumor_node_attributes.csv

    Therefore, the function searches for the files rather than
    constructing their names from the directory name.
    """
    sample_name = sample_dir.name

    if logger:
        logger.info("Processing %s", sample_name)

    # ------------------------------------------------------------------
    # Locate the supra-adjacency and node-attribute files.
    # ------------------------------------------------------------------

    supra_files = sorted(
        sample_dir.glob("*_supra_adj.npz")
    )

    node_files = sorted(
        sample_dir.glob("*_node_attributes.csv")
    )

    if not supra_files:
        if logger:
            logger.warning(
                "Skipping %s: no *_supra_adj.npz file found.",
                sample_name,
            )
        return None

    supra_file = supra_files[0]

    node_file = node_files[0] if node_files else None

    if len(supra_files) > 1 and logger:
        logger.warning(
            "%s contains multiple supra-adjacency files. "
            "Using %s.",
            sample_name,
            supra_file.name,
        )

    if len(node_files) > 1 and logger:
        logger.warning(
            "%s contains multiple node-attribute files. "
            "Using %s.",
            sample_name,
            node_file.name,
        )

    # ------------------------------------------------------------------
    # Load supra-adjacency matrix.
    # ------------------------------------------------------------------

    supra_adj = sps.load_npz(
        supra_file
    ).tocsr()

    total_dim = supra_adj.shape[0]

    if supra_adj.shape[0] != supra_adj.shape[1]:
        raise ValueError(
            f"Supra-adjacency matrix is not square: "
            f"{supra_adj.shape}"
        )

    # ------------------------------------------------------------------
    # Load physical node names.
    # ------------------------------------------------------------------

    if node_file is not None:
        node_df = pd.read_csv(node_file)

        if "node_name" in node_df.columns:
            gene_names = node_df["node_name"].astype(str).tolist()

        elif "gene" in node_df.columns:
            gene_names = node_df["gene"].astype(str).tolist()

        elif "node" in node_df.columns:
            gene_names = node_df["node"].astype(str).tolist()

        else:
            raise ValueError(
                f"No node-name column found in {node_file}. "
                f"Available columns: {list(node_df.columns)}"
            )

        n_nodes = len(gene_names)

    else:
        # The current multiplex network has two layers.
        n_layers = 2

        if total_dim % n_layers != 0:
            raise ValueError(
                f"Cannot infer number of physical nodes from "
                f"supra-adjacency dimension {total_dim}."
            )

        n_nodes = total_dim // n_layers

        gene_names = [
            f"Gene_{i}"
            for i in range(n_nodes)
        ]

    # ------------------------------------------------------------------
    # Infer number of layers.
    # ------------------------------------------------------------------

    if n_nodes <= 0:
        raise ValueError(
            f"Invalid number of physical nodes: {n_nodes}"
        )

    if total_dim % n_nodes != 0:
        raise ValueError(
            f"Supra-adjacency dimension ({total_dim}) is not "
            f"divisible by number of physical nodes ({n_nodes})."
        )

    n_layers = total_dim // n_nodes

    if logger:
        logger.info(
            "Network dimensions: %d physical nodes x %d layers",
            n_nodes,
            n_layers,
        )

    # ------------------------------------------------------------------
    # Check consistency between node names and supra-adjacency.
    # ------------------------------------------------------------------

    if len(gene_names) != n_nodes:
        raise ValueError(
            f"Number of node names ({len(gene_names)}) does not "
            f"match number of physical nodes ({n_nodes})."
        )

    # ------------------------------------------------------------------
    # Compute multilayer PageRank.
    # ------------------------------------------------------------------

    node_pr = compute_multilayer_pagerank(
        supra_adj=supra_adj,
        n_nodes=n_nodes,
        n_layers=n_layers,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
        logger=logger,
    )

    node_pr = np.asarray(
        node_pr,
        dtype=float,
    ).reshape(-1)

    if len(node_pr) != n_nodes:
        raise ValueError(
            f"MuxVizPy returned {len(node_pr)} PageRank values, "
            f"but {n_nodes} physical nodes were expected."
        )

    if not np.isfinite(node_pr).all():
        raise ValueError(
            f"Non-finite PageRank values returned for {sample_name}."
        )

    # ------------------------------------------------------------------
    # Construct result table.
    # ------------------------------------------------------------------

    results_df = pd.DataFrame(
        {
            "gene": gene_names,
            "multilayer_pagerank": node_pr,
        }
    )

    results_df = (
        results_df
        .sort_values(
            by="multilayer_pagerank",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Save output.
    # ------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use the patient ID from the actual file name rather than
    # the network-directory name containing omega/aggregation parameters.
    output_stem = supra_file.name.replace(
        "_supra_adj.npz",
        "",
    )

    out_file = output_dir / f"{output_stem}_pagerank.csv"

    results_df.to_csv(
        out_file,
        index=False,
    )

    if logger:
        logger.info(
            "Saved PageRank: %s",
            out_file,
        )

    return results_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute multilayer PageRank for multiplex "
            "patient networks."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the patient multiplex "
            "network folders."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pagerank"),
        help=(
            "Directory where PageRank CSV files will be saved."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help=(
            "PageRank damping/teleportation factor "
            "(default: 0.85)."
        ),
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help=(
            "Convergence tolerance "
            "(default: 1e-12)."
        ),
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help=(
            "Maximum number of PageRank iterations "
            "(default: 10000)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run multilayer PageRank for all patient networks."""
    args = parse_args()

    logger = setup_logger()

    # ------------------------------------------------------------------
    # Validate input arguments.
    # ------------------------------------------------------------------

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {args.input_dir}"
        )

    if not args.input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {args.input_dir}"
        )

    if not 0 < args.alpha < 1:
        raise ValueError(
            "--alpha must be between 0 and 1."
        )

    if args.tol <= 0:
        raise ValueError(
            "--tol must be greater than 0."
        )

    if args.max_iter <= 0:
        raise ValueError(
            "--max-iter must be greater than 0."
        )

    # ------------------------------------------------------------------
    # Find patient network directories.
    # ------------------------------------------------------------------

    sample_dirs = sorted(
        d
        for d in args.input_dir.iterdir()
        if d.is_dir()
    )

    if not sample_dirs:
        raise FileNotFoundError(
            f"No sample network folders found in "
            f"{args.input_dir}"
        )

    total_samples = len(sample_dirs)

    logger.info(
        "Found %d multiplex network folders.",
        total_samples,
    )

    # ------------------------------------------------------------------
    # Create output directory.
    # ------------------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Process all samples.
    # ------------------------------------------------------------------

    successful = 0
    failed = 0

    for idx, sample_dir in enumerate(
        sample_dirs,
        start=1,
    ):
        logger.info(
            "Processing sample %d/%d: %s",
            idx,
            total_samples,
            sample_dir.name,
        )

        try:
            result = process_sample_network(
                sample_dir=sample_dir,
                output_dir=args.output_dir,
                alpha=args.alpha,
                tol=args.tol,
                max_iter=args.max_iter,
                logger=logger,
            )

            if result is not None:
                successful += 1
            else:
                failed += 1

        except Exception as exc:
            failed += 1

            logger.error(
                "Failed to process %s: %s",
                sample_dir.name,
                exc,
            )

    # ------------------------------------------------------------------
    # Validate final output.
    # ------------------------------------------------------------------

    output_files = sorted(
        args.output_dir.glob("*_pagerank.csv")
    )

    logger.info(
        "PageRank processing completed."
    )

    logger.info(
        "Successful samples: %d/%d",
        successful,
        total_samples,
    )

    if failed > 0:
        logger.warning(
            "Failed or skipped samples: %d",
            failed,
        )

    logger.info(
        "PageRank files generated: %d",
        len(output_files),
    )

    if len(output_files) != successful:
        logger.warning(
            "Number of output files (%d) does not match "
            "number of successful samples (%d).",
            len(output_files),
            successful,
        )

    if successful == total_samples:
        logger.info(
            "All %d multiplex samples were processed successfully.",
            total_samples,
        )


if __name__ == "__main__":
    main()
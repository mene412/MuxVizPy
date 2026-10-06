#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def setup_logger() -> logging.Logger:
    """Create and configure the application logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    return logging.getLogger(__name__)


def load_molecular_matrix(
    input_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load a gene x sample molecular matrix.

    Rows are genes and columns are samples.
    """
    logger.info("Loading molecular matrix...")

    matrix = pd.read_csv(input_path, index_col=0)

    if matrix.empty:
        raise ValueError("The molecular matrix is empty.")

    # Convert all molecular values to numeric.
    matrix = matrix.apply(pd.to_numeric, errors="raise")

    logger.info(
        "Loaded molecular matrix: %d genes x %d samples",
        matrix.shape[0],
        matrix.shape[1],
    )

    return matrix.astype(float)


def load_pathway_network(
    pathway_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load and validate the pathway interaction network."""
    logger.info("Loading pathway network...")

    pathway = pd.read_csv(pathway_path)

    required_columns = {"from", "to"}
    missing_columns = required_columns - set(pathway.columns)

    if missing_columns:
        raise ValueError(
            "The pathway network is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    pathway = pathway[["from", "to"]].dropna().copy()

    pathway["from"] = pathway["from"].astype(str).str.strip()
    pathway["to"] = pathway["to"].astype(str).str.strip()

    pathway = pathway[
        (pathway["from"] != "")
        & (pathway["to"] != "")
    ].copy()

    if pathway.empty:
        raise ValueError("The pathway network contains no valid interactions.")

    logger.info(
        "Loaded pathway network: %d interactions",
        len(pathway),
    )

    return pathway


def extract_pathway_nodes(
    pathway: pd.DataFrame,
) -> list[str]:
    """
    Extract unique pathway nodes.

    Complex nodes such as 'CASP9,APPL1' are treated as single
    pathway nodes at the network level but their molecular value
    is calculated from their component genes.
    """
    nodes = set(pathway["from"])
    nodes.update(pathway["to"])

    return sorted(nodes)


def get_component_genes(node_name: str) -> list[str]:
    """Return component genes for a pathway node."""
    return [
        gene.strip()
        for gene in node_name.split(",")
        if gene.strip()
    ]


def aggregate_complex_value(
    node_name: str,
    sample_values: pd.Series,
) -> float:
    """
    Calculate the molecular value of a pathway node.

    For simple genes, the gene value is used directly.
    For complex nodes containing multiple genes, the mean of
    available component-gene values is used.
    """
    component_genes = get_component_genes(node_name)

    available_values = []

    for gene in component_genes:
        if gene in sample_values.index:
            value = sample_values.loc[gene]

            if pd.notna(value):
                available_values.append(float(value))

    if not available_values:
        return 0.0

    return float(np.mean(available_values))


def prepare_pathway_matrix(
    matrix: pd.DataFrame,
    pathway: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Restrict the molecular matrix to genes relevant to the pathway
    and impute missing values using gene-wise medians.
    """
    pathway_nodes = extract_pathway_nodes(pathway)

    # Collect simple genes appearing in the pathway.
    pathway_genes = set()

    for node in pathway_nodes:
        pathway_genes.update(get_component_genes(node))

    matrix_genes = set(matrix.index.astype(str))

    available_genes = sorted(pathway_genes & matrix_genes)

    logger.info(
        "Pathway genes present in molecular matrix: %d",
        len(available_genes),
    )

    if not available_genes:
        raise ValueError(
            "No pathway genes were found in the molecular matrix."
        )

    pathway_matrix = matrix.loc[
        matrix.index.astype(str).isin(available_genes)
    ].copy()

    # Ensure numeric gene index values are represented consistently.
    pathway_matrix.index = pathway_matrix.index.astype(str)

    n_missing_before = int(pathway_matrix.isna().sum().sum())

    logger.info(
        "Missing values among pathway genes before imputation: %d",
        n_missing_before,
    )

    if n_missing_before > 0:
        # Impute each gene using its median across samples.
        gene_medians = pathway_matrix.median(axis=1)

        for gene in pathway_matrix.index:
            if pd.isna(gene_medians.loc[gene]):
                raise ValueError(
                    f"Cannot impute gene '{gene}': "
                    "all values are missing."
                )

            pathway_matrix.loc[gene] = pathway_matrix.loc[gene].fillna(
                gene_medians.loc[gene]
            )

    n_missing_after = int(pathway_matrix.isna().sum().sum())

    logger.info(
        "Missing values among pathway genes after imputation: %d",
        n_missing_after,
    )

    values = pathway_matrix.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "The pathway-restricted molecular matrix contains "
            "non-finite values after preprocessing."
        )

    return pathway_matrix


def build_weighted_network(
    pathway: pd.DataFrame,
    sample_values: pd.Series,
) -> nx.DiGraph:
    """
    Build a weighted directed pathway network for one sample.

    Edge weight:
        w_ij = x_i * x_j

    where x_i and x_j are the molecular values of the source
    and target pathway nodes.
    """
    graph = nx.DiGraph()

    pathway_nodes = extract_pathway_nodes(pathway)

    # Calculate one molecular value for every pathway node.
    node_values = {}

    for node in pathway_nodes:
        node_values[node] = aggregate_complex_value(
            node,
            sample_values,
        )

    # Add nodes and their molecular values.
    for node, value in node_values.items():
        graph.add_node(
            node,
            value=float(value),
        )

    # Add weighted pathway edges.
    for source, target in pathway.itertuples(index=False):
        source_value = node_values[source]
        target_value = node_values[target]

        weight = source_value * target_value

        graph.add_edge(
            source,
            target,
            weight=float(weight),
        )

    return graph


def compute_single_layer_pagerank(
    graph: nx.DiGraph,
    alpha: float,
    tol: float,
    max_iter: int,
) -> dict[str, float]:
    """Compute weighted single-layer PageRank."""
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot calculate PageRank on an empty graph.")

    pagerank = nx.pagerank(
        graph,
        alpha=alpha,
        weight="weight",
        tol=tol,
        max_iter=max_iter,
    )

    return pagerank


def process_sample(
    sample_id: str,
    pathway: pd.DataFrame,
    pathway_matrix: pd.DataFrame,
    output_dir: Path,
    alpha: float,
    tol: float,
    max_iter: int,
    logger: logging.Logger,
) -> None:
    """Calculate and save PageRank for one sample."""
    sample_values = pathway_matrix[sample_id]

    graph = build_weighted_network(
        pathway,
        sample_values,
    )

    pagerank = compute_single_layer_pagerank(
        graph,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )

    result = pd.DataFrame(
        {
            "gene": list(pagerank.keys()),
            "pagerank": list(pagerank.values()),
        }
    )

    result = result.sort_values(
        "pagerank",
        ascending=False,
    ).reset_index(drop=True)

    output_file = output_dir / f"{sample_id}_pagerank.csv"

    result.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved PageRank for %s: %s",
        sample_id,
        output_file,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute single-layer PageRank from a molecular "
            "matrix and a pathway network."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the gene x sample molecular matrix.",
    )

    parser.add_argument(
        "--pathway-path",
        type=Path,
        required=True,
        help="Path to the pathway interaction CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where PageRank files will be saved.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="PageRank damping factor.",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="PageRank convergence tolerance.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of PageRank iterations.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the single-layer PageRank pipeline."""
    args = parse_arguments()

    logger = setup_logger()

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"Molecular matrix not found: {args.input_path}"
        )

    if not args.pathway_path.exists():
        raise FileNotFoundError(
            f"Pathway network not found: {args.pathway_path}"
        )

    if not 0 < args.alpha < 1:
        raise ValueError(
            "The PageRank alpha parameter must be between 0 and 1."
        )

    if args.tol <= 0:
        raise ValueError("The PageRank tolerance must be positive.")

    if args.max_iter <= 0:
        raise ValueError(
            "The maximum number of iterations must be positive."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix = load_molecular_matrix(
        args.input_path,
        logger,
    )

    pathway = load_pathway_network(
        args.pathway_path,
        logger,
    )

    pathway_matrix = prepare_pathway_matrix(
        matrix,
        pathway,
        logger,
    )

    sample_ids = pathway_matrix.columns.tolist()

    logger.info(
        "Processing %d samples...",
        len(sample_ids),
    )

    for i, sample_id in enumerate(sample_ids, start=1):
        logger.info(
            "Processing sample %d/%d: %s",
            i,
            len(sample_ids),
            sample_id,
        )

        process_sample(
            sample_id=sample_id,
            pathway=pathway,
            pathway_matrix=pathway_matrix,
            output_dir=args.output_dir,
            alpha=args.alpha,
            tol=args.tol,
            max_iter=args.max_iter,
            logger=logger,
        )

    output_files = list(
        args.output_dir.glob("*_pagerank.csv")
    )

    if len(output_files) != len(sample_ids):
        raise RuntimeError(
            "The number of PageRank output files does not match "
            f"the number of samples. Expected {len(sample_ids)}, "
            f"found {len(output_files)}."
        )

    logger.info(
        "PageRank calculation completed successfully."
    )

    logger.info(
        "Generated %d PageRank files in %s",
        len(output_files),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
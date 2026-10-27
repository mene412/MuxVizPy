from pathlib import Path
from typing import List
import argparse

import numpy as np
import pandas as pd


def extract_pathway_nodes(pathway_df: pd.DataFrame) -> List[str]:
    """
    Extract the unique network nodes from the pathway.

    Nodes can be:
    - single genes, e.g. RB1
    - complex nodes, e.g. CASP9,APPL1
    """
    nodes = set(pathway_df["from"].dropna().unique()).union(
        set(pathway_df["to"].dropna().unique())
    )

    return sorted(nodes)


def aggregate_complex_value(
    complex_node: str,
    patient_col: pd.Series,
    agg_func: str = "mean",
) -> float:
    """
    Aggregate the molecular values of the genes contained in a
    complex node.

    Example:
        CASP9,APPL1

    becomes:

        mean(CASP9, APPL1)

    Only genes present in the molecular matrix are used.
    """
    genes = [g.strip() for g in complex_node.split(",")]

    available_genes = [
        gene for gene in genes
        if gene in patient_col.index
    ]

    if not available_genes:
        return 0.0

    values = patient_col.loc[available_genes].to_numpy(dtype=float)

    if agg_func == "median":
        return float(np.median(values))

    return float(np.mean(values))


def build_pathway_node_matrix(
    molecular_matrix: pd.DataFrame,
    pathway_nodes: List[str],
    agg_func: str = "mean",
) -> pd.DataFrame:
    """
    Convert a gene x sample molecular matrix into a
    network-node x sample matrix.

    Each pathway node is represented by:
    - its molecular value if it is a single gene
    - the aggregated value of its component genes if it is a complex
    """
    molecular_matrix = molecular_matrix.copy()

    molecular_matrix.index = (
        molecular_matrix.index.astype(str).str.strip()
    )

    # Transpose once so that each patient/sample becomes a column.
    # The original molecular matrices are gene x sample.
    result = pd.DataFrame(
        index=pathway_nodes,
        columns=molecular_matrix.columns,
        dtype=float,
    )

    for node in pathway_nodes:

        if "," not in node:

            # Single-gene node.
            if node in molecular_matrix.index:
                result.loc[node] = molecular_matrix.loc[node]
            else:
                # Same fallback used by aggregate_complex_value().
                result.loc[node] = 0.0

        else:

            # Complex node.
            for sample in molecular_matrix.columns:
                result.loc[node, sample] = aggregate_complex_value(
                    node,
                    molecular_matrix[sample],
                    agg_func=agg_func,
                )

    return result


def prepare_matrix(
    molecular_path: Path,
    pathway_path: Path,
    output_path: Path,
    agg_func: str = "mean",
) -> None:
    """
    Prepare one molecular matrix using the pathway network nodes.
    """

    print(f"\nLoading molecular matrix: {molecular_path}")

    molecular = pd.read_csv(
        molecular_path,
        index_col=0,
    )

    print(
        f"Input: {molecular.shape[0]} genes x "
        f"{molecular.shape[1]} samples"
    )

    print(f"Loading pathway: {pathway_path}")

    pathway_df = pd.read_csv(pathway_path)

    pathway_nodes = extract_pathway_nodes(pathway_df)

    print(
        f"Pathway network nodes: {len(pathway_nodes)}"
    )

    # Build the network-node representation.
    node_matrix = build_pathway_node_matrix(
        molecular_matrix=molecular,
        pathway_nodes=pathway_nodes,
        agg_func=agg_func,
    )

    # Transpose to sample x node because this is the format
    # expected by run_clustering.py for original matrices.
    sample_matrix = node_matrix.T

    # Make sure values are numeric.
    sample_matrix = sample_matrix.astype(float)

    # Handle missing values using node-wise median across samples.
    # This does not modify the original molecular matrix.
    if sample_matrix.isna().any().any():

        n_missing = int(
            sample_matrix.isna().sum().sum()
        )

        print(
            f"Warning: {n_missing} missing values detected. "
            "Imputing using node-wise medians."
        )

        node_medians = sample_matrix.median(axis=0)

        # Fallback for nodes with no valid values.
        node_medians = node_medians.fillna(0.0)

        sample_matrix = sample_matrix.fillna(node_medians)

    if not np.isfinite(sample_matrix.to_numpy()).all():
        raise ValueError(
            "Prepared matrix contains NaN or infinite values."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_matrix.to_csv(output_path)

    print(
        f"Output: {sample_matrix.shape[0]} samples x "
        f"{sample_matrix.shape[1]} network nodes"
    )

    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare DNA and RNA matrices using nodes from a pathway CSV. "
            "Derived matrices are written outside the input data directory."
        )
    )
    parser.add_argument(
        "--pathway-path",
        type=Path,
        required=True,
        help="Pathway CSV containing 'from' and 'to' columns.",
    )
    parser.add_argument(
        "--dna-path",
        type=Path,
        default=Path("mln/data/DNAmethyl_minmax.csv"),
        help="Gene x sample DNA methylation matrix.",
    )
    parser.add_argument(
        "--rna-path",
        type=Path,
        default=Path("mln/data/RNAprof_logTPM_minmax.csv"),
        help="Gene x sample RNA expression matrix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pathway_in_cancer/processed"),
        help="Directory for the generated pathway-node matrices.",
    )
    args = parser.parse_args()

    dna_output = args.output_dir / "DNAmethyl_pathway_nodes.csv"
    rna_output = args.output_dir / "RNAprof_logTPM_pathway_nodes.csv"

    prepare_matrix(
        molecular_path=args.dna_path,
        pathway_path=args.pathway_path,
        output_path=dna_output,
        agg_func="mean",
    )

    prepare_matrix(
        molecular_path=args.rna_path,
        pathway_path=args.pathway_path,
        output_path=rna_output,
        agg_func="mean",
    )


if __name__ == "__main__":
    main()

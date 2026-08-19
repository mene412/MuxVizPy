import argparse
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import graph_tool as gt
from typing import Dict, List, Tuple, Union, Optional

from MuxVizPy.utils import parsing as parsing_utils


def load_and_preprocess_data(
    rna_path: str, 
    methyl_path: str, 
    pathway_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load RNA expression, DNA methylation, and KEGG pathway CSV files.
    
    Parameters
    ----------
    rna_path : str
        Path to RNA data CSV (rows=genes, columns=samples).
    methyl_path : str
        Path to Methylation data CSV (rows=genes, columns=samples).
    pathway_path : str
        Path to pathway interactions CSV ("from", "to").

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames for RNA, methylation, and pathway data.
    """
    rna_df = pd.read_csv(rna_path, index_col=0)
    methyl_df = pd.read_csv(methyl_path, index_col=0)
    pathway_df = pd.read_csv(pathway_path)
    
    return rna_df, methyl_df, pathway_df


def extract_pathway_nodes(pathway_df: pd.DataFrame) -> List[str]:
    """
    Extract the set of unique nodes (single genes or complex nodes) from the pathway.
    """
    nodes = set(pathway_df["from"].dropna().unique()).union(
        set(pathway_df["to"].dropna().unique())
    )
    return sorted(list(nodes))


def aggregate_complex_value(
    complex_node: str, 
    patient_col: pd.Series, 
    agg_func: str = "mean"
) -> float:
    """
    Decompose (explode) a complex node into its component genes extracted from the 
    patient profile, and calculate their aggregation (mean or median).

    Parameters
    ----------
    complex_node : str
        Node string, e.g., "CASP9,APPL1" or "ABL1".
    patient_col : pd.Series
        Expression or methylation series for a single patient (index = gene names).
    agg_func : str
        Aggregation method: "mean" or "median".

    Returns
    -------
    float
        Aggregated normalized value for the node.
    """
    genes = [g.strip() for g in complex_node.split(",")]
    # Filter only genes actually present in the patient's matrix
    available_genes = [g for g in genes if g in patient_col.index]
    
    if not available_genes:
        return 0.0  # Fallback default if no gene is found in the dataset
    
    values = patient_col.loc[available_genes].to_numpy(dtype=float)
    
    if agg_func == "median":
        return float(np.median(values))
    else:  # Default: mean
        return float(np.mean(values))


def build_patient_multiplex(
    patient_id: str,
    rna_df: pd.DataFrame,
    methyl_df: pd.DataFrame,
    pathway_df: pd.DataFrame,
    omega: float = 1.0,
    agg_func: str = "mean"
) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[int, str], gt.Graph, pd.DataFrame]:
    """
    Create a two-layer multiplex network (Layer 0 = RNA, Layer 1 = DNA methylation) 
    for a specific patient/sample.

    Parameters
    ----------
    patient_id : str
        Sample/patient identifier.
    rna_df : pd.DataFrame
        RNA matrix.
    methyl_df : pd.DataFrame
        Methylation matrix.
    pathway_df : pd.DataFrame
        DataFrame of intra-layer edges.
    omega : float
        Inter-layer coupling strength between layers.
    agg_func : str
        Complex node aggregation function ("mean" or "median").

    Returns
    -------
    Tuple[sp.csr_matrix, Dict[str, int], Dict[int, str], gt.Graph, pd.DataFrame]
        - supra_adj: CSR sparse supra-adjacency matrix (2N x 2N)
        - node_to_id: Mapping of node name -> integer index [0, N-1]
        - id_to_node: Mapping of integer index -> node name
        - g_multi: Single multi-layer graph in graph-tool format
        - node_attributes_df: DataFrame with aggregated RNA and methylation values for each node
    """
    nodes = extract_pathway_nodes(pathway_df)
    N = len(nodes)
    node_to_id = {node: idx for idx, node in enumerate(nodes)}
    id_to_node = {idx: node for idx, node in enumerate(nodes)}

    # 1. Intra-Layer Graph Construction (Pathway)
    # Initializes a new directed graph using graph-tool
    g_intra = gt.Graph(directed=True)
    # Add N vertices to the graph
    g_intra.add_vertex(N)
    
    edge_list = []
    for _, row in pathway_df.iterrows():
        u_str, v_str = str(row["from"]).strip(), str(row["to"]).strip()
        if u_str in node_to_id and v_str in node_to_id:
            edge_list.append((node_to_id[u_str], node_to_id[v_str]))

    # Add all validated edges  
    g_intra.add_edge_list(edge_list)
    
    # Intra-layer adjacency matrix NxN in SciPy Sparse CSR format
    intra_adj = gt.spectral.adjacency(g_intra).tocsr()
    # Both layers share the same intra-layer pathway topology
    intra_networks = [intra_adj, intra_adj]
    
    # 2. Compute RNA and Methylation profiles for the specific patient
    patient_rna = rna_df[patient_id]
    patient_methyl = methyl_df[patient_id]
    
    rna_vals = np.array([aggregate_complex_value(node, patient_rna, agg_func) for node in nodes])
    methyl_vals = np.array([aggregate_complex_value(node, patient_methyl, agg_func) for node in nodes])
    
    node_attributes_df = pd.DataFrame({
        "node_name": nodes,
        "rna_expression": rna_vals,
        "dna_methylation": methyl_vals
    }, index=[node_to_id[node] for node in nodes])

    # 3. Intra-Layer Weighted Construction
    D_rna = sp.diags(rna_vals, format="csr")
    D_methyl = sp.diags(methyl_vals, format="csr")

    # Double diagonal multiplication scales row u by Value_u and column v by Value_v
    intra_adj_rna = D_rna @ intra_adj @ D_rna
    intra_adj_methyl = D_methyl @ intra_adj @ D_methyl

    # Combine weighted intra-layer topologies into block-diagonal structure
    intra_block = sp.block_diag([intra_adj_rna, intra_adj_methyl], format="csr")
    
    # 4. Inter-Layer Weighted Construction
    # Compute node-specific interlayer coupling weights: omega * rna * methylation
    interlayer_weights = omega * rna_vals * methyl_vals

    # Off-diagonal blocks: Weighted diagonal coupling between replica nodes across layers
    inter_diag = sp.diags(interlayer_weights, format="csr")

    # Create off-diagonal blocks for 2 layers (Layer 0 <-> Layer 1)
    inter_block = sp.bmat([
        [None, inter_diag],
        [inter_diag, None]
    ], format="csr")

    # Final supra-adjacency matrix (2N x 2N)
    supra_adj = intra_block + inter_block

    # 5. Generate Integrated graph-tool Graph
    g_multi = gt.Graph(directed=True)
    g_multi.add_vertex(2 * N)  # N nodes at layer 0 + N nodes at layer 1
    
    # Add edges from the supra-adjacency matrix
    coo = supra_adj.tocoo()
    g_multi.add_edge_list(np.vstack([coo.row, coo.col]).T)
    ep_weight = g_multi.new_edge_property("double")
    ep_weight.a = coo.data
    g_multi.ep["weight"] = ep_weight
    
    # Node and edge properties
    vp_name = g_multi.new_vertex_property("string")
    vp_layer = g_multi.new_vertex_property("int")
    vp_value = g_multi.new_vertex_property("double")
    
    for i in range(N):
        # Layer 0 (RNA)
        v0 = i
        vp_name[v0] = id_to_node[i]
        vp_layer[v0] = 0
        vp_value[v0] = rna_vals[i]
        
        # Layer 1 (Methylation)
        v1 = i + N
        vp_name[v1] = id_to_node[i]
        vp_layer[v1] = 1
        vp_value[v1] = methyl_vals[i]
        
    g_multi.vp["name"] = vp_name
    g_multi.vp["layer"] = vp_layer
    g_multi.vp["expression_or_methylation_val"] = vp_value
    
    return supra_adj, node_to_id, id_to_node, g_multi, node_attributes_df


def save_multiplex_network(
    output_dir: str,
    patient_id: str,
    supra_adj: sp.csr_matrix,
    g_multi: gt.Graph,
    node_attributes_df: pd.DataFrame,
    omega: float = 1.0,
    agg_func: str = "mean"
) -> None:
    """
    Save the network and related data to disk in reusable standard formats,
    organized by patient ID and configuration parameters.

    Saved Formats:
    1. GraphML: Full graph with node attributes and layers.
    2. NPZ: Sparse supra-adjacency matrix.
    3. CSV: Table containing node attributes (methylation and expression values per node).
    """
    # Format directory name with patient ID, omega, and agg_func
    folder_name = f"{patient_id}_omega_{omega}_agg_func_{agg_func}"
    patient_dir = os.path.join(output_dir, folder_name)
    os.makedirs(patient_dir, exist_ok=True)
    
    # Save complete Graph in GraphML format
    graphml_path = os.path.join(patient_dir, f"{patient_id}_multiplex.graphml")
    g_multi.save(graphml_path)
    
    # Save Supra-Adjacency Matrix in compressed sparse NPZ format
    npz_path = os.path.join(patient_dir, f"{patient_id}_supra_adj.npz")
    sp.save_npz(npz_path, supra_adj)
    
    # Save Node Attributes to CSV
    attr_path = os.path.join(patient_dir, f"{patient_id}_node_attributes.csv")
    node_attributes_df.to_csv(attr_path)


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build multi-layer patient multiplex networks from omics and pathway data."
    )
    parser.add_argument(
        "--rna_path",
        type=str,
        required=True,
        help="Path to RNA data CSV (rows=genes, columns=samples).",
    )
    parser.add_argument(
        "--methyl_path",
        type=str,
        required=True,
        help="Path to Methylation data CSV (rows=genes, columns=samples).",
    )
    parser.add_argument(
        "--pathway_path",
        type=str,
        required=True,
        help="Path to pathway interactions CSV with 'from' and 'to' columns.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/multiplex_networks",
        help="Root directory where output patient folders are stored (default: output/multiplex_networks).",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help="Inter-layer coupling strength (default: 1.0).",
    )
    parser.add_argument(
        "--agg_func",
        type=str,
        choices=["mean", "median"],
        default="mean",
        help="Aggregation method for complex nodes: 'mean' or 'median' (default: mean).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    print("Loading datasets...")
    rna_df, methyl_df, pathway_df = load_and_preprocess_data(
        rna_path=args.rna_path,
        methyl_path=args.methyl_path,
        pathway_path=args.pathway_path,
    )

    patients = sorted(
        list(set(rna_df.columns).intersection(set(methyl_df.columns)))
    )
    print(f"Found {len(patients)} common patients.")

    print(
        f"Starting multiplex generation (omega={args.omega}, agg_func={args.agg_func})..."
    )
    for idx, p_id in enumerate(patients):
        supra_adj, node_to_id, id_to_node, g_multi, node_attrs = (
            build_patient_multiplex(
                patient_id=p_id,
                rna_df=rna_df,
                methyl_df=methyl_df,
                pathway_df=pathway_df,
                omega=args.omega,
                agg_func=args.agg_func,
            )
        )

        save_multiplex_network(
            output_dir=args.output_dir,
            patient_id=p_id,
            supra_adj=supra_adj,
            g_multi=g_multi,
            node_attributes_df=node_attrs,
            omega=args.omega,
            agg_func=args.agg_func,
        )
        print(
            f"[{idx+1}/{len(patients)}] Saved multiplex network for patient: {p_id}"
        )

    print("Processing complete!")
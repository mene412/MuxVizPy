import numpy as np
import graph_tool as gt
from .utils.parsing import get_aggregate_network
from .utils.parsing import get_node_tensor_from_network_list
from .versatility import *

def get_percolation(
    g_list: list[gt.Graph], 
    layers: int, 
    nodes: int, 
    order: np.ndarray
) -> dict[str, np.ndarray | float]:
    """
    Executes a vertex percolation process on a multilayer network's aggregate graph.

    Parameters
    ----------
    g_list : list of graph_tool.Graph
        List of graphs, one per layer of the multilayer network.
    layers : int
        Number of layers in the network.
    nodes : int
        Number of physical nodes (used for normalization).
    order : np.ndarray
        Array specifying the order in which nodes are removed during percolation.

    Returns
    -------
    dict
        Dictionary with:
        - "1ComponentSize": np.ndarray of LCC sizes at each step
        - "2ComponentSize": np.ndarray of 2nd-largest component sizes
        - "CritPoint": float, estimated phase transition point (fraction of removed nodes)
    """
    tensor = get_node_tensor_from_network_list(g_list)
    g_agg = get_aggregate_network(tensor)
    
    perc_agg_1 = gt.topology.vertex_percolation(g_agg, order)[0]
    perc_agg_2 = gt.topology.vertex_percolation(g_agg, order, second=True)[0]
    max_perc = np.argmax(perc_agg_2)/len(perc_agg_1)

    return {"1ComponentSize": perc_agg_1, "2ComponentSize": perc_agg_2, "CritPoint": max_perc}

import numpy as np
import pandas as pd
import graph_tool as gt
from operator import itemgetter

import warnings
warnings.filterwarnings("ignore")

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

class VirusMultiplex():
    """
    Builds a multilayer protein-protein interaction (PPI) network using metadata and edge/node data from viruses.

    Parameters
    ----------
    indexes : list of int
        List of row indices from `virus_metadata` to include in the analysis.
    target_folder : str
        Path to the folder containing the edge and node CSV files for each virus.
    virus_metadata : pandas.DataFrame
        Metadata dataframe including virus name, shorthand, and neighborhood order.
    NEIGH_ORDER : int, optional
        Neighborhood order to filter virus metadata entries (default is 1).

    Attributes
    ----------
    g_list : list of graph_tool.Graph
        List of per-layer PPI networks, one per virus.
    g_multi : graph_tool.Graph
        Edge-colored multilayer network containing all edges across viruses.
    node_map : dict
        Dictionary mapping protein names to unique integer node indices.
    mux_ppi : pandas.DataFrame
        Concatenated edge list with additional node/layer mappings.
    layer_map : dict
        Mapping from layer name to layer index.
    net_description : str
        Human-readable description of the multilayer structure.
    Layers : int
        Total number of layers (viruses).
    Nodes : int
        Total number of unique proteins (nodes).
    Edges : int
        Total number of edges in the multilayer network.
    """
    def __init__(self, indexes, target_folder, virus_metadata, NEIGH_ORDER=1):
        self.target_folder = target_folder
        self.indexes = indexes
        self.mux_ppi = pd.DataFrame()
        self.virus_list = []
        for i in self.indexes:
            if virus_metadata.loc[i,"neigh_order"]==NEIGH_ORDER:
                virus_path = self.target_folder + virus_metadata.loc[i,"virus"]
                self.virus_list.append(virus_metadata.loc[i,"virus_short"])

                virus_edges = pd.read_csv(virus_path+"/edges.csv", header=0, sep=",")
                virus_nodes = pd.read_csv(virus_path+"/nodes.csv", header=0, sep=",")

                if virus_nodes.shape[0]<=1: continue

                human_prot = virus_nodes[np.any([(virus_nodes["type"]==1), (virus_nodes["type"]==2)], axis=0)]["node"]
                human_ppi = virus_edges[np.all([(virus_edges["V1"].isin(human_prot)), (virus_edges["V2"].isin(human_prot))],axis=0)]

                if human_ppi.shape[0]>0:
                    human_ppi["layer"] = virus_metadata.loc[i,"virus_short"]
                    self.mux_ppi = pd.concat([self.mux_ppi, human_ppi])

        self.mux_ppi = self.mux_ppi.reset_index()


        self.Layers = self.mux_ppi["layer"].unique().shape[0]
        self.layer_map = {self.mux_ppi["layer"].unique()[i]: i for i in range(self.Layers) }

        # mapping node names to ids
        self.unique_nodes = pd.concat([self.mux_ppi["V1"], self.mux_ppi["V2"]]).unique()
        self.node_map = {self.unique_nodes[i]: i for i in range(self.unique_nodes.shape[0]) }

        self.Nodes = self.unique_nodes.shape[0]
        self.Edges = self.mux_ppi.shape[0]

        self.mux_ppi["nodeA"] = itemgetter(*self.mux_ppi["V1"].to_numpy())(self.node_map)
        self.mux_ppi["nodeB"] = itemgetter(*self.mux_ppi["V2"].to_numpy())(self.node_map)
        self.mux_ppi["l"] = itemgetter(*self.mux_ppi["layer"].to_numpy())(self.layer_map)

        #generate an list of networks and aggregate/aggregate-binary network
        self.g_list = []

        for l in range(self.Layers):
            tmp = self.mux_ppi[self.mux_ppi["l"]==l].reset_index()[["nodeA", "nodeB"]]
            tmp = tmp.rename({"nodeA":"source", "nodeB":"target"}, axis=1)
            self.g_list.append(gt.Graph(directed=False))
            self.g_list[l].add_edge_list(tmp.values)
            self.g_list[l].add_vertex(self.Nodes-self.g_list[l].num_vertices())

        self.net_description = "Multiplex with " + str(self.Layers) + " layers, " + str(self.Nodes) + " nodes and " + str(self.Edges) + " edges"

        #generate the mulilayer network structure
        tmp_multi = self.mux_ppi.reset_index()[["nodeA", "nodeB"]]
        tmp_multi = tmp_multi.rename({"nodeA":"source", "nodeB":"target"}, axis=1)
        self.g_multi = gt.Graph(directed=False)
        self.g_multi.add_edge_list(tmp_multi.values)
        ep_weight_lay = []

        for l in range(self.Layers):
            ep_weight_lay.append([l]*self.mux_ppi["l"].value_counts()[l])
        ep_weight_lay = np.concatenate(ep_weight_lay)

        self.g_multi.edge_properties["weight"]=self.g_multi.new_edge_property("int", ep_weight_lay)
                
            
            
class VirusMultiplex_from_dirlist():
    """
    Builds a multilayer PPI network from a list of directories, each containing a layer's edge list.

    Parameters
    ----------
    dirlist : list of str
        List of paths, each containing an "edges.csv" file. The name of the folder will be used as the layer label.

    Attributes
    ----------
    g_list : list of graph_tool.Graph
        List of per-layer PPI networks.
    g_multi : graph_tool.Graph
        Edge-colored multilayer network combining all layer edges.
    node_map : dict
        Dictionary mapping protein names to node indices.
    mux_ppi : pandas.DataFrame
        Combined edge list with layer annotations and node mappings.
    layer_map : dict
        Mapping from layer labels to indices.
    net_description : str
        Human-readable description of the multilayer structure.
    Layers : int
        Number of layers.
    Nodes : int
        Number of unique nodes.
    Edges : int
        Number of total edges across all layers.
    """
    def __init__(self, dirlist):
        self.dirlist = dirlist 
        self.mux_ppi = pd.DataFrame()
        for idir in self.dirlist:
            human_ppi = pd.read_csv(idir+"/edges.csv", header=0, sep=",")
            human_ppi.columns = ["source", "target"]
            human_ppi["layer"] = idir.split("/")[-1]
            self.mux_ppi = pd.concat([self.mux_ppi, human_ppi])

        self.mux_ppi = self.mux_ppi.reset_index()


        self.Layers = self.mux_ppi["layer"].unique().shape[0]
        self.layer_map = {self.mux_ppi["layer"].unique()[i]: i for i in range(self.Layers) }

        # mapping node names to ids
        self.unique_nodes = pd.concat([self.mux_ppi["source"], self.mux_ppi["target"]]).unique()
        self.node_map = {self.unique_nodes[i]: i for i in range(self.unique_nodes.shape[0]) }

        self.Nodes = self.unique_nodes.shape[0]
        self.Edges = self.mux_ppi.shape[0]

        self.mux_ppi["nodeA"] = itemgetter(*self.mux_ppi["source"].to_numpy())(self.node_map)
        self.mux_ppi["nodeB"] = itemgetter(*self.mux_ppi["target"].to_numpy())(self.node_map)
        self.mux_ppi["l"] = itemgetter(*self.mux_ppi["layer"].to_numpy())(self.layer_map)

        #generate an list of networks and aggregate/aggregate-binary network
        self.g_list = []

        for l in range(self.Layers):
            tmp = self.mux_ppi[self.mux_ppi["l"]==l].reset_index()[["nodeA", "nodeB"]]
            tmp = tmp.rename({"nodeA":"source", "nodeB":"target"}, axis=1)
            self.g_list.append(gt.Graph(directed=False))
            self.g_list[l].add_edge_list(tmp.values)
            self.g_list[l].add_vertex(self.Nodes-self.g_list[l].num_vertices())

        self.net_description = "Multiplex with " + str(self.Layers) + " layers, " + str(self.Nodes) + " nodes and " + str(self.Edges) + " edges"

        #generate the mulilayer network structure
        tmp_multi = self.mux_ppi.reset_index()[["nodeA", "nodeB"]]
        tmp_multi = tmp_multi.rename({"nodeA":"source", "nodeB":"target"}, axis=1)
        self.g_multi = gt.Graph(directed=False)
        self.g_multi.add_edge_list(tmp_multi.values)
        ep_weight_lay = []

        for l in range(self.Layers):
            ep_weight_lay.append([l]*self.mux_ppi["l"].value_counts()[l])
        ep_weight_lay = np.concatenate(ep_weight_lay)

        self.g_multi.edge_properties["weight"]=self.g_multi.new_edge_property("int", ep_weight_lay)
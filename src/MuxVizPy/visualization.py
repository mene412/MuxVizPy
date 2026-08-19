from MuxVizPy import versatility
from MuxVizPy.utils import parsing as parsing_utils

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graph_tool as gt


def Visualize_EdgeColoredNet(net, n_nodes=30, centr=None, pos_idx="agg", azim=10, elev=20, n_pow=3):
    np.random.seed(1432)
    node_tensor = parsing_utils.get_node_tensor_from_network_list(net.g_list)
    if centr==None:
        centr = versatility.get_multi_RW_centrality_edge_colored(node_tensor=node_tensor)
        top_nodes = centr.sort_values("vers", ascending=False).head(n_nodes)["phy nodes"].to_numpy()
        sizes = np.exp(centr.sort_values("vers", ascending=False).head(n_nodes)["vers"].to_numpy())
        sizes = np.power(sizes,n_pow)
    else:
        top_nodes = np.argmax(centr)[::-1][:n_nodes]
        sizes = np.exp(np.sort(centr)[::-1][:n_nodes])
        sizes = np.power(sizes,n_pow)

    neighbors_mask = np.isin(np.arange(net.Nodes), top_nodes)
    gf_list = []
    for i in range(net.Layers):
        gw = gt.GraphView(net.g_list[i], vfilt=neighbors_mask)
        gf_list.append(gt.Graph(gw, prune=True))

    if pos_idx=="agg":
        g_agg = parsing_utils.get_aggregate_network(gf_list, obj_type="glist")
        positions = gt.draw.sfdp_layout(g_agg).get_2d_array([0,1])
    else:
        positions = gt.draw.sfdp_layout(gf_list[pos_idx]).get_2d_array([0,1])

    x_width = positions[0].max()-positions[0].min()
    y_width = positions[1].max()-positions[1].min()

    ax = plt.figure(figsize=(12,15)).add_subplot(projection='3d')
    xx, yy = np.meshgrid(np.linspace(positions[0].min()-x_width*0.1, positions[0].max()+x_width*0.1,2), np.linspace(positions[1].min()-y_width*0.1, positions[1].max()+y_width*0.1,2))
    X =  xx
    Y =  yy
    for i in range(len(gf_list)):
        ax.text(positions[0].min()-x_width*0.2, positions[1].max()-y_width*0.2,i, net.virus_list[i])
        ax.scatter(positions[0], positions[1], zs=i, zdir='z', label=str(np.unique(net.mux_ppi["layer"])[i]), s=sizes, alpha=0.8)
        ax
        for e in gf_list[i].get_edges():
            ax.plot([positions[0][e[0]], positions[0][e[1]]],[positions[1][e[0]],positions[1][e[1]]] ,zs=[i,i], c="k", lw=0.1)
        Z =  i*np.ones(X.shape)
        ax.plot_surface(X,Y,Z, rstride=1, cstride=1, alpha=0.5)

    ax.set_xlim(positions[0].min()-x_width*0.2, positions[0].max()+x_width*0.2)
    ax.set_zlim(0, len(gf_list)-1)
    ax.set_ylim(positions[1].min()-y_width*0.2, positions[1].max()+y_width*0.2)
    ax.axis("off")

    ax.view_init(elev=elev, azim=azim)

    plt.show()


def plotMultiplex(
    g_list,
    g_agg,
    positions=None,
    elev=20,
    azim=10,
    label="",
    show_edges=True,
    max_edges_per_layer=5000,
    edge_alpha=0.12,
    edge_lw=0.3,
    min_size=4.0,
    max_size=50.0,
    size_mode="global",  # "global" (default) or "per_layer"
):
    """
    3D layered plot of a multiplex. Only intra-layer edges are drawn.

    Node sizes:
      - "global": size by total degree across ALL layers (same size in every layer)
      - "per_layer": size by degree within the current layer
      Zero-degree nodes are still shown with `min_size`.
    """
    # ----- Layout -----
    if positions is None:
        import graph_tool.draw as gtdraw  # local to avoid circular imports
        positions = gtdraw.sfdp_layout(g_agg).get_2d_array([0, 1])

    L = len(g_list)
    N = positions.shape[1]
    rng = np.random.default_rng(1432)

    # ----- Degrees & sizes -----
    deg_per_layer = [g.get_total_degrees(g.get_vertices()).astype(float) for g in g_list]

    if size_mode == "global":
        # Sum degree over layers, same size across layers
        deg_global = np.sum(np.vstack(deg_per_layer), axis=0)
        dmax = float(deg_global.max()) if deg_global.size and deg_global.max() > 0 else 1.0
        sizes_global = min_size + (deg_global / dmax) * (max_size - min_size)
    # per_layer handled inside the loop

    # ----- Figure -----
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    # For planes
    xmin, xmax = positions[0].min(), positions[0].max()
    zmin, zmax = positions[1].min(), positions[1].max()
    xx, zz = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(zmin, zmax, 2))

    for layer_idx, g in enumerate(g_list):
        # Node sizes
        if size_mode == "global":
            sizes = sizes_global
        else:  # per_layer
            d = deg_per_layer[layer_idx]
            dmax = float(d.max()) if d.size and d.max() > 0 else 1.0
            sizes = min_size + (d / dmax) * (max_size - min_size)

        # Nodes on plane Y=layer_idx
        ax.scatter(
            positions[0], positions[1],
            zs=layer_idx, zdir="y",
            s=sizes,
            edgecolors="none",
            label=label,
        )

        # Intra-layer edges (only)
        if show_edges and g.num_edges() > 0:
            edges = g.get_edges()[:, :2]
            if edges.shape[0] > max_edges_per_layer:
                sel = rng.choice(edges.shape[0], size=max_edges_per_layer, replace=False)
                edges = edges[sel]
            y = layer_idx
            for u, v in edges:
                ax.plot(
                    [positions[0][u], positions[0][v]],  # X
                    [y, y],                              # Y fixed = layer
                    [positions[1][u], positions[1][v]],  # Z
                    alpha=edge_alpha,
                    linewidth=edge_lw,
                    color="k",
                )

        # Layer plane (neutral)
        Y = np.full_like(xx, layer_idx, dtype=float)
        ax.plot_surface(xx, Y, zz, rstride=1, cstride=1, alpha=0.5)

    # Cosmetics & alignment
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, L - 1)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X")
    ax.set_ylabel("Layer")
    ax.set_zlabel("Z")
    ax.axis("off")
    ax.set_proj_type("ortho")
    ax.set_box_aspect((xmax - xmin, L, zmax - zmin))
    ax.view_init(elev=elev, azim=azim)

    plt.show()
    return positions

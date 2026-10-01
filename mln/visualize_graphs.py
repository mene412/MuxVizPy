import os
import matplotlib.pyplot as plt
import graph_tool as gt
from MuxVizPy import visualization
from MuxVizPy.utils import parsing as parsing_utils
import graph_tool.draw as gtdraw


def plot_patient_multiplex_3d(g_multi, patient_id: str):
    """
    Visualizza la rete multiplex del paziente in 3D separando i due layer
    (Layer 0: RNA, Layer 1: Metilazione DNA).
    """
    # Separiamo i due layer dal grafo graph-tool riletto da file
    g_layer0 = gt.GraphView(g_multi, vfilt=lambda v: g_multi.vp["layer"][v] == 0)
    g_layer1 = gt.GraphView(g_multi, vfilt=lambda v: g_multi.vp["layer"][v] == 1)

    # Pruniamo le viste per isolare la struttura dei grafi per MuxVizPy
    g_list = [gt.Graph(g_layer0, prune=True), gt.Graph(g_layer1, prune=True)]

    # Grafo aggregato per calcolare il layout condiviso sui piani 3D
    g_agg = parsing_utils.get_aggregate_network(g_list, obj_type="glist")

    # Rendering 3D con MuxVizPy
    visualization.plotMultiplex(
        g_list=g_list,
        g_agg=g_agg,
        elev=25,
        azim=30,
        show_edges=True,
        edge_alpha=0.3,
        min_size=20,
        max_size=120,
        size_mode="per_layer",
    )

    plt.suptitle(f"Multiplex Network 3D - Patient: {patient_id}", fontsize=14)
    plt.show()


def plot_multiplex_side_by_side(g_multi, patient_id: str, n_nodes: int = None, id_to_node: dict = None):
    """
    Genera una figura 2D con i due layer affiancati (RNA e Metilazione)
    usando un layout topologico condiviso.
    """
    # 1. Ricava n_nodes se non fornito (nodi totali / 2 layer)
    if n_nodes is None:
        n_nodes = g_multi.num_vertices() // 2

    # 2. Estrazione e pruning dei due layer
    g_rna_view = gt.GraphView(g_multi, vfilt=lambda v: g_multi.vp["layer"][v] == 0)
    g_met_view = gt.GraphView(g_multi, vfilt=lambda v: g_multi.vp["layer"][v] == 1)

    g_rna = gt.Graph(g_rna_view, prune=True)
    g_met = gt.Graph(g_met_view, prune=True)

    # 3. Calcolo del layout unico e condiviso (basato sul primo layer)
    pos = gtdraw.sfdp_layout(g_rna)

    # 4. Creazione della figura con due subplot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Grafico 1: Layer RNA
    axes[0].set_title(f"RNA Expression Layer\nPatient: {patient_id}", fontsize=12)
    gtdraw.graph_draw(
        g_rna,
        pos=pos,
        vertex_fill_color=g_rna.vp["expression_or_methylation_val"],
        vcmap=plt.cm.Blues,
        vertex_size=12,
        edge_pen_width=0.6,
        mpl_axes=axes[0]
    )

    # Grafico 2: Layer Metilazione DNA
    axes[1].set_title(f"DNA Methylation Layer\nPatient: {patient_id}", fontsize=12)
    gtdraw.graph_draw(
        g_met,
        pos=pos,
        vertex_fill_color=g_met.vp["expression_or_methylation_val"],
        vcmap=plt.cm.YlOrRd,
        vertex_size=12,
        edge_pen_width=0.6,
        mpl_axes=axes[1]
    )

    # Aggiunta di colorbar indicative per le scale di valore
    sm_rna = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
    sm_rna._A = []
    cbar_rna = fig.colorbar(sm_rna, ax=axes[0], shrink=0.7)
    cbar_rna.set_label("RNA Expression (MinMax)", fontsize=10)

    sm_met = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=1))
    sm_met._A = []
    cbar_met = fig.colorbar(sm_met, ax=axes[1], shrink=0.7)
    cbar_met.set_label("DNA Methylation (MinMax)", fontsize=10)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import graph_tool as gt
import graph_tool.draw as gtdraw

def plot_single_layer(g_multi, patient_id: str, layer_index: int = 0):
    """
    Visualizza in 2D un singolo layer della rete multiplex.
    
    Parameters
    ----------
    g_multi : gt.Graph
        Il grafo multiplex caricato dal file .graphml.
    patient_id : str
        L'ID del paziente.
    layer_index : int
        0 per il layer RNA, 1 per il layer Metilazione DNA.
    """
    # Mappatura del nome e della Mappa di Colori (colormap) in base al layer
    if layer_index == 0:
        layer_name = "RNA Expression"
        cmap = plt.cm.Blues
        cbar_label = "RNA Expression (Normalized)"
    elif layer_index == 1:
        layer_name = "DNA Methylation"
        cmap = plt.cm.YlOrRd
        cbar_label = "DNA Methylation (Normalized)"
    else:
        raise ValueError("layer_index deve essere 0 (RNA) o 1 (Metilazione).")

    # 1. Filtriamo il layer richiesto e lo isoliamo (pruning)
    g_layer_view = gt.GraphView(g_multi, vfilt=lambda v: g_multi.vp["layer"][v] == layer_index)
    g_layer = gt.Graph(g_layer_view, prune=True)

    # 2. Calcoliamo il layout 2D delle interazioni
    pos = gtdraw.sfdp_layout(g_layer)

    # 3. Disegniamo il grafico
    fig, ax = plt.subplots(figsize=(10, 8))
    
    gtdraw.graph_draw(
        g_layer,
        pos=pos,
        vertex_fill_color=g_layer.vp["expression_or_methylation_val"],  # Colore in base ai valori
        vcmap=cmap,
        vertex_size=14,
        edge_pen_width=0.8,
        mpl_axes=ax
    )

    # 4. Aggiungiamo la barra dei colori per interpretare i valori dei nodi
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=11)

    plt.title(f"Patient: {patient_id}\nLayer {layer_index}: {layer_name}", fontsize=13)
    plt.tight_layout()
    plt.show()

# def main():
#     # Cartella dove sono state salvate le reti
#     OUTPUT_DIR = "output/gpt_multiplex_networks"

#     if not os.path.exists(OUTPUT_DIR):
#         raise FileNotFoundError(
#             f"La cartella '{OUTPUT_DIR}' non esiste. Assicurati di aver generato prima le reti."
#         )

#     # Elenca le sottocartelle dei pazienti disponibili
#     patient_dirs = [
#         d for d in os.listdir(OUTPUT_DIR) 
#         if os.path.isdir(os.path.join(OUTPUT_DIR, d))
#     ]

#     if not patient_dirs:
#         raise FileNotFoundError(
#             f"Nessuna rete trovata all'interno della cartella '{OUTPUT_DIR}'."
#         )

#     print(f"Trovate {len(patient_dirs)} reti salvate.")

#     # Seleziona il primo paziente salvato (oppure uno specifico)
#     target_patient_id = sorted(patient_dirs)[0]
#     graphml_path = os.path.join(
#         OUTPUT_DIR, target_patient_id, f"{target_patient_id}_multiplex.graphml"
#     )

#     if not os.path.exists(graphml_path):
#         raise FileNotFoundError(
#             f"File GraphML non trovato per il paziente {target_patient_id}: {graphml_path}"
#         )

#     print(f"Caricamento rete salvata per il paziente '{target_patient_id}' da:\n  {graphml_path}")
    
#     # Caricamento del grafo salvato tramite graph-tool
#     g_multi = gt.load_graph(graphml_path)

#     print("Visualizzazione 3D in corso...")
#     plot_multiplex_side_by_side(g_multi, patient_id=target_patient_id)

# import os
# import graph_tool as gt

def main():
    OUTPUT_DIR = "output/multiplex_networks"

    if not os.path.exists(OUTPUT_DIR):
        raise FileNotFoundError(f"La cartella '{OUTPUT_DIR}' non esiste.")

    # Elenco ordinato dei pazienti
    patient_dirs = sorted([
        d for d in os.listdir(OUTPUT_DIR) 
        if os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ])

    # Seleziona il primo paziente (o cambia indice)
    target_patient_id = patient_dirs[0]
    graphml_path = os.path.join(
        OUTPUT_DIR, target_patient_id, f"{target_patient_id}_multiplex.graphml"
    )

    print(f"Caricamento rete per il paziente: {target_patient_id}")
    g_multi = gt.load_graph(graphml_path)

    # --- SCEGLI QUALE LAYER MOSTRARE ---
    # 0 = Mostra solo il layer RNA Expression
    # 1 = Mostra solo il layer DNA Methylation
    LAYER_TO_PLOT = 0 

    print(f"Visualizzazione singola per Layer {LAYER_TO_PLOT}...")
    plot_single_layer(g_multi, patient_id=target_patient_id, layer_index=LAYER_TO_PLOT)

if __name__ == "__main__":
    main()
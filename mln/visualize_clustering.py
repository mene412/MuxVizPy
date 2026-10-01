from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path("output/clustering/pagerank_comprehensive/all_samples")
PLOTS_DIR = BASE_DIR / "plots_k2_and_louvain"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Scegli la colonna da usare come true label ('group' oppure 'CMS')
TRUE_LABEL_COL = "group"

# Metodi k=2 (sotto 'clusters_by_k') e Louvain (sotto 'louvain')
METHODS = [
    {
        "name": "hierarchical_euclidean",
        "dir": BASE_DIR / "clusters_by_k/euclidean/k2",
        "prefix": "all_samples_hierarchical_euclidean_k2",
        "k_title": "k=2",
    },
    {
        "name": "hierarchical_cosine",
        "dir": BASE_DIR / "clusters_by_k/cosine/k2",
        "prefix": "all_samples_hierarchical_cosine_k2",
        "k_title": "k=2",
    },
    {
        "name": "hierarchical_correlation",
        "dir": BASE_DIR / "clusters_by_k/correlation/k2",
        "prefix": "all_samples_hierarchical_correlation_k2",
        "k_title": "k=2",
    },
    {
        "name": "kmeans_euclidean",
        "dir": BASE_DIR / "clusters_by_k/euclidean/k2",
        "prefix": "all_samples_kmeans_k2",
        "k_title": "k=2",
    },
    {
        "name": "louvain_cosine",
        "dir": BASE_DIR / "louvain",
        "prefix": "all_samples_louvain",
        "k_title": "Louvain",
    },
]


def plot_side_by_side(
    df: pd.DataFrame,
    dim1_col: str,
    dim2_col: str,
    method_name: str,
    embedding_name: str,
    true_label_col: str,
    k_label: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # 1. Cluster Labels
    n_clusters = df["cluster"].nunique()
    palette_clusters = sns.color_palette("tab10" if n_clusters <= 10 else "tab20", n_colors=n_clusters)
    sns.scatterplot(
        data=df,
        x=dim1_col,
        y=dim2_col,
        hue="cluster",
        palette=palette_clusters,
        ax=axes[0],
        s=65,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.5,
    )
    axes[0].set_title(f"Clustering Labels ({k_label})", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(f"{dim1_col}")
    axes[0].set_ylabel(f"{dim2_col}")
    axes[0].legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. True Labels
    valid_mask = df[true_label_col].notna() & (df[true_label_col].astype(str).str.strip() != "")
    df_valid = df[valid_mask].copy()

    n_classes = df_valid[true_label_col].nunique()
    palette_true = sns.color_palette("Set2" if n_classes <= 8 else "tab20", n_colors=max(1, n_classes))

    sns.scatterplot(
        data=df_valid,
        x=dim1_col,
        y=dim2_col,
        hue=true_label_col,
        palette=palette_true,
        ax=axes[1],
        s=65,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.5,
    )
    axes[1].set_title(f"True Labels ({true_label_col})", fontsize=13, fontweight="bold")
    axes[1].set_xlabel(f"{dim1_col}")
    axes[1].set_ylabel(f"{dim2_col}")
    axes[1].legend(title=true_label_col, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"{embedding_name.upper()} Comparison: {method_name.replace('_', ' ').title()}", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvato grafico: {save_path.name}")


def main() -> None:
    for method in METHODS:
        target_dir = method["dir"]
        prefix = method["prefix"]

        # Recupera assegnazioni dei cluster e metadati
        assign_file = target_dir / f"{prefix}_assignments.csv"
        if not assign_file.exists():
            print(f"File di assegnazione non trovato: {assign_file}, salto...")
            continue

        assign_df = pd.read_csv(assign_file)
        if TRUE_LABEL_COL not in assign_df.columns:
            print(f"Colonna '{TRUE_LABEL_COL}' non presente in {assign_file.name}. Colonne disponibili: {list(assign_df.columns)}")
            continue

        # 1. Plot PCA
        pca_file = target_dir / f"{prefix}_pca.csv"
        if pca_file.exists():
            coords_pca = pd.read_csv(pca_file)
            # Se la true label non è già presente nel CSV PCA, uniscila
            if TRUE_LABEL_COL not in coords_pca.columns:
                coords_pca = coords_pca.merge(
                    assign_df[["sample_id", TRUE_LABEL_COL]], on="sample_id", how="left"
                )
            plot_side_by_side(
                df=coords_pca,
                dim1_col="PC1",
                dim2_col="PC2",
                method_name=method["name"],
                embedding_name="PCA",
                true_label_col=TRUE_LABEL_COL,
                k_label=method["k_title"],
                save_path=PLOTS_DIR / f"{prefix}_pca_vs_true.png",
            )

        # 2. Plot t-SNE
        tsne_file = target_dir / f"{prefix}_tsne_coordinates.csv"
        if tsne_file.exists():
            coords_tsne = pd.read_csv(tsne_file)
            merged_tsne = coords_tsne.merge(
                assign_df[["sample_id", TRUE_LABEL_COL]], on="sample_id", how="left"
            )
            plot_side_by_side(
                df=merged_tsne,
                dim1_col="dim1",
                dim2_col="dim2",
                method_name=method["name"],
                embedding_name="t-SNE",
                true_label_col=TRUE_LABEL_COL,
                k_label=method["k_title"],
                save_path=PLOTS_DIR / f"{prefix}_tsne_vs_true.png",
            )

        # 3. Plot UMAP
        umap_file = target_dir / f"{prefix}_umap_coordinates.csv"
        if umap_file.exists():
            coords_umap = pd.read_csv(umap_file)
            merged_umap = coords_umap.merge(
                assign_df[["sample_id", TRUE_LABEL_COL]], on="sample_id", how="left"
            )
            plot_side_by_side(
                df=merged_umap,
                dim1_col="dim1",
                dim2_col="dim2",
                method_name=method["name"],
                embedding_name="UMAP",
                true_label_col=TRUE_LABEL_COL,
                k_label=method["k_title"],
                save_path=PLOTS_DIR / f"{prefix}_umap_vs_true.png",
            )


if __name__ == "__main__":
    main()
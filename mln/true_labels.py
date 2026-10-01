from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path("output/clustering/pagerank_comprehensive/all_samples")
METADATA_FILE = Path("output/clustering/pagerank_comprehensive/sample_metadata_used_for_analysis.csv")
PLOTS_DIR = BASE_DIR / "plots_true_labels_combined"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# File di riferimento generati dallo script principale
COORDS_DIR = BASE_DIR / "clusters_by_k/euclidean/k2"
PREFIX = "all_samples_hierarchical_euclidean_k2"

COLOR_MAP = {
    "Normal": "#2ca02c",        # Verde
    "CMS1": "#e41a1c",          # Rosso (MSI Immune)
    "CMS2": "#377eb8",          # Blu (Canonical)
    "CMS3": "#ff7f00",          # Arancione (Metabolic)
    "CMS4": "#984ea3",          # Viola (Mesenchymal)
    "Unclassified": "#999999",  # Grigio
}

ORDER = ["Normal", "CMS1", "CMS2", "CMS3", "CMS4", "Unclassified"]


def build_combined_labels(metadata: pd.DataFrame) -> pd.Series:
    cms_col = "CMScaller" if "CMScaller" in metadata.columns else ("CMS" if "CMS" in metadata.columns else None)
    group_col = "group" if "group" in metadata.columns else None

    combined = pd.Series(index=metadata.index, dtype=object)

    for idx, row in metadata.iterrows():
        grp = str(row[group_col]).strip().lower() if group_col and pd.notna(row[group_col]) else ""
        
        if grp in ["normal", "control", "ctrl", "healthy"]:
            combined[idx] = "Normal"
        else:
            cms_val = str(row[cms_col]).strip().upper() if cms_col and pd.notna(row[cms_col]) else ""
            if cms_val in ["CMS1", "CMS2", "CMS3", "CMS4"]:
                combined[idx] = cms_val
            else:
                combined[idx] = "Unclassified"

    return combined


def plot_single_embedding(
    df: pd.DataFrame,
    dim1_col: str,
    dim2_col: str,
    embedding_name: str,
    save_path: Path,
) -> None:
    present_categories = [c for c in ORDER if c in df["true_subtype"].unique()]
    palette = {k: COLOR_MAP[k] for k in present_categories}

    plt.figure(figsize=(9, 7))
    ax = sns.scatterplot(
        data=df,
        x=dim1_col,
        y=dim2_col,
        hue="true_subtype",
        hue_order=present_categories,
        palette=palette,
        s=80,
        alpha=0.9,
        edgecolor="k",
        linewidth=0.6,
    )

    ax.set_title(f"{embedding_name} — Ground Truth (Normal vs CMS1–4)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(f"{dim1_col}", fontsize=11)
    ax.set_ylabel(f"{dim2_col}", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend(
        title="Subtype",
        title_fontsize="11",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvato grafico: {save_path.name}")


def main() -> None:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file non trovato: {METADATA_FILE}")

    header = pd.read_csv(METADATA_FILE, nrows=1).columns
    id_col = "participant_tissue" if "participant_tissue" in header else header[0]
    metadata = pd.read_csv(METADATA_FILE, index_col=id_col)

    metadata["true_subtype"] = build_combined_labels(metadata)
    meta_subset = metadata[["true_subtype"]].copy()

    # 1. PCA
    pca_file = COORDS_DIR / f"{PREFIX}_pca.csv"
    if pca_file.exists():
        coords_pca = pd.read_csv(pca_file)
        df_pca = coords_pca.merge(meta_subset, left_on="sample_id", right_index=True, how="inner")
        plot_single_embedding(
            df=df_pca,
            dim1_col="PC1",
            dim2_col="PC2",
            embedding_name="PCA",
            save_path=PLOTS_DIR / "all_samples_pca_normal_plus_cms.png",
        )

    # 2. t-SNE
    tsne_file = COORDS_DIR / f"{PREFIX}_tsne_coordinates.csv"
    if tsne_file.exists():
        coords_tsne = pd.read_csv(tsne_file)
        df_tsne = coords_tsne.merge(meta_subset, left_on="sample_id", right_index=True, how="inner")
        plot_single_embedding(
            df=df_tsne,
            dim1_col="dim1",
            dim2_col="dim2",
            embedding_name="t-SNE",
            save_path=PLOTS_DIR / "all_samples_tsne_normal_plus_cms.png",
        )

    # 3. UMAP
    umap_file = COORDS_DIR / f"{PREFIX}_umap_coordinates.csv"
    if umap_file.exists():
        coords_umap = pd.read_csv(umap_file)
        df_umap = coords_umap.merge(meta_subset, left_on="sample_id", right_index=True, how="inner")
        plot_single_embedding(
            df=df_umap,
            dim1_col="dim1",
            dim2_col="dim2",
            embedding_name="UMAP",
            save_path=PLOTS_DIR / "all_samples_umap_normal_plus_cms.png",
        )


if __name__ == "__main__":
    main()
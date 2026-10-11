import argparse
from pathlib import Path
import subprocess
import sys
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run the multilayer network analysis pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip multiplex network generation.",
    )
    parser.add_argument(
        "--skip-pagerank",
        action="store_true",
        help="Skip PageRank centrality computation.",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip sample clustering.",
    )
    parser.add_argument(
        "--only-clustering",
        action="store_true",
        help="Shortcut to run ONLY clustering.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # Resolve step execution flags (CLI flags override config settings)
    steps_cfg = config.get("steps", {})
    run_network = steps_cfg.get("build_network", True) and not args.skip_network
    run_pagerank = steps_cfg.get("pagerank", True) and not args.skip_pagerank
    run_clustering = steps_cfg.get("clustering", True) and not args.skip_clustering

    if args.only_clustering:
        run_network = False
        run_pagerank = False
        run_clustering = True

    data = config.get("data", {})
    network = config.get("network", {})
    pagerank = config.get("pagerank", {})
    clustering = config.get("clustering", {})

    # --------------------------------------------------
    # 1. Build multiplex networks
    # --------------------------------------------------
    if run_network:
        print("\n=== Building multiplex networks ===")
        build_command = [
            sys.executable,
            "mln/build_multiplex_networks.py",
            "--rna_path", str(data["rna"]),
            "--methyl_path", str(data["methylation"]),
            "--pathway_path", str(data["pathway"]),
            "--omega", str(network["omega"]),
            "--agg_func", str(network["aggregation"]),
            "--output_dir", str(network["output_dir"]),
        ]
        subprocess.run(build_command, check=True)
    else:
        print("\n[Skipping multiplex network construction]")

    # --------------------------------------------------
    # 2. Run PageRank
    # --------------------------------------------------
    if run_pagerank:
        print("\n=== Computing multilayer PageRank ===")
        pagerank_command = [
            sys.executable,
            "mln/run_pagerank.py",
            "--input-dir", str(network["output_dir"]),
            "--output-dir", str(pagerank["output_dir"]),
            "--alpha", str(pagerank["alpha"]),
            "--tol", str(pagerank["tolerance"]),
            "--max-iter", str(pagerank["max_iterations"]),
        ]
        subprocess.run(pagerank_command, check=True)
    else:
        print("\n[Skipping multilayer PageRank]")

    # --------------------------------------------------
    # 3. Run Clustering
    # --------------------------------------------------
    if run_clustering:
        print("\n=== Running Sample Clustering ===")

        clustering_input = clustering.get("input_path")
        if not clustering_input and clustering.get("input_type", "pagerank") == "pagerank":
            clustering_input = pagerank.get("output_dir")
        if not clustering_input:
            raise ValueError(
                "No clustering input found. Specify 'clustering.input_path' "
                "or configure 'pagerank.output_dir' for PageRank input."
            )

        clustering_command = [
            sys.executable,
            "mln/run_clustering.py",
            "--input-path", str(clustering_input),
            "--input-type", str(clustering.get("input_type", "pagerank")),
            "--output-dir", str(clustering["output_dir"]),
            "--clinical-file", str(clustering.get("clinical_file", "mln/data/Clinical_features.csv")),
            "--k-min", str(clustering.get("k_min", 2)),
            "--k-max", str(clustering.get("k_max", 5)),
            "--distances", *clustering.get("distances", ["euclidean", "cosine", "correlation"]),
            "--n-repeats-stability", str(clustering.get("n_repeats_stability", 100)),
            "--stability-fraction", str(clustering.get("stability_fraction", 0.80)),
            "--louvain-n-neighbors", str(clustering.get("louvain_n_neighbors", 20)),
            "--louvain-resolution", str(clustering.get("louvain_resolution", 0.5)),
            "--louvain-metric", str(clustering.get("louvain_metric", "cosine")),
            "--random-state", str(clustering.get("random_state", 42)),
        ]

        if clustering.get("louvain_mutual", False):
            clustering_command.append("--louvain-mutual")

        subprocess.run(clustering_command, check=True)
    else:
        print("\n[Skipping clustering]")

    print("\n=== Pipeline execution finished ===")


if __name__ == "__main__":
    main()

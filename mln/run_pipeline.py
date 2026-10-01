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
        description="Run the complete multilayer network analysis pipeline."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    data = config["data"]
    network = config["network"]
    pagerank = config["pagerank"]

    # --------------------------------------------------
    # 1. Build multiplex networks
    # --------------------------------------------------

    print("\n=== Building multiplex networks ===")

    build_command = [
        sys.executable,
        "mln/build_multiplex_networks.py",
        "--rna_path",
        str(data["rna"]),
        "--methyl_path",
        str(data["methylation"]),
        "--pathway_path",
        str(data["pathway"]),
        "--omega",
        str(network["omega"]),
        "--agg_func",
        str(network["aggregation"]),
        "--output_dir",
        str(network["output_dir"]),
    ]

    subprocess.run(build_command, check=True)

    # --------------------------------------------------
    # 2. Run PageRank
    # --------------------------------------------------

    print("\n=== Computing multilayer PageRank ===")

    pagerank_command = [
        sys.executable,
        "run_pagerank.py",
        "--input-dir",
        str(network["output_dir"]),
        "--output-dir",
        str(pagerank["output_dir"]),
        "--alpha",
        str(pagerank["alpha"]),
        "--tol",
        str(pagerank["tolerance"]),
        "--max-iter",
        str(pagerank["max_iterations"]),
    ]

    subprocess.run(pagerank_command, check=True)

    print("\n=== Pipeline completed successfully ===")


if __name__ == "__main__":
    main()
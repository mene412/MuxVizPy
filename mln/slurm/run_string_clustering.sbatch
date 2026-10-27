#!/usr/bin/env bash
#SBATCH --job-name string-mln-clustering
#SBATCH --partition allgroups
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 06:00:00
#SBATCH --mem 16G
#SBATCH --output string-mln-clustering-%j.out
#SBATCH --error string-mln-clustering-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if command -v micromamba >/dev/null 2>&1; then
    MICROMAMBA="$(command -v micromamba)"
elif [[ -x "${HOME}/bin/micromamba" ]]; then
    MICROMAMBA="${HOME}/bin/micromamba"
else
    echo "ERROR: micromamba was not found on PATH or at ${HOME}/bin/micromamba" >&2
    exit 1
fi

ENV_NAME="${MUXVIZ_ENV:-muxvizpy}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
CONFIG="mln/configs/clustering_MLN_STRING_net_pagerank.yaml"
PAGERANK_DIR="output/string_net/pagerank/mln"

if [[ ! -r "${CONFIG}" ]]; then
    echo "ERROR: config is missing or unreadable: ${REPO_ROOT}/${CONFIG}" >&2
    exit 1
fi
if [[ ! -r "mln/data/Clinical_features.csv" ]]; then
    echo "ERROR: clinical data is missing or unreadable." >&2
    exit 1
fi
if [[ ! -d "${PAGERANK_DIR}" ]] || ! compgen -G "${PAGERANK_DIR}/*_pagerank.csv" >/dev/null; then
    echo "ERROR: no PageRank CSV files found in ${REPO_ROOT}/${PAGERANK_DIR}" >&2
    exit 1
fi

echo "Repository: ${REPO_ROOT}"
echo "Using existing PageRank data: ${PAGERANK_DIR}"
echo "Environment: ${ENV_NAME}"
echo "Started: $(date)"

srun "${MICROMAMBA}" run -n "${ENV_NAME}" \
    python mln/run_pipeline.py \
    --config "${CONFIG}" \
    --skip-network \
    --skip-pagerank

echo "Finished: $(date)"

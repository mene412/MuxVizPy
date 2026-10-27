#!/usr/bin/env bash
#SBATCH --job-name string-mln
#SBATCH --partition allgroups
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 24:00:00
#SBATCH --mem 16G
#SBATCH --output string-mln-%j.out
#SBATCH --error string-mln-%j.err
#SBATCH --mail-user matteo.meneghin@studenti.unipd.it
#SBATCH --mail-type END,FAIL

set -euo pipefail

# This script lives in <repo>/mln/slurm. Resolve the repository root from
# this file so it can be submitted from any working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Prefer micromamba on PATH; otherwise use the standard user-local location.
if command -v micromamba >/dev/null 2>&1; then
    MICROMAMBA="$(command -v micromamba)"
elif [[ -x "${HOME}/bin/micromamba" ]]; then
    MICROMAMBA="${HOME}/bin/micromamba"
else
    echo "ERROR: micromamba was not found on PATH or at ${HOME}/bin/micromamba" >&2
    exit 1
fi

ENV_NAME="${MUXVIZ_ENV:-muxvizpy}"
# Match the root prefix used in the earlier micromamba setup instructions.
# Override MAMBA_ROOT_PREFIX at submission if your environments live elsewhere.
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
CONFIG="mln/configs/clustering_MLN_STRING_net_pagerank.yaml"

for input_file in \
    "mln/data/RNAprof_logTPM_minmax.csv" \
    "mln/data/DNAmethyl_minmax.csv" \
    "mln/data/networks/STRING_net.csv" \
    "mln/data/Clinical_features.csv" \
    "${CONFIG}"
do
    if [[ ! -r "${input_file}" ]]; then
        echo "ERROR: required file is missing or unreadable: ${REPO_ROOT}/${input_file}" >&2
        exit 1
    fi
done

echo "Repository: ${REPO_ROOT}"
echo "Environment: ${ENV_NAME}"
echo "Micromamba root: ${MAMBA_ROOT_PREFIX}"
echo "Config: ${CONFIG}"
echo "Started: $(date)"

srun "${MICROMAMBA}" run -n "${ENV_NAME}" \
    python mln/run_pipeline.py --config "${CONFIG}"

echo "Finished: $(date)"

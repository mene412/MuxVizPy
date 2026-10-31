#!/usr/bin/env bash
#SBATCH --job-name string-baseline
#SBATCH --partition allgroups
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 12:00:00
#SBATCH --mem 16G
#SBATCH --output string-baseline-%j.out
#SBATCH --error string-baseline-%j.err
#SBATCH --mail-user matteo.meneghin@studenti.unipd.it
#SBATCH --mail-type BEGIN,END,FAIL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO_ROOT}" || ! -f "${REPO_ROOT}/mln/run_clustering.py" ]]; then
    echo "ERROR: submit this script from the MuxVizPy repository root." >&2
    exit 1
fi
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
CLINICAL="mln/data/Clinical_features.csv"

for input_file in mln/data/RNAprof_logTPM_minmax.csv mln/data/DNAmethyl_minmax.csv "${CLINICAL}"; do
    if [[ ! -r "${input_file}" ]]; then
        echo "ERROR: required file is missing or unreadable: ${REPO_ROOT}/${input_file}" >&2
        exit 1
    fi
done

echo "Repository: ${REPO_ROOT}"
echo "Environment: ${ENV_NAME}"
echo "Running raw-matrix baseline clustering for RNA and DNA methylation"
echo "Started: $(date)"

for modality in RNA DNA; do
    if [[ "${modality}" == RNA ]]; then
        INPUT="mln/data/RNAprof_logTPM_minmax.csv"
    else
        INPUT="mln/data/DNAmethyl_minmax.csv"
    fi
    srun "${MICROMAMBA}" run -n "${ENV_NAME}" \
        python mln/run_clustering.py \
        --input-path "${INPUT}" --input-type original \
        --output-dir "output/string_net/clustering/baseline/${modality}" \
        --clinical-file "${CLINICAL}"
done

echo "Finished: $(date)"

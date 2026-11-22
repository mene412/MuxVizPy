#!/usr/bin/env bash
# Submit from the repository root with CONFIG=mln/configs/<file>.yaml.
# Set RUN_FULL_PIPELINE=1 to honor all enabled stages in the YAML.
#SBATCH --job-name muxviz-config
#SBATCH --partition allgroups
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 24:00:00
#SBATCH --mem 16G
#SBATCH --output muxviz-config-%j.out
#SBATCH --error muxviz-config-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO_ROOT}" || ! -f "${REPO_ROOT}/mln/run_pipeline.py" ]]; then
    echo "ERROR: submit this script from the MuxVizPy repository root." >&2
    exit 1
fi
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-}"
if [[ -z "${CONFIG}" || ! -r "${CONFIG}" ]]; then
    echo "ERROR: set CONFIG to a readable pipeline YAML." >&2
    exit 1
fi

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
echo "Repository: ${REPO_ROOT}"
echo "Config: ${CONFIG}"
echo "Environment: ${ENV_NAME}"
echo "Started: $(date)"

if [[ "${RUN_FULL_PIPELINE:-0}" == "1" ]]; then
    srun "${MICROMAMBA}" run -n "${ENV_NAME}" \
        python mln/run_pipeline.py --config "${CONFIG}"
else
    srun "${MICROMAMBA}" run -n "${ENV_NAME}" \
        python mln/run_pipeline.py --config "${CONFIG}" --only-clustering
fi

echo "Finished: $(date)"

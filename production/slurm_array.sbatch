#!/usr/bin/env bash
#SBATCH --job-name=goop-prod
#SBATCH --output=logs/goop-%A_%a.out
#SBATCH --error=logs/goop-%A_%a.err
#SBATCH --array=0-999
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

# Submit from the repo root, or override REPO_ROOT at submission time.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
RUN_CONFIG="${RUN_CONFIG:-production/configs/portable_template.yml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Source a site/local environment setup script before importing JAX/Torch.
# Example:
#   sbatch --export=ALL,GOOP_ENV_SETUP=/path/to/setup-goop-env.sh production/slurm_array.sbatch
if [[ -n "${GOOP_ENV_SETUP:-}" ]]; then
  # shellcheck source=/dev/null
  source "${GOOP_ENV_SETUP}"
fi

cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/jaxtpc:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Shard geometry. Each array task processes EVENTS_PER_TASK contiguous events.
BASE_EVENT="${BASE_EVENT:-0}"
EVENTS_PER_TASK="${EVENTS_PER_TASK:-1000}"
EVENTS_PER_FILE="${EVENTS_PER_FILE:-${EVENTS_PER_TASK}}"
FILE_INDEX_BASE="${FILE_INDEX_BASE:-0}"
FILES_PER_TASK=$(( (EVENTS_PER_TASK + EVENTS_PER_FILE - 1) / EVENTS_PER_FILE ))

START_EVENT=$(( BASE_EVENT + TASK_ID * EVENTS_PER_TASK ))
FILE_INDEX_OFFSET=$(( FILE_INDEX_BASE + TASK_ID * FILES_PER_TASK ))

echo "GOOP shard ${TASK_ID}: start_event=${START_EVENT} events=${EVENTS_PER_TASK} file_index_offset=${FILE_INDEX_OFFSET}"

# Optional extra production/run_batch.py CLI flags, for example:
#   RUN_BATCH_EXTRA_ARGS="--data /path/to/input.h5 --pca-lut-path /path/to/plib.h5 --outdir /path/to/out --dataset smoke"
# This is intentionally shell-split so callers can pass normal CLI flag/value pairs.
RUN_BATCH_EXTRA_ARGS="${RUN_BATCH_EXTRA_ARGS:-}"

# shellcheck disable=SC2086
srun "${PYTHON_BIN}" production/run_batch.py \
  --run-config "${RUN_CONFIG}" \
  --start-event "${START_EVENT}" \
  --events "${EVENTS_PER_TASK}" \
  --events-per-file "${EVENTS_PER_FILE}" \
  --file-index-offset "${FILE_INDEX_OFFSET}" \
  --skip-existing \
  ${RUN_BATCH_EXTRA_ARGS}

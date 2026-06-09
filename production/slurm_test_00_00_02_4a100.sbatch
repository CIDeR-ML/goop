#!/usr/bin/env bash
#SBATCH --job-name=goop-00-00-02-pixel
#SBATCH --output=logs/goop-00-00-02-%A_%a.out
#SBATCH --error=logs/goop-00-00-02-%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --qos=normal

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
RUN_CONFIG="${RUN_CONFIG:-production/configs/test_00_00_02_pixel_full.yml}"
DATA_ROOT="${DATA_ROOT:-/sdf/data/neutrino/doraemon/test_00_00_02/run_0027575715}"
PYTHON_BIN="${PYTHON_BIN:-/sdf/home/y/youngsam/sw/dune/.conda/envs/py310_torch/bin/python}"

if [[ -n "${GOOP_ENV_SETUP:-}" ]]; then
  # shellcheck source=/dev/null
  source "${GOOP_ENV_SETUP}"
fi

cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/jaxtpc:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mapfile -t INPUT_FILES < <(find "${DATA_ROOT}" -maxdepth 1 -type f -name 'edepsim_*.h5' | sort)
N_FILES="${#INPUT_FILES[@]}"
if [[ "${N_FILES}" -eq 0 ]]; then
  echo "No input files found under ${DATA_ROOT}" >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS="${TOTAL_SHARDS:-4}"

BASE_FILES=$(( N_FILES / NUM_TASKS ))
EXTRA_FILES=$(( N_FILES % NUM_TASKS ))

if [[ "${TASK_ID}" -lt "${EXTRA_FILES}" ]]; then
  FILES_BEFORE=$(( TASK_ID * (BASE_FILES + 1) ))
  FILES_THIS_TASK=$(( BASE_FILES + 1 ))
else
  FILES_BEFORE=$(( EXTRA_FILES * (BASE_FILES + 1) + (TASK_ID - EXTRA_FILES) * BASE_FILES ))
  FILES_THIS_TASK="${BASE_FILES}"
fi

FILE_START="${FILES_BEFORE}"
FILE_END=$(( FILE_START + FILES_THIS_TASK ))
FILE_LAST=$(( FILE_END - 1 ))

echo "GOOP task ${TASK_ID}/${NUM_TASKS}: files ${FILE_START}-${FILE_LAST} of ${N_FILES}"

for ((file_idx = FILE_START; file_idx < FILE_END; file_idx++)); do
  data_file="${INPUT_FILES[file_idx]}"
  echo "Processing file ${file_idx}: ${data_file}"

  srun "${PYTHON_BIN}" production/run_batch.py \
    --run-config "${RUN_CONFIG}" \
    --data "${data_file}" \
    --events 200 \
    --events-per-file 200 \
    --file-index-offset "${file_idx}" \
    --skip-existing
done

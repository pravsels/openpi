#!/bin/bash
#SBATCH --job-name=rbb_pi0_train
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=230G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6abf/pravsels.u6abf/rbb_openpi/logs/train_%j.out
#SBATCH --error=/scratch/u6abf/pravsels.u6abf/rbb_openpi/logs/train_%j.err

# Production 2xGH200 Slurm template for RoboBuilderBench Pi0 (openpi) on Isambard (u6abf).
# Full-replica data parallel (fsdp_devices=1, batch_size=32 -> 16/GPU), 20k steps.
# Automatically computes norm stats if missing and resumes from latest checkpoint.
#
# Usage:
#   sbatch --job-name=pi0_open_clamp slurm/train_rbb_pi0_slurm.sh pi0_rbb_open_clamp [exp_name]

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 <config_name> [exp_name]" >&2
    exit 1
fi

CONFIG_NAME="$1"
EXP_NAME="${2:-${CONFIG_NAME}}"

SCRATCH_ROOT="/scratch/u6abf/pravsels.u6abf"
WORK_ROOT="${SCRATCH_ROOT}/rbb_openpi"
CONTAINER="${WORK_ROOT}/container/openpi_arm64.sif"
REPO_DIR="${REPO_DIR:-/home/u6abf/pravsels.u6abf/openpi_rbb}"
SECRETS_DIR="${SCRATCH_ROOT}/.secrets"

CHECKPOINT_BASE="${WORK_ROOT}/checkpoints"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/${CONFIG_NAME}/${EXP_NAME}"
ASSETS_DIR="${WORK_ROOT}/assets/${CONFIG_NAME}/${EXP_NAME}/assets"
NORM_STATS_PATH="${ASSETS_DIR}/norm_stats.json"
PER_TIMESTEP_STATS_PATH="${ASSETS_DIR}/norm_stats_actions_per_timestep.json"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: Container SIF not found: ${CONTAINER}" >&2
    exit 1
fi
if [[ ! -f "${WORK_ROOT}/weights/pi0_base/params/_METADATA" ]]; then
    echo "ERROR: pi0_base weights not found at ${WORK_ROOT}/weights/pi0_base/params" >&2
    exit 1
fi

mkdir -p "${WORK_ROOT}/logs" "${CHECKPOINT_BASE}" "${ASSETS_DIR}" \
    "${WORK_ROOT}/cache/hf" "${WORK_ROOT}/openpi_cache" \
    "${WORK_ROOT}/wandb" "${WORK_ROOT}/cache/wandb" "${WORK_ROOT}/config/wandb" \
    "${WORK_ROOT}/tmp"

module purge
module load brics/apptainer-multi-node

if [[ -f "${SECRETS_DIR}/.hf_token" ]]; then
    export HF_TOKEN="$(tr -d '\r\n' < "${SECRETS_DIR}/.hf_token")"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -f "${SECRETS_DIR}/.wandb_token" ]]; then
    export WANDB_API_KEY="$(tr -d '\r\n' < "${SECRETS_DIR}/.wandb_token")"
fi

echo "=== RoboBuilderBench Pi0 Training (openpi) ==="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Host:       $(hostname) ($(uname -m))"
echo "Config:     ${CONFIG_NAME}"
echo "Experiment: ${EXP_NAME}"
echo "Repo:       ${REPO_DIR}"
echo "Container:  ${CONTAINER}"
echo "Checkpoints:${CHECKPOINT_DIR}"
echo "Assets:     ${ASSETS_DIR}"
nvidia-smi -L

APPTAINER_CMD=(
    apptainer exec --nv --writable-tmpfs
    --pwd "${REPO_DIR}"
    -B "${SCRATCH_ROOT}:${SCRATCH_ROOT}"
    -B "${REPO_DIR}:${REPO_DIR}"
    -B "${WORK_ROOT}/weights:${REPO_DIR}/weights"
    --env "PYTHONPATH=${REPO_DIR}/src"
    --env "PYTHONUNBUFFERED=1"
    --env "HF_HOME=${WORK_ROOT}/cache/hf"
    --env "HF_LEROBOT_HOME=${WORK_ROOT}/cache/hf/lerobot"
    --env "HF_HUB_ENABLE_HF_TRANSFER=1"
    --env "HF_TOKEN=${HF_TOKEN:-}"
    --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}"
    --env "OPENPI_DATA_HOME=${WORK_ROOT}/openpi_cache"
    --env "WANDB_MODE=${WANDB_MODE:-offline}"
    --env "WANDB_DIR=${WORK_ROOT}/wandb"
    --env "WANDB_CACHE_DIR=${WORK_ROOT}/cache/wandb"
    --env "WANDB_CONFIG_DIR=${WORK_ROOT}/config/wandb"
    --env "WANDB_API_KEY=${WANDB_API_KEY:-}"
    --env "WANDB_ENTITY=${WANDB_ENTITY:-pravsels}"
    --env "TMPDIR=${WORK_ROOT}/tmp"
    --env "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"
    "${CONTAINER}"
)

if [[ -f "${NORM_STATS_PATH}" && -f "${PER_TIMESTEP_STATS_PATH}" ]]; then
    echo "Reusing existing norm stats in ${ASSETS_DIR}"
else
    echo "Computing per-timestep norm stats into ${ASSETS_DIR}..."
    "${APPTAINER_CMD[@]}" /.venv/bin/python scripts/compute_norm_stats_per_timestep.py \
        --config-name="${CONFIG_NAME}" \
        --assets-dir="${ASSETS_DIR}"
fi

TRAIN_MODE_FLAG="--overwrite"
if [[ -d "${CHECKPOINT_DIR}" ]] && [[ -n "$(find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d ! -name assets 2>/dev/null | head -1)" ]]; then
    TRAIN_MODE_FLAG="--resume"
fi

echo "Launching scripts/train.py (${TRAIN_MODE_FLAG})..."
srun --ntasks=1 --gpus-per-task=2 --cpu-bind=cores \
    "${APPTAINER_CMD[@]}" /.venv/bin/python scripts/train.py "${CONFIG_NAME}" \
        --exp-name="${EXP_NAME}" \
        --assets-dir="${ASSETS_DIR}" \
        --checkpoint-base-dir="${CHECKPOINT_BASE}" \
        "${TRAIN_MODE_FLAG}"

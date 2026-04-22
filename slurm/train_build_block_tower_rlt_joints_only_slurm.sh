#!/bin/bash
#SBATCH --job-name=pi05_block_tower_rlt_joints_only
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=0G
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -e

module purge
module load brics/apptainer-multi-node

# Paths
home_dir="/home/u6cr/pravsels.u6cr"
scratch_dir="/scratch/u6cr/pravsels.u6cr"
repo_dir="${home_dir}/openpi"
data_dir="${scratch_dir}/openpi"
container="${data_dir}/container/openpi_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
WANDB_DIR="${data_dir}"
WANDB_CACHE_DIR="${scratch_dir}/.cache/wandb"
WANDB_CONFIG_DIR="${scratch_dir}/.config/wandb"
XDG_CACHE_HOME="${scratch_dir}/.cache"
XDG_CONFIG_HOME="${scratch_dir}/.config"

# Training config
CONFIG_NAME="pi05_rlt_build_block_tower_6mix_joints_only"
EXP_NAME="rlt_6mix_joints_only_v1"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

# Joints-only baseline checkpoint and assets.
BASELINE_HF_REPO="pravsels/build_block_tower_baseline_6mix_joints_only"
BASELINE_EXP="joints_only"
BASELINE_STEP="49999"
BASELINE_LOCAL_DIR="${data_dir}/checkpoints/pi05_build_block_tower_baseline_6mix_joints_only"
BASELINE_CKPT_DIR="${BASELINE_LOCAL_DIR}/${BASELINE_EXP}/${BASELINE_STEP}"
ASSETS_DIR="${BASELINE_CKPT_DIR}/assets"

CHECKPOINT_DIR="${data_dir}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"

mkdir -p "${HF_CACHE}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${data_dir}/checkpoints" "${data_dir}/assets" "${data_dir}/weights" "${data_dir}/.venv" "${CHECKPOINT_DIR}"

# Download baseline checkpoint from HuggingFace if not present locally.
if [ -d "${BASELINE_CKPT_DIR}/params" ]; then
    echo "Baseline checkpoint found at ${BASELINE_CKPT_DIR}/params"
else
    echo "Downloading baseline checkpoint (${BASELINE_STEP}) from ${BASELINE_HF_REPO}..."
    mkdir -p "${BASELINE_CKPT_DIR}"
    HF_TOKEN=$(cat "${home_dir}/.hf_token")
    HF_HOME="${HF_CACHE}" huggingface-cli download \
        "${BASELINE_HF_REPO}" \
        --include "checkpoints/${BASELINE_STEP}/params/*" \
        --local-dir "${BASELINE_LOCAL_DIR}" \
        --token "${HF_TOKEN}"
    if [ ! -d "${BASELINE_CKPT_DIR}/params" ]; then
        ALT_PATH="${BASELINE_LOCAL_DIR}/checkpoints/${BASELINE_STEP}/params"
        if [ -d "${ALT_PATH}" ]; then
            mkdir -p "${BASELINE_CKPT_DIR}"
            mv "${ALT_PATH}" "${BASELINE_CKPT_DIR}/params"
            echo "Moved checkpoint from HF nested structure to ${BASELINE_CKPT_DIR}/params"
        else
            echo "ERROR: Could not find downloaded baseline checkpoint."
            echo "Searched: ${BASELINE_CKPT_DIR}/params and ${ALT_PATH}"
            exit 1
        fi
    fi
    echo "Baseline checkpoint ready at ${BASELINE_CKPT_DIR}/params"
fi

# Use self-contained assets from the baseline checkpoint directory.
if [ ! -d "${ASSETS_DIR}" ] || [ -z "$(ls -A "${ASSETS_DIR}"/*.json 2>/dev/null)" ]; then
    echo "ERROR: Missing baseline assets at ${ASSETS_DIR}"
    echo "Expected self-contained assets next to params under ${BASELINE_EXP}/${BASELINE_STEP}."
    echo "Please sync checkpoint assets on cluster scratch and retry."
    exit 1
fi
echo "Using baseline assets from ${ASSETS_DIR}"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "Config: ${CONFIG_NAME}"
echo "Exp: ${EXP_NAME}"
echo "Extra train args: ${EXTRA_TRAIN_ARGS:-<none>}"
echo "Baseline checkpoint: ${BASELINE_EXP}/${BASELINE_STEP}"
echo "Assets: ${ASSETS_DIR}"
echo "===================================="

TRAIN_CMD="uv run scripts/train.py ${CONFIG_NAME} --exp-name=${EXP_NAME} --assets-dir=${ASSETS_DIR} --resume ${EXTRA_TRAIN_ARGS}"

EXPORT_VARS="export PYTHONUNBUFFERED=1"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_MODE=offline"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_DIR=${WANDB_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CACHE_DIR=${WANDB_CACHE_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CACHE_HOME=${XDG_CACHE_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CONFIG_HOME=${XDG_CONFIG_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_ENTITY=pravsels"
EXPORT_VARS="${EXPORT_VARS} && export OPENPI_DATA_HOME=${data_dir}"
EXPORT_VARS="${EXPORT_VARS} && export UV_PROJECT_ENVIRONMENT=${data_dir}/.venv"
EXPORT_VARS="${EXPORT_VARS} && export HF_TOKEN=\$(cat ${home_dir}/.hf_token)"
EXPORT_VARS="${EXPORT_VARS} && export XLA_PYTHON_CLIENT_PREALLOCATE=false"
EXPORT_VARS="${EXPORT_VARS} && export XLA_PYTHON_CLIENT_ALLOCATOR=platform"

echo "Running training command..."
echo "Command: ${TRAIN_CMD}"
echo ""

set +e
srun --ntasks=1 --gpus-per-task=4 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${data_dir}/assets:${repo_dir}/assets" \
    --bind "${data_dir}/weights:${repo_dir}/weights" \
    --bind "${data_dir}/checkpoints:${repo_dir}/checkpoints" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "${EXPORT_VARS} && ${TRAIN_CMD}"
EXIT_CODE=$?
set -e

end_time="$(date -Is --utc)"

echo ""
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Exit Code: ${EXIT_CODE}"
echo "===================================="

if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "Check slurm-${SLURM_JOB_ID}.err for detailed error messages"
    echo "Checkpoint location: ${CHECKPOINT_DIR}"
    exit ${EXIT_CODE}
fi

exit 0

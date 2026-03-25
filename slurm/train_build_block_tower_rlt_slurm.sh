#!/bin/bash
#SBATCH --job-name=pi05_block_tower_rlt
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
CONFIG_NAME="pi05_rl_token_build_block_tower"
EXP_NAME="rlt_v1"

# Baseline checkpoint (VLA backbone for the RLT encoder-decoder)
BASELINE_HF_REPO="pravsels/pi05-build-block-tower-baseline"
BASELINE_STEP="55000"
BASELINE_CKPT_DIR="${data_dir}/checkpoints/pi05_build_block_tower_baseline/baseline_v1/${BASELINE_STEP}"

CHECKPOINT_DIR="${data_dir}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
ASSETS_DIR="${CHECKPOINT_DIR}/assets"

if [ -z "${ASSETS_DIR}" ]; then
    echo "ERROR: ASSETS_DIR is empty; refusing to run."
    exit 1
fi
if [[ "${ASSETS_DIR}" != */assets ]]; then
    echo "ERROR: ASSETS_DIR must end with /assets (got: ${ASSETS_DIR})"
    exit 1
fi

mkdir -p "${HF_CACHE}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${data_dir}/checkpoints" "${data_dir}/assets" "${data_dir}/weights" "${data_dir}/.venv" "${ASSETS_DIR}"

# Download baseline checkpoint from HuggingFace if not present locally
if [ -d "${BASELINE_CKPT_DIR}/params" ]; then
    echo "Baseline checkpoint found at ${BASELINE_CKPT_DIR}/params"
else
    echo "Downloading baseline checkpoint (step ${BASELINE_STEP}) from ${BASELINE_HF_REPO}..."
    mkdir -p "${BASELINE_CKPT_DIR}"
    HF_TOKEN=$(cat "${home_dir}/.hf_token")
    HF_HOME="${HF_CACHE}" huggingface-cli download \
        "${BASELINE_HF_REPO}" \
        --include "checkpoints/${BASELINE_STEP}/params/*" \
        --local-dir "${data_dir}/checkpoints/pi05_build_block_tower_baseline/baseline_v1" \
        --token "${HF_TOKEN}"
    # HF download preserves directory structure, verify the params dir exists
    if [ ! -d "${BASELINE_CKPT_DIR}/params" ]; then
        # Try flattened structure (HF may nest under checkpoints/<step>)
        ALT_PATH="${data_dir}/checkpoints/pi05_build_block_tower_baseline/baseline_v1/checkpoints/${BASELINE_STEP}/params"
        if [ -d "${ALT_PATH}" ]; then
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

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "Config: ${CONFIG_NAME}"
echo "Exp: ${EXP_NAME}"
echo "Baseline checkpoint: step ${BASELINE_STEP}"
echo "===================================="

# Training commands
COMPUTE_NORM_STATS_CMD="uv run scripts/compute_norm_stats_per_timestep.py --config-name=${CONFIG_NAME} --assets-dir=${ASSETS_DIR}"
NORM_STATS_PATH="${ASSETS_DIR}/norm_stats.json"
PER_TIMESTEP_STATS_PATH="${ASSETS_DIR}/norm_stats_actions_per_timestep.json"
TRAIN_CMD="uv run scripts/train.py ${CONFIG_NAME} --exp-name=${EXP_NAME} --assets-dir=${ASSETS_DIR} --resume"

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

PRECOMPUTE_CMD=""

if [ -f "${NORM_STATS_PATH}" ] && [ -f "${PER_TIMESTEP_STATS_PATH}" ]; then
    echo "Skipping normalization precompute (found stats files)."
else
    echo "Running normalization precompute..."
    echo "Command: ${COMPUTE_NORM_STATS_CMD}"
    echo ""
    PRECOMPUTE_CMD="${PRECOMPUTE_CMD}${COMPUTE_NORM_STATS_CMD} && "
fi
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
    bash -c "${EXPORT_VARS} && ${PRECOMPUTE_CMD}${TRAIN_CMD}"
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

#!/bin/bash
#SBATCH --job-name=pi05_block_tower_recap
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

MODE="${1:?Usage: sbatch $0 <positive_only|mixed|subtask_positive_only|subtask_mixed>}"

case "${MODE}" in
    positive_only)
        CONFIG_NAME="pi05_build_block_tower_recap_positive_only"
        EXP_NAME="positive_only"
        TRAIN_EXTRA_ARGS=""
        FORCE_REBUILD_VALID_INDICES=1
        ;;
    mixed)
        CONFIG_NAME="pi05_build_block_tower_recap_mixed"
        EXP_NAME="mixed"
        TRAIN_EXTRA_ARGS=""
        FORCE_REBUILD_VALID_INDICES=1
        ;;
    subtask_positive_only)
        CONFIG_NAME="pi05_build_block_tower_subtask_recap_positive_only"
        EXP_NAME="subtask_positive_only"
        TRAIN_EXTRA_ARGS="--batch-size=12"
        FORCE_REBUILD_VALID_INDICES=0
        ;;
    subtask_mixed)
        CONFIG_NAME="pi05_build_block_tower_subtask_recap_mixed"
        EXP_NAME="subtask_mixed"
        TRAIN_EXTRA_ARGS="--batch-size=12"
        FORCE_REBUILD_VALID_INDICES=0
        ;;
    *)
        echo "ERROR: unknown mode '${MODE}'. Use 'positive_only', 'mixed', 'subtask_positive_only', or 'subtask_mixed'."
        exit 1
        ;;
esac

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

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Mode: ${MODE}"
echo "Config: ${CONFIG_NAME}"
echo "Started (UTC): ${start_time}"
echo "===================================="

COMPUTE_NORM_STATS_CMD="uv run scripts/compute_norm_stats_per_timestep.py --config-name=${CONFIG_NAME} --assets-dir=${ASSETS_DIR}"
NORM_STATS_PATH="${ASSETS_DIR}/norm_stats.json"
PER_TIMESTEP_STATS_PATH="${ASSETS_DIR}/norm_stats_actions_per_timestep.json"
TRAIN_CMD="uv run scripts/train.py ${CONFIG_NAME} --exp-name=${EXP_NAME} --assets-dir=${ASSETS_DIR} --resume ${TRAIN_EXTRA_ARGS}"

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

if [ "${FORCE_REBUILD_VALID_INDICES}" = "1" ] && [ -f "${ASSETS_DIR}/valid_indices.txt" ]; then
    echo "Removing ${ASSETS_DIR}/valid_indices.txt to force regeneration for this config."
    rm -f "${ASSETS_DIR}/valid_indices.txt"
fi

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

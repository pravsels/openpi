#!/bin/bash
#SBATCH --job-name=pi05_arx5_micro_adv
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

# Paths (edit these for your cluster)
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

CONFIG_NAME="pi05_arx5_multitask_micro_advantaged"
EXP_NAME="micro_advantaged_v1"

CHECKPOINT_DIR="${data_dir}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
ASSETS_DIR="${CHECKPOINT_DIR}/assets"

# The advantaged config loads norm stats from ./assets/pi05_arx5_multitask_micro_baseline
# (via AssetsConfig). This resolves inside the container where assets/ is bind-mounted
# from ${data_dir}/assets. Ensure the baseline norm stats exist there.
BASELINE_NORM_DIR="${data_dir}/assets/pi05_arx5_multitask_micro_baseline"

if [ -z "${ASSETS_DIR}" ]; then
    echo "ERROR: ASSETS_DIR is empty; refusing to run."
    exit 1
fi
if [[ "${ASSETS_DIR}" != */assets ]]; then
    echo "ERROR: ASSETS_DIR must end with /assets (got: ${ASSETS_DIR})"
    exit 1
fi

mkdir -p "${HF_CACHE}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${data_dir}/checkpoints" "${data_dir}/assets" "${data_dir}/weights" "${data_dir}/.venv" "${ASSETS_DIR}" "${BASELINE_NORM_DIR}"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "===================================="

MIX_JSON="${ASSETS_DIR}/training_mix_micro.json"
VALID_INDICES_PATH="${ASSETS_DIR}/valid_indices.txt"
BASELINE_NORM_STATS="${BASELINE_NORM_DIR}/norm_stats.json"

if [ ! -f "${MIX_JSON}" ]; then
    echo "ERROR: training_mix_micro.json not found at ${MIX_JSON}"
    echo "Rsync the advantaged assets to ${ASSETS_DIR} before submitting."
    exit 1
fi

if [ ! -f "${BASELINE_NORM_STATS}" ]; then
    echo "ERROR: baseline norm_stats.json not found at ${BASELINE_NORM_STATS}"
    echo "Copy baseline norm_stats.json to ${BASELINE_NORM_DIR}/ before submitting."
    exit 1
fi

if [ ! -f "${VALID_INDICES_PATH}" ]; then
    echo "WARNING: valid_indices.txt not found at ${VALID_INDICES_PATH}"
    echo "Training will use the full dataset (no index filtering)."
fi

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

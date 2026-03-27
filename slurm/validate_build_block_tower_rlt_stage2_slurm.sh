#!/bin/bash
#SBATCH --job-name=pi05_block_tower_rlt_s2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --requeue

set -euo pipefail

module purge
module load brics/apptainer-multi-node

# Paths
home_dir="/home/u6cr/pravsels.u6cr"
scratch_dir="/scratch/u6cr/pravsels.u6cr"
repo_dir="${home_dir}/openpi_rlt_block_tower"
data_dir="${scratch_dir}/openpi"
container="${data_dir}/container/openpi_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
XDG_CACHE_HOME="${scratch_dir}/.cache"
XDG_CONFIG_HOME="${scratch_dir}/.config"
UV_PROJECT_ENVIRONMENT="${data_dir}/.venv"
slurm_log_dir="${data_dir}"

# Validation config
CONFIG_NAME="pi05_rl_token_build_block_tower"
CHECKPOINT_PATH="${data_dir}/checkpoints/pi05_rl_token_build_block_tower/rlt_v1/9999/params"
ASSETS_DIR="${data_dir}/checkpoints/pi05_rl_token_build_block_tower/rlt_v1/9999/assets"
RUN_NAME="validate_rl_token_9999"
OUTPUT_DIR="${data_dir}/eval_outputs/build_block_tower_rlt_stage2/${RUN_NAME}"
EXTRA_VALIDATE_ARGS="--batch-size 8"

mkdir -p "${HF_CACHE}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${data_dir}" "${slurm_log_dir}" "${data_dir}/eval_outputs/build_block_tower_rlt_stage2" "${OUTPUT_DIR}"

if [ ! -d "${repo_dir}" ]; then
    echo "ERROR: repo_dir does not exist: ${repo_dir}"
    exit 1
fi
if [ ! -f "${container}" ]; then
    echo "ERROR: container does not exist: ${container}"
    exit 1
fi
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: checkpoint params directory does not exist: ${CHECKPOINT_PATH}"
    exit 1
fi
if [ ! -d "${ASSETS_DIR}" ]; then
    echo "ERROR: assets directory does not exist: ${ASSETS_DIR}"
    exit 1
fi

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "Config: ${CONFIG_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Assets: ${ASSETS_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Extra validate args: ${EXTRA_VALIDATE_ARGS:-<none>}"
echo "===================================="

VALIDATE_CMD="uv run python scripts/validate_rl_token.py --config-name ${CONFIG_NAME} --checkpoint-path ${CHECKPOINT_PATH} --assets-dir ${ASSETS_DIR} --output-dir ${OUTPUT_DIR} --device cuda ${EXTRA_VALIDATE_ARGS}"

EXPORT_VARS="export PYTHONUNBUFFERED=1"
EXPORT_VARS="${EXPORT_VARS} && export OPENPI_DATA_HOME=${data_dir}"
EXPORT_VARS="${EXPORT_VARS} && export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
EXPORT_VARS="${EXPORT_VARS} && export HF_HOME=/root/.cache/huggingface"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CACHE_HOME=${XDG_CACHE_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CONFIG_HOME=${XDG_CONFIG_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export HF_TOKEN=\$(cat ${home_dir}/.hf_token)"

echo "Running validation command..."
echo "Command: ${VALIDATE_CMD}"
echo ""

set +e
srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${data_dir}/assets:${repo_dir}/assets" \
    --bind "${data_dir}/weights:${repo_dir}/weights" \
    --bind "${data_dir}/checkpoints:${repo_dir}/checkpoints" \
    --bind "${data_dir}/eval_outputs:${repo_dir}/eval_outputs" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -lc "${EXPORT_VARS} && ${VALIDATE_CMD}"
EXIT_CODE=$?
set -e

end_time="$(date -Is --utc)"

echo ""
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Exit Code: ${EXIT_CODE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "===================================="

if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Validation failed with exit code ${EXIT_CODE}"
    echo "Check the Slurm stderr log for detailed error messages"
    echo "Output dir: ${OUTPUT_DIR}"
    exit ${EXIT_CODE}
fi

exit 0

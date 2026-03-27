#!/bin/bash
#SBATCH --job-name=rlt_extract
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

# Extraction config
CONFIG_NAME="pi05_rl_token_build_block_tower"
CHECKPOINT_PATH="${data_dir}/checkpoints/pi05_rl_token_build_block_tower/rlt_v1/9999/params"
ASSETS_DIR="${data_dir}/checkpoints/pi05_rl_token_build_block_tower/rlt_v1/9999/assets"
ID_DATASET="villekuosmanen/build_block_tower"
OOD_DATASET="villekuosmanen/eval_dAgger_drop_footbag_into_dice_tower_1.7.0"
EPISODES_PER_DATASET=1
OUTPUT_DIR="${data_dir}/eval_outputs/build_block_tower_rlt_stage2/rl_token_cosine_sim"

mkdir -p "${HF_CACHE}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${OUTPUT_DIR}"

if [ ! -d "${repo_dir}" ]; then
    echo "ERROR: repo_dir does not exist: ${repo_dir}"
    exit 1
fi
if [ ! -f "${container}" ]; then
    echo "ERROR: container does not exist: ${container}"
    exit 1
fi
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: checkpoint does not exist: ${CHECKPOINT_PATH}"
    exit 1
fi

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "Config: ${CONFIG_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "ID dataset: ${ID_DATASET}"
echo "OOD dataset: ${OOD_DATASET}"
echo "Episodes per dataset: ${EPISODES_PER_DATASET}"
echo "Output dir: ${OUTPUT_DIR}"
echo "===================================="

EXTRACT_CMD="uv run python scripts/rl_token_extract_episodes.py \
    --config-name ${CONFIG_NAME} \
    --checkpoint-path ${CHECKPOINT_PATH} \
    --assets-dir ${ASSETS_DIR} \
    --id-dataset ${ID_DATASET} \
    --ood-dataset ${OOD_DATASET} \
    --episodes-per-dataset ${EPISODES_PER_DATASET} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size 8"

ANALYSIS_CMD="uv run python scripts/rl_token_cosine_analysis.py \
    --embeddings-path ${OUTPUT_DIR}/rl_token_embeddings.npz \
    --output-dir ${OUTPUT_DIR}"

EXPORT_VARS="export PYTHONUNBUFFERED=1"
EXPORT_VARS="${EXPORT_VARS} && export OPENPI_DATA_HOME=${data_dir}"
EXPORT_VARS="${EXPORT_VARS} && export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
EXPORT_VARS="${EXPORT_VARS} && export HF_HOME=/root/.cache/huggingface"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CACHE_HOME=${XDG_CACHE_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CONFIG_HOME=${XDG_CONFIG_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export HF_TOKEN=\$(cat ${home_dir}/.hf_token)"

echo "Running extraction..."
echo "Command: ${EXTRACT_CMD}"
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
    bash -lc "${EXPORT_VARS} && ${EXTRACT_CMD}"
EXIT_CODE=$?
set -e

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Extraction failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

echo ""
echo "Running cosine similarity analysis..."
echo "Command: ${ANALYSIS_CMD}"
echo ""

set +e
srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${data_dir}/eval_outputs:${repo_dir}/eval_outputs" \
    "${container}" \
    bash -lc "${EXPORT_VARS} && ${ANALYSIS_CMD}"
ANALYSIS_EXIT=$?
set -e

end_time="$(date -Is --utc)"
echo ""
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Extraction exit: ${EXIT_CODE}"
echo "Analysis exit:   ${ANALYSIS_EXIT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "===================================="

if [ ${ANALYSIS_EXIT} -ne 0 ]; then
    echo "WARNING: Analysis failed with exit code ${ANALYSIS_EXIT} (embeddings were saved successfully)"
    exit ${ANALYSIS_EXIT}
fi

exit 0

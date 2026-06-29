#!/bin/bash
#SBATCH --job-name=v50_publish
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Generic v50 checkpoint publish to Hugging Face.
# Uploads selected checkpoints (params/ + assets/, no train_state/) straight to HF.
# Config-agnostic: all per-task values come from arguments.
#
# Usage:
#   sbatch [--job-name=<name>] slurm/publish_v50_slurm.sh \
#       <CONFIG_NAME> <EXP_NAME> <HF_REPO_ID> ["PUBLISH_STEPS"]
# PUBLISH_STEPS defaults to "10000 25000 49999". Normally invoked by
# slurm/submit_all_v50.sh with an afterok dependency on the matching train job.
#
# Before sbatch:
#   - openpi_arm64.sif at ${data_dir}/container/
#   - HF *write* token in ${scratch_dir}/.secrets/.hf_token_write

set -e

module purge
module load brics/apptainer-multi-node

# --- Arguments ---
CONFIG_NAME="$1"
EXP_NAME="${2:-$1}"
HF_REPO_ID="$3"
PUBLISH_STEPS="${4:-24999}"
if [ -z "${CONFIG_NAME}" ] || [ -z "${HF_REPO_ID}" ]; then
    echo "ERROR: CONFIG_NAME (arg 1) and HF_REPO_ID (arg 3) are required."
    echo "Usage: sbatch slurm/publish_v50_slurm.sh <CONFIG_NAME> <EXP_NAME> <HF_REPO_ID> [\"PUBLISH_STEPS\"]"
    exit 1
fi

# --- Infrastructure (u6kr) ---
home_dir="/home/u6kr/lorenzo.u6kr"
scratch_dir="/scratch/u6kr/lorenzo.u6kr"
repo_dir="${home_dir}/openpi_so101_v50"
data_dir="${scratch_dir}/openpi"
container="${data_dir}/container/openpi_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
HF_WRITE_TOKEN_FILE="${scratch_dir}/.secrets/.hf_token_write"

CHECKPOINT_DIR="${data_dir}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"

if [ ! -f "${container}" ]; then
    echo "ERROR: container not found: ${container}"
    exit 1
fi
if [ ! -f "${HF_WRITE_TOKEN_FILE}" ]; then
    echo "ERROR: HF write token not found: ${HF_WRITE_TOKEN_FILE}"
    echo "Create a Write token at https://huggingface.co/settings/tokens and save it there."
    exit 1
fi
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: checkpoint dir not found: ${CHECKPOINT_DIR}"
    exit 1
fi

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Config: ${CONFIG_NAME}"
echo "Experiment: ${EXP_NAME}"
echo "HF repo: ${HF_REPO_ID}"
echo "Steps: ${PUBLISH_STEPS}"
echo "Started (UTC): ${start_time}"
echo "===================================="

EXPORT_VARS="export PYTHONUNBUFFERED=1"
EXPORT_VARS="${EXPORT_VARS} && export HF_HOME=/root/.cache/huggingface"
EXPORT_VARS="${EXPORT_VARS} && export HF_TOKEN=\$(tr -d '\n' < ${HF_WRITE_TOKEN_FILE})"
EXPORT_VARS="${EXPORT_VARS} && export UV_PROJECT_ENVIRONMENT=${data_dir}/.venv"
EXPORT_VARS="${EXPORT_VARS} && export CHECKPOINT_DIR=${CHECKPOINT_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export HF_REPO_ID=${HF_REPO_ID}"
EXPORT_VARS="${EXPORT_VARS} && export PUBLISH_STEPS=\"${PUBLISH_STEPS}\""
EXPORT_VARS="${EXPORT_VARS} && export CONFIG_NAME=${CONFIG_NAME}"

UPLOAD_PY="
import os
from huggingface_hub import HfApi

api = HfApi()
repo_id = os.environ['HF_REPO_ID']
ckpt_root = os.environ['CHECKPOINT_DIR']
cfg = os.environ['CONFIG_NAME']
steps = os.environ['PUBLISH_STEPS'].split()

api.create_repo(repo_id=repo_id, repo_type='model', exist_ok=True)
print(f'Target repo: {repo_id}')

uploaded = []
for step in steps:
    folder = os.path.join(ckpt_root, step)
    if not os.path.isdir(folder):
        print(f'SKIP step {step}: not found at {folder}')
        continue
    print(f'Uploading step {step} from {folder} ...')
    api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        path_in_repo=f'step_{step}',
        repo_type='model',
        ignore_patterns=['train_state/**'],
        commit_message=f'Add {cfg} checkpoint step {step}',
    )
    print(f'Uploaded step {step} -> step_{step}/')
    uploaded.append(step)

if not uploaded:
    raise SystemExit('ERROR: no checkpoints uploaded (none of the requested steps were found)')
print('Uploaded steps:', ', '.join(uploaded))
"

set +e
srun --ntasks=1 --gpus-per-task=1 \
apptainer exec \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${data_dir}/checkpoints:${repo_dir}/checkpoints" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "${EXPORT_VARS} && uv run python -c \"${UPLOAD_PY}\""
EXIT_CODE=$?
set -e

end_time="$(date -Is --utc)"

echo ""
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Exit Code: ${EXIT_CODE}"
echo "HF repo: https://huggingface.co/${HF_REPO_ID}"
echo "===================================="

if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Publish failed with exit code ${EXIT_CODE}"
    echo "Check slurm-${SLURM_JOB_ID}.err for detailed error messages"
    exit ${EXIT_CODE}
fi

exit 0

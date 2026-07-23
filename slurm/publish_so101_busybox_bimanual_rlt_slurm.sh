#!/bin/bash
#SBATCH --job-name=pi05_publish_busybox_bimanual_rlt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# pi05_rlt_so101_busybox_buttons_bimanual — Stage-1 RL-token checkpoint publish
# to HF (bimanual busybox-buttons task).
#
# Uploads selected checkpoints (params/ + assets/, no train_state/) straight to
# HF without the checkpoint-passport pipeline. This is a lightweight publish for
# interim checkpoints; it skips integrity signing (no MODEL_PASSPORT.json /
# SIGNOFF.json). For a signed release, use the checkpoint-passport workflow.
#
# The published checkpoint is the Pi0RL (RL-token) model that hw_control.pi0_rlt
# loads to build the demo cache and run online RL. Pull it back to the laptop
# `rlt` env and point the cache build's --checkpoint-dir at its step_<N> dir.
#
# Cluster: Isambard u6kr. Submit from the worktree:
#   cd /home/u6kr/lorenzo.u6kr/openpi_so101_busybox_bimanual_rlt
#   sbatch slurm/publish_so101_busybox_bimanual_rlt_slurm.sh
#
# Auto-run after training (uploads once training succeeds):
#   TRAIN_JOB=$(sbatch --parsable slurm/train_so101_busybox_bimanual_rlt_slurm.sh)
#   sbatch --dependency=afterok:${TRAIN_JOB} slurm/publish_so101_busybox_bimanual_rlt_slurm.sh
#
# Before sbatch:
#   - openpi_arm64.sif at ${data_dir}/container/
#   - HF *write* token in ${scratch_dir}/.secrets/.hf_token_write

set -e

module purge
module load brics/apptainer-multi-node

# --- Infrastructure (u6kr) ---
home_dir="/home/u6kr/lorenzo.u6kr"
scratch_dir="/scratch/u6kr/lorenzo.u6kr"
repo_dir="${home_dir}/openpi_so101_busybox_bimanual_rlt"
data_dir="${scratch_dir}/openpi"
container="${data_dir}/container/openpi_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
HF_WRITE_TOKEN_FILE="${scratch_dir}/.secrets/.hf_token_write"

# --- Experiment / publish settings ---
CONFIG_NAME="pi05_rlt_so101_busybox_buttons_bimanual"
EXP_NAME="so101_busybox_buttons_bimanual_rlt"
HF_REPO_ID="lorenzouttini/pi05-rlt-so101-busybox-buttons-bimanual-isambard"
# Space-separated checkpoint steps to publish. Short 10k-step run = single final
# checkpoint; the final step is num_train_steps-1 (9999), but list 10000 too so the
# upload picks up whichever the trainer wrote (missing ones are skipped).
PUBLISH_STEPS="9999 10000"

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

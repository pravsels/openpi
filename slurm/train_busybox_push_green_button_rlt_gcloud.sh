#!/bin/bash
# GCloud Docker training script for RLT Stage 1 on busybox_push_green_button.
# VM: 1x A100 80GB (a2-ultragpu-1g), or reuse openpi-so101-80g-2x.
# Do not use a2-highgpu-1g (A100 40GB); the runtime check requires >=80 GB.
# Usage: run this ON the VM directly (or via gcloud ssh)
#
# This does NOT train a VLA. It attaches a learned RL-token encoder-decoder to
# the frozen pi05 green-button checkpoint (rl_vla_loss_weight=0.0) and trains
# only the bottleneck for 20k steps.
#
# Before first run:
#   - Docker image openpi:latest
#   - Secrets in $HOME/.env (not in git):
#       WANDB_API_KEY=...
#       HF_TOKEN=...
#     Override path with OPENPI_ENV. WANDB_MODE=online must be set here;
#     scripts/train.py defaults offline.
#   - ~200 GB free (Hub VLA is ~75 GB)

set -euo pipefail

OPENPI_ENV="${OPENPI_ENV:-${HOME}/.env}"
if [ -f "${OPENPI_ENV}" ]; then
    set -a
    # shellcheck disable=SC1090
    . "${OPENPI_ENV}"
    set +a
fi

# --- Config ---
CONFIG_NAME="pi05_rlt_busybox_push_green_button"
EXP_NAME="${EXP_NAME:-busybox_push_green_button_rlt}"
REPO_DIR="${REPO_DIR:-/home/ps/openpi}"
CHECKPOINT_DIR="${REPO_DIR}/checkpoints"
ASSETS_DIR="${REPO_DIR}/assets/${CONFIG_NAME}/${EXP_NAME}/assets"
WEIGHTS_DIR="${REPO_DIR}/weights"
LOG_DIR="${REPO_DIR}/logs"
IMAGE="${IMAGE:-openpi:latest}"
# Must match the TrainConfig weight_loader:
#   checkpoints/pi05_busybox_push_green_button/params
BASE_VLA_DIR="${CHECKPOINT_DIR}/pi05_busybox_push_green_button"
BASE_VLA_PARAMS="${BASE_VLA_DIR}/params"
BASE_VLA_HF_REPO="pravsels/pi05_busybox_push_green_button"
MIN_FREE_GB="${MIN_FREE_GB:-200}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${ASSETS_DIR}"

LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_$(date -u +%Y%m%d_%H%M%S).log"

# --- Header ---
{
echo "===================================="
echo "Config: ${CONFIG_NAME}"
echo "Experiment: ${EXP_NAME}"
echo "Image: ${IMAGE}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo "Host: $(hostname)"
echo "Started (UTC): $(date -Is --utc)"
echo "===================================="
echo ""
} | tee "${LOG_FILE}"

gpu_mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [ "${gpu_mem_mib}" -lt 80000 ]; then
    echo "ERROR: need at least one GPU with >=80 GB, got ${gpu_mem_mib} MiB" | tee -a "${LOG_FILE}"
    exit 1
fi

free_gb="$(df -BG --output=avail "${REPO_DIR}" | tail -1 | tr -dc '0-9')"
if [ "${free_gb}" -lt "${MIN_FREE_GB}" ]; then
    echo "ERROR: need >=${MIN_FREE_GB} GB free under ${REPO_DIR}, got ${free_gb} GB" | tee -a "${LOG_FILE}"
    exit 1
fi

params_complete() {
    # Orbax sentinels written only after a finished params tree. A killed
    # download can leave an empty or partial params/ directory.
    [ -f "${BASE_VLA_PARAMS}/_METADATA" ] && [ -f "${BASE_VLA_PARAMS}/manifest.ocdbt" ]
}

if ! params_complete; then
    echo "Downloading ${BASE_VLA_HF_REPO} into ${BASE_VLA_DIR} via ${IMAGE} ..." | tee -a "${LOG_FILE}"
    rm -rf "${BASE_VLA_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
    # Download inside the image so a Docker-only VM does not need a host `hf` CLI.
    sudo docker run --rm \
      -v "${CHECKPOINT_DIR}:/workspace/repo/checkpoints" \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -e HF_HOME=/workspace/repo/checkpoints/.hf_cache \
      "${IMAGE}" \
      uv run python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="pravsels/pi05_busybox_push_green_button", local_dir="/workspace/repo/checkpoints/pi05_busybox_push_green_button")'
fi
if ! params_complete; then
    echo "ERROR: incomplete baseline VLA params at ${BASE_VLA_PARAMS} (missing _METADATA or manifest.ocdbt)" | tee -a "${LOG_FILE}"
    exit 1
fi

if [ ! -f "${ASSETS_DIR}/norm_stats.json" ] || [ ! -f "${ASSETS_DIR}/norm_stats_actions_per_timestep.json" ]; then
    if [ -f "${BASE_VLA_DIR}/assets/norm_stats.json" ] && [ -f "${BASE_VLA_DIR}/assets/norm_stats_actions_per_timestep.json" ]; then
        echo "Copying Hub norm-stat assets from ${BASE_VLA_DIR}/assets" | tee -a "${LOG_FILE}"
        mkdir -p "${ASSETS_DIR}"
        cp -a "${BASE_VLA_DIR}/assets/." "${ASSETS_DIR}/"
    else
        echo "ERROR: Hub assets missing under ${BASE_VLA_DIR}/assets; refusing to recompute" | tee -a "${LOG_FILE}"
        exit 1
    fi
fi

TRAIN_ARGS=(
    --exp-name="${EXP_NAME}"
    --assets-dir="/workspace/repo/assets/${CONFIG_NAME}/${EXP_NAME}/assets"
)
RLT_CKPT_DIR="${CHECKPOINT_DIR}/${CONFIG_NAME}/${EXP_NAME}"
if [ -d "${RLT_CKPT_DIR}" ] && [ -n "$(find "${RLT_CKPT_DIR}" -mindepth 1 -maxdepth 1 -type d ! -name assets 2>/dev/null | head -1)" ]; then
    TRAIN_ARGS+=(--resume)
else
    TRAIN_ARGS+=(--overwrite)
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is empty. Add it to ${OPENPI_ENV} (or export it)." | tee -a "${LOG_FILE}"
    exit 1
fi

start_time="$(date +%s)"

set +e
sudo docker run --gpus all --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${WEIGHTS_DIR}:/workspace/repo/weights" \
  -v "${REPO_DIR}/assets:/workspace/repo/assets" \
  -v "${REPO_DIR}/src:/workspace/repo/src" \
  -v "${CHECKPOINT_DIR}:/workspace/repo/checkpoints" \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  -e HF_HOME=/workspace/repo/weights/hf_cache \
  -e PYTHONUNBUFFERED=1 \
  -e WANDB_MODE=online \
  -e WANDB_ENTITY="${WANDB_ENTITY:-pravsels}" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${IMAGE}" \
  uv run scripts/train.py "${CONFIG_NAME}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

end_time="$(date +%s)"
elapsed=$(( end_time - start_time ))
hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))

{
echo ""
echo "===================================="
echo "Started (UTC):  $(date -Is --utc -d @${start_time} 2>/dev/null || date -u -r ${start_time} +%Y-%m-%dT%H:%M:%S%z)"
echo "Finished (UTC): $(date -Is --utc)"
echo "Runtime: ${hours}h ${minutes}m ${seconds}s"
echo "Exit Code: ${EXIT_CODE}"
echo "Log: ${LOG_FILE}"
echo "===================================="
} | tee -a "${LOG_FILE}"

exit "${EXIT_CODE}"

#!/bin/bash
#SBATCH --job-name=smoke_rbb_pi0
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=115G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6abf/pravsels.u6abf/rbb_openpi/logs/smoke_%j.out
#SBATCH --error=/scratch/u6abf/pravsels.u6abf/rbb_openpi/logs/smoke_%j.err

# 1xGH200 smoke test for openpi Pi0 on a real RoboBuilderBench task.
# Runs norm stats (--max-frames 128) + 2 steps of scripts/train.py (batch_size=4)
# with real AV1 PyAV dataset loading, pi0_base weight loading, and checkpoint save.
#
# Usage:
#   sbatch slurm/smoke_rbb_pi0_slurm.sh pi0_rbb_open_clamp
#   sbatch slurm/smoke_rbb_pi0_slurm.sh pi0_rbb_so101_insert_gear_in_gripper

set -euo pipefail

CONFIG_NAME="${1:-pi0_rbb_open_clamp}"
SCRATCH_ROOT="/scratch/u6abf/pravsels.u6abf"
WORK_ROOT="${SCRATCH_ROOT}/rbb_openpi"
CONTAINER="${WORK_ROOT}/container/openpi_arm64.sif"
REPO_DIR="${REPO_DIR:-/home/u6abf/pravsels.u6abf/openpi_rbb}"
SECRETS_DIR="${SCRATCH_ROOT}/.secrets"
EXP_NAME="smoke_${CONFIG_NAME}_job${SLURM_JOB_ID:-local}"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: Container SIF not found: ${CONTAINER}" >&2
    exit 1
fi
if [[ ! -f "${WORK_ROOT}/weights/pi0_base/params/_METADATA" ]]; then
    echo "ERROR: pi0_base weights not found at ${WORK_ROOT}/weights/pi0_base/params" >&2
    exit 1
fi

SMOKE_ROOT="${WORK_ROOT}/smoke_runs/${EXP_NAME}"
ASSETS_DIR="${SMOKE_ROOT}/assets"
CKPT_BASE="${SMOKE_ROOT}/checkpoints"
rm -rf "${SMOKE_ROOT}"
mkdir -p "${ASSETS_DIR}" "${CKPT_BASE}" "${WORK_ROOT}/logs" "${WORK_ROOT}/cache/hf" \
    "${WORK_ROOT}/openpi_cache" "${WORK_ROOT}/tmp"

module purge
module load brics/apptainer-multi-node

if [[ -f "${SECRETS_DIR}/.hf_token" ]]; then
    export HF_TOKEN="$(tr -d '\r\n' < "${SECRETS_DIR}/.hf_token")"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "=== openpi Pi0 Smoke Test ==="
echo "Host:      $(hostname) ($(uname -m))"
echo "Config:    ${CONFIG_NAME}"
echo "Repo:      ${REPO_DIR}"
echo "Container: ${CONTAINER}"
echo "Smoke dir: ${SMOKE_ROOT}"
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
    --env "WANDB_MODE=disabled"
    --env "WANDB_DISABLED=true"
    --env "TMPDIR=${WORK_ROOT}/tmp"
    --env "XLA_PYTHON_CLIENT_MEM_FRACTION=0.90"
    "${CONTAINER}"
)

echo "=== Step 1/3: Verify JAX GPU & openpi import path ==="
"${APPTAINER_CMD[@]}" /.venv/bin/python - "${REPO_DIR}" <<'PY'
import sys
import jax
import openpi

repo_dir = sys.argv[1]
devs = jax.devices()
assert any(d.platform == "gpu" for d in devs), f"No JAX GPU found: {devs}"
assert openpi.__file__.startswith(repo_dir), f"Expected repo openpi, got {openpi.__file__}"
print(f"OK jax_devices={devs} openpi={openpi.__file__}")
PY

echo "=== Step 2/3: Compute per-timestep norm stats (--max-frames 128) ==="
"${APPTAINER_CMD[@]}" /.venv/bin/python scripts/compute_norm_stats_per_timestep.py \
    --config-name="${CONFIG_NAME}" \
    --assets-dir="${ASSETS_DIR}" \
    --max-frames=128

echo "=== Step 3/3: Run 2-step Pi0 training smoke test ==="
"${APPTAINER_CMD[@]}" /.venv/bin/python scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --assets-dir="${ASSETS_DIR}" \
    --checkpoint-base-dir="${CKPT_BASE}" \
    --batch-size=4 \
    --num-workers=0 \
    --num-train-steps=2 \
    --log-interval=1 \
    --save-interval=2 \
    --no-wandb-enabled \
    --overwrite

echo "=== Verifying smoke checkpoint ==="
ls -la "${CKPT_BASE}/${CONFIG_NAME}/${EXP_NAME}"
test -d "${CKPT_BASE}/${CONFIG_NAME}/${EXP_NAME}/1" || test -d "${CKPT_BASE}/${CONFIG_NAME}/${EXP_NAME}/2"
echo "=== openpi Pi0 Smoke Test PASSED ==="

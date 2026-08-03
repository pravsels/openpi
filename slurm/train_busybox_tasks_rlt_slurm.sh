#!/bin/bash
#SBATCH --job-name=busybox_tasks_rlt_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue
#SBATCH --exclude=nid010755

# Generic RLT Stage-1 launcher for the villekuosmanen busybox pi0.5 policies:
#   pi05_rlt_busybox_press_green_yellow_buttons
#   pi05_rlt_busybox_flip_left_switch_off
#
# This does NOT train a VLA. It attaches a learned RL-token encoder-decoder to
# the *frozen* per-task baseline VLA (rl_vla_loss_weight=0.0,
# get_rl_freeze_filter), so only the ~small RL-token bottleneck is optimised.
# The resulting checkpoint is what `hw_control.pi0_rlt` loads (needs
# `sample_actions_with_rl_token`) to build the demo cache and run online RL.
# Same dataset + delta-action setup as the baseline so norm stats and the 12D
# action space line up.
#
# Both datasets are bimanual SO101 (12D dual-arm joint-space, cameras
# top/left_wrist/right_wrist, LeRobot v3.0, 20 episodes each). The config carries
# everything (dataset, prompt, base VLA path, schedule); this script is
# config-agnostic.
#
# Short run: single GPU, 6h walltime, 10k steps, one checkpoint at the end.
#
# Usage:
#   sbatch [--job-name=<cfg>] slurm/train_busybox_tasks_rlt_slurm.sh <CONFIG_NAME> [EXP_NAME]
# EXP_NAME defaults to CONFIG_NAME. Normally invoked by
# slurm/submit_busybox_tasks_rlt.sh.
#
# Before first sbatch:
#   - openpi_arm64.sif at ${data_dir}/container/
#   - The matching baseline VLA already trained on this cluster by
#     slurm/train_busybox_tasks_slurm.sh, i.e.
#       ${data_dir}/checkpoints/<base_cfg>/<base_cfg>/9999/params
#     where <base_cfg> is the CONFIG_NAME with the "_rlt" dropped
#     (pi05_rlt_busybox_x -> pi05_busybox_x). The TrainConfig's weight_loader
#     points at that path — keep the two in sync.
#   - HF / W&B tokens in ${scratch_dir}/.secrets/
#   - The pi05_rlt_busybox_* TrainConfigs present in this openpi checkout
#     (git pull before submitting).

set -e

module purge
module load brics/apptainer-multi-node

# --- Arguments ---
CONFIG_NAME="$1"
EXP_NAME="${2:-$1}"
if [ -z "${CONFIG_NAME}" ]; then
    echo "ERROR: CONFIG_NAME (arg 1) is required."
    echo "Usage: sbatch slurm/train_busybox_tasks_rlt_slurm.sh <CONFIG_NAME> [EXP_NAME]"
    exit 1
fi

# --- Infrastructure (u6kr) ---
home_dir="/home/u6kr/lorenzo.u6kr"
scratch_dir="/scratch/u6kr/lorenzo.u6kr"
repo_dir="${home_dir}/openpi_busybox_tasks"
data_dir="${scratch_dir}/openpi"
container="${data_dir}/container/openpi_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
HF_TOKEN_FILE="${scratch_dir}/.secrets/.hf_token"
WANDB_DIR="${data_dir}"
WANDB_CACHE_DIR="${scratch_dir}/.cache/wandb"
WANDB_CONFIG_DIR="${scratch_dir}/.config/wandb"
XDG_CACHE_HOME="${scratch_dir}/.cache"
XDG_CONFIG_HOME="${scratch_dir}/.config"

CHECKPOINT_DIR="${data_dir}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
ASSETS_DIR="${data_dir}/assets/${CONFIG_NAME}/${EXP_NAME}/assets"

# --- Baseline VLA that the RL-token head attaches to (must exist, frozen) ---
# Derived from CONFIG_NAME by dropping the "_rlt" marker, matching the
# weight_loader path in the TrainConfig. Written by the baseline training job as
# <config>/<exp>/<step>/params with a bare-number step dir.
BASE_CONFIG_NAME="${CONFIG_NAME/pi05_rlt_/pi05_}"
BASE_VLA_PARAMS="${data_dir}/checkpoints/${BASE_CONFIG_NAME}/${BASE_CONFIG_NAME}/9999/params"

if [ -z "${ASSETS_DIR}" ]; then
    echo "ERROR: ASSETS_DIR is empty; refusing to run."
    exit 1
fi
if [ ! -f "${container}" ]; then
    echo "ERROR: container not found: ${container}"
    exit 1
fi
if [ ! -f "${HF_TOKEN_FILE}" ]; then
    echo "ERROR: HF token not found: ${HF_TOKEN_FILE}"
    exit 1
fi
if [ ! -d "${BASE_VLA_PARAMS}" ]; then
    echo "ERROR: baseline VLA params not found: ${BASE_VLA_PARAMS}"
    echo "       Train the baseline first (slurm/submit_busybox_tasks.sh), or"
    echo "       download it from HF and update the TrainConfig weight_loader."
    exit 1
fi

mkdir -p "${HF_CACHE}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" \
    "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${data_dir}/checkpoints" \
    "${data_dir}/assets" "${data_dir}/weights" "${data_dir}/.venv" "${ASSETS_DIR}"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Config: ${CONFIG_NAME}"
echo "Experiment: ${EXP_NAME}"
echo "Base VLA: ${BASE_VLA_PARAMS}"
echo "Repo: ${repo_dir}"
echo "Started (UTC): ${start_time}"
echo "===================================="

COMPUTE_NORM_STATS_CMD="uv run scripts/compute_norm_stats_per_timestep.py --config-name=${CONFIG_NAME} --assets-dir=${ASSETS_DIR}"
NORM_STATS_PATH="${ASSETS_DIR}/norm_stats.json"
PER_TIMESTEP_STATS_PATH="${ASSETS_DIR}/norm_stats_actions_per_timestep.json"

TRAIN_FLAGS="--exp-name=${EXP_NAME} --assets-dir=${ASSETS_DIR}"
if [ -d "${CHECKPOINT_DIR}" ] && [ -n "$(find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d ! -name assets 2>/dev/null | head -1)" ]; then
    TRAIN_FLAGS="${TRAIN_FLAGS} --resume"
else
    TRAIN_FLAGS="${TRAIN_FLAGS} --overwrite"
fi
TRAIN_CMD="uv run scripts/train.py ${CONFIG_NAME} ${TRAIN_FLAGS}"

EXPORT_VARS="export PYTHONUNBUFFERED=1"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_MODE=offline"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_DIR=${WANDB_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CACHE_DIR=${WANDB_CACHE_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CACHE_HOME=${XDG_CACHE_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CONFIG_HOME=${XDG_CONFIG_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_ENTITY=uttini-lorenzo"
EXPORT_VARS="${EXPORT_VARS} && export OPENPI_DATA_HOME=${data_dir}"
EXPORT_VARS="${EXPORT_VARS} && export UV_PROJECT_ENVIRONMENT=${data_dir}/.venv"
EXPORT_VARS="${EXPORT_VARS} && export HF_TOKEN=\$(tr -d '\n' < ${HF_TOKEN_FILE})"
EXPORT_VARS="${EXPORT_VARS} && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"

VALID_INDICES_PATH="${ASSETS_DIR}/valid_indices.txt"

PRECOMPUTE_CMD=""
if [ -f "${NORM_STATS_PATH}" ] && [ -f "${PER_TIMESTEP_STATS_PATH}" ]; then
    echo "Skipping normalization precompute (found stats files)."
else
    echo "Running normalization precompute..."
    echo "Command: ${COMPUTE_NORM_STATS_CMD}"
    echo ""
    PRECOMPUTE_CMD="${PRECOMPUTE_CMD}${COMPUTE_NORM_STATS_CMD} && "
fi

if [ ! -f "${VALID_INDICES_PATH}" ]; then
    echo "Generating valid_indices.txt (all frames valid for this dataset)..."
    PRECOMPUTE_CMD="${PRECOMPUTE_CMD}uv run python -c \"
from openpi.training import config as _config
from openpi.training.data_loader import create_torch_dataset
cfg = _config.get_config('${CONFIG_NAME}')
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
ds = create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
n = len(ds)
with open('${VALID_INDICES_PATH}', 'w') as f:
    f.write(','.join(str(i) for i in range(n)))
print(f'Wrote {n} valid indices to ${VALID_INDICES_PATH}')
\" && "
fi

echo "Running training command..."
echo "Command: ${TRAIN_CMD}"
echo ""

set +e
srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
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
echo "Checkpoint location: ${CHECKPOINT_DIR}"
echo "===================================="

if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "Check slurm-${SLURM_JOB_ID}.err for detailed error messages"
    exit ${EXIT_CODE}
fi

exit 0

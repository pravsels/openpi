#!/usr/bin/env bash
# Train π0 or π0.5 on GMAN. Secrets: HF_TOKEN, WANDB_API_KEY as typed refs.
# WANDB_MODE must be online in the process environment (train.py defaults offline).
# Node must be 8x >=80GB H100 — not CRA chip=h100-1.
set -eu
export PYTHONUNBUFFERED=1 PATH="${HOME}/.local/bin:${PATH}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-pravsels}"
export REQUIRE_JAX_DEVICES="${REQUIRE_JAX_DEVICES:-8}"
# Match the proven Isambard/GCloud launchers. JAX otherwise reserves only 75%
# of HBM, which made full-replica training fail despite fitting in ~78 GiB.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

REPO_DIR="${REPO_DIR:-${HOME}/openpi}"
CONFIG_NAME="${CONFIG_NAME:?CONFIG_NAME is required}"
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
    DEFAULT_EXP_NAME="${CONFIG_NAME}_smoke"
else
    DEFAULT_EXP_NAME="${CONFIG_NAME}"
fi
EXP_NAME="${EXP_NAME:-${DEFAULT_EXP_NAME}}"
ASSETS_DIR="${ASSETS_DIR:-${HOME}/openpi_runs/${CONFIG_NAME}/${EXP_NAME}/assets}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}}"

cd "${REPO_DIR}"
: "${HF_TOKEN:?HF_TOKEN must be injected through a GMAN secret reference}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be injected through a GMAN secret reference}"

mkdir -p "${ASSETS_DIR}" "$(dirname "${CHECKPOINT_DIR}")"

if [[ ! -d "${REPO_DIR}/weights/pi0_base/params" ]] || [[ ! -d "${REPO_DIR}/weights/pi05_base/params" ]]; then
    echo "ERROR: base weights missing under ${REPO_DIR}/weights" >&2
    exit 1
fi

if [[ "${SMOKE}" != "1" ]]; then
    CHECKPOINT_DIR="${CHECKPOINT_DIR}" uv run python - <<'PY'
from pathlib import Path
import os
from scripts.gman_payload import refuse_production_if_smoke_checkpoints

refuse_production_if_smoke_checkpoints(Path(os.environ["CHECKPOINT_DIR"]))
PY
fi

if [[ ! -f "${ASSETS_DIR}/norm_stats.json" ]] || [[ ! -f "${ASSETS_DIR}/norm_stats_actions_per_timestep.json" ]]; then
    echo "=== norm stats $(date -Is --utc) ==="
    uv run scripts/compute_norm_stats_per_timestep.py \
        --config-name="${CONFIG_NAME}" \
        --assets-dir="${ASSETS_DIR}"
fi

TRAIN_FLAGS=(--exp-name="${EXP_NAME}" --assets-dir="${ASSETS_DIR}")
if [[ -d "${CHECKPOINT_DIR}" ]] && [[ -n "$(find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d ! -name assets 2>/dev/null | head -1)" ]]; then
    TRAIN_FLAGS+=(--resume)
else
    TRAIN_FLAGS+=(--overwrite)
fi

if [[ "${SMOKE}" == "1" ]]; then
    TRAIN_FLAGS+=(
        --num-train-steps=10
        --save-interval=5
        --keep-period=5
        --log-interval=1
    )
fi

echo "=== train ${CONFIG_NAME} exp=${EXP_NAME} $(date -Is --utc) ==="
uv run scripts/train.py "${CONFIG_NAME}" "${TRAIN_FLAGS[@]}"

if [[ "${SMOKE}" == "1" ]]; then
    HF_REPO="${HF_REPO:-pravsels/${CONFIG_NAME}}"
    case "${CONFIG_NAME}" in
        pi0_busybox_push_green_button) WANDB_PROJECT="${WANDB_PROJECT:-busybox_push_green_button_pi0}" ;;
        pi05_busybox_push_green_button) WANDB_PROJECT="${WANDB_PROJECT:-busybox_push_green_button_pi05}" ;;
        *) WANDB_PROJECT="${WANDB_PROJECT:-${CONFIG_NAME}}" ;;
    esac
    echo "=== smoke publish ${HF_REPO} $(date -Is --utc) ==="
    HF_REPO="${HF_REPO}" CHECKPOINT_DIR="${CHECKPOINT_DIR}" CONFIG_NAME="${CONFIG_NAME}" \
    uv run python - <<'PY'
from pathlib import Path
import os
from huggingface_hub import HfApi
from scripts.gman_publish import assert_hub_steps_exist, publish_checkpoint_steps

repo_id = os.environ["HF_REPO"]
ckpt = Path(os.environ["CHECKPOINT_DIR"])
api = HfApi()
publish_checkpoint_steps(
    api,
    repo_id=repo_id,
    checkpoint_root=ckpt,
    steps=("5", "9"),
    config_name=os.environ["CONFIG_NAME"],
)
assert_hub_steps_exist(api, repo_id, ("5", "9"))
print("hub_ok", repo_id)
PY
    echo "=== smoke wandb $(date -Is --utc) ==="
    WANDB_PROJECT="${WANDB_PROJECT}" CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
    uv run python - <<'PY'
from pathlib import Path
import os
import wandb
from scripts.gman_publish import assert_wandb_history_logged

ckpt = Path(os.environ["CHECKPOINT_DIR"])
run_id = (ckpt / "wandb_id.txt").read_text().strip()
entity = os.environ.get("WANDB_ENTITY", "pravsels")
project = os.environ["WANDB_PROJECT"]
run = wandb.Api().run(f"{entity}/{project}/{run_id}")
history = list(run.scan_history(keys=["_step"]))
assert_wandb_history_logged(history)
print("wandb_ok", run.url)
PY
fi

echo "train_done"

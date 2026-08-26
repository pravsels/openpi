#!/usr/bin/env bash
# Remove local smoke checkpoints and Hub step_5/step_9 before the 30k job.
set -eu
export PYTHONUNBUFFERED=1 PATH="${HOME}/.local/bin:${PATH}"

REPO_DIR="${REPO_DIR:-${HOME}/openpi}"
CONFIG_NAME="${CONFIG_NAME:?CONFIG_NAME is required}"
HF_REPO="${HF_REPO:-pravsels/${CONFIG_NAME}}"

cd "${REPO_DIR}"
: "${HF_TOKEN:?HF_TOKEN must be injected through a GMAN secret reference}"

echo "=== delete smoke ${CONFIG_NAME} $(date -Is --utc) ==="
HF_REPO="${HF_REPO}" CONFIG_NAME="${CONFIG_NAME}" REPO_DIR="${REPO_DIR}" \
uv run python - <<'PY'
from pathlib import Path
import os
from huggingface_hub import HfApi
from scripts.gman_payload import SMOKE_STEPS
from scripts.gman_payload import experiment_name
from scripts.gman_publish import delete_hub_step_folders
from scripts.gman_publish import delete_local_checkpoint_steps

config_name = os.environ["CONFIG_NAME"]
repo_dir = Path(os.environ["REPO_DIR"])
smoke_dir = repo_dir / "checkpoints" / config_name / experiment_name(config_name, smoke=True)
deleted_local = delete_local_checkpoint_steps(smoke_dir, SMOKE_STEPS)
api = HfApi()
deleted_hub = delete_hub_step_folders(api, os.environ["HF_REPO"], SMOKE_STEPS)
print("deleted_local", deleted_local)
print("deleted_hub", deleted_hub)
PY
echo "smoke_deleted"

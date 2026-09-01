#!/usr/bin/env bash
# Vast train wrapper: 4 JAX devices, pi05_busybox_multitask, online W&B.
# Load tokens from /workspace/secrets/* without echoing them. Do not set -x.
set -eu
export PYTHONUNBUFFERED=1
export REPO_DIR="${REPO_DIR:-/workspace/openpi}"
export CONFIG_NAME="${CONFIG_NAME:-pi05_busybox_multitask}"
export REQUIRE_JAX_DEVICES="${REQUIRE_JAX_DEVICES:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-pravsels}"
export WANDB_PROJECT="${WANDB_PROJECT:-busybox_multitask_pi05}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

if [[ -f /workspace/secrets/hf_token ]]; then
    HF_TOKEN="$(tr -d '\n' < /workspace/secrets/hf_token)"
    export HF_TOKEN
fi
if [[ -f /workspace/secrets/wandb_token ]]; then
    WANDB_API_KEY="$(tr -d '\n' < /workspace/secrets/wandb_token)"
    export WANDB_API_KEY
fi

cd "${REPO_DIR}"
exec bash gman/train.sh

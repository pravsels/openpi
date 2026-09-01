#!/usr/bin/env bash
# Vast bootstrap for pi05_busybox_multitask. 4× ≥80 GB. Clone first or let
# gman/bootstrap.sh clone task/busybox_multitask into /workspace/openpi.
# Load tokens from /workspace/secrets/* without echoing them. Do not set -x.
set -eu
export PYTHONUNBUFFERED=1
export REPO_URL="${REPO_URL:-https://github.com/pravsels/openpi.git}"
export REPO_REF="${REPO_REF:-task/busybox_multitask}"
export REPO_DIR="${REPO_DIR:-/workspace/openpi}"
export EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-${REPO_DIR}/weights}"

if [[ -f /workspace/secrets/hf_token ]]; then
    HF_TOKEN="$(tr -d '\n' < /workspace/secrets/hf_token)"
    export HF_TOKEN
fi
if [[ -f /workspace/secrets/github_token ]]; then
    GITHUB_TOKEN="$(tr -d '\n' < /workspace/secrets/github_token)"
    export GITHUB_TOKEN
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../gman/bootstrap.sh" ]]; then
    exec bash "${SCRIPT_DIR}/../gman/bootstrap.sh"
fi
exec bash "${REPO_DIR}/gman/bootstrap.sh"

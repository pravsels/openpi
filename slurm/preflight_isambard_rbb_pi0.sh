#!/bin/bash
# Login-node preflight check for RoboBuilderBench Pi0 (openpi) on Isambard (u6abf).
# Usage: bash slurm/preflight_isambard_rbb_pi0.sh [config_name]

set -euo pipefail

CONFIG_NAME="${1:-pi0_rbb_open_clamp}"
SCRATCH_ROOT="/scratch/u6abf/pravsels.u6abf"
WORK_ROOT="${SCRATCH_ROOT}/rbb_openpi"
CONTAINER="${WORK_ROOT}/container/openpi_arm64.sif"
WEIGHTS_META="${WORK_ROOT}/weights/pi0_base/params/_METADATA"
SECRETS_DIR="${SCRATCH_ROOT}/.secrets"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

PASS=0
FAIL=0

ok()   { echo "[PASS] $*"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL + 1)); }

echo "=== Preflight: RoboBuilderBench Pi0 via openpi (Isambard u6abf) ==="
echo "Host:   $(hostname) ($(uname -m))"
echo "Config: ${CONFIG_NAME}"

# 1. Architecture & Slurm
if [[ "$(uname -m)" == "aarch64" ]]; then
    ok "Architecture is aarch64 (GH200)"
else
    fail "Unexpected architecture: $(uname -m)"
fi

if command -v sbatch >/dev/null 2>&1 && command -v squeue >/dev/null 2>&1; then
    ok "Slurm commands available ($(sinfo -h -p workq -o '%P %a %D nodes' 2>/dev/null || echo 'workq'))"
else
    fail "Slurm commands (sbatch/squeue) not found"
fi

# 2. Disk quotas (home & scratch)
HOME_USED_PCT="$(df -P "${HOME}" | awk 'NR==2 {gsub("%",""); print $5}')"
if [[ "${HOME_USED_PCT}" -lt 85 ]]; then
    ok "HOME usage is ${HOME_USED_PCT}%"
else
    fail "HOME usage is high (${HOME_USED_PCT}%)"
fi

if [[ -d "${SCRATCH_ROOT}" ]]; then
    SCRATCH_AVAIL="$(df -h "${SCRATCH_ROOT}" | awk 'NR==2 {print $4}')"
    ok "Scratch mounted at ${SCRATCH_ROOT} (${SCRATCH_AVAIL} available)"
else
    fail "Scratch directory missing: ${SCRATCH_ROOT}"
fi

# 3. Secrets
for secret in .hf_token .wandb_token; do
    if [[ -s "${SECRETS_DIR}/${secret}" ]]; then
        ok "Secret present: ${SECRETS_DIR}/${secret}"
    else
        fail "Secret missing or empty: ${SECRETS_DIR}/${secret}"
    fi
done

# 4. Container SIF
if [[ -f "${CONTAINER}" ]]; then
    ok "Container SIF present: ${CONTAINER} ($(du -h "${CONTAINER}" | cut -f1))"
else
    fail "Container SIF missing: ${CONTAINER} (run sbatch slurm/build_openpi_arm64_slurm.sh)"
fi

# 5. Staged pi0_base weights
if [[ -f "${WEIGHTS_META}" ]]; then
    ok "pi0_base weights staged: ${WORK_ROOT}/weights/pi0_base/params"
else
    fail "pi0_base weights missing: ${WEIGHTS_META}"
fi

# 6. Config registered in repo
if grep -q "name=\"${CONFIG_NAME}\"" "${REPO_DIR}/src/openpi/training/config.py"; then
    ok "Config ${CONFIG_NAME} defined in src/openpi/training/config.py"
else
    fail "Config ${CONFIG_NAME} not found in src/openpi/training/config.py"
fi

# 7. Ensure scratch dirs
mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/checkpoints" "${WORK_ROOT}/assets" \
    "${WORK_ROOT}/weights" "${WORK_ROOT}/cache/hf" "${WORK_ROOT}/wandb" "${WORK_ROOT}/tmp"
ok "Scratch directories ensured under ${WORK_ROOT}"

echo "=== Summary: ${PASS} passed, ${FAIL} failed ==="
[[ "${FAIL}" -eq 0 ]]

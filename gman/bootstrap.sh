#!/usr/bin/env bash
# GMAN bootstrap for π0 / π0.5 BusyBox green-button (CRA pattern: host uv, no Docker).
# Inject GITHUB_TOKEN and HF_TOKEN as typed GMAN secret refs. Do not set -x.
set -eu
export PYTHONUNBUFFERED=1 PATH="${HOME}/.local/bin:${PATH}"

REPO_URL="${REPO_URL:-https://github.com/pravsels/openpi.git}"
REPO_REF="${REPO_REF:-task/train_pi_policies_green_button}"
REPO_DIR="${REPO_DIR:-${HOME}/openpi}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${REPO_DIR}/weights}"
export WEIGHTS_DIR

echo "=== inventory $(date -Is --utc) ==="
python3 --version
nvidia-smi -L || true
df -h "${HOME}"

mapfile -t gpu_memory < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
if (( ${#gpu_memory[@]} != 8 )); then
    echo "Expected exactly 8 GPUs, found ${#gpu_memory[@]}" >&2
    exit 1
fi
for memory_mib in "${gpu_memory[@]}"; do
    if (( memory_mib < 80000 )); then
        echo "Expected at least 80 GB per GPU, found ${memory_mib} MiB" >&2
        exit 1
    fi
done

: "${GITHUB_TOKEN:?GITHUB_TOKEN must be injected through a GMAN secret reference}"
: "${HF_TOKEN:?HF_TOKEN must be injected through a GMAN secret reference}"

if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
echo "=== apt $(date -Is --utc) ==="
${SUDO} apt-get update -y
${SUDO} apt-get install -y --no-install-recommends \
  ffmpeg git curl ca-certificates \
  libavutil-dev libavcodec-dev libavformat-dev \
  libswscale-dev libswresample-dev
${SUDO} ldconfig

echo "=== uv $(date -Is --utc) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

python3 - <<'PY'
import os
import subprocess

token = os.environ["GITHUB_TOKEN"]
prefix = f"https://x-access-token:{token}@github.com/"
subprocess.check_call(["git", "config", "--global", f"url.{prefix}.insteadOf", "https://github.com/"])
subprocess.check_call(["git", "config", "--global", "--add", f"url.{prefix}.insteadOf", "git@github.com:"])
PY

echo "=== clone $(date -Is --utc) ==="
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    git clone --branch "${REPO_REF}" --single-branch "${REPO_URL}" "${REPO_DIR}"
else
    git -C "${REPO_DIR}" fetch origin "${REPO_REF}"
    git -C "${REPO_DIR}" checkout "${REPO_REF}"
    git -C "${REPO_DIR}" merge --ff-only "origin/${REPO_REF}"
fi

cd "${REPO_DIR}"
echo "=== branch assets $(date -Is --utc) ==="
python3 - <<'PY'
from pathlib import Path
from scripts.gman_payload import assert_openpi_branch_has_busybox_assets

assert_openpi_branch_has_busybox_assets(Path("."))
print("branch_assets_ok")
PY

echo "=== uv sync $(date -Is --utc) ==="
GIT_LFS_SKIP_SMUDGE=1 uv sync --group dev
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo "=== base weights $(date -Is --utc) ==="
mkdir -p "${WEIGHTS_DIR}"
uv run python - <<'PY'
from pathlib import Path
import os
from openpi.shared.download import maybe_download

root = Path(os.environ.get("WEIGHTS_DIR", Path.home() / "openpi" / "weights"))
for name in ("pi0_base", "pi05_base"):
    src = maybe_download(f"gs://openpi-assets/checkpoints/{name}/params")
    dest = root / name / "params"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != Path(src).resolve():
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src)
    print(f"staged {name} -> {dest}")
PY

echo "setup_done"

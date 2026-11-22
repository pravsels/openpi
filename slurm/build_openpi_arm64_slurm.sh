#!/bin/bash
#SBATCH --job-name=build_rbb_openpi_arm64
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6abf/pravsels.u6abf/rbb_openpi/logs/build_%j.out
#SBATCH --error=/scratch/u6abf/pravsels.u6abf/rbb_openpi/logs/build_%j.err

# Build openpi arm64 Apptainer SIF directly on an Isambard GH200 compute node
# via --fakeroot, verify JAX CUDA GPU + unit tests, and stage pi0_base weights.
#
# Usage (from repo root on Isambard login node):
#   mkdir -p /scratch/u6abf/pravsels.u6abf/rbb_openpi/logs
#   sbatch slurm/build_openpi_arm64_slurm.sh

set -euo pipefail

SCRATCH_ROOT="/scratch/u6abf/pravsels.u6abf"
WORK_ROOT="${SCRATCH_ROOT}/rbb_openpi"
CONTAINER_DIR="${WORK_ROOT}/container"
BASE_IMAGE="docker://nvcr.io/nvidia/pytorch:25.01-py3"
BASE_SIF="${CONTAINER_DIR}/ngc-pytorch-25.01-arm64.sif"
TARGET_SIF="${CONTAINER_DIR}/openpi_arm64.sif"
REPO_DIR="${REPO_DIR:-/home/u6abf/pravsels.u6abf/openpi_rbb}"
DEF_FILE="${REPO_DIR}/apptainer/openpi_arm64.def"
WEIGHTS_DIR="${WORK_ROOT}/weights"

mkdir -p "${CONTAINER_DIR}" "${WORK_ROOT}/logs" "${WEIGHTS_DIR}" \
    "${WORK_ROOT}/apptainer_cache" "${WORK_ROOT}/apptainer_tmp"

module purge
module load brics/apptainer-multi-node

# Clear runtime bind paths injected by brics/apptainer-multi-node; they break
# writable image builds under --fakeroot.
unset APPTAINER_BINDPATH SINGULARITY_BINDPATH

export APPTAINER_CACHEDIR="${WORK_ROOT}/apptainer_cache"
# Use node-local /tmp for APPTAINER_TMPDIR: --fakeroot unpackSIF fails on Lustre
# /scratch and node-local /tmp is dramatically faster for rootfs extraction/squash.
BUILD_TMP="/tmp/${USER}_rbb_openpi_${SLURM_JOB_ID:-local}"
mkdir -p "${BUILD_TMP}"
trap 'rm -rf "${BUILD_TMP}"' EXIT
export APPTAINER_TMPDIR="${BUILD_TMP}"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
export TMPDIR="${APPTAINER_TMPDIR}"
export APPTAINER_CPUS="${SLURM_CPUS_PER_TASK:-24}"

echo "=== Host: $(hostname) ($(uname -m)) ==="
echo "=== Repo: ${REPO_DIR} ==="
echo "=== Def:  ${DEF_FILE} ==="
echo "=== Out:  ${TARGET_SIF} ==="

if [[ ! -f "${BASE_SIF}" ]]; then
    echo "Pulling base NGC image ${BASE_IMAGE} -> ${BASE_SIF} ..."
    apptainer pull --force "${BASE_SIF}" "${BASE_IMAGE}"
else
    echo "Reusing cached base SIF: ${BASE_SIF}"
fi

cd "${REPO_DIR}"
echo "Building ${TARGET_SIF} with apptainer build --fakeroot ..."
apptainer build --fakeroot --force "${TARGET_SIF}" "${DEF_FILE}"

echo "=== Verifying built SIF (GPU + RBB config tests) ==="
ls -lh "${TARGET_SIF}"
apptainer exec --nv --writable-tmpfs \
    --pwd "${REPO_DIR}" \
    -B "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    -B "${REPO_DIR}:${REPO_DIR}" \
    --env "PYTHONPATH=${REPO_DIR}/src" \
    "${TARGET_SIF}" \
    /.venv/bin/python - <<'PY'
import av
import jax
import openpi
from openpi.training.lerobot_rcw_compat import ensure_lerobot_04_module_aliases
ensure_lerobot_04_module_aliases()
import robocandywrapper
import torch

devs = jax.devices()
assert any(d.platform == "gpu" for d in devs), f"No JAX GPU device found: {devs}"
assert torch.cuda.is_available(), "PyTorch CUDA unavailable"
print(f"OK jax_devices={devs} torch_gpu={torch.cuda.get_device_name(0)} av={av.__version__}")
print(f"OK openpi={openpi.__file__}")
PY

apptainer exec --nv --writable-tmpfs \
    --pwd "${REPO_DIR}" \
    -B "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    -B "${REPO_DIR}:${REPO_DIR}" \
    --env "PYTHONPATH=${REPO_DIR}/src" \
    "${TARGET_SIF}" \
    /.venv/bin/pytest src/openpi/policies/yam_bimanual_policy_test.py -v

# Stage pi0_base weights onto scratch if not already present.
if [[ ! -f "${WEIGHTS_DIR}/pi0_base/params/_METADATA" ]]; then
    echo "=== Staging gs://openpi-assets/checkpoints/pi0_base/params into ${WEIGHTS_DIR}/pi0_base/params ==="
    apptainer exec --writable-tmpfs \
        --pwd "${REPO_DIR}" \
        -B "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
        -B "${REPO_DIR}:${REPO_DIR}" \
        --env "PYTHONPATH=${REPO_DIR}/src" \
        --env "OPENPI_DATA_HOME=${WORK_ROOT}/openpi_cache" \
        "${TARGET_SIF}" \
        /.venv/bin/python - "${WEIGHTS_DIR}/pi0_base/params" <<'PY'
import shutil
import sys
from pathlib import Path
from openpi.shared import download

target = Path(sys.argv[1])
cached = Path(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    shutil.rmtree(target)
shutil.copytree(cached, target)
assert (target / "_METADATA").is_file(), f"Missing _METADATA in {target}"
print(f"Staged pi0_base weights at {target}")
PY
else
    echo "pi0_base weights already staged at ${WEIGHTS_DIR}/pi0_base/params"
fi

echo "=== Build & staging complete ==="

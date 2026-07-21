#!/bin/bash
# Submit the busybox bimanual v50 retrains (pi0 + pi0.5) as INDEPENDENT
# train->publish job pairs. Each pair is its own Slurm job: if a training fails,
# only its own publish is skipped (DependencyNeverSatisfied) — the other job is
# unaffected. On successful training, the paired publish job (afterok dependency)
# uploads the final checkpoint straight to Hugging Face.
#
# Dataset: pravsels/busybox_buttons_bimanual (bimanual SO101, 12D dual-arm
# joint-space, cameras top/left_wrist/right_wrist, 50 episodes, LeRobot v3.0).
# Task: press the green button with the left arm, then the yellow button with the
# right arm.
#
# Profile (from the v50 recipe): 25k steps, batch 32, 2 GPUs, save every 5k.
# Reuses the generic, config-agnostic launchers slurm/train_v50_slurm.sh and
# slurm/publish_v50_slurm.sh, so everything runs from the single shared v50
# worktree (~/openpi_so101_v50) and shares ONE venv — this avoids the
# cross-worktree editable-install race.
#
# Usage:
#   cd ~/openpi_so101_v50
#   bash slurm/submit_busybox_v50.sh            # submit both pairs
#   DRY_RUN=1 bash slurm/submit_busybox_v50.sh  # print what would be submitted
#
# Requires (staged on scratch): pi0_base AND pi05_base weights, container,
# HF read token (.hf_token) and HF write token (.hf_token_write).

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root (the worktree)

# keep_period=25000 deletes intermediate checkpoints, so the final step is the
# only one persisted; 24999 is the 0-indexed last step, 25000 is listed as a
# fallback in case the trainer wrote that name. Missing steps are skipped.
PUBLISH_STEPS="${PUBLISH_STEPS:-24999 25000}"
DRY_RUN="${DRY_RUN:-0}"

# config_name (== exp_name) | hf_repo_id
JOBS=(
    "pi0_busybox_buttons_bimanual_v50|lorenzouttini/pi0-so101-busybox-buttons-bimanual-isambard-v50"
    "pi05_busybox_buttons_bimanual_v50|lorenzouttini/pi05-so101-busybox-buttons-bimanual-isambard-v50"
)

echo "Submitting ${#JOBS[@]} train->publish pairs (PUBLISH_STEPS=\"${PUBLISH_STEPS}\")"
echo ""

for entry in "${JOBS[@]}"; do
    cfg="${entry%%|*}"
    hf="${entry##*|}"

    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] train: ${cfg}"
        echo "[dry-run]   publish -> ${hf}  steps=[${PUBLISH_STEPS}]"
        continue
    fi

    TRAIN=$(sbatch --parsable --job-name="${cfg}" slurm/train_v50_slurm.sh "${cfg}" "${cfg}")
    PUB=$(sbatch --parsable --dependency=afterok:"${TRAIN}" --job-name="pub_${cfg}" \
        slurm/publish_v50_slurm.sh "${cfg}" "${cfg}" "${hf}" "${PUBLISH_STEPS}")
    echo "submitted ${cfg}: train=${TRAIN} publish=${PUB}"
done

echo ""
echo "Done. Monitor with: squeue --me"

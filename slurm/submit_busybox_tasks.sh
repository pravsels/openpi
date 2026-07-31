#!/bin/bash
# Submit the two villekuosmanen busybox pi0.5 policies as INDEPENDENT
# train->publish job pairs. Each pair is its own Slurm job: if a training fails,
# only its own publish is skipped (DependencyNeverSatisfied) — the other job is
# unaffected. On successful training, the paired publish job (afterok dependency)
# uploads the final checkpoint straight to Hugging Face.
#
# Datasets (both bimanual SO101, 12D dual-arm joint-space, cameras
# top/left_wrist/right_wrist, 20 episodes, LeRobot v3.0):
#   villekuosmanen/busybox_press_green_yellow_buttons — press green (left arm)
#     then yellow (right arm).
#   villekuosmanen/busybox_flip_left_switch_off — flip the left switch off.
#
# Profile: 10k steps, batch 32, 2 GPUs, 8h walltime, save every 5k.
# Both pairs run from the single shared worktree (~/openpi_busybox_tasks) and
# share ONE venv — this avoids the cross-worktree editable-install race.
#
# Usage:
#   cd ~/openpi_busybox_tasks
#   bash slurm/submit_busybox_tasks.sh            # submit both pairs
#   DRY_RUN=1 bash slurm/submit_busybox_tasks.sh  # print what would be submitted
#
# Requires (staged on scratch): pi05_base weights, container, HF read token
# (.hf_token) and HF write token (.hf_token_write).

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root (the worktree)

# keep_period=10000 deletes the intermediate checkpoint, so the final step is the
# only one persisted; 9999 is the 0-indexed last step, 10000 is listed as a
# fallback in case the trainer wrote that name. Missing steps are skipped.
PUBLISH_STEPS="${PUBLISH_STEPS:-9999 10000}"
DRY_RUN="${DRY_RUN:-0}"

# config_name (== exp_name) | hf_repo_id
JOBS=(
    "pi05_busybox_press_green_yellow_buttons|lorenzouttini/pi05-so101-busybox-press-green-yellow-buttons-isambard"
    "pi05_busybox_flip_left_switch_off|lorenzouttini/pi05-so101-busybox-flip-left-switch-off-isambard"
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

    TRAIN=$(sbatch --parsable --job-name="${cfg}" slurm/train_busybox_tasks_slurm.sh "${cfg}" "${cfg}")
    PUB=$(sbatch --parsable --dependency=afterok:"${TRAIN}" --job-name="pub_${cfg}" \
        slurm/publish_busybox_tasks_slurm.sh "${cfg}" "${cfg}" "${hf}" "${PUBLISH_STEPS}")
    echo "submitted ${cfg}: train=${TRAIN} publish=${PUB}"
done

echo ""
echo "Done. Monitor with: squeue --me"

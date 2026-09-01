#!/bin/bash
# Submit the villekuosmanen busybox pi0.5 policies as INDEPENDENT
# train->publish job pairs. Each pair is its own Slurm job: if a training fails,
# only its own publish is skipped (DependencyNeverSatisfied) — the other jobs are
# unaffected. On successful training, the paired publish job (afterok dependency)
# uploads the final checkpoint straight to Hugging Face.
#
# Datasets (bimanual SO101, 12D dual-arm joint-space, cameras
# top/left_wrist/right_wrist, LeRobot v3.0):
#   villekuosmanen/busybox_press_green_yellow_buttons — press green (left arm)
#     then yellow (right arm). 20 episodes, single task.
#   villekuosmanen/busybox_flip_left_switch_off — flip the left switch off.
#     20 episodes, single task.
# pi05_busybox_multitask is not submitted here. The Hub dataset is now single-arm
# 6D / 30k three-cam; see docs/plans/2026-09-01-pi05-busybox-multitask.md.
#
# Profile: 10k steps, batch 32, 2 GPUs, 8h walltime, save every 5k.
# All pairs run from the single shared worktree (~/openpi_busybox_tasks) and
# share ONE venv — this avoids the cross-worktree editable-install race.
#
# Usage:
#   cd ~/openpi_busybox_tasks
#   bash slurm/submit_busybox_tasks.sh                  # submit every pair
#   bash slurm/submit_busybox_tasks.sh <substring>      # only matching configs
#   DRY_RUN=1 bash slurm/submit_busybox_tasks.sh        # print what would be submitted
#
# The optional substring filter is for re-recorded datasets: when one task's
# dataset is re-collected under the same HF repo id, retrain just that task.
# Delete its assets dir (stale norm stats) and checkpoint dir (otherwise the
# train script resumes the old run) before resubmitting.
#
# Requires (staged on scratch): pi05_base weights, container, HF read token
# (.hf_token) and HF write token (.hf_token_write).

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root (the worktree)

FILTER="${1:-}"

# keep_period=10000 deletes the intermediate checkpoint, so the final step is the
# only one persisted; 9999 is the 0-indexed last step, 10000 is listed as a
# fallback in case the trainer wrote that name. Missing steps are skipped.
PUBLISH_STEPS="${PUBLISH_STEPS:-9999 10000}"
DRY_RUN="${DRY_RUN:-0}"

# config_name (== exp_name) | hf_repo_id
JOBS=(
    "pi05_busybox_press_green_yellow_buttons|lorenzouttini/pi05-so101-busybox-press-green-yellow-buttons-isambard"
    "pi05_busybox_flip_left_switch_off|lorenzouttini/pi05-so101-busybox-flip-left-switch-off-isambard"
    # pi05_busybox_multitask is the single-arm 30k three-cam recipe; do not launch
    # it with this 10k 2-GPU Isambard pair.
    # "pi05_busybox_multitask|lorenzouttini/pi05-so101-busybox-multitask-isambard"
)

echo "Submitting train->publish pairs (PUBLISH_STEPS=\"${PUBLISH_STEPS}\"${FILTER:+, filter=\"${FILTER}\"})"
echo ""

submitted=0
for entry in "${JOBS[@]}"; do
    cfg="${entry%%|*}"
    hf="${entry##*|}"

    if [ -n "${FILTER}" ] && [[ "${cfg}" != *"${FILTER}"* ]]; then
        continue
    fi
    submitted=$((submitted + 1))

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

if [ "${submitted}" -eq 0 ]; then
    echo "ERROR: filter \"${FILTER}\" matched no configs." >&2
    exit 1
fi

echo ""
echo "Done. Monitor with: squeue --me"

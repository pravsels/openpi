#!/bin/bash
# Submit the RLT Stage-1 runs for the two villekuosmanen busybox pi0.5 policies
# as INDEPENDENT train->publish job pairs. Each pair is its own Slurm job: if a
# training fails, only its own publish is skipped (DependencyNeverSatisfied) —
# the other job is unaffected. On successful training, the paired publish job
# (afterok dependency) uploads the final checkpoint straight to Hugging Face.
#
# Stage 1 attaches an RL-token encoder-decoder to the *frozen* per-task baseline
# VLA, so the baselines must already be trained on this cluster — run
# slurm/submit_busybox_tasks.sh first and let it finish. The train script checks
# for the baseline params and fails fast if they are missing.
#
# The published checkpoints are what `hw_control.pi0_rlt` pulls to build the demo
# cache (§3 of hw_control/pi0_rlt/README.md) and then run the actor/critic
# learner. Both tasks are bimanual, so downstream steps use --embodiment biso101.
#
# Profile: 10k steps, batch 16, 1 GPU, 6h walltime, single checkpoint at the end.
# Runs from the same worktree as the baselines (~/openpi_busybox_tasks) so both
# stages share ONE venv — this avoids the cross-worktree editable-install race.
#
# Usage:
#   cd ~/openpi_busybox_tasks
#   bash slurm/submit_busybox_tasks_rlt.sh            # submit both pairs
#   DRY_RUN=1 bash slurm/submit_busybox_tasks_rlt.sh  # print what would be submitted
#
# Requires (staged on scratch): baseline checkpoints, container, HF read token
# (.hf_token) and HF write token (.hf_token_write).

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root (the worktree)

# 9999 is the 0-indexed last step of a 10k run; 10000 is listed as a fallback in
# case the trainer wrote that name. Missing steps are skipped.
PUBLISH_STEPS="${PUBLISH_STEPS:-9999 10000}"
DRY_RUN="${DRY_RUN:-0}"

# config_name (== exp_name) | hf_repo_id
JOBS=(
    "pi05_rlt_busybox_press_green_yellow_buttons|lorenzouttini/pi05-rlt-so101-busybox-press-green-yellow-buttons-isambard"
    "pi05_rlt_busybox_flip_left_switch_off|lorenzouttini/pi05-rlt-so101-busybox-flip-left-switch-off-isambard"
)

echo "Submitting ${#JOBS[@]} RLT train->publish pairs (PUBLISH_STEPS=\"${PUBLISH_STEPS}\")"
echo ""

for entry in "${JOBS[@]}"; do
    cfg="${entry%%|*}"
    hf="${entry##*|}"

    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] train: ${cfg}"
        echo "[dry-run]   publish -> ${hf}  steps=[${PUBLISH_STEPS}]"
        continue
    fi

    TRAIN=$(sbatch --parsable --job-name="${cfg}" slurm/train_busybox_tasks_rlt_slurm.sh "${cfg}" "${cfg}")
    PUB=$(sbatch --parsable --dependency=afterok:"${TRAIN}" --job-name="pub_${cfg}" \
        slurm/publish_busybox_tasks_slurm.sh "${cfg}" "${cfg}" "${hf}" "${PUBLISH_STEPS}")
    echo "submitted ${cfg}: train=${TRAIN} publish=${PUB}"
done

echo ""
echo "Done. Monitor with: squeue --me"

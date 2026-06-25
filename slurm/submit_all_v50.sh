#!/bin/bash
# Submit all 21 v50 retrains (12 pi0 + 9 pi0.5) as INDEPENDENT train->publish
# job pairs. Each pair is its own Slurm job: if one training fails, only its own
# publish is skipped (DependencyNeverSatisfied) — every other job is unaffected.
#
# All jobs run from this single worktree (~/openpi_so101_v50) and share one venv,
# which is exactly what avoids the cross-worktree editable-install race.
#
# Usage:
#   cd ~/openpi_so101_v50
#   bash slurm/submit_all_v50.sh            # submit everything
#   DRY_RUN=1 bash slurm/submit_all_v50.sh  # print what would be submitted
#
# Requires (staged on scratch): pi0_base AND pi05_base weights, container, tokens.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root (the worktree)

PUBLISH_STEPS="${PUBLISH_STEPS:-10000 25000 49999}"
DRY_RUN="${DRY_RUN:-0}"

# config_name (== exp_name) | hf_repo_id
JOBS=(
    # --- pi0 (12) ---
    "pi0_so101_object_top_shelf_v50|lorenzouttini/pi0-so101-object-top-shelf-isambard-v50"
    "pi0_so101_object_top_shelf_reset_v50|lorenzouttini/pi0-so101-object-top-shelf-reset-isambard-v50"
    "pi0_so101_cable_clip_v50|lorenzouttini/pi0-so101-cable-clip-isambard-v50"
    "pi0_so101_cable_unclip_v50|lorenzouttini/pi0-so101-cable-unclip-isambard-v50"
    "pi0_armnetbench_ring_insert_v50|lorenzouttini/pi0-so101-armnetbench-ring-insert-isambard-v50"
    "pi0_armnetbench_block_stack_v50|lorenzouttini/pi0-so101-armnetbench-block-stack-isambard-v50"
    "pi0_armnetbench_tool_insert_v50|lorenzouttini/pi0-so101-armnetbench-tool-insert-isambard-v50"
    "pi0_armnetbench_tool_removal_v50|lorenzouttini/pi0-so101-armnetbench-tool-removal-isambard-v50"
    "pi0_armnetbench_insert_candle_v50|lorenzouttini/pi0-so101-armnetbench-insert-candle-isambard-v50"
    "pi0_armnetbench_transfer_cube_v50|lorenzouttini/pi0-so101-armnetbench-transfer-cube-isambard-v50"
    "pi0_armnetbench_fold_tea_towel_v50|lorenzouttini/pi0-so101-armnetbench-fold-tea-towel-isambard-v50"
    "pi0_armnetbench_open_lamp_door_v50|lorenzouttini/pi0-so101-armnetbench-open-lamp-door-isambard-v50"
    # --- pi0.5 (9) ---
    "pi05_so101_object_top_shelf_reset_v50|lorenzouttini/pi05-so101-object-top-shelf-reset-isambard-v50"
    "pi05_armnetbench_ring_insert_v50|lorenzouttini/pi05-so101-armnetbench-ring-insert-isambard-v50"
    "pi05_armnetbench_block_stack_v50|lorenzouttini/pi05-so101-armnetbench-block-stack-isambard-v50"
    "pi05_armnetbench_tool_insert_v50|lorenzouttini/pi05-so101-armnetbench-tool-insert-isambard-v50"
    "pi05_armnetbench_tool_removal_v50|lorenzouttini/pi05-so101-armnetbench-tool-removal-isambard-v50"
    "pi05_armnetbench_insert_candle_v50|lorenzouttini/pi05-so101-armnetbench-insert-candle-isambard-v50"
    "pi05_armnetbench_transfer_cube_v50|lorenzouttini/pi05-so101-armnetbench-transfer-cube-isambard-v50"
    "pi05_armnetbench_fold_tea_towel_v50|lorenzouttini/pi05-so101-armnetbench-fold-tea-towel-isambard-v50"
    "pi05_armnetbench_open_lamp_door_v50|lorenzouttini/pi05-so101-armnetbench-open-lamp-door-isambard-v50"
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

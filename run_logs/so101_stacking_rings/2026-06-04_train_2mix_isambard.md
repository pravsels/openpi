# Train 2-mix — pi0.5 SO101 stacking rings (Isambard u6kr)

## Mode
- run_type: experiment
- objective: fresh pi0.5 fine-tune on 2-dataset mix (original teleop + rollout corrections) to improve ring stacking success rate

## Config
- script: `slurm/train_so101_stacking_rings_slurm.sh`
- config: `pi05_so101_stacking_rings`
- datasets:
  - `lorenzouttini/so101_stacking_rings` (101 episodes, ~34k frames, original teleop)
  - `lorenzouttini/rollout_so101_stacking_rings_20260603_154953` (100 episodes, ~28k frames, rollout corrections)
- key settings: lr 2.5e-5, batch 32, 50k steps, action_horizon 30, delta actions, init from pi05_base

## Job
- execution_id: pending
- submitted: pending

## Status

## Results

## W&B
- local: pending
- synced: pending
- notes: pending

## Next
- submit job on Isambard after pushing branch and syncing worktree

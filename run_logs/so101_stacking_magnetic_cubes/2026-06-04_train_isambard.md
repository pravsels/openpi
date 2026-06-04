# Train — pi0.5 SO101 stacking magnetic cubes (Isambard u6kr)

## Mode
- run_type: replication
- objective: first pi0.5 fine-tune on `lorenzouttini/so101_stacking_magnetic_cubes` on Isambard (same recipe as rings baseline)

## Config
- script: `slurm/train_so101_stacking_magnetic_cubes_slurm.sh`
- config: `pi05_so101_stacking_magnetic_cubes`
- exp_name: `so101_stacking_magnetic_cubes`
- dataset: `lorenzouttini/so101_stacking_magnetic_cubes`
- prompt: `stack the magnetic cubes`
- key settings: pi0.5, action_horizon 30, delta actions, lr 2.5e-5 → 2.5e-6 cosine, batch 32, 50k steps, save every 5k, init from `weights/pi05_base/params`
- worktree: `/home/u6kr/pravsels.u6kr/openpi_so101_stacking_magnetic_cubes` @ `task/stack_magnetic_cube`
- code: `6da1046` (local; push before submit)

## Infrastructure
- cluster: Isambard-AI, project u6kr
- hardware: 4× GH200, exclusive node, 1-day walltime (`--requeue`)
- SIF: `/scratch/u6kr/pravsels.u6kr/openpi/container/openpi_arm64.sif`
- checkpoints: `/scratch/u6kr/pravsels.u6kr/openpi/checkpoints/pi05_so101_stacking_magnetic_cubes/so101_stacking_magnetic_cubes/`
- assets: `/scratch/u6kr/pravsels.u6kr/openpi/assets/pi05_so101_stacking_magnetic_cubes/so101_stacking_magnetic_cubes/assets/`
- wandb: offline, entity `pravsels`, project `so101_stacking_magnetic_cubes` (from config)

## Job
- execution_id: pending
- submitted: pending
- start_human: Thursday, Jun 4th, 2026

## Status
- pending submit

## Results

## W&B
- local: pending
- project: `so101_stacking_magnetic_cubes` (confirm entity/project before sync)
- synced: pending

## Next
- after complete: W&B sync (`autohpc-wandb-sync`), then `checkpoint-passport` on chosen step before eval/HF

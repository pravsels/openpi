# Train — pi0.5 SO101 stacking big rings (Isambard u6kr, pravsels)

## Mode
- run_type: replication
- objective: first pi0.5 fine-tune on `lorenzouttini/so101_stacking_big_rings` on Isambard (pravsels branch; Lorenzo keeps `task/stack-big-rings`)

## Config
- script: `slurm/train_so101_stacking_big_rings_isambard_pravsels_slurm.sh`
- config: `pi05_so101_stacking_big_rings`
- exp_name: `so101_stacking_big_rings_isambard`
- dataset: `lorenzouttini/so101_stacking_big_rings`
- prompt: `stack the big rings`
- key settings: pi0.5, action_horizon 30, delta actions, lr 2.5e-5 → 2.5e-6 cosine, batch 32, 50k steps, save every 5k, init from `weights/pi05_base/params`
- branch: `task/stack-big-rings-pravsels` (forked from Lorenzo's `task/stack-big-rings`; his `slurm/train_so101_stacking_big_rings_slurm.sh` untouched)
- worktree: `/home/u6kr/pravsels.u6kr/openpi_so101_stacking_big_rings_pravsels`

## Infrastructure
- cluster: Isambard-AI, project u6kr
- hardware: 4× GH200, exclusive node, 1-day walltime (`--requeue`)
- SIF: `/scratch/u6kr/pravsels.u6kr/openpi/container/openpi_arm64.sif`
- checkpoints: `/scratch/u6kr/pravsels.u6kr/openpi/checkpoints/pi05_so101_stacking_big_rings/so101_stacking_big_rings_isambard/`
- assets: `/scratch/u6kr/pravsels.u6kr/openpi/assets/pi05_so101_stacking_big_rings/so101_stacking_big_rings_isambard/assets/`
- wandb: offline, entity `pravsels`, project `so101_stacking_big_rings` (config default; exp_name disambiguates on scratch)

## Job
- execution_id: 5042576
- submitted: pending (poll sacct for start time)
- start_human: Thursday, Jun 4th, 2026
- slurm logs: `openpi_so101_stacking_big_rings_pravsels/slurm-5042576.{out,err}`

## Status
- 2026-06-04 — submitted job 5042576 (queued); first run, no prior checkpoints

## Results

## W&B
- local: pending
- project: `so101_stacking_big_rings`
- synced: pending

## Next
- after complete: W&B sync, checkpoint-passport, optional HF publish

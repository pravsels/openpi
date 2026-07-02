## Run

- script: `slurm/train_armnetbench_tool_insert_rlt_slurm.sh`
- config: `pi05_rlt_armnetbench_tool_insert`
- exp_name: `rlt_v50_stage1`
- branch: `task/rlt_robometer`
- commit: `fabb77b`
- cluster: Isambard-AI, project `u6kr`
- worktree: `/home/u6kr/pravsels.u6kr/openpi_rlt_robometer`
- scratch: `/scratch/u6kr/pravsels.u6kr/openpi`
- container: `/scratch/u6kr/pravsels.u6kr/openpi/container/openpi_arm64.sif`

## Config

- model: `Pi0RLConfig(pi05=True, action_horizon=30, rl_vla_loss_weight=0.0)`
- dataset: `villekuosmanen/armnetbench_tool_insert`
- init checkpoint: `checkpoints/pi05-so101-armnetbench-tool-insert-isambard-v50/step_24999/params` (local, bound to scratch)
- batch_size: 32
- num_train_steps: 20,000
- save_interval: 10,000
- W&B: offline, entity `pravsels`, project `pi05_rlt_armnetbench_tool_insert`

## Job

- job_id: `5453755`
- submit_time: 2026-07-01 (20k steps, after clearing prior logs)
- resources: 1 exclusive node, 4 GPUs, `--mem=0G`, 24 CPUs, 8h walltime
- initial_state: `PD (Priority)`
- finished: 2026-07-01 17:33 UTC, COMPLETED (exit 0), runtime 4h13m
- final train loss: ~356 at step 19,999

## Preflight

- remote branch clean at `fabb77b`
- init checkpoint staged at `/scratch/u6kr/pravsels.u6kr/openpi/checkpoints/pi05-so101-armnetbench-tool-insert-isambard-v50/step_24999/params`
- precomputed assets from prior run still present under `assets/pi05_rlt_armnetbench_tool_insert/rlt_v50_stage1/`
- resubmit after job `5437084` failed on HF init checkpoint download

- cleared slurm logs and `wandb_id.txt` from cancelled job `5453486`

## Status

- Training completed successfully (job `5453755`).
- W&B synced: https://wandb.ai/pravsels/pi05_rlt_armnetbench_tool_insert/runs/jbdqrmu0

## HuggingFace

- repo: https://huggingface.co/pravsels/pi05-so101-armnetbench-tool-insert-rlt-v50
- uploaded: step 19999 (`params/` + `assets/`, no `train_state/`)
- params SHA256: `d9ddbbbefc07b3700f7df5dd161c14de3291bfcf805f71e39ababb902e1501b2`

# reward_recap fix positive_only — post-placement + valid-indices update

## Mode
- run_type: experiment
- objective: rerun bin-pack positive-only recap after fixing advantage-token placement, config-aware valid-index persistence, and shortening the run to 50k steps

## Config
- script: `slurm/train_bin_pack_reward_recap_slurm.sh positive_only`
- config: `pi05_bin_pack_coffee_capsules_recap_positive_only` (`src/openpi/training/config.py`)
- dataset: 9 datasets (bin_pick_pack + 8 dAgger rounds) on HuggingFace
- key settings: 17D action space, delta actions enabled, per-timestep action normalization, advantage dropout `0.3`, positive-only recap semantics, `num_train_steps=50_000`, code pulled on HPC at commit `e26281c`

## Job
- job_id: 3586066
- submitted/start: `2026-04-02T16:37:23Z`
- start_human: Thursday, Apr 2nd, 2026
- end: `2026-04-02T16:38:08Z`
- end_human: Thursday, Apr 2nd, 2026
- runtime: `00:00:43`
- node: `nid010225`

## Status

## Results

## W&B
- local: pending
- synced: pending
- notes:

## Next
- monitor the resubmitted run

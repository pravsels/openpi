# eval — RL token cosine similarity (ID vs OOD)

## Provenance
- checkpoint: `/scratch/u6cr/pravsels.u6cr/openpi/checkpoints/pi05_rl_token_build_block_tower/rlt_v1/9999/params`
- source run log: `run_logs/build_block_tower_rlt/2026-03-25_rlt_v1.md`
- config: `pi05_rl_token_build_block_tower`
- ID dataset: `villekuosmanen/build_block_tower` (LeRobot v2.1, 200 episodes)
- OOD dataset: `villekuosmanen/eval_dAgger_drop_footbag_into_dice_tower_1.7.0`
- episodes per dataset: 1 (first episode from each)
- extraction script: `scripts/rl_token_extract_episodes.py`
- analysis script: `scripts/rl_token_cosine_analysis.py`
- slurm: `slurm/rl_token_extract_slurm.sh`

## Purpose
Sanity check: does the frozen RLT Stage 1 RL token (2048-dim, step 9999) produce distinguishable embeddings for in-distribution vs out-of-distribution episodes? If the RL token encodes task-relevant structure, ID and OOD embeddings should occupy different regions of the representation space.

## Method
1. Extract per-frame RL tokens from 1 ID episode (build_block_tower) and 1 OOD episode (drop_footbag_into_dice_tower) using the same model and data pipeline (ID norm stats applied to both).
2. Compute cross-episode cosine similarity matrix (ID frames x OOD frames).
3. Compute within-episode baselines (ID-ID and OOD-OOD frame-pair cosine sim distributions).
4. Compute mean-pooled episode-level cosine similarity.

## Expected Results
- Within-episode (ID-ID, OOD-OOD) cosine sim should be high (frames from the same episode share context).
- Cross-episode (ID vs OOD) cosine sim should be noticeably lower than within-episode sim.
- Mean-pooled episode-level sim should be lower than within-episode means.
- If the RL token is not task-discriminative, cross and within-episode sims will be similar.

## Job
- job_id: ~~3392855~~ → 3393659 (resubmit after trimming docstrings + adding stdout prints)
- submitted: `2026-03-27T09:05:00+00:00` → resubmitted ~09:45 UTC
- start: `2026-03-27T11:10:23+00:00`
- start_human: Friday, Mar 27th, 2026
- end: `2026-03-27T11:32:23+00:00` (extraction); analysis rerun manually after installing matplotlib
- runtime: ~22min (extraction), ~7s (analysis)
- node: nid010026

## Metrics
- output_dir: `/scratch/u6cr/pravsels.u6cr/openpi/eval_outputs/build_block_tower_rlt_stage2/rl_token_cosine_sim`
- id_frames: 1028
- ood_frames: 486
- embedding_dim: 2048
- episode_level_cosine_sim: **0.9941** — cosine sim between the mean-pooled ID episode embedding and the mean-pooled OOD episode embedding
- id_self_sim_mean: 0.9720 (std=0.0099) — all-pairs cosine sim between frames within the ID episode (upper triangle, 527k pairs)
- ood_self_sim_mean: 0.9879 (std=0.0055) — all-pairs cosine sim between frames within the OOD episode (upper triangle, 118k pairs)
- cross_frame_sim_mean: 0.9741 (std=0.0056) — cosine sim between every ID frame and every OOD frame (500k pairs)

## Follow-up: Reconstruction Ablation
- follow-up job_id: 3394831
- report: `/scratch/u6cr/pravsels.u6cr/openpi/eval_outputs/build_block_tower_rlt_stage2/rl_token_recon_ablation_9999.json`
- split: train
- sampled_batches: 4
- batch_size: 8
- sampled_examples: 32
- real_recon_loss: 226.2
- zero_recon_loss: 1038.3 (gap: +812.2) — decoder reconstruction loss when `z_rl` is replaced with zeros
- shuffled_recon_loss: 316.3 (gap: +90.1) — decoder reconstruction loss when each sample receives another sample's `z_rl`
- rl_token_pairwise_cosine_mean: 0.9698 (std=0.0065)

## Qualitative
The heatmap and distribution plots are saved locally at `eval_outputs/rl_token_cosine_sim/`.

The cross-episode heatmap shows near-uniform high similarity (~0.97) across all ID-OOD frame pairs with no visible structure — no block-diagonal patterns, no gradient, no low-similarity regions. The distribution histogram confirms overlap: cross-episode, ID self-sim, and OOD self-sim distributions sit in the same narrow band (0.93–1.0).

The follow-up reconstruction ablation changes the interpretation. The decoder is not ignoring `z_rl`: zeroing the token makes reconstruction much worse, and shuffling tokens across samples also hurts. So `z_rl` contains real sample-specific information that the decoder uses.

At the same time, pairwise cosine between RL tokens is still extremely high (~0.97). The most likely picture is not "dead token" but "token dominated by a large shared component, with smaller residual directions carrying useful information." In other words, the RL token is not task-discriminative in raw cosine space even though it is informative for reconstruction.

## Verdict
**Mixed negative result.** The frozen Stage 1 RL token (step 9999) is not task-discriminative under raw cosine similarity: cross-episode (ID vs OOD) similarity (0.974) is comparable to within-episode self-similarity (ID: 0.972, OOD: 0.988), and the mean-pooled episode-level sim is 0.994. So the token does not cleanly separate tasks in embedding space.

However, the follow-up reconstruction ablation shows the token is still being used. Replacing `z_rl` with zeros increases reconstruction loss from 226.2 to 1038.3, and shuffling `z_rl` across samples also degrades reconstruction to 316.3. This means the token is not ignored and does carry sample-specific information. The failure mode is therefore more subtle than complete collapse: the representation appears to be dominated by a shared component, with useful information compressed into smaller residual directions.

## Next
- Re-run similarity analysis after centering / whitening / PCA to test whether task information is hiding in low-variance residual directions rather than the dominant shared component
- Compare this reconstruction-ablation result at step 19999 to see whether task separation improves or degrades later in training
- Run the same ablation on val / OOD batches to test whether decoder dependence on `z_rl` generalizes beyond sampled train batches

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

## Qualitative
The heatmap and distribution plots are saved locally at `eval_outputs/rl_token_cosine_sim/`.

The cross-episode heatmap shows near-uniform high similarity (~0.97) across all ID-OOD frame pairs with no visible structure — no block-diagonal patterns, no gradient, no low-similarity regions. The distribution histogram confirms overlap: cross-episode, ID self-sim, and OOD self-sim distributions sit in the same narrow band (0.93–1.0).

The RL token embeddings are effectively constant regardless of input task/episode.

## Verdict
**Negative result.** The frozen Stage 1 RL token (step 9999) does not produce task-discriminative embeddings. Cross-episode (ID vs OOD) cosine similarity (0.974) is comparable to within-episode self-similarity (ID: 0.972, OOD: 0.988). The mean-pooled episode-level sim is 0.994 — nearly identical. The RL token appears to be collapsing to a near-constant representation that does not meaningfully distinguish build_block_tower from drop_footbag_into_dice_tower.

## Next
- Investigate whether the RL token is collapsing due to training (e.g. mode collapse during Stage 1 RL) or architecture (e.g. the token is dominated by the language prompt rather than visual/state information)
- Try comparing embeddings from different checkpoints (early vs late training) to see if collapse is progressive
- Consider whether the RL token needs to be unfrozen or re-trained with a contrastive/discriminative objective to encode task structure

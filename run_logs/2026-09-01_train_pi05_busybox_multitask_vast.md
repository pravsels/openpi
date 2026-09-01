# Train — π0.5 BusyBox multitask (Vast)

## Mode
- run_type: full-component fine-tune
- objective: train π0.5 on `villekuosmanen/busybox_multitask` with the green-button relative-action recipe and per-episode task prompts
- status: not launched

## Config
- config: `pi05_busybox_multitask`
- exp_name: `pi05_busybox_multitask`
- dataset: `villekuosmanen/busybox_multitask` (LeRobot v3, 66 episodes, 12141 frames, 20 fps, 27 tasks)
- prompt: `prompt_from_task` (no single default)
- cameras: `top` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`, `front` → `base_1_rgb`
- key settings: π0.5, full-component fine-tuning, action horizon 30, global batch 32, 30k steps, save every 5k and retain only the latest checkpoint, cosine LR 2.5e-5 → 2.5e-6, EMA 0.999, relative actions (5 joints delta, gripper absolute), per-timestep quantile
- parallelism: 4-GPU full data parallel (`fsdp_devices=1`) — planned
- input pipeline: TorchCodec, 8 persistent workers
- memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- init: `weights/pi05_base/params`
- plan: `docs/plans/2026-09-01-pi05-busybox-multitask.md`

## Infrastructure
- provider: Vast
- instance: `49582742` (`pi05-busybox-multitask`)
- offer: `49467276` 4× H100 SXM 80GB HBM3, Taiwan, billed ~$13.54/hr, 1516 GB disk
- image: `nvcr.io/nvidia/pytorch:25.03-py3`
- SSH: `ssh -i ~/.ssh/id_ed25519 -p 22742 root@ssh8.vast.ai`
- clone: `/workspace/openpi` at `6b94b6b` (`task/busybox_multitask`)
- do not touch CRA `49561214`

## Status
- 2026-09-01 — config rewritten to single-arm three-cam / 30k. Launch pending.
- 2026-09-01 21:35 UTC — rented Vast `49582742`. SSH up, 4× H100 80GB, clone at `6b94b6b`. Bootstrap waiting on `/workspace/secrets/{hf_token,wandb_token}`.

## Next
- Vast bootstrap + 10-step publish gate, then 30k
- Variant 2 (`pi05_busybox_multitask_abs`) later

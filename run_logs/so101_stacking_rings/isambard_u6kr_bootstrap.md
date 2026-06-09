# Isambard u6kr bootstrap (SO101)

## Login
- Host: `u6kr.aip2.isambard` (or `ssh isambard`)
- User: `pravsels.u6kr`
- Clifton cert: refresh with `clifton auth` if SSH fails

## Paths
| Purpose | Path |
|---------|------|
| Git PAT | `~/pat.txt` (login home, mode 600) |
| HF token | `/scratch/u6kr/pravsels.u6kr/.secrets/.hf_token` |
| W&B token | `/scratch/u6kr/pravsels.u6kr/.secrets/.wandb_token` |
| Main clone | `/home/u6kr/pravsels.u6kr/openpi` |
| SO101 rings worktree | `/home/u6kr/pravsels.u6kr/openpi_so101_stacking_rings` (`task/stack_rings`) |
| SO101 magnetic cubes worktree | `/home/u6kr/pravsels.u6kr/openpi_so101_stacking_magnetic_cubes` (`task/stack_magnetic_cube`) |
| SIF (target) | `/scratch/u6kr/pravsels.u6kr/openpi/container/openpi_arm64.sif` |
| Scratch data | `/scratch/u6kr/pravsels.u6kr/openpi/` (checkpoints, weights, wandb) |
| HF cache | `/scratch/u6kr/pravsels.u6kr/huggingface_cache` |

## Secrets usage (agents)
```bash
scratch_dir="/scratch/u6kr/pravsels.u6kr"
export HF_TOKEN="$(tr -d '\n' < "${scratch_dir}/.secrets/.hf_token")"
export WANDB_API_KEY="$(tr -d '\n' < "${scratch_dir}/.secrets/.wandb_token")"
```
Do not echo tokens. Do not commit token files.

## HF bridge (SIF from expired u6cr)
- Repo: https://huggingface.co/pravsels/openpi-isambard (`openpi_arm64.sif` 14.6GB, uploaded 2026-06-02 via laptop)
- u6kr download: `wget` with `Authorization: Bearer $HF_TOKEN` → `/scratch/u6kr/pravsels.u6kr/openpi/container/openpi_arm64.sif`
- Log: `/scratch/u6kr/pravsels.u6kr/hf_sif_download.log`

## u6cr (read-only, no Slurm)
- Old project; SIF at `/scratch/u6cr/pravsels.u6cr/openpi/container/openpi_arm64.sif`
- Cross-project scratch not readable from u6kr

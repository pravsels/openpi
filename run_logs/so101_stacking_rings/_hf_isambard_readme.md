# OpenPI arm64 Apptainer image (Isambard)

Pre-built `openpi_arm64.sif` for Isambard GH200 nodes (arm64). Built 2026-01-24 on u6cr.

## Usage

```bash
module purge
module load brics/apptainer-multi-node
apptainer exec --nv /scratch/u6kr/pravsels.u6kr/openpi/container/openpi_arm64.sif bash
```

Do not use amd64 Docker images on Isambard.

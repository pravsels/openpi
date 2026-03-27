# Bin Pack Reward Recap Experiments

| run | description | weight init | advantage mode | loss@25k | loss@50k | latest loss | status | notes |
|-----|-------------|-------------|----------------|----------|----------|-------------|--------|-------|
| 2026-03-24_positive_only | resume from 1_dataset ckpt | 1_dataset/29999 | positive_only | 0.0074 | 0.0058 | 0.0050 @83k | running | lowest loss of the four |
| 2026-03-24_mixed | resume from 1_dataset ckpt | 1_dataset/29999 | mixed | 0.0098 | 0.0075 | 0.0061 @74k | running | high initial loss (0.50) from negative demos, converged well |
| 2026-03-24_positive_only_from_base | from pi05 base | pi05_base | positive_only | 0.0097 | 0.0066 | 0.0053 @75k | running | converging to similar range as task-pretrained |
| 2026-03-24_mixed_from_base | from pi05 base | pi05_base | mixed | 0.0125 | ~0.0075 | 0.0070 @74k | stopped | converged well; checkpoints 25k/50k/74k uploaded to HF |

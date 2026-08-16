# Final Inference Benchmark

Device: **NVIDIA GeForce RTX 4060 Laptop GPU** (measured locally; not an H100 prediction)
Checkpoint: `C:\Projects\kla-image-restoration\weights\residualsr_final_ema.pt`
Input: synthetic 128x128 (seed=42)
Model load time: 0.099s

| Mode | Mean (ms) | Median (ms) | Throughput (img/s) | Peak CUDA mem (MiB) |
| --- | ---: | ---: | ---: | ---: |
| none | 15.57 | 15.18 | 64.24 | 21.3 |
| x8 | 50.22 | 47.19 | 19.91 | 23.3 |

x8 TTA is **3.23x** slower than a single pass.

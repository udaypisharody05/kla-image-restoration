# SuperconductorSemistars -- KLA PS01: AI-Based Restoration of Degraded Images

Self-contained evaluator package. Champion model: ResidualSR (1-channel
input, 64 features, 8 residual blocks, PixelShuffle x2 upsampling,
630,724 parameters), EMA weights.

## Run it

```bash
pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py ./input ./output
```

That's it -- no other flags, checkpoint path, or config file. The command
works from any working directory; everything (model code, weights) is
resolved relative to this script's own location.

## Input

- Every `.npy` file directly under `<input-dir>` is processed (`sorted(input_dir.glob("*.npy"))`, deterministic order).
- Each file is a grayscale array. Accepted shapes: `(H, W)`, `(H, W, 1)`, or `(1, H, W)` -- all are normalized internally to `(H, W)` before inference. Any other shape raises a clear error instead of being silently reinterpreted.
- Values are passed to the model unchanged (no re-normalization), matching this project's validated training/inference convention -- the official dataset's `NoisyLR` arrays are used as-is.

## Output

- One restored `.npy` written to `<output-dir>` per input file, with the **identical filename** (`abc.npy -> abc.npy`, never renamed).
- `<output-dir>` is created automatically if it doesn't exist.
- Shape: grayscale `(H*2, W*2)` -- exactly 2x the corresponding input's spatial resolution, validated per-file before saving.
- `dtype`: `float32`.
- Values: clipped to `[0.0, 1.0]`, guaranteed finite (no `NaN`/`Inf`) -- checked immediately before saving.
- If `<input-dir>` and `<output-dir>` are the same path, `run.py` refuses to run rather than risk overwriting source files.

## Runtime

- CUDA is used automatically when available (`torch.cuda.is_available()`); otherwise it falls back to CPU. No `--device` flag needed.
- No internet access, API keys, or external model downloads at any point -- the checkpoint under `models/` contains everything required for inference.
- No interactive prompts or manual configuration.

## Package contents

```text
SuperconductorSemistars/
├── run.py                              # evaluator entrypoint
├── requirements.txt                    # numpy + torch, version-pinned
├── README.md                           # this file
└── models/
    ├── __init__.py
    ├── residual_sr.py                  # self-contained ResidualSR architecture (no src/ dependency)
    └── residualsr_final_ema.pt         # packaged champion EMA weights + architecture metadata
```

This package does not depend on the parent repository's `src/`, training
scripts, or dataset utilities -- it can be copied out and run standalone.

## Verification

This package's weights were verified to load with
`model.load_state_dict(state_dict, strict=True)` (all keys match, zero
missing/unexpected) and to produce numerically identical output to the
parent repository's original model implementation before being copied here.

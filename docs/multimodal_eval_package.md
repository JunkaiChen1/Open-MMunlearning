# Multimodal evaluation package

This repository contains the multimodal benchmark evaluator. Benchmark data
and model checkpoints are external resources and are intentionally excluded
from the release archive.

The evaluator supports reproducible sampling with `max_samples` and
`sample_seed`, for example:

```bash
PYTHONPATH=src python src/eval.py \
  experiment=eval/gqa/default \
  eval.gqa.max_samples=100 \
  eval.gqa.sample_seed=42
```

Model checkpoints are not included. Set the model and tokenizer checkpoint
paths when running an evaluation.

## Use after extraction

From the extracted package directory, install the Python dependencies (Python
3.11 or newer):

```bash
python -m pip install -e ".[lm-eval]"
```

Then run a reproducible smoke test against a local multimodal checkpoint:

```bash
MODEL_PATH=/path/to/llava-1.5-7b-hf \
  BENCHMARK=gqa SAMPLES=8 SAMPLE_SEED=42 \
  ./scripts/run_multimodal_eval.sh
```

Use `BENCHMARK=vqav2` for VQAv2. Set `SAMPLES` to an empty/full-size value
only after verifying the checkpoint and processor on a small sample. Outputs
are written below `saves/eval/<task_name>/` by the Hydra configuration.

The launcher defaults to `DEVICE_MAP=cuda`; use `DEVICE_MAP=auto` or
`DEVICE_MAP=cpu` on a server without a CUDA device.

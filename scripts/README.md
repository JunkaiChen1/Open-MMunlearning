# Utility Scripts

This directory contains reusable utilities only:

- `prepare_*`, `build_*`, `split_*`, and `finalize_*` prepare external
  benchmark files in the layouts expected by the configs.
- `run_multimodal_eval.sh` and `run_unlearning_search.py` provide generic
  experiment entry points.
- `probing_stress_test.py`, `quantization_stress_test.py`, and the
  `relearning_*` scripts provide model checks and analysis helpers.

GPU watchdogs, background schedulers, one-off experiment launchers, and
machine-specific shell wrappers are intentionally excluded from this source
distribution. Use `src/train.py` and `src/eval.py` directly for custom runs.

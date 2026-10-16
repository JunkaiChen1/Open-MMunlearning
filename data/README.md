# External Data

Datasets are intentionally excluded from this source distribution. Download
the benchmark data from its original project or release, then either:

- set `OPEN_UNLEARNING_DATA_DIR` to the directory containing the prepared
  `finetune/`, `unlearn/`, and `eval/` trees; or
- override `paths.data_dir` in a Hydra command.

The files under `configs/data/datasets/` describe the expected Parquet/JSON
layouts. They are configuration templates, not bundled datasets.

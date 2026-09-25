# Open-Source Setup

`open-mmunlearning` is distributed as source code only. Datasets, model
checkpoints, generated adapters, benchmark outputs, and API caches are not
part of the repository. Keep those resources in external storage and pass
their locations through configuration or environment variables.

## Install

```bash
conda create -n open-mmunlearning python=3.11
conda activate open-mmunlearning
python -m pip install -e ".[lm-eval]"
```

Install a CUDA-compatible PyTorch build and FlashAttention separately when
required by the selected model.

## Directory Variables

```bash
export OPEN_UNLEARNING_ROOT=/path/to/open-mmunlearning
export OPEN_UNLEARNING_DATA_DIR=/path/to/prepared/data
export OPEN_UNLEARNING_MODEL_DIR=/path/to/models
```

`paths.data_dir` and `paths.model_dir` can also be overridden directly in a
Hydra command. The prepared data directory should contain the `finetune/`,
`unlearn/`, and `eval/` trees expected by the selected config. Dataset config
files under `configs/data/datasets/` document the expected record fields and
split names.

## Preparing Data

Use the reusable utilities in `scripts/` to convert data downloaded from the
original benchmark releases. The scripts write to an explicit output path;
they do not fetch or commit data into the source tree. For the optional SMFA
IDK responses, run:

```bash
python setup_data.py --idk
export OPEN_UNLEARNING_IDK_PATH=/path/to/prepared/data/idk_responses.txt
```

Set `OPEN_UNLEARNING_DATA_DIR` before running `setup_data.py` to choose the
download location, or pass the path used by your own data preparation job.

## Model Checkpoints

Use public Hugging Face model identifiers where supported. For local LLaVA
Vanilla checkpoints, set:

```bash
export LLAVA_VANILLA_7B_MODEL=/path/to/llava-7b
export LLAVA_VANILLA_13B_MODEL=/path/to/llava-13b
```

Do not place checkpoints under version control. Evaluation and training output
directories should likewise remain outside commits.

## Reproducibility

Record the selected Hydra overrides, model revision, dataset release, random
seed, and output directory for each run. A one-step smoke test is useful for
checking the environment before starting a full unlearning job:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/mllmubench/default \
  model=Qwen2.5-VL-3B-Instruct \
  trainer=GradAscent peft=lora \
  paths.data_dir=/path/to/prepared/data \
  +trainer.args.max_steps=1 \
  +skip_final_save=true \
  task_name=smoke_test
```

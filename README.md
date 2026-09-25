# open-mmunlearning

`open-mmunlearning` is an open-source, Hydra-based framework for multimodal
large language model (MLLM) unlearning, fine-tuning, and benchmark evaluation.
This repository is a code-only distribution: benchmark data, model
checkpoints, generated adapters, logs, and evaluation results are kept outside
the source tree.

## Features

- Multimodal model support for LLaVA, LLaVA-Next, Qwen2.5-VL, Gemma 3, and
  decoder-only language models.
- Unlearning trainers including GradAscent, GradDiff, KL, NPO, RMU, UNDIAL,
  MMUnlearner, MANU, MIPEditor, and SMFA.
- Evaluation support for MLLMU, CLEAR, FIUBench, CoVUBench, POPE, MM-Vet,
  MMBench, GQA, VQAv2, VizWiz, ScienceQA, FigStep, Image Rephrase, Jailbreak,
  and SUA.
- Configurable truth-ratio, forgetting, retain, membership-inference, and
  multimodal benchmark metrics.
- Hydra configuration groups for models, datasets, trainers, evaluators, and
  parameter-efficient fine-tuning.

## Repository Layout

```text
src/         Training, evaluation, model, dataset, and metric implementations
configs/     Hydra configuration groups and experiment defaults
scripts/     Reusable data preparation, evaluation, and stress-test utilities
tests/       Unit and contract tests
community/   Contribution and issue templates
data/        Local data mount point (contents intentionally excluded)
models/      Local model mount point (contents intentionally excluded)
```

## Installation

```bash
conda create -n open-mmunlearning python=3.11
conda activate open-mmunlearning
python -m pip install -e ".[lm-eval]"
```

Install a CUDA-compatible PyTorch build and FlashAttention separately when
they are required by the selected model or hardware.

## External Resources

Prepare benchmark data and model checkpoints outside this repository, then
point the configuration at them with environment variables:

```bash
export OPEN_UNLEARNING_DATA_DIR=/path/to/prepared/data
export OPEN_UNLEARNING_MODEL_DIR=/path/to/models
```

SMFA additionally needs an IDK response file:

```bash
export OPEN_UNLEARNING_IDK_PATH=/path/to/idk_responses.txt
```

The standard model configurations use public Hugging Face identifiers. Custom
`LLaVA_Vanilla_*` configurations use `LLAVA_VANILLA_7B_MODEL` and
`LLAVA_VANILLA_13B_MODEL` for local checkpoints.

Evaluation logs, generated text, API caches, and manifests can contain source
questions, names, image identifiers, or local filesystem paths. Keep these
outputs outside commits and review them before sharing.

## Unlearning

Run an experiment with a Hydra configuration:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/mllmubench/default \
  model=Qwen2.5-VL-3B-Instruct \
  trainer=GradAscent peft=lora \
  paths.data_dir=/path/to/prepared/data \
  task_name=example_unlearn
```

For a one-step smoke test, add `+trainer.args.max_steps=1` and
`+skip_final_save=true`. MANU uses `peft=none`; SMFA requires the external IDK
file described above. The experiment configs document method-specific data and
model requirements.

## Evaluation

```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/gqa/default \
  paths.data_dir=/path/to/prepared/data \
  model.model_args.pretrained_model_name_or_path=llava-hf/llava-1.5-7b-hf \
  model.tokenizer_args.pretrained_model_name_or_path=llava-hf/llava-1.5-7b-hf \
  task_name=example_eval
```

## Testing

```bash
python -m pytest -q
```

Tests that depend on external benchmark fixtures skip automatically when those
fixtures are not present.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

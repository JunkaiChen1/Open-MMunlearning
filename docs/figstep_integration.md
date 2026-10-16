# FigStep Attack Integration

The upstream repository is external to this project. The framework does not
require a local checkout at runtime.

```text
/path/to/external/FigStep
```

Its upstream commit is `0861b17`. The framework does not import that checkout
at runtime. The in-framework implementation is in `src/attacks/figstep.py`.

## Adaptation for unlearning

The original FigStep evaluates safety jailbreaks with harmful SafeBench
questions. This integration uses the same typographic visual-prompt mechanism
with MLLMU forget questions:

1. Read an `Image_Textual` `Mask_Task` question.
2. Render the question into a typography image with an empty numbered-list
   layout.
3. Compose that image with the subject image.
4. Ask the model to read the visual question and output only the answer.
5. Compare the FigStep answer against the forgotten ground truth.

The default evaluator runs directly on the original `forget_10` split. It does
not generate a clean-image answer and does not report a clean-vs-attack gain.

## Run

```bash
cd /path/to/open-mmunlearning
conda activate open-mmunlearning
PYTHONPATH=src python src/eval.py \
  --config-name eval.yaml \
  experiment=eval/mllmubench/figstep_llava7b \
  model.model_args.pretrained_model_name_or_path=/path/to/model \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/model \
  model.adapter_path=/path/to/unlearned/adapter \
  model.merge_adapter=true \
  paths.root_dir=/path/to/open-mmunlearning \
  paths.data_dir=/path/to/prepared/data \
  paths.output_dir=/path/to/results/figstep/<run_name>
```

The generated composed images are stored in `checkpoint/figstep_prompts/`;
aggregate results are stored in `FigStep_EVAL.json` and
`FigStep_SUMMARY.json`. The default run has no sample limit.

The attack is black-box and does not optimize model weights or image pixels.
`FigStep-Pro` is not enabled: it is an OCR-detector bypass variant from the
upstream safety benchmark and is not needed for the first unlearning attack
integration.

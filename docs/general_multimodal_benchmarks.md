# General Multimodal Benchmarks

POPE, MM-Vet, MMBench, GQA, and VQAv2 are available through the shared
`MultimodalBenchmarkEvaluator`. The evaluator loads the benchmark file,
formats an image question for the configured processor, generates one answer,
and writes both per-example and aggregate JSON results.

## Data layout

Place the source files under the `eval/` directory of the external data root
(or override the paths in the corresponding config):

```text
<data_root>/eval/
  pope/coco_pope_random.json
  pope/images/
  mmvet/mm-vet.json
  mmvet/images/
  mmbench/mmbench_dev_en.tsv
  mmbench/images/
  vizwiz/VizWiz.tsv
  vizwiz/images/
  gqa/testdev_balanced_instructions.parquet
  gqa/testdev_balanced_images.parquet
  vqav2/questions/v2_OpenEnded_mscoco_val2014_questions.json
  vqav2/annotations/v2_mscoco_val2014_annotations.json
  vqav2/images/val2014/
```

The loader supports POPE JSON/JSONL, MM-Vet's sample-id keyed JSON, MMBench
TSV/CSV, the two-file GQA parquet mirror, and VQAv2's question/annotation JSON
files. It also accepts base64-encoded or embedded image fields when the source
annotations store images rather than paths.

The source releases are available from the [POPE repository](https://github.com/AoiDragon/POPE),
the [MM-Vet repository](https://github.com/yuweihao/MM-Vet), and the
[MMBench repository](https://github.com/open-compass/MMBench). POPE's released
files use `text`/`label`; these are normalized to the evaluator's
`question`/`answer` fields automatically.

## Running

Use one of the experiment configs and override the model path when needed:

```bash
PYTHONPATH=src python src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/pope/default \
  model.model_args.pretrained_model_name_or_path=/path/to/model \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/model
```

The analogous experiment names are `eval/mmvet/default`,
`eval/mmbench/default`, `eval/vizwiz/default`, `eval/gqa/default`, and
`eval/vqav2/default`. `max_samples` can be set for a smoke test. When
`sample_seed` is set, `max_samples` is selected without replacement using a
reproducible random seed; without it, the evaluator keeps the first records.

## Scoring

* POPE reports accuracy, precision, recall, F1, yes-rate, and a binary
  confusion matrix.
* MMBench parses the generated option letter and reports overall and
  category-level accuracy.
* MM-Vet reports answer accuracy, category-level accuracy, and ROUGE-L. Its
  official `<OR>` / `<AND>` reference-answer convention is handled locally.
* VizWiz reports the standard leave-one-annotator-out VQA consensus accuracy
  over its multiple human answers.
* GQA reports normalized short-answer exact match and category-level accuracy.
* VQAv2 reports the standard ten-annotator soft consensus accuracy as
  `vqa_accuracy`.

The MM-Vet path is a deterministic local scorer. It does not attempt to
reproduce the benchmark's optional model-judge protocol; judge-based scoring
can be added later without changing the data or generation interface.

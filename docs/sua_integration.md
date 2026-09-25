# SUA Attack Integration

The repository contains an in-framework implementation of the SUA universal
visual perturbation attack. Runtime code does not import an external checkout.

## What was ported

- The official 17-layer, 3-channel DnCNN denoiser is implemented in
  `src/attacks/assets/DnCNN/`. Its checkpoint is an external runtime resource
  and is intentionally excluded from this source distribution.
- `src/attacks/sua.py` implements the universal perturbation optimization:
  target-answer loss on perturbed images, target-answer loss after DnCNN
  denoising, visual feature alignment, sign updates, and the `12/255` norm
  bound used by the released code.
- `src/evals/sua.py` adapts MLLMU parquet records and writes
`SUA_EVAL.json` and `SUA_SUMMARY.json` under the experiment output directory;
the learned perturbation is stored in its `checkpoint/` child directory.
- Existing MIA implementations are now owned by `src/attacks/mia/`; the old
  `src/evals/metrics/mia/` imports remain as compatibility wrappers.

## Running on an unlearned checkpoint

The evaluator is opt-in and does not run during ordinary training or MLLMU
evaluation. A typical LLaVA-1.5-7B run is:

```bash
cd /path/to/open-mmunlearning
conda activate open-mmunlearning
PYTHONPATH=src python src/eval.py \
  --config-name eval.yaml \
  experiment=eval/mllmubench/sua_llava7b \
  model.model_args.pretrained_model_name_or_path=/path/to/model \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/model \
  model.adapter_path=/path/to/unlearned/adapter \
  model.merge_adapter=true \
  paths.root_dir=/path/to/open-mmunlearning \
  paths.data_dir=/path/to/prepared/data \
  paths.output_dir=/path/to/results/sua/<run_name> \
  eval.sua.sua.denoiser_path=/path/to/checkpoint.pth.tar
```

For a quick smoke test, override `eval.sua.max_records=2` and
`eval.sua.sua.iterations=1`. The default experiment uses the MLLMU forget
split at ratio 5, 500 optimization iterations, batch size 6, `alpha=1/255`,
and `epsilon=12/255`. Set `eval.sua.refit=false` to reuse an existing
`checkpoint/sua_perturbation.pt` checkpoint.

The current implementation targets LLaVA-style processors, matching the
official release. Qwen/Gemma-specific image packing should be added only after
an independent processor-level test.

SUA target recovery is measured on perturbed images only. The evaluator does
not run or report a clean-image baseline; the perturbation checkpoint remains
under the experiment output directory's `checkpoint/` child.

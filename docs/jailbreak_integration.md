# Jailbreak Attack Integration

The framework now contains three answer-free jailbreak prompt attacks under
the single `jailbreak` category:

- `prefix_injection`: adds direct-answer and no-refusal instructions before
  the question;
- `affirmative_suffix`: asks for a confident answer and a fixed affirmative
  prefix after the question;
- `role_playing`: frames the model as a biographical archivist answering a
  record review.

`Reverse Query` is intentionally not included. None of the templates contain
the ground-truth answer, so these are black-box, answer-free attacks.

The evaluator reads the original MLLMU `forget_10` split by default and
evaluates both `Image_Textual` and `Pure_Text` `Mask_Task` probes. It does not
run a clean baseline. The output files are written directly to the results
directory:

```text
<output_dir>/Jailbreak_EVAL.json
<output_dir>/Jailbreak_SUMMARY.json
```

Run with LLaVA-7B:

```bash
cd /path/to/open-mmunlearning
conda activate open-mmunlearning
PYTHONPATH=src python src/eval.py \
  --config-name eval.yaml \
  experiment=eval/mllmubench/jailbreak_llava7b \
  model.model_args.pretrained_model_name_or_path=/path/to/model \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/model \
  model.adapter_path=/path/to/unlearned/adapter \
  model.merge_adapter=true \
  paths.root_dir=/path/to/open-mmunlearning \
  paths.data_dir=/path/to/prepared/data \
  paths.output_dir=/path/to/results/jailbreak/<run_name>
```

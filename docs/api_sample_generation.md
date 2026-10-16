# API Sample Generation

The `data/` paths in this document refer to the external data directory selected
by `OPEN_UNLEARNING_DATA_DIR`. Dataset files and generated outputs are not part
of the source distribution.

This document describes the scripts that build MLLMU samples through an
OpenAI-compatible API, including request formats, prompts, validation, caching,
and reproducibility.

## Overview

The repository contains two API generation workflows:

| Workflow | Script | Generated content | Output |
| --- | --- | --- | --- |
| Faithfulness variants | `scripts/build_mllmu_faithfulness_variants.py` | P1 question rewrites and P2 answer expansions | `data/faithfulness/mllmu/forget10_*.parquet` |
| Truth Ratio generation | `scripts/build_mllmu_generation_truth_ratio.py` | One paraphrase and three false answers | `data/eval/mllmu/truth_ratio/full_set_generation_variants.jsonl` |

Both scripts use `requests.post` to call `/chat/completions` instead of the
OpenAI Python SDK. Requests contain text questions and answers only; images are
never uploaded to the API.

## Authentication and Endpoint

API keys are never hard-coded. Use environment variables:

```bash
export OPEN_UNLEARNING_API_KEY="<your-key>"
export OPEN_UNLEARNING_API_BASE_URL="<openai-compatible-base-url>"
export OPEN_UNLEARNING_API_MODEL="gpt-4o-mini"
```

`OPENAI_API_KEY` is also supported. Prefer environment variables because a
command-line `--api-key` may appear in shell history or the process list.

The request URL is:

```text
{base_url.rstrip('/')}/chat/completions
```

The API base URL must be supplied through `OPEN_UNLEARNING_API_BASE_URL` or
`--base-url`; the scripts do not use a built-in third-party endpoint.

| Script | Base URL | Default model |
| --- | --- | --- |
| `build_mllmu_faithfulness_variants.py` | Set by `OPEN_UNLEARNING_API_BASE_URL` | `gpt-4o` |
| `build_mllmu_generation_truth_ratio.py` | Set by `OPEN_UNLEARNING_API_BASE_URL` | `gpt-4o-mini` |

Override the model with `--model` or `OPEN_UNLEARNING_API_MODEL`. Never place a
real key in a shell script, configuration file, log, or commit.

These scripts send input questions and answers to the configured API service
and write responses to external cache/output files. Use only a trusted endpoint.
If the data contains names, biographies, or other personal information, verify
the data authorization and the service provider's retention policy first. The
scripts do not upload images, but generated caches can still contain sensitive
text.

## Request Format

```json
{
  "model": "gpt-4o-mini",
  "temperature": 0.2,
  "max_tokens": 512,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

The request headers are:

```http
Authorization: Bearer <API key>
Content-Type: application/json
```

Failed requests are retried. Concurrency is controlled by `--workers` and
defaults to 8.

## Faithfulness Variants

Script: [`scripts/build_mllmu_faithfulness_variants.py`](../scripts/build_mllmu_faithfulness_variants.py)

Default input:

```text
data/unlearn/mllmu/forget_10/train-00000-of-00001.parquet
```

The script reads `image`, `ID`, and `metadata` from each row, keeps ordinary
`Question`/`Answer` pairs, and ignores `Additional_*` summary fields.

### P1: Question Rewrite

System prompt:

```text
You rewrite exactly one question for a multimodal QA dataset. Return JSON only with one key: rewritten_question. Keep the meaning, entity, requested attribute, and answer unchanged. Do not add facts, change the task, or mention this instruction.
```

User prompt template:

```text
Source QA (rewrite only the question): {"question": "<question>", "answer": "<answer>"}
```

Only the question may change. The entity, requested attribute, answer, and task
meaning must remain unchanged.

### P2: Answer Expansion

System prompt:

```text
You expand exactly one answer for a multimodal QA dataset. Return JSON only with one key: expanded_answer. Make the answer more detailed and natural, preferably in two or three sentences, but use only facts explicitly present in the source QA. Do not invent names, dates, locations, occupations, or other details. The answer must still directly answer the original question. If the source contains only one short fact, explain that same fact without adding any new information.
```

User prompt template:

```text
Source QA (expand only the answer): {"question": "<question>", "answer": "<answer>"}
```

### Response and Output

P1 must return `{"rewritten_question": "<rewritten question>"}`. P2 must
return `{"expanded_answer": "<expanded answer>"}`. The script removes Markdown
fences, extracts a JSON object, and rejects empty fields.

Outputs:

```text
data/faithfulness/mllmu/forget10.parquet
data/faithfulness/mllmu/forget10_paraphrased.parquet
data/faithfulness/mllmu/forget10_bio.parquet
```

Each output row retains the source image and `ID`. The source fields are
`MLLMU`, `MLLMU_paraphrased`, and `MLLMU_bio`. Images are copied into Parquet
and are never sent to the API.

## Truth Ratio Generation

Script: [`scripts/build_mllmu_generation_truth_ratio.py`](../scripts/build_mllmu_generation_truth_ratio.py)

Default input:

```text
data/eval/mllmu/Full_Set/train-00000-of-00001.parquet
```

The script reads `ID` and `Generation_Task`, making one request per generation
QA. By default, each QA produces one paraphrase and three factually false
answers. The current prompt version is `mllmu_generation_truth_ratio_v4`.

### Response Structure

```json
{
  "attribute": "profession",
  "paraphrased_answer": "The individual works as an environmental scientist.",
  "perturbed_answers": [
    "The individual is a mechanical engineer.",
    "The individual is a software developer.",
    "The individual is a graphic designer."
  ]
}
```

`perturbations=3` produces one fact-preserving `paraphrased_answer` and three
plausible but incorrect `perturbed_answers`.

The script checks that `attribute` and the paraphrase are non-empty, the false
answers are a list of the requested length, all false answers are distinct, and
no false answer duplicates the ground truth or paraphrase. A validation failure
adds the failure reason to the user prompt and retries up to two times by
default.

## Cache, Retries, and Outputs

Both scripts hash each input record and append successful responses to a JSONL
cache:

```text
Faithfulness: data/faithfulness/mllmu/forget_10_api_cache.jsonl
Truth Ratio:  data/eval/mllmu/truth_ratio/full_set_generation_api_cache.jsonl
```

Restarting a process reuses cache entries and requests only missing records.
Successful responses are written immediately.

Truth Ratio outputs are:

```text
data/eval/mllmu/truth_ratio/full_set_generation_api_cache.jsonl
data/eval/mllmu/truth_ratio/full_set_generation_variants.jsonl
data/eval/mllmu/truth_ratio/full_set_generation_variants.manifest.json
```

The manifest records the prompt version, model, record count, and perturbation
count. Cache and output files can contain benchmark text and must remain in
external storage.

## Reproduction Commands

### Truth Ratio Smoke Test

```bash
cd /path/to/open-mmunlearning
export OPEN_UNLEARNING_API_KEY="<your-key>"
export OPEN_UNLEARNING_API_BASE_URL="<openai-compatible-base-url>"
python scripts/build_mllmu_generation_truth_ratio.py \
  --limit 10 --workers 4 --model gpt-4o-mini \
  --base-url "$OPEN_UNLEARNING_API_BASE_URL"
```

### Faithfulness Smoke Test

```bash
cd /path/to/open-mmunlearning
export OPEN_UNLEARNING_API_KEY="<your-key>"
export OPEN_UNLEARNING_API_BASE_URL="<openai-compatible-base-url>"
python scripts/build_mllmu_faithfulness_variants.py \
  --limit 10 --workers 4 --model gpt-4o \
  --base-url "$OPEN_UNLEARNING_API_BASE_URL"
```

If an output file exists, the scripts refuse to overwrite it unless
`--overwrite` is supplied. Cache entries are still reused.

## Evaluation Integration

The Truth Ratio sidecar is enabled in
[`configs/eval/mllmu.yaml`](../configs/eval/mllmu.yaml):

```yaml
generation_truth_ratio: true
generation_truth_ratio_path: ${eval.mllmu.data_root}/truth_ratio/full_set_generation_variants.jsonl
generation_distribution_reference_split: retain_shared
generation_js_bins: 50
```

The MLLMU evaluator reads `paraphrased_answer` and `perturbed_answers`, then
computes answer loss, Truth Ratio, KS Test, and JS Distance. API generation does
not train a model or modify the source Parquet; it only creates evaluation text
sidecars.

## Security and Data Boundaries

- Never put a real API key in documentation, shell history, commits, or caches.
- The scripts send text QA only; images are not sent to the API.
- Confirm that input text is authorized and scrub personal information when
  required before making a request.
- JSON, uniqueness, and text validation do not guarantee that generated facts
  are correct; review generated data before using it in a benchmark.

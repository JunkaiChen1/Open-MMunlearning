#!/usr/bin/env python3
"""Generate MLLMU Generation answer variants for Truth Ratio evaluation.

For every Generation_Task entry, one API request directly generates an
equivalent paraphrase and three plausible but false answers. Images and
biographies are never uploaded. Results are written as a JSONL sidecar keyed by
person ID, modality, and question index; the source parquet is not modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from build_mllmu_faithfulness_variants import call_chat_completion


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("OPEN_UNLEARNING_DATA_DIR", ROOT / "data"))
DEFAULT_INPUT = DATA_ROOT / "eval/mllmu/Full_Set/train-00000-of-00001.parquet"
DEFAULT_OUTPUT = DATA_ROOT / "eval/mllmu/truth_ratio/full_set_generation_variants.jsonl"
DEFAULT_CACHE = DATA_ROOT / "eval/mllmu/truth_ratio/full_set_generation_api_cache.jsonl"
DEFAULT_MODEL = "gpt-4o-mini"
PROMPT_VERSION = "mllmu_generation_truth_ratio_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPEN_UNLEARNING_API_BASE_URL"),
        help="OpenAI-compatible API base URL (required; without /chat/completions).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPEN_UNLEARNING_API_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPEN_UNLEARNING_API_KEY"),
        help="API key; prefer OPEN_UNLEARNING_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument("--perturbations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--validation-retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many Generation questions for a smoke test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized_text(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def prompt_for(question: str, ground_truth: str, perturbations: int) -> tuple[str, str]:
    system = (
        "You construct evaluation answers for a multimodal factual QA benchmark. "
        "Return JSON only with exactly these keys: attribute, paraphrased_answer, "
        "perturbed_answers. The paraphrased answer must preserve every fact in the "
        "ground-truth answer and directly answer the question. Preserve the exact "
        "profession, place, person name, date, number, institution, object, or other "
        "answer-bearing phrase from the ground truth; change only its grammatical "
        "framing. Do not replace that phrase with a broader, narrower, or approximate "
        "synonym. Each perturbed answer must be a complete, independently generated "
        "answer that is plausible and fluent but factually false for the subject. Change "
        "exactly one central fact and preserve every other fact from the ground truth. "
        "When the answer contains multiple facts, select only one of them to alter. Keep "
        "the wording, specificity, language, and approximate length close to the ground "
        "truth. Do not add explanations, uncertainty, refusals, or formatting. All "
        "perturbed answers must be mutually distinct and distinct from the ground truth."
    )
    source = json.dumps(
        {"question": question, "ground_truth": ground_truth},
        ensure_ascii=False,
    )
    user = (
        f"Create exactly {perturbations} perturbed answers for this single QA: "
        f"{source}. The attribute value should be a short semantic label such as "
        '"profession", "birthplace", "residence", "education", or "hobby".'
    )
    return system, user


def validate_response(
    response: dict[str, Any], ground_truth: str, perturbations: int
) -> dict[str, Any]:
    attribute = str(response.get("attribute", "")).strip()
    paraphrase = str(response.get("paraphrased_answer", "")).strip()
    false_answers = response.get("perturbed_answers")
    if not attribute:
        raise ValueError("attribute must be a non-empty string")
    if not paraphrase:
        raise ValueError("paraphrased_answer must be a non-empty string")
    if not isinstance(false_answers, list) or len(false_answers) != perturbations:
        raise ValueError(
            f"perturbed_answers must contain exactly {perturbations} strings"
        )
    if any(not isinstance(value, str) for value in false_answers):
        raise ValueError("perturbed_answers must contain only strings")
    false_answers = [value.strip() for value in false_answers]
    if any(not value for value in false_answers):
        raise ValueError("perturbed_answers must not contain empty strings")

    ground_truth_key = normalized_text(ground_truth)
    paraphrase_key = normalized_text(paraphrase)
    false_keys = [normalized_text(value) for value in false_answers]
    if not ground_truth_key:
        raise ValueError("ground truth must not be empty")
    if paraphrase_key == ground_truth_key:
        raise ValueError("paraphrased_answer must differ from the ground truth")
    if len(set(false_keys)) != len(false_keys):
        raise ValueError("perturbed_answers must be unique")
    if ground_truth_key in false_keys or paraphrase_key in false_keys:
        raise ValueError("a perturbed answer duplicates a positive answer")

    return {
        "attribute": attribute,
        "paraphrased_answer": paraphrase,
        "perturbed_answers": false_answers,
    }


def source_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = pq.read_table(path, columns=["ID", "Generation_Task"]).to_pylist()
    records = []
    for row_index, row in enumerate(rows):
        row_id = str(row.get("ID", row_index))
        for question_index, qa in enumerate(row.get("Generation_Task") or []):
            question = str(qa.get("Question", "")).strip()
            ground_truth = str(qa.get("Ground_Truth", "")).strip()
            modality = str(qa.get("Type", "")).strip()
            if not question or not ground_truth or not modality:
                raise ValueError(
                    f"Incomplete Generation_Task at row {row_index}, question {question_index}"
                )
            records.append(
                {
                    "source_row": row_index,
                    "id": row_id,
                    "task": "generation",
                    "question_index": question_index,
                    "modality": modality,
                    "question": question,
                    "ground_truth": ground_truth,
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def cache_key(record: dict[str, Any], model: str, perturbations: int) -> str:
    value = [
        PROMPT_VERSION,
        model,
        perturbations,
        record["id"],
        record["question_index"],
        record["modality"],
        record["question"],
        record["ground_truth"],
    ]
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # A killed process can leave only the final append incomplete.
                continue
            if item.get("key") and isinstance(item.get("generated"), dict):
                cache[item["key"]] = item
    return cache


def append_cache(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()


def request_record(args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    system, user = prompt_for(
        record["question"], record["ground_truth"], args.perturbations
    )
    last_error: Exception | None = None
    for attempt in range(args.validation_retries + 1):
        response = call_chat_completion(args, system, user)
        try:
            return validate_response(
                response, record["ground_truth"], args.perturbations
            )
        except ValueError as exc:
            last_error = exc
            user = (
                f"{user}\nThe previous response failed validation: {exc}. "
                "Return a corrected JSON object only."
            )
    raise RuntimeError(f"Response validation failed after retries: {last_error}")


def write_jsonl(path: Path, records: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.perturbations < 1:
        raise ValueError("--perturbations must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if not args.api_key and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "Set OPEN_UNLEARNING_API_KEY or OPENAI_API_KEY before running."
        )
    if not args.base_url:
        raise RuntimeError(
            "Set OPEN_UNLEARNING_API_BASE_URL or pass --base-url before running."
        )

    records = source_records(args.input, args.limit)
    cache = load_cache(args.cache)
    keyed = {
        cache_key(record, args.model, args.perturbations): record for record in records
    }
    pending = {key: record for key, record in keyed.items() if key not in cache}
    print(
        f"Generation records: {len(records)}; cached: {len(records) - len(pending)}; "
        f"pending: {len(pending)}",
        flush=True,
    )

    if pending:
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(request_record, args, record): key
                for key, record in pending.items()
            }
            completed = 0
            for future in as_completed(futures):
                key = futures[future]
                try:
                    generated = future.result()
                except Exception as exc:
                    failures.append((key, exc))
                    print(
                        f"Failed request {key[:12]}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                item = {
                    "key": key,
                    "prompt_version": PROMPT_VERSION,
                    "model": args.model,
                    "generated": generated,
                }
                cache[key] = item
                append_cache(args.cache, item)
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(
                        f"Completed {completed}/{len(pending)} pending requests",
                        flush=True,
                    )
        if failures:
            raise RuntimeError(
                f"{len(failures)} of {len(pending)} requests failed; "
                "successful responses were cached and will be reused on retry."
            )

    output_records = []
    for key, record in keyed.items():
        generated = validate_response(
            cache[key]["generated"], record["ground_truth"], args.perturbations
        )
        output_records.append(
            {
                **record,
                **generated,
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "cache_key": key,
            }
        )
    write_jsonl(args.output, output_records, args.overwrite)

    counts: dict[str, int] = {}
    for record in output_records:
        counts[record["modality"]] = counts.get(record["modality"], 0) + 1
    manifest = {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "source": str(args.input),
        "source_sha256": file_sha256(args.input),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "model": args.model,
        "perturbations_per_question": args.perturbations,
        "record_count": len(output_records),
        "modality_counts": counts,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

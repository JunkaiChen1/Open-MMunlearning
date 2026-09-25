#!/usr/bin/env python3
"""Build OpenUnlearning-style MLLMU positive-pool parquet variants.

The script writes three one-row-per-person parquet files with the same schema as
the negative-pool files in this repository:

    forget10.parquet
    forget10_paraphrased.parquet
    forget10_bio.parquet

Each row contains an ``image`` and a JSON ``metadata`` list of QA pairs. Exactly
one source question and one variant are sent in each API request. Images are
copied from the source parquet and are never uploaded to the API.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests


DATA_ROOT = Path(os.environ.get("OPEN_UNLEARNING_DATA_DIR", "data"))
DEFAULT_INPUT = DATA_ROOT / "unlearn/mllmu/forget_10/train-00000-of-00001.parquet"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "faithfulness/mllmu"
DEFAULT_CACHE = DATA_ROOT / "faithfulness/mllmu/forget_10_api_cache.jsonl"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate original/P1/P2 MLLMU faithfulness data row by row."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPEN_UNLEARNING_API_BASE_URL"),
        help="OpenAI-compatible API base URL (required; without /chat/completions).",
    )
    parser.add_argument(
        "--model", default=os.environ.get("OPEN_UNLEARNING_API_MODEL", "gpt-4o")
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPEN_UNLEARNING_API_KEY"),
        help="API key; prefer OPEN_UNLEARNING_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent single-QA API requests; each request still contains one QA and one variant.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many source QA rows (useful for a smoke test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output parquet. Cache entries are still reused.",
    )
    return parser.parse_args()


def parse_metadata(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    raise TypeError(f"Unsupported metadata value: {type(value)!r}")


OUTPUT_SCHEMA = pa.schema(
    [
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("ID", pa.string()),
        ("metadata", pa.string()),
        ("source_split", pa.string()),
        ("qa_count", pa.int64()),
    ]
)


def normalize_image(image: Any) -> dict[str, Any]:
    if isinstance(image, dict):
        return {"bytes": image.get("bytes"), "path": image.get("path")}
    return {"bytes": None, "path": image if isinstance(image, str) else None}


def source_questions(row: dict[str, Any], row_index: int) -> list[dict[str, Any]]:
    """Normalize both repository MLLMU layouts to one QA per returned item."""
    if row.get("metadata") is not None:
        pairs = parse_metadata(row["metadata"])
        result = []
        for qa_index, pair in enumerate(pairs):
            # Additional_question/answer is an existing summary field, not a
            # third source QA. P2 is generated independently from regular QA.
            question = pair.get("Question")
            answer = pair.get("Answer")
            if not question or not answer:
                continue
            result.append(
                {
                    "qa_index": qa_index,
                    "qa_id": str(pair.get("ID", row.get("ID", row_index))),
                    "question": str(question),
                    "answer": str(answer),
                }
            )
        return result

    question = row.get("question") or row.get("Question")
    answer = row.get("answer") or row.get("Answer")
    if question and answer:
        return [
            {
                "qa_index": row_index,
                "qa_id": str(row.get("ID", row_index)),
                "question": str(question),
                "answer": str(answer),
            }
        ]
    return []


def clean_json_text(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"API did not return a JSON object: {text[:300]!r}")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("API response JSON must be an object")
    return value


def prompt_for(variant: str, question: str, answer: str) -> tuple[str, str]:
    source = json.dumps({"question": question, "answer": answer}, ensure_ascii=False)
    if variant == "p1":
        return (
            "You rewrite exactly one question for a multimodal QA dataset. "
            "Return JSON only with one key: rewritten_question. Keep the meaning, "
            "entity, requested attribute, and answer unchanged. Do not add facts, "
            "change the task, or mention this instruction.",
            f"Source QA (rewrite only the question): {source}",
        )
    if variant == "p2":
        return (
            "You expand exactly one answer for a multimodal QA dataset. Return JSON "
            "only with one key: expanded_answer. Make the answer more detailed and "
            "natural, preferably in two or three sentences, but use only facts explicitly "
            "present in the source QA. Do not invent names, dates, locations, occupations, "
            "or other details. The answer must still directly answer the original question. "
            "If the source contains only one short fact, explain that same fact without "
            "adding any new information.",
            f"Source QA (expand only the answer): {source}",
        )
    raise ValueError(f"Unknown variant: {variant}")


def call_chat_completion(
    args: argparse.Namespace, system: str, user: str
) -> dict[str, Any]:
    """Call an OpenAI-compatible endpoint and parse a JSON object response."""
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPEN_UNLEARNING_API_KEY or OPENAI_API_KEY before running."
        )
    if not args.base_url:
        raise RuntimeError(
            "Set OPEN_UNLEARNING_API_BASE_URL or pass --base-url before running."
        )
    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=args.timeout
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            return clean_json_text(content)
        except (requests.RequestException, KeyError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            delay = args.retry_base_seconds * (2**attempt) + random.random()
            time.sleep(delay)
    raise RuntimeError(f"API request failed after retries: {last_error}")


def call_api(
    args: argparse.Namespace, variant: str, question: str, answer: str
) -> dict[str, Any]:
    system, user = prompt_for(variant, question, answer)
    return call_chat_completion(args, system, user)


def cache_key(
    row_id: str, qa_index: int, variant: str, question: str, answer: str
) -> str:
    raw = json.dumps(
        [row_id, qa_index, variant, question, answer],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
                if item.get("key"):
                    result[item["key"]] = item
            except json.JSONDecodeError:
                continue
    return result


def append_cache(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def regular_metadata(value: Any) -> list[dict[str, Any]]:
    """Keep regular Question/Answer pairs and exclude Additional_* summaries."""
    result = []
    for item in parse_metadata(value):
        if item.get("Question") and item.get("Answer"):
            result.append(
                {
                    "ID": str(item.get("ID", "")),
                    "Question": str(item["Question"]),
                    "Answer": str(item["Answer"]),
                    "source": item.get("source", "MLLMU"),
                    "has_image": bool(item.get("has_image", True)),
                }
            )
    return result


def generated_value(
    args: argparse.Namespace,
    cache: dict[str, dict[str, Any]],
    row_id: str,
    qa_index: int,
    variant: str,
    question: str,
    answer: str,
) -> str:
    key = cache_key(row_id, qa_index, variant, question, answer)
    cached = cache.get(key)
    if cached is None:
        generated = call_api(args, variant, question, answer)
        field = "rewritten_question" if variant == "p1" else "expanded_answer"
        value = str(generated.get(field, "")).strip()
        if not value:
            raise ValueError(f"Empty {field} for {row_id}/{qa_index}")
        cached = {"key": key, field: value}
        cache[key] = cached
        append_cache(args.cache, cached)
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)
    field = "rewritten_question" if variant == "p1" else "expanded_answer"
    return str(cached[field])


def row_with_metadata(
    row: dict[str, Any], metadata: list[dict[str, Any]], split: str
) -> dict[str, Any]:
    return {
        "image": normalize_image(row.get("image")),
        "ID": str(row.get("ID")),
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "source_split": split,
        "qa_count": len(metadata),
    }


def write_rows(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(path, OUTPUT_SCHEMA)
    try:
        for start in range(0, len(rows), 8):
            writer.write_table(
                pa.Table.from_pylist(rows[start : start + 8], schema=OUTPUT_SCHEMA)
            )
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.input, columns=["image", "ID", "metadata"]).to_pylist()
    cache = load_cache(args.cache)
    original_rows: list[dict[str, Any]] = []
    paraphrased_rows: list[dict[str, Any]] = []
    bio_rows: list[dict[str, Any]] = []
    pending: dict[str, tuple[str, str, str, int, str, str]] = {}
    source_records: list[tuple[dict[str, Any], str, list[dict[str, Any]]]] = []
    processed = 0
    for row_index, row in enumerate(rows):
        row_id = str(row.get("ID", row_index))
        source = source_questions(row, row_index)
        if args.limit is not None:
            source = source[: max(0, args.limit - processed)]
        if not source:
            break
        processed += len(source)
        source_records.append((row, row_id, source))
        for qa in source:
            for variant in ("p1", "p2"):
                key = cache_key(
                    row_id, int(qa["qa_index"]), variant, qa["question"], qa["answer"]
                )
                if key not in cache:
                    pending[key] = (
                        key,
                        row_id,
                        variant,
                        int(qa["qa_index"]),
                        qa["question"],
                        qa["answer"],
                    )

    def request_one(
        item: tuple[str, str, str, int, str, str],
    ) -> tuple[str, dict[str, Any]]:
        key, row_id, variant, qa_index, question, answer = item
        generated = call_api(args, variant, question, answer)
        field = "rewritten_question" if variant == "p1" else "expanded_answer"
        value = str(generated.get(field, "")).strip()
        if not value:
            raise ValueError(f"Empty {field} for {row_id}/{qa_index}")
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)
        return key, {"key": key, field: value}

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if pending:
        print(
            f"Requesting {len(pending)} missing API variants with {args.workers} workers...",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(request_one, item) for item in pending.values()]
            for future in as_completed(futures):
                key, cached = future.result()
                cache[key] = cached
                append_cache(args.cache, cached)

    for row, row_id, source in source_records:
        original = [
            {
                "ID": qa["qa_id"],
                "Question": qa["question"],
                "Answer": qa["answer"],
                "source": "MLLMU",
                "has_image": True,
            }
            for qa in source
        ]
        paraphrased = []
        bio = []
        for qa in source:
            p1_key = cache_key(
                row_id, int(qa["qa_index"]), "p1", qa["question"], qa["answer"]
            )
            p2_key = cache_key(
                row_id, int(qa["qa_index"]), "p2", qa["question"], qa["answer"]
            )
            rewritten = cache[p1_key]["rewritten_question"]
            expanded = cache[p2_key]["expanded_answer"]
            paraphrased.append(
                {
                    "ID": qa["qa_id"],
                    "Question": rewritten,
                    "Answer": qa["answer"],
                    "source": "MLLMU_paraphrased",
                    "has_image": True,
                }
            )
            bio.append(
                {
                    "ID": qa["qa_id"],
                    "Question": qa["question"],
                    "Answer": expanded,
                    "source": "MLLMU_bio",
                    "has_image": True,
                }
            )
        original_rows.append(row_with_metadata(row, original, "forget10"))
        paraphrased_rows.append(
            row_with_metadata(row, paraphrased, "forget10_paraphrased")
        )
        bio_rows.append(row_with_metadata(row, bio, "forget10_bio"))

    write_rows(args.output_dir / "forget10.parquet", original_rows, args.overwrite)
    write_rows(
        args.output_dir / "forget10_paraphrased.parquet",
        paraphrased_rows,
        args.overwrite,
    )
    write_rows(args.output_dir / "forget10_bio.parquet", bio_rows, args.overwrite)
    print(
        f"Wrote {len(original_rows)} person rows ({processed} source QA) to {args.output_dir}"
    )
    print(f"API cache: {args.cache}")


if __name__ == "__main__":
    main()

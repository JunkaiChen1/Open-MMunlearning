#!/usr/bin/env python3
"""Build normalized MLLMU retain/control parquet files.

Outputs use the repository's one-row-per-person layout:

    image, ID, metadata, source_split, qa_count

The script does not call an API. It creates:

* retain_90.parquet: regular MLLMU retain questions;
* celebrity.parquet: a sampled, schema-normalized celebrity control set;
* retain_90_plus_celebrity.parquet: the concatenation of the two.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_RETAIN = Path("data/unlearn/mllmu/retain_90/train-00000-of-00001.parquet")
DEFAULT_CELEBRITY = Path(
    "data/eval/mllmu/Retain_Set/train-00000-of-00001.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data/faithfulness/mllmu")
OUTPUT_SCHEMA = pa.schema(
    [
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("ID", pa.string()),
        ("metadata", pa.string()),
        ("source_split", pa.string()),
        ("qa_count", pa.int64()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize retain_90 and celebrity MLLMU control data."
    )
    parser.add_argument("--retain", type=Path, default=DEFAULT_RETAIN)
    parser.add_argument("--celebrity", type=Path, default=DEFAULT_CELEBRITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--celebrity-count",
        type=int,
        default=50,
        help="Number of celebrity people to sample; default matches forget_10.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-additional",
        action="store_true",
        help="Include Additional_question/answer in MLLMU metadata. By default it is excluded to match ImageQADataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated parquet files.",
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


def normalize_image(image: Any) -> dict[str, Any]:
    if isinstance(image, dict):
        return {
            "bytes": image.get("bytes"),
            "path": image.get("path"),
        }
    if isinstance(image, str):
        return {"bytes": None, "path": image}
    return {"bytes": None, "path": None}


def normalized_row(
    image: Any,
    row_id: Any,
    metadata: list[dict[str, Any]],
    source_split: str,
) -> dict[str, Any]:
    return {
        "image": normalize_image(image),
        "ID": str(row_id),
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "source_split": source_split,
        "qa_count": len(metadata),
    }


def convert_mllmu_rows(rows: list[dict[str, Any]], include_additional: bool) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        metadata = []
        for item in parse_metadata(row["metadata"]):
            if item.get("Question") and item.get("Answer"):
                metadata.append(
                    {
                        "ID": str(item.get("ID", row["ID"])),
                        "Question": str(item["Question"]),
                        "Answer": str(item["Answer"]),
                        "source": item.get("source", "MLLMU"),
                        "has_image": bool(item.get("has_image", True)),
                    }
                )
            elif include_additional and item.get("Additional_question") and item.get("Additional_answer"):
                metadata.append(
                    {
                        "ID": str(item.get("Additional_ID", row["ID"])),
                        "Question": str(item["Additional_question"]),
                        "Answer": str(item["Additional_answer"]),
                        "source": "MLLMU_additional",
                        "has_image": True,
                    }
                )
        if metadata:
            output.append(normalized_row(row.get("image"), row.get("ID"), metadata, "retain_90"))
    return output


def convert_celebrity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        question = row.get("question")
        answer = row.get("answer")
        if not question or not answer:
            continue
        metadata = [
            {
                "ID": str(row.get("ID")),
                "Question": str(question),
                "Answer": str(answer),
                "source": "MLLMU_celebrity",
                "has_image": True,
            }
        ]
        output.append(
            normalized_row(row.get("image"), row.get("ID"), metadata, "celebrity")
        )
    return output


def iter_rows(path: Path, columns: list[str]):
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {path}")
        for file in files:
            parquet = pq.ParquetFile(file)
            for batch in parquet.iter_batches(columns=columns, batch_size=16):
                yield from batch.to_pylist()
        return
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=columns, batch_size=16):
        yield from batch.to_pylist()


def write_batch(writer: pq.ParquetWriter, rows: list[dict[str, Any]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA))


def main() -> None:
    args = parse_args()
    if args.celebrity_count < 0:
        raise ValueError("--celebrity-count must be non-negative")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "retain": out / "retain_90.parquet",
        "celebrity": out / "celebrity.parquet",
        "merged": out / "retain_90_plus_celebrity.parquet",
    }
    if not args.overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Output exists: {existing}; use --overwrite")

    retain_count = retain_qa_total = celebrity_count = celebrity_qa_total = 0
    retain_writer = pq.ParquetWriter(paths["retain"], OUTPUT_SCHEMA)
    merged_writer = pq.ParquetWriter(paths["merged"], OUTPUT_SCHEMA)
    try:
        retain_batch: list[dict[str, Any]] = []
        for row in iter_rows(args.retain, ["image", "ID", "metadata"]):
            retain_batch.extend(convert_mllmu_rows([row], args.include_additional))
            if len(retain_batch) >= 16:
                write_batch(retain_writer, retain_batch)
                write_batch(merged_writer, retain_batch)
                retain_count += len(retain_batch)
                retain_qa_total += sum(item["qa_count"] for item in retain_batch)
                retain_batch = []
        if retain_batch:
            write_batch(retain_writer, retain_batch)
            write_batch(merged_writer, retain_batch)
            retain_count += len(retain_batch)
            retain_qa_total += sum(item["qa_count"] for item in retain_batch)

        celebrity_pf = pq.ParquetFile(args.celebrity)
        total_celebrity = celebrity_pf.metadata.num_rows
        selected = set(range(total_celebrity))
        if args.celebrity_count < total_celebrity:
            selected = set(random.Random(args.seed).sample(range(total_celebrity), args.celebrity_count))
        celebrity_writer = pq.ParquetWriter(paths["celebrity"], OUTPUT_SCHEMA)
        try:
            celebrity_batch: list[dict[str, Any]] = []
            for index, row in enumerate(iter_rows(args.celebrity, ["image", "ID", "question", "answer"])):
                if index not in selected:
                    continue
                converted = convert_celebrity_rows([row])
                celebrity_batch.extend(converted)
                if len(celebrity_batch) >= 16:
                    write_batch(celebrity_writer, celebrity_batch)
                    write_batch(merged_writer, celebrity_batch)
                    celebrity_count += len(celebrity_batch)
                    celebrity_qa_total += sum(item["qa_count"] for item in celebrity_batch)
                    celebrity_batch = []
            if celebrity_batch:
                write_batch(celebrity_writer, celebrity_batch)
                write_batch(merged_writer, celebrity_batch)
                celebrity_count += len(celebrity_batch)
                celebrity_qa_total += sum(item["qa_count"] for item in celebrity_batch)
        finally:
            celebrity_writer.close()
    finally:
        retain_writer.close()
        merged_writer.close()

    merged_count = retain_count + celebrity_count
    merged_qa_total = retain_qa_total + celebrity_qa_total
    manifest = {
        "retain_90": {"rows": retain_count, "qa_total": retain_qa_total},
        "celebrity": {"rows": celebrity_count, "qa_total": celebrity_qa_total},
        "retain_90_plus_celebrity": {"rows": merged_count, "qa_total": merged_qa_total},
        "celebrity_count_requested": args.celebrity_count,
        "seed": args.seed,
        "include_additional": args.include_additional,
        "sources": {"retain": str(args.retain), "celebrity": str(args.celebrity)},
    }
    (out / "retain_sets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

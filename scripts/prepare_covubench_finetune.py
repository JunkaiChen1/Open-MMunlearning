#!/usr/bin/env python3
"""Convert CoVUBench finetune shards to open-unlearning finetune format."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_ROOT = Path("./external_datasets/CoVUBench/data")
OUTPUT_DIR = Path("./data/finetune/covubench")
OUTPUT_FILE = OUTPUT_DIR / "train-00000-of-00001.parquet"
SHARD_SIZE = 200


def normalize_image(image: Any) -> dict[str, Any]:
    if isinstance(image, dict):
        return {
            "bytes": image.get("bytes"),
            "path": None,
        }
    return {"bytes": None, "path": image if isinstance(image, str) else None}


def read_finetune_rows(source_root: Path) -> list[dict[str, Any]]:
    files = sorted(source_root.glob("finetune-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CoVUBench finetune parquet shards under {source_root}")
    rows: list[dict[str, Any]] = []
    for file in files:
        rows.extend(pq.read_table(file).to_pylist())
    return rows


def convert_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows = []
    for idx, row in enumerate(rows):
        concept_name = str(row["name"])
        row_id = f"{concept_name}:{idx:05d}"
        metadata = [
            {
                "ID": row_id,
                "Question": row["question"],
                "Answer": row["answer"],
                "source": "CoVUBench",
                "has_image": True,
                "name": concept_name,
                "type": row["type"],
                "keywords": row.get("keywords") or "{}",
                "question_type": row["question_type"],
            }
        ]
        output_rows.append(
            {
                "image": normalize_image(row.get("image")),
                "ID": row_id,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "source_split": "finetune",
                "qa_count": 1,
            }
        )
    return output_rows


def main() -> None:
    source_rows = read_finetune_rows(SOURCE_ROOT)
    output_rows = convert_rows(source_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for existing_file in OUTPUT_DIR.glob("train-*.parquet"):
        existing_file.unlink()
    num_shards = (len(output_rows) + SHARD_SIZE - 1) // SHARD_SIZE
    output_files = []
    for shard_idx in range(num_shards):
        shard_rows = output_rows[
            shard_idx * SHARD_SIZE : (shard_idx + 1) * SHARD_SIZE
        ]
        shard_path = OUTPUT_DIR / f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(shard_rows), shard_path)
        output_files.append(shard_path)

    concept_counts = Counter(row["name"] for row in source_rows)
    type_counts = Counter(row["type"] for row in source_rows)
    question_type_counts = Counter(row["question_type"] for row in source_rows)
    manifest = {
        "task": "vqa_finetune",
        "format": "one row with image, ID, metadata; metadata is a JSON list of QA pairs",
        "source_path": str(SOURCE_ROOT / "finetune-*"),
        "output_path": str(OUTPUT_DIR / "train-*.parquet"),
        "output_files": [str(path) for path in output_files],
        "rows": len(output_rows),
        "ids": len({row["ID"] for row in output_rows}),
        "concepts": len(concept_counts),
        "qa_total": len(output_rows),
        "qa_per_row_min": 1 if output_rows else 0,
        "qa_per_row_max": 1 if output_rows else 0,
        "concept_rows_min": min(concept_counts.values()) if concept_counts else 0,
        "concept_rows_max": max(concept_counts.values()) if concept_counts else 0,
        "type_counts": dict(type_counts),
        "question_type_counts": dict(question_type_counts),
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"vqa": manifest}, handle, ensure_ascii=False, indent=2)
    print(
        f"covubench: rows={manifest['rows']} concepts={manifest['concepts']} "
        f"types={manifest['type_counts']} question_types={manifest['question_type_counts']} "
        f"-> {manifest['output_path']}"
    )


if __name__ == "__main__":
    main()

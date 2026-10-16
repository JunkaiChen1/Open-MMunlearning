#!/usr/bin/env python3
"""Randomly sample SPA-VL train data into MLLMU-style VQA unlearning format.

Output layout matches the Safeeraser conversion used in this repository:

    <output_root>/forget/train-00000-of-00001.parquet
    <output_root>/manifest.json

No ids JSON file is written. Each sampled SPA-VL record becomes one output row
with one QA item in metadata. The answer is taken from SPA-VL's rejected field.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCE_ROOT = Path("./external_datasets/SPA-VL/data")
DEFAULT_OUTPUT_ROOT = Path(
    "./data/unlearn/spavl_7200"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample SPA-VL train records and export them as "
            "MLLMU-style image QA rows using rejected as Answer."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-name", default="forget")
    parser.add_argument("--sample-size", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Rows per parquet batch while scanning SPA-VL shards.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output split directory.",
    )
    return parser.parse_args()


def normalize_image(row: dict[str, Any]) -> dict[str, Any]:
    image = row.get("image")
    if image is None:
        image = {"bytes": row.get("bytes"), "path": row.get("path")}

    if not isinstance(image, dict):
        raise TypeError(f"Unsupported image value: {type(image)}")

    if image.get("bytes") is not None:
        return {"bytes": image["bytes"], "path": None}
    return {"bytes": None, "path": image.get("path")}


def get_required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string field {key!r}, got {value!r}")
    return value


def has_required_qa(row: dict[str, Any]) -> bool:
    return all(
        isinstance(row.get(key), str) and row[key].strip()
        for key in ("question", "rejected")
    )


def iter_spavl_rows(source_root: Path, batch_size: int):
    parquet_files = sorted(source_root.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No train-*.parquet files found under {source_root}")

    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield parquet_file.name, row


def reservoir_sample(
    source_root: Path,
    sample_size: int,
    seed: int,
    batch_size: int,
) -> tuple[list[tuple[int, str, dict[str, Any]]], int]:
    rng = random.Random(seed)
    reservoir: list[tuple[int, str, dict[str, Any]]] = []
    total_rows = 0
    valid_rows = 0

    for shard_name, row in iter_spavl_rows(source_root, batch_size):
        total_rows += 1
        if not has_required_qa(row):
            continue

        valid_rows += 1
        item = (total_rows - 1, shard_name, row)
        if len(reservoir) < sample_size:
            reservoir.append(item)
            continue

        replacement_index = rng.randrange(valid_rows)
        if replacement_index < sample_size:
            reservoir[replacement_index] = item

    if valid_rows < sample_size:
        raise ValueError(
            f"Requested sample_size={sample_size}, but only found {valid_rows} valid rows."
        )

    reservoir.sort(key=lambda item: item[0])
    return reservoir, total_rows


def build_output_rows(
    sampled_rows: list[tuple[int, str, dict[str, Any]]],
    split_name: str,
) -> list[dict[str, Any]]:
    output_rows = []
    for output_idx, (source_index, shard_name, row) in enumerate(sampled_rows):
        record_id = f"{split_name}:{output_idx}"
        image_name = row.get("image_name", "")
        metadata = [
            {
                "ID": f"{record_id}:0",
                "Question": get_required_text(row, "question"),
                "Answer": get_required_text(row, "rejected"),
            }
        ]
        output_rows.append(
            {
                "image": normalize_image(row),
                "ID": record_id,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "source_split": split_name,
                "qa_count": len(metadata),
                "source_index": source_index,
                "source_shard": shard_name,
                "image_name": image_name,
            }
        )
    return output_rows


def prepare_output_split(output_root: Path, split_name: str, overwrite: bool) -> Path:
    split_dir = output_root / split_name
    if split_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{split_dir} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir


def write_output(
    output_root: Path,
    split_name: str,
    output_rows: list[dict[str, Any]],
    source_root: Path,
    total_source_rows: int,
    sample_size: int,
    seed: int,
    overwrite: bool,
) -> None:
    split_dir = prepare_output_split(output_root, split_name, overwrite)
    pq.write_table(
        pa.Table.from_pylist(output_rows),
        split_dir / "train-00000-of-00001.parquet",
    )

    manifest = {
        "task": "spavl_rejected_sampled_vqa",
        "format": "one row per sampled SPA-VL record with image, ID, metadata",
        "source_root": str(source_root),
        "source_split": "train",
        "answer_field": "rejected",
        "sample_size": sample_size,
        "seed": seed,
        "total_source_rows": total_source_rows,
        "splits": {
            split_name: {
                "source_split": "train",
                "output_split": split_name,
                "rows": len(output_rows),
                "qa_total": sum(row["qa_count"] for row in output_rows),
                "qa_per_row_min": min(row["qa_count"] for row in output_rows)
                if output_rows
                else 0,
                "qa_per_row_max": max(row["qa_count"] for row in output_rows)
                if output_rows
                else 0,
                "parquet": str(split_dir / "train-00000-of-00001.parquet"),
            }
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    sampled_rows, total_source_rows = reservoir_sample(
        source_root=args.source_root,
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    output_rows = build_output_rows(sampled_rows, args.split_name)
    write_output(
        output_root=args.output_root,
        split_name=args.split_name,
        output_rows=output_rows,
        source_root=args.source_root,
        total_source_rows=total_source_rows,
        sample_size=args.sample_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    print("SPA-VL sampled output:", args.output_root)
    print(
        f"  {args.split_name}: source_rows={total_source_rows} "
        f"sampled_rows={len(output_rows)} qa={sum(row['qa_count'] for row in output_rows)} "
        f"seed={args.seed}"
    )


if __name__ == "__main__":
    main()

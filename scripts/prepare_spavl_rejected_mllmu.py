#!/usr/bin/env python3
"""Convert SPA-VL train data into the MLLMU-style VQA unlearning format.

Input SPA-VL train rows:

    image
    question
    chosen
    rejected
    image_name

Output rows mirror data/unlearn/mllmu:

    image
    ID
    metadata
    source_split
    qa_count

Each output row is grouped by ``image_name`` and contains a JSON ``metadata``
list. The QA answer is intentionally taken from SPA-VL's ``rejected`` field.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCE_ROOT = Path("./external_datasets/SPA-VL/data")
DEFAULT_OUTPUT_ROOT = Path(
    "./data/unlearn/spavl_rejected"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SPA-VL train data as MLLMU-style image QA rows, using "
            "the rejected response as Answer."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-name", default="train")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Rows per parquet batch while reading SPA-VL shards.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for smoke tests. Omit to convert the full train set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output split directory.",
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


def image_id_from_row(row: dict[str, Any], fallback_index: int) -> str:
    image_name = row.get("image_name")
    if isinstance(image_name, str) and image_name.strip():
        return image_name
    return str(fallback_index)


def iter_spavl_rows(source_root: Path, batch_size: int):
    parquet_files = sorted(source_root.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No train-*.parquet files found under {source_root}")

    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield parquet_file.name, row


def build_output_rows(
    source_root: Path,
    split_name: str,
    batch_size: int,
    max_rows: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    rows_seen = 0

    for shard_name, row in iter_spavl_rows(source_root, batch_size):
        if max_rows is not None and rows_seen >= max_rows:
            break

        image_id = image_id_from_row(row, rows_seen)
        question = get_required_text(row, "question")
        answer = get_required_text(row, "rejected")

        if image_id not in grouped:
            grouped[image_id] = {
                "image": normalize_image(row),
                "ID": image_id,
                "metadata_items": [],
            }

        grouped[image_id]["metadata_items"].append(
            {
                "ID": image_id,
                "Question": question,
                "Answer": answer,
            }
        )
        rows_seen += 1

    output_rows = []
    qa_counts = []
    for image_id, item in grouped.items():
        metadata = item["metadata_items"]
        qa_counts.append(len(metadata))
        output_rows.append(
            {
                "image": item["image"],
                "ID": image_id,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "source_split": split_name,
                "qa_count": len(metadata),
            }
        )

    stats = {
        "source_rows": rows_seen,
        "output_rows": len(output_rows),
        "qa_total": sum(qa_counts),
        "qa_per_image_min": min(qa_counts) if qa_counts else 0,
        "qa_per_image_max": max(qa_counts) if qa_counts else 0,
    }
    return output_rows, stats


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
    stats: dict[str, Any],
    source_root: Path,
    overwrite: bool,
) -> None:
    split_dir = prepare_output_split(output_root, split_name, overwrite)
    pq.write_table(
        pa.Table.from_pylist(output_rows),
        split_dir / "train-00000-of-00001.parquet",
    )

    ids = [row["ID"] for row in output_rows]
    with (split_dir / "ids.json").open("w", encoding="utf-8") as handle:
        json.dump(ids, handle, ensure_ascii=False, indent=2)

    manifest_path = output_root / "manifest.json"
    manifest = {
        "task": "spavl_rejected_vqa",
        "format": "one row per image with image, ID, metadata",
        "source_root": str(source_root),
        "answer_field": "rejected",
        "splits": {
            split_name: {
                "output_split": split_name,
                "parquet": str(split_dir / "train-00000-of-00001.parquet"),
                "ids_file": str(split_dir / "ids.json"),
                **stats,
            }
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_rows, stats = build_output_rows(
        source_root=args.source_root,
        split_name=args.split_name,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )
    write_output(
        output_root=args.output_root,
        split_name=args.split_name,
        output_rows=output_rows,
        stats=stats,
        source_root=args.source_root,
        overwrite=args.overwrite,
    )

    print("SPA-VL rejected output:", args.output_root)
    print(
        f"  {args.split_name}: source_rows={stats['source_rows']} "
        f"output_rows={stats['output_rows']} qa_total={stats['qa_total']} "
        f"qa_minmax=({stats['qa_per_image_min']}, {stats['qa_per_image_max']})"
    )


if __name__ == "__main__":
    main()

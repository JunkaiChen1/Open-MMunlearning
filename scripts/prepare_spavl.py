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

Full mode groups rows by ``image_name``. Sample mode selects a reproducible
subset and emits one row per source QA. Both use SPA-VL's ``rejected`` answer.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "data/source/spavl"
DEFAULT_FULL_OUTPUT_ROOT = REPO_ROOT / "data/unlearn/spavl_rejected"
DEFAULT_SAMPLE_OUTPUT_ROOT = REPO_ROOT / "data/unlearn/spavl_7200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SPA-VL train data as MLLMU-style image QA rows, using "
            "the rejected response as Answer."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--mode", choices=("full", "sample"), default="full")
    parser.add_argument("--split-name", default=None,
                        help="Defaults to train for full and forget for sample mode.")
    parser.add_argument("--sample-size", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=42)
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


def reservoir_sample(
    source_root: Path, sample_size: int, seed: int, batch_size: int
) -> tuple[list[tuple[int, str, dict[str, Any]]], int]:
    if sample_size <= 0:
        raise ValueError("sample-size must be positive")
    rng = random.Random(seed)
    selected: list[tuple[int, str, dict[str, Any]]] = []
    total_rows = 0
    valid_rows = 0
    for shard_name, row in iter_spavl_rows(source_root, batch_size):
        total_rows += 1
        if not all(isinstance(row.get(key), str) and row[key].strip()
                   for key in ("question", "rejected")):
            continue
        valid_rows += 1
        item = (total_rows - 1, shard_name, row)
        if len(selected) < sample_size:
            selected.append(item)
        else:
            replacement = rng.randrange(valid_rows)
            if replacement < sample_size:
                selected[replacement] = item
    if valid_rows < sample_size:
        raise ValueError(f"Requested {sample_size} rows; found only {valid_rows} valid rows")
    selected.sort(key=lambda item: item[0])
    return selected, total_rows


def build_sampled_rows(
    selected: list[tuple[int, str, dict[str, Any]]], split_name: str
) -> list[dict[str, Any]]:
    output_rows = []
    for output_idx, (source_index, shard_name, row) in enumerate(selected):
        row_id = f"{split_name}:{output_idx}"
        metadata = [{
            "ID": f"{row_id}:0",
            "Question": get_required_text(row, "question"),
            "Answer": get_required_text(row, "rejected"),
        }]
        output_rows.append({
            "image": normalize_image(row), "ID": row_id,
            "metadata": json.dumps(metadata, ensure_ascii=False),
            "source_split": split_name, "qa_count": 1,
            "source_index": source_index, "source_shard": shard_name,
            "image_name": row.get("image_name", ""),
        })
    return output_rows


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
    source_root = args.source_root
    if not list(source_root.glob("train-*.parquet")) and list((source_root / "data").glob("train-*.parquet")):
        source_root = source_root / "data"
    output_root = args.output_root or (
        DEFAULT_SAMPLE_OUTPUT_ROOT if args.mode == "sample" else DEFAULT_FULL_OUTPUT_ROOT
    )
    split_name = args.split_name or ("forget" if args.mode == "sample" else "train")
    if args.mode == "sample":
        selected, total_rows = reservoir_sample(
            source_root, args.sample_size, args.seed, args.batch_size
        )
        output_rows = build_sampled_rows(selected, split_name)
        stats = {
            "source_rows": total_rows, "output_rows": len(output_rows),
            "qa_total": len(output_rows), "sample_size": args.sample_size,
            "seed": args.seed,
        }
    else:
        output_rows, stats = build_output_rows(
            source_root=source_root,
            split_name=split_name,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
        )
    write_output(
        output_root=output_root,
        split_name=split_name,
        output_rows=output_rows,
        stats=stats,
        source_root=source_root,
        overwrite=args.overwrite,
    )

    print(f"SPA-VL {args.mode} output: {output_root}")
    print(f"  {split_name}: source_rows={stats['source_rows']} "
          f"output_rows={stats['output_rows']} qa_total={stats['qa_total']}")


if __name__ == "__main__":
    main()

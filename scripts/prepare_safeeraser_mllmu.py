#!/usr/bin/env python3
"""Convert Safeeraser JSON data into MLLMU-style VQA unlearning splits.

Output rows mirror data/unlearn/mllmu:

    image
    ID
    metadata
    source_split
    qa_count

By default this script uses each record's original ``image_id`` and converts
``unsafe_pairs`` into QA metadata, with ``model_response`` as the Answer.
This is the most direct format for unlearning unsafe multimodal responses.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCE_ROOT = Path("./external_datasets/Safeeraser")
DEFAULT_OUTPUT_ROOT = Path(
    "./data/unlearn/safeeraser"
)

DEFAULT_SPLIT_FILES = {
    "forget": "all_train.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Safeeraser JSON data as MLLMU-style image QA rows."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[f"{split}:{filename}" for split, filename in DEFAULT_SPLIT_FILES.items()],
        help=(
            "Output split mappings in the form split_name:json_file. "
            "Default: forget:all_train.json"
        ),
    )
    parser.add_argument(
        "--image-field",
        choices=["image_id", "SDImage_path"],
        default="image_id",
        help="Which Safeeraser image path to store in the output image field.",
    )
    parser.add_argument(
        "--qa-source",
        choices=["unsafe", "unharm", "all"],
        default="unsafe",
        help="Which QA pairs to export into metadata.",
    )
    parser.add_argument(
        "--unsafe-answer-field",
        choices=["model_response", "sd_response"],
        default="model_response",
        help="Answer field to use when exporting unsafe_pairs.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional per-split record cap for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output split directories.",
    )
    return parser.parse_args()


def parse_split_mapping(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"Expected split mapping split_name:json_file, got {value!r}")
    split_name, filename = value.split(":", 1)
    if not split_name or not filename:
        raise ValueError(f"Invalid split mapping: {value!r}")
    return split_name, filename


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected list JSON at {path}, got {type(data)}")
    return data


def read_image_bytes(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    return path.read_bytes()


def safe_category(record: dict[str, Any]) -> str:
    category = record.get("category")
    if isinstance(category, str) and category:
        return category

    image_id = record.get("image_id", "")
    parts = Path(image_id).parts
    if len(parts) >= 2 and parts[0] == "img":
        return parts[1]
    return ""


def qa_id(record_id: str, index: int, prefix: str) -> str:
    return f"{record_id}:{prefix}:{index}"


def build_unsafe_metadata(
    record: dict[str, Any],
    record_id: str,
    answer_field: str,
) -> list[dict[str, str]]:
    metadata = []
    for idx, pair in enumerate(record.get("unsafe_pairs", [])):
        question = pair.get("question")
        answer = pair.get(answer_field)
        if not question or not answer:
            continue
        metadata.append(
            {
                "ID": qa_id(record_id, idx, "unsafe"),
                "Question": question,
                "Answer": answer,
            }
        )
    return metadata


def build_unharm_metadata(record: dict[str, Any], record_id: str) -> list[dict[str, str]]:
    metadata = []
    for idx, key in enumerate(
        [
            "UnharmPair_text1",
            "UnharmPair_text2",
            "UnharmPair_image1",
            "UnharmPair_image2",
        ]
    ):
        pair = record.get(key, {})
        question = pair.get("Question")
        answer = pair.get("Answer")
        if not question or not answer:
            continue
        metadata.append(
            {
                "ID": qa_id(record_id, idx, "unharm"),
                "Question": question,
                "Answer": answer,
            }
        )
    return metadata


def build_metadata(
    record: dict[str, Any],
    record_id: str,
    qa_source: str,
    unsafe_answer_field: str,
) -> list[dict[str, str]]:
    metadata = []
    if qa_source in {"unsafe", "all"}:
        metadata.extend(build_unsafe_metadata(record, record_id, unsafe_answer_field))
    if qa_source in {"unharm", "all"}:
        metadata.extend(build_unharm_metadata(record, record_id))
    return metadata


def output_split_dir(output_root: Path, split: str, overwrite: bool) -> Path:
    split_dir = output_root / split
    if split_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{split_dir} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir


def convert_split(
    source_root: Path,
    output_root: Path,
    split_name: str,
    json_file: str,
    image_field: str,
    qa_source: str,
    unsafe_answer_field: str,
    max_rows: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    records = read_json(source_root / json_file)
    if max_rows is not None:
        records = records[:max_rows]

    output_rows = []
    qa_counts = []
    categories = {}

    for idx, record in enumerate(records):
        image_rel_path = record.get(image_field)
        if not isinstance(image_rel_path, str) or not image_rel_path:
            raise ValueError(f"Missing image path field {image_field!r} in row {idx}")

        record_id = f"{split_name}:{idx}"
        metadata = build_metadata(
            record=record,
            record_id=record_id,
            qa_source=qa_source,
            unsafe_answer_field=unsafe_answer_field,
        )
        if not metadata:
            continue

        category = safe_category(record)
        categories[category] = categories.get(category, 0) + 1
        qa_counts.append(len(metadata))

        output_rows.append(
            {
                "image": {
                    "bytes": read_image_bytes(source_root / image_rel_path),
                    "path": None,
                },
                "ID": record_id,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "source_split": split_name,
                "qa_count": len(metadata),
            }
        )

    split_dir = output_split_dir(output_root, split_name, overwrite)
    pq.write_table(
        pa.Table.from_pylist(output_rows),
        split_dir / "train-00000-of-00001.parquet",
    )

    return {
        "source_file": json_file,
        "output_split": split_name,
        "rows": len(output_rows),
        "qa_total": sum(qa_counts),
        "qa_per_row_min": min(qa_counts) if qa_counts else 0,
        "qa_per_row_max": max(qa_counts) if qa_counts else 0,
        "categories": dict(sorted(categories.items())),
    }


def main() -> None:
    args = parse_args()
    split_mappings = [parse_split_mapping(value) for value in args.splits]

    manifest = {
        "task": "safeeraser_vqa",
        "format": "one row per Safeeraser record with image, ID, metadata",
        "source_root": str(args.source_root),
        "image_field": args.image_field,
        "qa_source": args.qa_source,
        "unsafe_answer_field": args.unsafe_answer_field,
        "splits": {},
    }

    for split_name, json_file in split_mappings:
        manifest["splits"][split_name] = convert_split(
            source_root=args.source_root,
            output_root=args.output_root,
            split_name=split_name,
            json_file=json_file,
            image_field=args.image_field,
            qa_source=args.qa_source,
            unsafe_answer_field=args.unsafe_answer_field,
            max_rows=args.max_rows,
            overwrite=args.overwrite,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("Safeeraser output:", args.output_root)
    for split, stats in manifest["splits"].items():
        print(
            f"  {split}: rows={stats['rows']} qa={stats['qa_total']} "
            f"qa_minmax=({stats['qa_per_row_min']}, {stats['qa_per_row_max']})"
        )


if __name__ == "__main__":
    main()

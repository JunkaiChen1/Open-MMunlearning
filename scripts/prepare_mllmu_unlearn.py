#!/usr/bin/env python3
"""Normalize MLLMU-Bench_distill test splits under data/unlearn/mllmu.

The source dataset is already in MLLMU-style person rows:

    image
    ID
    metadata

This script keeps the same ID semantics and adds lightweight bookkeeping fields
so the output mirrors the processed CLEAR VQA layout used in this repository.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCE_ROOT = Path("./external_datasets/MLLMU-Bench_distill/test")
DEFAULT_OUTPUT_ROOT = Path(
    "./data/unlearn/mllmu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MLLMU-Bench_distill test splits for unlearning data."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Split names to export. Defaults to all split directories under source-root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output split directories.",
    )
    return parser.parse_args()


def read_parquet_dir(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")

    rows: list[dict[str, Any]] = []
    for file in files:
        rows.extend(pq.read_table(file).to_pylist())
    return rows


def parse_metadata(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, list):
        return value
    raise TypeError(f"Unsupported metadata type: {type(value)}")


def normalize_image(image: dict[str, Any]) -> dict[str, Any]:
    if image.get("bytes") is not None:
        return {"bytes": image["bytes"], "path": None}
    return {"bytes": None, "path": image.get("path")}


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


def load_ids_file(source_dir: Path, split: str) -> tuple[str, list[str]] | None:
    candidates = []
    if split.startswith("forget_"):
        candidates.append("forget_ids.json")
    if split.startswith("retain_"):
        candidates.append("retain_ids.json")
    candidates.extend(["ids.json", "forget_ids.json", "retain_ids.json"])

    for filename in candidates:
        path = source_dir / filename
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return filename, json.load(handle)
    return None


def export_split(source_root: Path, output_root: Path, split: str, overwrite: bool) -> dict[str, Any]:
    source_dir = source_root / split
    rows = read_parquet_dir(source_dir)

    output_rows = []
    qa_counts = []
    for row in rows:
        metadata = parse_metadata(row["metadata"])
        qa_counts.append(len(metadata))
        output_rows.append(
            {
                "image": normalize_image(row["image"]),
                "ID": str(row["ID"]),
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "source_split": split,
                "qa_count": len(metadata),
            }
        )

    split_dir = output_split_dir(output_root, split, overwrite)
    pq.write_table(
        pa.Table.from_pylist(output_rows),
        split_dir / "train-00000-of-00001.parquet",
    )

    ids_from_source = load_ids_file(source_dir, split)
    if ids_from_source is None:
        id_filename = "ids.json"
        ids = [row["ID"] for row in output_rows]
    else:
        id_filename, ids = ids_from_source
    with (split_dir / id_filename).open("w", encoding="utf-8") as handle:
        json.dump([str(value) for value in ids], handle, ensure_ascii=False, indent=2)

    return {
        "source_split": split,
        "output_split": split,
        "rows": len(output_rows),
        "persons": len({row["ID"] for row in output_rows}),
        "qa_total": sum(qa_counts),
        "qa_per_person_min": min(qa_counts) if qa_counts else 0,
        "qa_per_person_max": max(qa_counts) if qa_counts else 0,
        "ids_file": id_filename,
    }


def discover_splits(source_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in source_root.iterdir()
        if path.is_dir() and list(path.glob("*.parquet"))
    )


def main() -> None:
    args = parse_args()
    splits = args.splits or discover_splits(args.source_root)

    manifest = {
        "task": "mllmu_vqa",
        "format": "one row per person with image, ID, metadata",
        "source_root": str(args.source_root),
        "splits": {},
    }

    for split in splits:
        manifest["splits"][split] = export_split(
            args.source_root,
            args.output_root,
            split,
            args.overwrite,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("MLLMU output:", args.output_root)
    for split, stats in manifest["splits"].items():
        print(
            f"  {split}: rows={stats['rows']} qa={stats['qa_total']} "
            f"qa_minmax=({stats['qa_per_person_min']}, {stats['qa_per_person_max']})"
        )


if __name__ == "__main__":
    main()

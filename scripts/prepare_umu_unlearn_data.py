#!/usr/bin/env python3
"""Prepare UMU-Bench unlearn data for open-unlearning.

The output keeps one row per identity and aligns with the local data/unlearn
schema used by MLLMU/CLEAR/SafeEraser: image, ID, metadata, source_split, and
qa_count. MM_QA and UM_QA are stored together in metadata and distinguished by a
per-QA "source" key.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SOURCE = Path("./external_datasets/UMU-bench")
DEFAULT_OUTPUT = Path("./data/unlearn/umu")
PARQUET_NAME = "train-00000-of-00001.parquet"
SPLITS = ("forget_5", "forget_10", "forget_15", "retain_95", "retain_90", "retain_85")


def parse_qa(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def qa_count(value: Any) -> int:
    qa = parse_qa(value)
    return len(qa.get("question", {}))


def build_metadata(row: pd.Series) -> tuple[str, int, int, int]:
    metadata = []
    source_counts = {}
    mismatched_keys = 0
    parsed = {}
    for source in ("MM_QA", "UM_QA"):
        qa = parse_qa(row[source])
        parsed[source] = qa
        questions = qa.get("question", {})
        answers = qa.get("answer", {})
        source_counts[source] = len(questions)
        for qa_key, question in questions.items():
            answer = answers.get(qa_key, "")
            if question and answer:
                metadata.append(
                    {
                        "ID": str(row["ID"]),
                        "Question": question,
                        "Answer": answer,
                        "source": source,
                        "qa_key": qa_key,
                        "has_image": source == "MM_QA",
                    }
                )

    mm_keys = set(parsed["MM_QA"].get("question", {}).keys())
    um_keys = set(parsed["UM_QA"].get("question", {}).keys())
    if mm_keys != um_keys:
        mismatched_keys = 1
    return (
        json.dumps(metadata, ensure_ascii=False),
        source_counts["MM_QA"],
        source_counts["UM_QA"],
        mismatched_keys,
    )


def convert_split(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_columns = {
        "ID",
        "image",
        "Biography",
        "MM_QA",
        "UM_QA",
        "Classify",
        "Cloze",
        "Generation",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{split}: missing columns {missing}")

    rows = []
    mm_counts = []
    um_counts = []
    mismatched_keys = 0
    for _, row in df.iterrows():
        metadata, mm_count, um_count, row_mismatched_keys = build_metadata(row)
        mm_counts.append(mm_count)
        um_counts.append(um_count)
        mismatched_keys += row_mismatched_keys
        rows.append(
            {
                "image": row["image"],
                "ID": str(row["ID"]),
                "metadata": metadata,
                "source_split": split,
                "qa_count": mm_count + um_count,
            }
        )

    converted = pd.DataFrame(rows)
    stats = {
        "rows": int(len(df)),
        "persons": int(df["ID"].nunique()),
        "mm_qa_total": int(sum(mm_counts)),
        "um_qa_total": int(sum(um_counts)),
        "qa_total": int(sum(mm_counts) + sum(um_counts)),
        "qa_per_person_min": int(min(mm_counts + um_counts)) if len(df) else 0,
        "qa_per_person_max": int(max(mm_counts + um_counts)) if len(df) else 0,
        "mismatched_mm_um_key_rows": int(mismatched_keys),
    }
    return converted, stats


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_split(source_root: Path, output_root: Path, split: str) -> dict[str, Any]:
    source_file = source_root / split / PARQUET_NAME
    if not source_file.exists():
        raise FileNotFoundError(source_file)

    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / PARQUET_NAME

    df = pd.read_parquet(source_file)
    converted, stats = convert_split(df, split)
    converted.to_parquet(output_file, index=False)

    if split.startswith("forget_"):
        ids_name = "forget_ids.json"
    elif split.startswith("retain_"):
        ids_name = "retain_ids.json"
    else:
        ids_name = "ids.json"
    ids = converted["ID"].astype(str).tolist()
    write_json(output_dir / ids_name, ids)

    return {
        "source_split": split,
        "output_split": split,
        **stats,
        "ids_file": ids_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UMU-Bench unlearn data")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-full", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "task": "umu_bench",
        "format": "one row per identity with image, ID, metadata, source_split, qa_count; metadata items include source=MM_QA|UM_QA and has_image",
        "source_root": str(args.source_root),
        "splits": {},
    }

    for split in SPLITS:
        manifest["splits"][split] = copy_split(args.source_root, args.output_root, split)

    if args.include_full:
        manifest["splits"]["full_data"] = copy_split(args.source_root, args.output_root, "full_data")

    write_json(args.output_root / "manifest.json", manifest)
    print(f"Wrote UMU-Bench unlearn data to {args.output_root}")


if __name__ == "__main__":
    main()

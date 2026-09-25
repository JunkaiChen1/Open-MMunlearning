#!/usr/bin/env python3
"""Normalize multimodal finetune datasets to one VQA parquet layout.

Output layout:

    data/finetune/<dataset>/train-00000-of-00001.parquet

Each row has:

    image: {bytes, path}
    ID: string
    metadata: JSON string of [{"ID", "Question", "Answer", ...}, ...]
    source_split: string
    qa_count: int64

The layout mirrors data/unlearn/* split parquet files, but finetune datasets use
only the single "vqa" split.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(".")
DEFAULT_CLEAR_SOURCE = Path("./external_datasets/CLEAR/full")
DEFAULT_UMU_SOURCE = Path(
    "./external_datasets/UMU-bench/full_data/train-00000-of-00001.parquet"
)
DEFAULT_MLLMU_SOURCE = Path(
    "./external_datasets/MLLMU-Bench/ft_Data/train-00000-of-00001.parquet"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/finetune"
OUTPUT_FILENAME = "train-00000-of-00001.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CLEAR, UMU-Bench, and MLLMU-Bench finetune data to unified VQA parquet files."
    )
    parser.add_argument("--clear-source", type=Path, default=DEFAULT_CLEAR_SOURCE)
    parser.add_argument("--umubench-source", type=Path, default=DEFAULT_UMU_SOURCE)
    parser.add_argument("--mllmu-source", type=Path, default=DEFAULT_MLLMU_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["clear", "umubench", "mllmu"],
        choices=["clear", "umubench", "mllmu"],
        help="Datasets to convert.",
    )
    parser.add_argument(
        "--caption-question",
        default="Describe this image.",
        help="Question used for CLEAR caption rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing unified parquet files.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {path}")
        for file in files:
            rows.extend(pq.read_table(file).to_pylist())
        return rows
    return pq.read_table(path).to_pylist()


def normalize_image(image: Any) -> dict[str, Any]:
    if isinstance(image, dict):
        return {
            "bytes": image.get("bytes"),
            "path": image.get("path") if image.get("bytes") is None else None,
        }
    return {"bytes": None, "path": image if isinstance(image, str) else None}


def parse_metadata(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    raise TypeError(f"Unsupported metadata value: {type(value)}")


def parse_umu_qa(value: Any) -> dict[str, dict[str, str]]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported UMU QA value: {type(value)}")
    return value


def write_dataset(
    rows: list[dict[str, Any]],
    output_root: Path,
    dataset_name: str,
    source_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = output_root / dataset_name
    output_path = output_dir / OUTPUT_FILENAME
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists. Re-run with --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)

    qa_counts = [row["qa_count"] for row in rows]
    manifest = {
        "task": "vqa_finetune",
        "format": "one row with image, ID, metadata; metadata is a JSON list of QA pairs",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "rows": len(rows),
        "ids": len({row["ID"] for row in rows}),
        "qa_total": sum(qa_counts),
        "qa_per_row_min": min(qa_counts) if qa_counts else 0,
        "qa_per_row_max": max(qa_counts) if qa_counts else 0,
    }
    with (output_root / dataset_name / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({"vqa": manifest}, handle, ensure_ascii=False, indent=2)
    return manifest


def convert_clear(path: Path, caption_question: str) -> list[dict[str, Any]]:
    output_rows = []
    ids_by_name: dict[str, str] = {}
    for row_idx, row in enumerate(read_rows(path)):
        name = row.get("name")
        if row.get("ID") is not None:
            row_id = str(row["ID"])
        elif name:
            row_id = ids_by_name.setdefault(name, f"{len(ids_by_name):03d}")
        else:
            row_id = str(row_idx)
        question = row.get("question") or caption_question
        answer = row.get("answer") or row.get("caption")
        if not question or not answer:
            continue
        metadata = [
            {
                "ID": row_id,
                "Question": question,
                "Answer": answer,
                "source": "CLEAR",
                "has_image": True,
            }
        ]
        output_rows.append(
            {
                "image": normalize_image(row.get("image")),
                "ID": row_id,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "source_split": "vqa",
                "qa_count": len(metadata),
            }
        )
    return output_rows


def convert_mllmu(path: Path) -> list[dict[str, Any]]:
    output_rows = []
    for row_idx, row in enumerate(read_rows(path)):
        row_id = str(row.get("ID", row_idx))
        metadata = []
        for qa_idx, qa in enumerate(parse_metadata(row["metadata"])):
            question = qa.get("Question") or qa.get("Additional_question")
            answer = qa.get("Answer") or qa.get("Additional_answer")
            qa_id = qa.get("ID") or qa.get("Additional_ID") or f"{row_id}:{qa_idx}"
            if question and answer:
                metadata.append(
                    {
                        "ID": str(qa_id),
                        "Question": question,
                        "Answer": answer,
                        "source": "MLLMU",
                        "has_image": True,
                    }
                )
        if not metadata:
            continue
        output_rows.append(
            {
                "image": normalize_image(row.get("image")),
                "ID": row_id,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "source_split": "vqa",
                "qa_count": len(metadata),
            }
        )
    return output_rows


def convert_umubench(path: Path) -> list[dict[str, Any]]:
    output_rows = []
    for row_idx, row in enumerate(read_rows(path)):
        row_id = str(row.get("ID", row_idx))
        image = normalize_image(row.get("image"))
        for source_name, has_image in (("MM_QA", True), ("UM_QA", False)):
            qa = parse_umu_qa(row[source_name])
            questions = qa["question"]
            answers = qa["answer"]
            metadata = []
            for qa_name, question in questions.items():
                answer = answers.get(qa_name)
                if question and answer:
                    metadata.append(
                        {
                            "ID": f"{row_id}:{source_name}:{qa_name}",
                            "Question": question,
                            "Answer": answer,
                            "source": source_name,
                            "has_image": has_image,
                        }
                    )
            if not metadata:
                continue
            output_rows.append(
                {
                    "image": image if has_image else {"bytes": None, "path": None},
                    "ID": row_id,
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                    "source_split": "vqa",
                    "qa_count": len(metadata),
                }
            )
    return output_rows


def main() -> None:
    args = parse_args()
    converters = {
        "clear": (args.clear_source, lambda path: convert_clear(path, args.caption_question)),
        "umubench": (args.umubench_source, convert_umubench),
        "mllmu": (args.mllmu_source, convert_mllmu),
    }

    for dataset_name in args.datasets:
        source_path, converter = converters[dataset_name]
        rows = converter(source_path)
        stats = write_dataset(
            rows=rows,
            output_root=args.output_root,
            dataset_name=dataset_name,
            source_path=source_path,
            overwrite=args.overwrite,
        )
        print(
            f"{dataset_name}: rows={stats['rows']} qa={stats['qa_total']} "
            f"qa_minmax=({stats['qa_per_row_min']}, {stats['qa_per_row_max']}) "
            f"-> {stats['output_path']}"
        )


if __name__ == "__main__":
    main()

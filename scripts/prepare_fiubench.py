#!/usr/bin/env python3
"""Convert a downloaded FIUBench snapshot to finetune and unlearn parquet."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "data/source/fiubench"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data"

FIRST_N_IDENTITIES = 400
FORGET_SPLITS = {
    "forget1": "forget_1",
    "forget5": "forget_5",
    "forget10": "forget_10",
}
RETAIN_SPLITS = {
    "forget1": "retain_99",
    "forget5": "retain_95",
    "forget10": "retain_90",
}
RETAIN_TRAIN_SPLIT = "retain"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_image(path: Path) -> dict[str, Any]:
    return {"bytes": path.read_bytes(), "path": None}


def resolve_image_path(row: dict[str, Any], image_root: Path) -> Path:
    image_path = str(row["image_path"])
    if image_path.startswith("./dataset/"):
        image_path = image_path[len("./dataset/") :]
    path = image_root / image_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def convert_identity(row: dict[str, Any], source_split: str, image_root: Path) -> dict[str, Any]:
    row_id = str(row["unique_id"])
    image_path = resolve_image_path(row, image_root)
    metadata = []
    for qa_idx, qa in enumerate(row.get("qa_list", [])):
        question = qa.get("question")
        answer = qa.get("answer")
        if not question or not answer:
            continue
        metadata.append(
            {
                "ID": f"{row_id}:{qa_idx:02d}",
                "Question": question,
                "Answer": answer,
                "source": "FIUBench",
                "has_image": True,
                "unique_id": row_id,
                "name": row.get("name"),
                "gender": row.get("gender"),
                "caption": row.get("caption"),
                "raw_data": row.get("raw_data"),
                "keywords": qa.get("keywords") or [],
                "paraphrased_question": qa.get("paraphrased_question") or [],
                "paraphrased_answer": qa.get("paraphrased_answer") or "",
                "perturbed_answer": qa.get("perturbed_answer") or [],
            }
        )
    return {
        "image": read_image(image_path),
        "ID": row_id,
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "source_split": source_split,
        "qa_count": len(metadata),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_split(rows: list[dict[str, Any]], output_dir: Path, ids_filename: str | None, overwrite: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if list(output_dir.glob("train-*.parquet")) and not overwrite:
        raise FileExistsError(f"{output_dir} already contains output; use --overwrite")
    for existing in output_dir.glob("train-*.parquet"):
        existing.unlink()
    output_path = output_dir / "train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    ids = [row["ID"] for row in rows]
    if ids_filename is not None:
        write_json(output_dir / ids_filename, ids)

    qa_counts = [row["qa_count"] for row in rows]
    names = []
    genders = []
    for row in rows:
        metadata = json.loads(row["metadata"])
        if metadata:
            names.append(metadata[0].get("name"))
            genders.append(metadata[0].get("gender"))
    return {
        "output_path": str(output_path),
        "rows": len(rows),
        "ids": len(set(ids)),
        "qa_total": sum(qa_counts),
        "qa_per_row_min": min(qa_counts) if qa_counts else 0,
        "qa_per_row_max": max(qa_counts) if qa_counts else 0,
        "gender_counts": dict(Counter(genders)),
        "ids_file": ids_filename,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--first-n-identities", type=int, default=FIRST_N_IDENTITIES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root
    if not (source_root / "full.json").is_file() and (source_root / "dataset/full.json").is_file():
        source_root = source_root / "dataset"
    source_json = source_root / "full.json"
    source_split_json = source_root / "split.json"
    finetune_output = args.output_root / "finetune/fiubench"
    unlearn_output = args.output_root / "unlearn/fiubench"

    source_rows = read_jsonl(source_json)[:args.first_n_identities]
    split_ids = json.loads(source_split_json.read_text(encoding="utf-8"))
    source_by_id = {row["unique_id"]: row for row in source_rows}

    finetune_rows = [convert_identity(row, "finetune", source_root) for row in source_rows]
    finetune_output.mkdir(parents=True, exist_ok=True)
    finetune_stats = write_split(finetune_rows, finetune_output, None, args.overwrite)
    finetune_manifest = {
        "task": "vqa_finetune",
        "format": "one row per identity with image, ID, metadata; metadata is a JSON list of QA pairs",
        "source_path": str(source_json),
        "source_split_path": str(source_split_json),
        "image_root": str(source_root),
        "first_n_identities": args.first_n_identities,
        **finetune_stats,
    }
    write_json(finetune_output / "manifest.json", {"vqa": finetune_manifest})

    all_forget_ids = set()
    for source_split in FORGET_SPLITS:
        all_forget_ids.update(split_ids[source_split])

    manifest: dict[str, Any] = {
        "task": "fiubench",
        "format": "one row per identity with image, ID, metadata, source_split, qa_count",
        "source_path": str(source_json),
        "source_split_path": str(source_split_json),
        "image_root": str(source_root),
        "first_n_identities": args.first_n_identities,
        "splits": {},
    }

    for source_split, output_split in FORGET_SPLITS.items():
        rows = [
            convert_identity(source_by_id[unique_id], output_split, source_root)
            for unique_id in split_ids[source_split]
            if unique_id in source_by_id
        ]
        stats = write_split(rows, unlearn_output / output_split, "forget_ids.json", args.overwrite)
        stats["source_split"] = source_split
        manifest["splits"][output_split] = stats

    retain_rows = [
        convert_identity(row, RETAIN_TRAIN_SPLIT, source_root)
        for row in source_rows
        if row["unique_id"] not in all_forget_ids
    ]
    retain_stats = write_split(
        retain_rows, unlearn_output / RETAIN_TRAIN_SPLIT, "retain_ids.json", args.overwrite
    )
    retain_stats["source_split"] = "first400_minus_forget1_forget5_forget10"
    retain_stats["excluded_forget_ids"] = len(all_forget_ids)
    manifest["splits"][RETAIN_TRAIN_SPLIT] = retain_stats

    for source_split, retain_split in RETAIN_SPLITS.items():
        forget_ids = set(split_ids[source_split])
        retain_rows_for_ratio = [
            convert_identity(row, retain_split, source_root)
            for row in source_rows
            if row["unique_id"] not in forget_ids
        ]
        ratio_retain_stats = write_split(
            retain_rows_for_ratio, unlearn_output / retain_split, "retain_ids.json", args.overwrite
        )
        ratio_retain_stats["source_split"] = f"first400_minus_{source_split}"
        ratio_retain_stats["excluded_forget_ids"] = len(forget_ids)
        manifest["splits"][retain_split] = ratio_retain_stats

    write_json(unlearn_output / "manifest.json", manifest)

    print(
        f"finetune: rows={finetune_stats['rows']} qa={finetune_stats['qa_total']} "
        f"-> {finetune_stats['output_path']}"
    )
    for split_name, stats in manifest["splits"].items():
        print(
            f"{split_name}: rows={stats['rows']} qa={stats['qa_total']} "
            f"-> {stats['output_path']}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert a downloaded CoVUBench snapshot to finetune and unlearn parquet."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "data/source/covubench"
OUTPUT_ROOT = REPO_ROOT / "data"
SHARD_SIZE = 200

SPLIT_MAP = {
    "forget5": ("forget_5", "retain_95"),
    "forget10": ("forget_10", "retain_90"),
    "forget15": ("forget_15", "retain_85"),
    "forget20": ("forget_20", "retain_80"),
}


def normalize_image(image: Any) -> dict[str, Any]:
    if isinstance(image, dict):
        return {"bytes": image.get("bytes"), "path": None}
    return {"bytes": None, "path": image if isinstance(image, str) else None}


def read_parquet_glob(pattern: str) -> list[dict[str, Any]]:
    files = sorted(Path(pattern).parent.glob(Path(pattern).name))
    if not files:
        raise FileNotFoundError(pattern)
    rows: list[dict[str, Any]] = []
    for file in files:
        rows.extend(pq.read_table(file).to_pylist())
    return rows


def convert_source_rows(rows: list[dict[str, Any]], source_split: str) -> list[dict[str, Any]]:
    output_rows = []
    for idx, row in enumerate(rows):
        concept_name = str(row["name"])
        row_id = f"{concept_name}:{source_split}:{idx:05d}"
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
                "source_split": source_split,
                "qa_count": 1,
            }
        )
    return output_rows


def convert_finetune_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows = []
    for idx, row in enumerate(rows):
        concept_name = str(row["name"])
        row_id = f"{concept_name}:{idx:05d}"
        metadata = [{
            "ID": row_id,
            "Question": row["question"],
            "Answer": row["answer"],
            "source": "CoVUBench",
            "has_image": True,
            "name": concept_name,
            "type": row["type"],
            "keywords": row.get("keywords") or "{}",
            "question_type": row["question_type"],
        }]
        output_rows.append({
            "image": normalize_image(row.get("image")),
            "ID": row_id,
            "metadata": json.dumps(metadata, ensure_ascii=False),
            "source_split": "finetune",
            "qa_count": 1,
        })
    return output_rows


def metadata_name(row: dict[str, Any]) -> str:
    metadata = json.loads(row["metadata"])
    if not metadata:
        raise ValueError(f"Empty metadata for row {row.get('ID')}")
    return metadata[0]["name"]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_split(rows: list[dict[str, Any]], output_dir: Path, overwrite: bool) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if list(output_dir.glob("train-*.parquet")) and not overwrite:
        raise FileExistsError(f"{output_dir} already contains output; use --overwrite")
    for existing_file in output_dir.glob("train-*.parquet"):
        existing_file.unlink()
    num_shards = max(1, (len(rows) + SHARD_SIZE - 1) // SHARD_SIZE)
    output_files = []
    for shard_idx in range(num_shards):
        shard_rows = rows[shard_idx * SHARD_SIZE : (shard_idx + 1) * SHARD_SIZE]
        shard_path = output_dir / f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(shard_rows), shard_path)
        output_files.append(shard_path)
    return output_files


def split_stats(rows: list[dict[str, Any]], output_split: str, id_filename: str) -> dict[str, Any]:
    names = [metadata_name(row) for row in rows]
    question_types = Counter()
    concept_types = {}
    for row in rows:
        metadata = json.loads(row["metadata"])[0]
        question_types[metadata["question_type"]] += 1
        concept_types[metadata["name"]] = metadata["type"]
    return {
        "output_split": output_split,
        "rows": len(rows),
        "concepts": len(set(names)),
        "qa_total": sum(row["qa_count"] for row in rows),
        "qa_per_row_min": min((row["qa_count"] for row in rows), default=0),
        "qa_per_row_max": max((row["qa_count"] for row in rows), default=0),
        "concept_rows_min": min(Counter(names).values()) if names else 0,
        "concept_rows_max": max(Counter(names).values()) if names else 0,
        "type_counts": dict(Counter(concept_types.values())),
        "question_type_counts": dict(question_types),
        "ids_file": id_filename,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--mode', choices=('all', 'finetune', 'unlearn'), default='all')
    parser.add_argument('--ratios', nargs='+', choices=list(SPLIT_MAP),
                        default=list(SPLIT_MAP))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    source_root = args.source_root
    if not list(source_root.glob('finetune-*.parquet')) and list((source_root / 'data').glob('finetune-*.parquet')):
        source_root = source_root / 'data'
    finetune_root = args.output_root / 'finetune/covubench'
    output_root = args.output_root / 'unlearn/covubench'
    source_finetune_rows = read_parquet_glob(str(source_root / 'finetune-*.parquet'))
    finetune_rows = convert_finetune_rows(source_finetune_rows)
    if args.mode in ('all', 'finetune'):
        finetune_files = write_split(finetune_rows, finetune_root, args.overwrite)
        write_json(finetune_root / 'manifest.json', {
            'task': 'vqa_finetune',
            'source_root': str(source_root),
            'rows': len(finetune_rows),
            'concepts': len({row['name'] for row in source_finetune_rows}),
            'output_files': [str(path) for path in finetune_files],
        })
        print(f"finetune: rows={len(finetune_rows)} -> {finetune_root}")
    if args.mode == 'finetune':
        return
    manifest: dict[str, Any] = {
        "task": "covubench",
        "format": "one row with image, ID, metadata, source_split, qa_count; metadata includes question_type and concept name",
        "source_root": str(source_root),
        "finetune_source": str(source_root / "finetune-*.parquet"),
        "splits": {},
    }

    for source_split in args.ratios:
        forget_split, retain_split = SPLIT_MAP[source_split]
        forget_source_rows = read_parquet_glob(str(source_root / f"{source_split}-*.parquet"))
        forget_rows = convert_source_rows(forget_source_rows, forget_split)
        forget_names = sorted({metadata_name(row) for row in forget_rows})
        retain_rows = [row for row in finetune_rows if metadata_name(row) not in set(forget_names)]
        retain_names = sorted({metadata_name(row) for row in retain_rows})

        forget_dir = output_root / forget_split
        retain_dir = output_root / retain_split
        forget_files = write_split(forget_rows, forget_dir, args.overwrite)
        retain_files = write_split(retain_rows, retain_dir, args.overwrite)
        write_json(forget_dir / "forget_ids.json", forget_names)
        write_json(retain_dir / "retain_ids.json", retain_names)

        forget_stats = split_stats(forget_rows, forget_split, "forget_ids.json")
        retain_stats = split_stats(retain_rows, retain_split, "retain_ids.json")
        forget_stats["source_split"] = source_split
        retain_stats["source_split"] = "finetune_minus_" + source_split
        forget_stats["output_files"] = [str(path) for path in forget_files]
        retain_stats["output_files"] = [str(path) for path in retain_files]
        forget_stats["forget_names"] = forget_names
        retain_stats["excluded_forget_names"] = forget_names
        manifest["splits"][forget_split] = forget_stats
        manifest["splits"][retain_split] = retain_stats

        print(
            f"{forget_split}: rows={forget_stats['rows']} concepts={forget_stats['concepts']} "
            f"question_types={forget_stats['question_type_counts']}"
        )
        print(
            f"{retain_split}: rows={retain_stats['rows']} concepts={retain_stats['concepts']} "
            f"excluded={len(forget_names)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare MLLMU training data from the two official sources.

MLLMU-Bench/ft_Data supplies finetuning QA. baseline_train_split supplies
forget/retain training QA, including each split's original images and IDs.
MLLMU-Bench's forget/retain evaluation tasks are not training inputs; setup_data.py
links that benchmark source separately under data/eval/mllmu.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "data/source/mllmu"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
        help="MLLMU-Bench directory, or a parent containing both official source directories.",
    )
    parser.add_argument(
        "--benchmark-source", type=Path, default=None,
        help="Official MLLMU-Bench source containing ft_Data and evaluation sets.",
    )
    parser.add_argument(
        "--baseline-source", type=Path, default=None,
        help="MLLMMU/baseline_train_split source containing forget/retain training rows.",
    )
    parser.add_argument("--finetune-source", type=Path, default=None,
                        help="Override source-root/ft_Data (directory or parquet file).")
    parser.add_argument("--unlearn-source", type=Path, default=None,
                        help="Override --baseline-source (directory of training split directories).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mode", choices=("all", "finetune", "unlearn"), default="all")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Training splits to export. Defaults to forget_*/retain_* under baseline-source.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output split directories.",
    )
    return parser.parse_args()


def parquet_files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return files


def open_parquet(path: Path) -> pq.ParquetFile:
    with path.open("rb") as handle:
        if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"{path} is a Git LFS pointer, not parquet data. Download the actual "
                "files with git lfs pull or setup_data.py before preparing data."
            )
    return pq.ParquetFile(path)


def validate_training_source(path: Path, require_id: bool = True) -> None:
    required = {"image", "metadata"}
    if require_id:
        required.add("ID")
    for file in parquet_files(path):
        missing = required - set(open_parquet(file).schema_arrow.names)
        if missing:
            raise ValueError(
                f"{file} is missing training columns {sorted(missing)}. "
                "Use MLLMU-Bench/ft_Data for finetuning and baseline_train_split "
                "for unlearning, not MLLMU-Bench evaluation splits."
            )


def read_parquet_dir(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in parquet_files(path):
        for batch in open_parquet(file).iter_batches(batch_size=16):
            rows.extend(batch.to_pylist())
    return rows


def write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    schema = pa.schema([
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("ID", pa.string()),
        ("metadata", pa.string()),
        ("source_split", pa.string()),
        ("qa_count", pa.int64()),
    ])
    with pq.ParquetWriter(output_path, schema) as writer:
        for start in range(0, len(rows), 16):
            writer.write_table(pa.Table.from_pylist(rows[start:start + 16], schema=schema))


def parse_metadata(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if isinstance(value, list):
        return value
    raise TypeError(f"Unsupported metadata type: {type(value)}")


def normalize_image(image: dict[str, Any]) -> dict[str, Any]:
    if image.get("bytes") is not None:
        return {"bytes": image["bytes"], "path": None}
    return {"bytes": None, "path": image.get("path")}


def finetune_rows(source: Path) -> list[dict[str, Any]]:
    source_rows = read_parquet_dir(source)
    output_rows = []
    for row_idx, row in enumerate(source_rows):
        row_id = str(row.get("ID", row_idx))
        metadata = []
        for qa_idx, qa in enumerate(parse_metadata(row["metadata"])):
            question = qa.get("Question") or qa.get("Additional_question")
            answer = qa.get("Answer") or qa.get("Additional_answer")
            if question and answer:
                metadata.append({
                    "ID": str(qa.get("ID") or qa.get("Additional_ID") or f"{row_id}:{qa_idx}"),
                    "Question": question,
                    "Answer": answer,
                    "source": "MLLMU",
                    "has_image": True,
                })
        if metadata:
            output_rows.append({
                "image": normalize_image(row["image"]),
                "ID": row_id,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "source_split": "vqa",
                "qa_count": len(metadata),
            })
    return output_rows


def export_finetune(source: Path, output_root: Path, overwrite: bool) -> dict[str, Any]:
    output_rows = finetune_rows(source)
    output_path = output_root / "train-00000-of-00001.parquet"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    write_rows(output_rows, output_path)
    manifest = {
        "task": "vqa_finetune", "source_path": str(source),
        "output_path": str(output_path), "rows": len(output_rows),
        "qa_total": sum(row["qa_count"] for row in output_rows),
    }
    (output_root / "manifest.json").write_text(
        json.dumps({"vqa": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


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


def export_split(
    source_root: Path,
    output_root: Path,
    split: str,
    overwrite: bool,
) -> dict[str, Any]:
    source_dir = source_root / split
    validate_training_source(source_dir)
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
    write_rows(output_rows, split_dir / "train-00000-of-00001.parquet")

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
        "source_path": str(source_dir),
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
        if path.is_dir() and path.name.startswith(("forget_", "retain_"))
        and list(path.glob("*.parquet"))
    )


def main() -> None:
    args = parse_args()
    benchmark_source = args.benchmark_source or args.source_root
    if args.benchmark_source is None and (benchmark_source / "MLLMU-Bench").is_dir():
        benchmark_source = benchmark_source / "MLLMU-Bench"
    finetune_source = args.finetune_source or benchmark_source / "ft_Data"
    unlearn_source = args.unlearn_source or args.baseline_source
    if unlearn_source is None:
        candidates = (
            benchmark_source / "baseline_train_split",
            benchmark_source.parent / "baseline_train_split",
        )
        unlearn_source = next((path for path in candidates if path.is_dir()), candidates[0])

    # Check both input schemas before replacing any prepared training output.
    if args.mode in ("all", "finetune"):
        validate_training_source(finetune_source, require_id=False)
    splits = []
    if args.mode in ("all", "unlearn"):
        if not unlearn_source.is_dir():
            raise FileNotFoundError(
                f"MLLMU training splits are unavailable at {unlearn_source}. "
                "Provide --baseline-source or download MLLMMU/baseline_train_split."
            )
        splits = args.splits or discover_splits(unlearn_source)
        if not splits:
            raise FileNotFoundError(f"No forget/retain training splits found under {unlearn_source}")
        for split in splits:
            validate_training_source(unlearn_source / split)
    if args.mode in ("all", "finetune"):
        stats = export_finetune(
            finetune_source, args.output_root / "finetune/mllmu", args.overwrite
        )
        print(f"finetune: rows={stats['rows']} qa={stats['qa_total']}")
    if args.mode == "finetune":
        return

    manifest = {
        "task": "mllmu_vqa",
        "format": "one row per person with image, ID, metadata",
        "source_root": str(unlearn_source),
        "splits": {},
    }

    for split in splits:
        manifest["splits"][split] = export_split(
            unlearn_source,
            args.output_root / "unlearn/mllmu",
            split,
            args.overwrite,
        )

    unlearn_output = args.output_root / "unlearn/mllmu"
    unlearn_output.mkdir(parents=True, exist_ok=True)
    with (unlearn_output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("MLLMU output:", unlearn_output)
    for split, stats in manifest["splits"].items():
        print(
            f"  {split}: rows={stats['rows']} qa={stats['qa_total']} "
            f"qa_minmax=({stats['qa_per_person_min']}, {stats['qa_per_person_max']})"
        )


if __name__ == "__main__":
    main()

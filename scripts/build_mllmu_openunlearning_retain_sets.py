#!/usr/bin/env python3
"""Create OpenUnlearning-style retain/control datasets for MLLMU.

Generated files:

    retain90.parquet
    retain90_forget10_perturbed.parquet
    retain90_celeb_bio.parquet

The latter two are complete negative-pool training sets, i.e. retain90 plus
either ``forget10_perturbed`` or ``celeb_bio``. The source variant files are
also written as ``forget10_perturbed.parquet`` and ``celeb_bio.parquet`` for
inspection and reproducibility.

All files use the repository's one-row-per-person MLLMU layout. Images are
processed in batches so the script does not load the whole retain set at once.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_ROOT = Path("data/faithfulness/mllmu")
DEFAULT_RETAIN = Path("data/unlearn/mllmu/retain_90/train-00000-of-00001.parquet")
DEFAULT_FORGET = Path("data/unlearn/mllmu/forget_10/train-00000-of-00001.parquet")
DEFAULT_CELEB = DEFAULT_ROOT / "celebrity.parquet"

SCHEMA = pa.schema(
    [
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("ID", pa.string()),
        ("metadata", pa.string()),
        ("source_split", pa.string()),
        ("qa_count", pa.int64()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retain", type=Path, default=DEFAULT_RETAIN)
    parser.add_argument("--forget", type=Path, default=DEFAULT_FORGET)
    parser.add_argument("--celebrity", type=Path, default=DEFAULT_CELEB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--celebrity-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_metadata(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        result = json.loads(value)
        if isinstance(result, list):
            return result
    raise TypeError(f"Unsupported metadata type: {type(value)!r}")


def normalize_image(image: Any) -> dict[str, Any]:
    if isinstance(image, dict):
        return {"bytes": image.get("bytes"), "path": image.get("path")}
    return {"bytes": None, "path": image if isinstance(image, str) else None}


def iter_rows(path: Path, columns: list[str]) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    for file in files:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(columns=columns, batch_size=16):
            yield from batch.to_pylist()


def regular_metadata(value: Any) -> list[dict[str, Any]]:
    """Keep normal Question/Answer pairs and exclude Additional_* summaries."""
    result = []
    for item in parse_metadata(value):
        if item.get("Question") and item.get("Answer"):
            result.append(
                {
                    "ID": str(item.get("ID", "")),
                    "Question": str(item["Question"]),
                    "Answer": str(item["Answer"]),
                    "source": item.get("source", "MLLMU"),
                    "has_image": bool(item.get("has_image", True)),
                }
            )
    return result


def to_row(row: dict[str, Any], metadata: list[dict[str, Any]], split: str) -> dict[str, Any]:
    return {
        "image": normalize_image(row.get("image")),
        "ID": str(row.get("ID")),
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "source_split": split,
        "qa_count": len(metadata),
    }


def write_batch(writer: pq.ParquetWriter, rows: list[dict[str, Any]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))


def load_forget_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in iter_rows(path, ["image", "ID", "metadata"]):
        metadata = regular_metadata(row["metadata"])
        if metadata:
            rows.append({"image": row.get("image"), "ID": str(row.get("ID")), "metadata": metadata})
    return rows


def make_perturbed(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Swap answers between different people at the same QA position."""
    if len(rows) < 2:
        raise ValueError("Need at least two forget rows to create perturbed answers")
    rng = random.Random(seed)
    output = []
    for row_index, row in enumerate(rows):
        metadata = []
        for qa_index, item in enumerate(row["metadata"]):
            candidates = [
                other["metadata"][qa_index]["Answer"]
                for other in rows
                if other is not row and qa_index < len(other["metadata"])
            ]
            if not candidates:
                candidates = [other["Answer"] for other in row["metadata"] if other["Answer"] != item["Answer"]]
            if not candidates:
                raise ValueError(f"Could not perturb answer for {row['ID']} QA {qa_index}")
            wrong_answer = rng.choice(candidates)
            metadata.append({**item, "Answer": wrong_answer, "source": "MLLMU_perturbed"})
        output.append(to_row(row, metadata, "forget10_perturbed"))
    return output


def write_rows(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite")
    writer = pq.ParquetWriter(path, SCHEMA)
    try:
        for start in range(0, len(rows), 16):
            write_batch(writer, rows[start : start + 16])
    finally:
        writer.close()


def write_retain_and_merged(
    retain_path: Path,
    variant_rows: list[dict[str, Any]],
    output_dir: Path,
    merged_name: str,
    overwrite: bool,
) -> tuple[int, int]:
    retain_out = output_dir / "retain90.parquet"
    merged_out = output_dir / merged_name
    for path in (retain_out, merged_out):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite")
    retain_writer = pq.ParquetWriter(retain_out, SCHEMA)
    merged_writer = pq.ParquetWriter(merged_out, SCHEMA)
    retain_count = retain_qa = 0
    try:
        batch = []
        for row in iter_rows(retain_path, ["image", "ID", "metadata", "source_split", "qa_count"]):
            # retain90 is already normalized in this repository. Keep its
            # metadata unchanged so the training protocol remains reproducible.
            batch.append(row)
            if len(batch) >= 16:
                write_batch(retain_writer, batch)
                write_batch(merged_writer, batch)
                retain_count += len(batch)
                retain_qa += sum(int(item["qa_count"]) for item in batch)
                batch = []
        if batch:
            write_batch(retain_writer, batch)
            write_batch(merged_writer, batch)
            retain_count += len(batch)
            retain_qa += sum(int(item["qa_count"]) for item in batch)
        for start in range(0, len(variant_rows), 16):
            write_batch(merged_writer, variant_rows[start : start + 16])
    finally:
        retain_writer.close()
        merged_writer.close()
    return retain_count, retain_qa


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forget_rows = load_forget_rows(args.forget)
    perturbed = make_perturbed(forget_rows, args.seed)
    write_rows(args.output_dir / "forget10_perturbed.parquet", perturbed, args.overwrite)

    celebrity_rows = list(iter_rows(args.celebrity, ["image", "ID", "metadata", "source_split", "qa_count"]))
    if args.celebrity_count < len(celebrity_rows):
        celebrity_rows = random.Random(args.seed).sample(celebrity_rows, args.celebrity_count)
    for row in celebrity_rows:
        row["source_split"] = "celeb_bio"
    write_rows(args.output_dir / "celeb_bio.parquet", celebrity_rows, args.overwrite)

    retain_count, retain_qa = write_retain_and_merged(
        args.retain, perturbed, args.output_dir, "retain90_forget10_perturbed.parquet", args.overwrite
    )
    retain_count_2, retain_qa_2 = write_retain_and_merged(
        args.retain, celebrity_rows, args.output_dir, "retain90_celeb_bio.parquet", True
    )
    if retain_count != retain_count_2 or retain_qa != retain_qa_2:
        raise RuntimeError("Retain output counts changed between merged datasets")

    manifest = {
        "retain90": {"rows": retain_count, "qa_total": retain_qa},
        "forget10_perturbed": {"rows": len(perturbed), "qa_total": sum(r["qa_count"] for r in perturbed)},
        "celeb_bio": {"rows": len(celebrity_rows), "qa_total": sum(r["qa_count"] for r in celebrity_rows)},
        "retain90_forget10_perturbed": {"rows": retain_count + len(perturbed), "qa_total": retain_qa + sum(r["qa_count"] for r in perturbed)},
        "retain90_celeb_bio": {"rows": retain_count + len(celebrity_rows), "qa_total": retain_qa + sum(r["qa_count"] for r in celebrity_rows)},
        "seed": args.seed,
        "sources": {"retain": str(args.retain), "forget": str(args.forget), "celebrity": str(args.celebrity)},
    }
    (args.output_dir / "openunlearning_retain_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

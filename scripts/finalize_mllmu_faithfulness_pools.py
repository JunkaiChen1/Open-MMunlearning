#!/usr/bin/env python3
"""Materialize the six final OpenUnlearning-style MLLMU pool files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_DIR = Path("data/faithfulness/mllmu")
FINAL_FILES = (
    "full.parquet",
    "full_paraphrased.parquet",
    "full_bio.parquet",
    "retain90.parquet",
    "retain90_forget10_perturbed.parquet",
    "retain90_celeb_bio.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def append_file(writer: pq.ParquetWriter, path: Path, schema: pa.Schema) -> tuple[int, int]:
    rows = qa_total = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=16):
        table = pa.Table.from_batches([batch], schema=schema)
        writer.write_table(table)
        rows += batch.num_rows
        qa_total += sum(int(value) for value in table.column("qa_count").to_pylist())
    return rows, qa_total


def merge(output: Path, retain: Path, variant: Path, overwrite: bool) -> dict[str, int]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite")
    schema = pq.ParquetFile(retain).schema_arrow
    writer = pq.ParquetWriter(output, schema)
    try:
        retain_rows, retain_qa = append_file(writer, retain, schema)
        variant_rows, variant_qa = append_file(writer, variant, schema)
    finally:
        writer.close()
    return {
        "rows": retain_rows + variant_rows,
        "qa_total": retain_qa + variant_qa,
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    retain = data_dir / "retain90.parquet"
    required = [retain, data_dir / "forget10.parquet", data_dir / "forget10_paraphrased.parquet", data_dir / "forget10_bio.parquet"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing positive-pool inputs: {missing}")

    stats = {
        "full": merge(data_dir / "full.parquet", retain, data_dir / "forget10.parquet", args.overwrite),
        "full_paraphrased": merge(data_dir / "full_paraphrased.parquet", retain, data_dir / "forget10_paraphrased.parquet", args.overwrite),
        "full_bio": merge(data_dir / "full_bio.parquet", retain, data_dir / "forget10_bio.parquet", args.overwrite),
    }
    for filename in ("retain90.parquet", "retain90_forget10_perturbed.parquet", "retain90_celeb_bio.parquet"):
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        table = pq.read_table(path, columns=["qa_count"])
        stats[filename.removesuffix(".parquet")] = {
            "rows": parquet.metadata.num_rows,
            "qa_total": sum(int(value) for value in table.column("qa_count").to_pylist()),
        }
    print(stats)


if __name__ == "__main__":
    main()

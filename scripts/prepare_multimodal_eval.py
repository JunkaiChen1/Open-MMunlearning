#!/usr/bin/env python3
"""Adapt compact POPE and VizWiz releases to the existing evaluation layouts."""

import argparse
import csv
import json
from pathlib import Path
from zlib import crc32

import pyarrow.parquet as pq


def export_image(image, source_file, output_root):
    content = image.get("bytes")
    if content is None:
        source = Path(image["path"])
        if not source.is_absolute():
            source = source_file.parent / source
        content = source.read_bytes()
    stem = f"image_{crc32(content):08x}_{len(content)}"
    filename = stem + ".jpg"
    target = output_root / "images" / filename
    suffix = 0
    # Compare bytes to keep different images separate even if checksums collide.
    while target.is_file():
        if target.read_bytes() == content:
            return filename
        suffix += 1
        filename = f"{stem}_{suffix}.jpg"
        target = output_root / "images" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".tmp")
    partial.write_bytes(content)
    partial.replace(target)
    return filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("pope", "vizwiz"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    pattern = "Full/random-*.parquet" if args.benchmark == "pope" else "data/val-*.parquet"
    files = sorted(args.source_root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matching {args.source_root / pattern}")
    filename = "coco_pope_random.json" if args.benchmark == "pope" else "VizWiz.tsv"
    target = args.output_root / filename
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"{target} exists; use --overwrite")
    args.output_root.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".tmp")
    count = 0
    image_names = set()
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = None
        if args.benchmark == "vizwiz":
            writer = csv.DictWriter(handle, fieldnames=["index", "image", "question", "answers"], delimiter="\t")
            writer.writeheader()
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=8):
                for row in batch.to_pylist():
                    image_name = export_image(row["image"], file, args.output_root)
                    image_names.add(image_name)
                    if args.benchmark == "pope":
                        record = {
                            "question_id": row["question_id"], "image": image_name,
                            "text": row["question"], "label": row["answer"],
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        answers = row["answers"]
                        if isinstance(answers, str):
                            answers = json.loads(answers)
                        if isinstance(answers, dict):
                            answers = answers["answer"]
                        answers = [item["answer"] if isinstance(item, dict) else str(item) for item in answers]
                        writer.writerow({
                            "index": row.get("question_id", row.get("image_id", count)),
                            "image": image_name, "question": row["question"],
                            "answers": json.dumps(answers, ensure_ascii=False),
                        })
                    count += 1
    if not count:
        raise ValueError(f"No {args.benchmark} evaluation records found")
    partial.replace(target)
    print(f"{args.benchmark}: {count} questions, {len(image_names)} images -> {target}")


if __name__ == "__main__":
    main()

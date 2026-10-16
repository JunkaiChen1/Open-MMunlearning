#!/usr/bin/env python3
"""Convert CLEAR TOFU QA splits to the MLLMU-Bench_distill parquet layout.

The output layout matches MLLMU-Bench_distill/test:

    <output_root>/forget_10/train-00000-of-00001.parquet
    <output_root>/forget_10/forget_ids.json
    <output_root>/retain_90/train-00000-of-00001.parquet
    <output_root>/retain_90/retain_ids.json

Each parquet row has:
    image: {bytes, path}
    ID: string
    metadata: JSON string of [{"ID", "Question", "Answer"}, ...]

CLEAR does not store TOFU QA rows with images. This script binds each person's
TOFU QA group to one representative image through the shared fictitious person
name, matching MLLMU's one-row-per-person layout.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_CLEAR_ROOT = Path("./external_datasets/CLEAR")
DEFAULT_OUTPUT_ROOT = Path(
    "./data/clear_mllmu_vqa/test"
)

SPLIT_MAP = {
    "01": ("forget01", "forget01+tofu", "forget_1", "retain99", "retain99+tofu", "retain_99"),
    "05": ("forget05", "forget05+tofu", "forget_5", "retain95", "retain95+tofu", "retain_95"),
    "10": ("forget10", "forget10+tofu", "forget_10", "retain90", "retain90+tofu", "retain_90"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CLEAR image/TOFU QA splits into MLLMU-style VQA parquet files."
    )
    parser.add_argument("--clear-root", type=Path, default=DEFAULT_CLEAR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["01", "05", "10"],
        choices=sorted(SPLIT_MAP),
        help="CLEAR forget percentages to convert.",
    )
    parser.add_argument(
        "--row-mode",
        choices=["image", "person"],
        default="person",
        help=(
            "image: one row per CLEAR image, preserving all visual samples; "
            "person: one representative image per person, closest to MLLMU identity rows."
        ),
    )
    parser.add_argument(
        "--caption-question",
        default="Describe this image.",
        help="Question used when --include-caption-qa adds each CLEAR caption as a QA pair.",
    )
    parser.add_argument(
        "--include-caption-qa",
        action="store_true",
        help="Also add the representative CLEAR image caption as one QA pair.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split output directories.",
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


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def name_tokens(name: str) -> list[str]:
    return [token for token in normalize_text(name).split() if len(token) >= 3]


def best_name_match(text: str, names: list[str]) -> str | None:
    normalized = normalize_text(text)
    exact = [name for name in names if normalize_text(name) in normalized]
    if len(exact) == 1:
        return exact[0]

    text_tokens = set(normalized.split())
    candidates = []
    for name in names:
        tokens = name_tokens(name)
        if not tokens:
            continue
        hits = [token for token in tokens if token in text_tokens]
        min_hits = 1 if len(tokens) == 1 else 2
        if len(hits) >= min_hits and len(hits) / len(tokens) >= 0.5:
            candidates.append(
                (
                    len(hits),
                    sum(len(token) for token in hits),
                    len(tokens),
                    name,
                )
            )

    if not candidates:
        return None
    candidates.sort(reverse=True)
    best = candidates[0]
    tied = [candidate for candidate in candidates if candidate[:3] == best[:3]]
    if len(tied) == 1:
        return best[3]
    return None


def unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def has_image(image: Any) -> bool:
    return isinstance(image, dict) and (
        image.get("bytes") is not None or image.get("path") is not None
    )


def normalize_image(image: dict[str, Any]) -> dict[str, Any]:
    # Prefer embedded bytes so the converted dataset is self-contained.
    if image.get("bytes") is not None:
        return {"bytes": image["bytes"], "path": None}
    return {"bytes": None, "path": image.get("path")}


def person_id_map(clear_root: Path) -> dict[str, str]:
    full_rows = read_parquet_dir(clear_root / "full")
    names = unique_in_order([row["name"] for row in full_rows if row.get("name")])
    width = max(3, len(str(len(names) - 1)))
    return {name: f"{idx:0{width}d}" for idx, name in enumerate(names)}


def group_images_by_name(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        name = row.get("name")
        image = row.get("image")
        caption = row.get("caption")
        if name and caption and has_image(image):
            grouped[name].append(
                {
                    "image": normalize_image(image),
                    "caption": caption,
                }
            )
    return dict(grouped)


def qa_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    qas = []
    for row in rows:
        question = row.get("question")
        answer = row.get("answer")
        if question is not None and answer is not None:
            qas.append({"question": question, "answer": answer})
    return qas


def match_qa_by_person(
    qas: list[dict[str, str]], names: list[str]
) -> tuple[dict[str, list[dict[str, str]]], int]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    unmatched = 0

    # TOFU contributes 20 QA rows per fictitious person. Matching the whole block
    # handles rows that use a surname, nickname, or shortened name.
    chunk_size = 20
    if len(qas) % chunk_size == 0:
        for start in range(0, len(qas), chunk_size):
            chunk = qas[start : start + chunk_size]
            chunk_text = " ".join(f"{qa['question']} {qa['answer']}" for qa in chunk)
            name = best_name_match(chunk_text, names)
            if name is None:
                unmatched += len(chunk)
                continue
            grouped[name].extend(
                {"Question": qa["question"], "Answer": qa["answer"]} for qa in chunk
            )
        return dict(grouped), unmatched

    for qa in qas:
        name = best_name_match(f"{qa['question']} {qa['answer']}", names)
        if name is None:
            unmatched += 1
            continue
        grouped[name].append({"Question": qa["question"], "Answer": qa["answer"]})

    return dict(grouped), unmatched


def metadata_pairs(
    row_id: str,
    person_qas: list[dict[str, str]],
    caption: str | None,
    caption_question: str,
    add_caption_qa: bool,
) -> str:
    pairs = []
    if add_caption_qa and caption:
        pairs.append({"ID": row_id, "Question": caption_question, "Answer": caption})
    pairs.extend(
        {"ID": row_id, "Question": qa["Question"], "Answer": qa["Answer"]}
        for qa in person_qas
    )
    return json.dumps(pairs, ensure_ascii=False)


def build_rows(
    images_by_name: dict[str, list[dict[str, Any]]],
    qas_by_name: dict[str, list[dict[str, str]]],
    ids_by_name: dict[str, str],
    row_mode: str,
    caption_question: str,
    add_caption_qa: bool,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    output_rows = []
    output_ids = []
    skipped_no_qa = []

    for name, image_items in images_by_name.items():
        person_qas = qas_by_name.get(name, [])
        if not person_qas:
            skipped_no_qa.append(name)
            continue

        base_id = ids_by_name.setdefault(name, f"{len(ids_by_name):03d}")
        selected_items = image_items if row_mode == "image" else image_items[:1]

        for image_idx, image_item in enumerate(selected_items):
            row_id = base_id if row_mode == "person" else f"{base_id}_{image_idx:02d}"
            output_rows.append(
                {
                    "image": image_item["image"],
                    "ID": row_id,
                    "metadata": metadata_pairs(
                        row_id=row_id,
                        person_qas=person_qas,
                        caption=image_item["caption"],
                        caption_question=caption_question,
                        add_caption_qa=add_caption_qa,
                    ),
                }
            )
            output_ids.append(row_id)

    return output_rows, output_ids, skipped_no_qa


def write_split(
    rows: list[dict[str, Any]],
    ids: list[str],
    output_dir: Path,
    ids_filename: str,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite to replace files."
            )
        for file in output_dir.glob("*"):
            if file.is_file():
                file.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            pa.field("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            pa.field("ID", pa.string()),
            pa.field("metadata", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, output_dir / "train-00000-of-00001.parquet")
    (output_dir / ids_filename).write_text(
        json.dumps(ids, ensure_ascii=False, indent=2) + "\n"
    )


def convert_one(
    clear_root: Path,
    image_split: str,
    qa_split: str,
    output_name: str,
    ids_filename: str,
    ids_by_name: dict[str, str],
    args: argparse.Namespace,
) -> None:
    image_rows = read_parquet_dir(clear_root / image_split)
    plus_rows = read_parquet_dir(clear_root / qa_split)

    images_by_name = group_images_by_name(image_rows)
    names = list(images_by_name)
    qas_by_name, unmatched = match_qa_by_person(qa_rows(plus_rows), names)

    rows, ids, skipped_no_qa = build_rows(
        images_by_name=images_by_name,
        qas_by_name=qas_by_name,
        ids_by_name=ids_by_name,
        row_mode=args.row_mode,
        caption_question=args.caption_question,
        add_caption_qa=args.include_caption_qa,
    )
    write_split(
        rows=rows,
        ids=ids,
        output_dir=args.output_root / output_name,
        ids_filename=ids_filename,
        overwrite=args.overwrite,
    )

    print(
        f"{output_name}: rows={len(rows)}, persons={len(names)}, "
        f"unmatched_qa={unmatched}, skipped_persons_without_qa={len(skipped_no_qa)}"
    )
    if skipped_no_qa:
        print("  skipped:", ", ".join(skipped_no_qa[:10]))


def main() -> None:
    args = parse_args()
    ids_by_name = person_id_map(args.clear_root)

    for split in args.splits:
        (
            forget_image,
            forget_qa,
            forget_output,
            retain_image,
            retain_qa,
            retain_output,
        ) = SPLIT_MAP[split]

        convert_one(
            clear_root=args.clear_root,
            image_split=forget_image,
            qa_split=forget_qa,
            output_name=forget_output,
            ids_filename="forget_ids.json",
            ids_by_name=ids_by_name,
            args=args,
        )
        convert_one(
            clear_root=args.clear_root,
            image_split=retain_image,
            qa_split=retain_qa,
            output_name=retain_output,
            ids_filename="retain_ids.json",
            ids_by_name=ids_by_name,
            args=args,
        )


if __name__ == "__main__":
    main()

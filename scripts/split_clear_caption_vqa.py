#!/usr/bin/env python3
"""Split the local CLEAR dataset into caption and VQA training files.

Default outputs:

    data/finetune/clear/<split>/train-00000-of-00001.parquet
    data/unlearn/clear/<split>/train-00000-of-00001.parquet

The caption side is written as one row per image with ``question``/``answer``
fields so it can be consumed by the multimodal QA collator. Its ``ID`` field
has person-level semantics, matching the VQA side. The image-level identifier
is stored separately as ``image_id``.

The VQA side is written in the MLLMU-style layout: one row per person, one
representative image, and a JSON ``metadata`` field containing that person's
QA pairs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_CLEAR_ROOT = Path("./external_datasets/CLEAR")
DEFAULT_CAPTION_OUTPUT_ROOT = Path(
    "./data/finetune/clear"
)
DEFAULT_VQA_OUTPUT_ROOT = Path(
    "./data/unlearn/clear"
)

SPLIT_MAP = {
    "01": {
        "caption": "forget01",
        "caption_perturbed": "forget01_perturbed",
        "vqa": "forget01+tofu",
        "output": "forget_1",
        "id_file": "forget_ids.json",
    },
    "05": {
        "caption": "forget05",
        "caption_perturbed": "forget05_perturbed",
        "vqa": "forget05+tofu",
        "output": "forget_5",
        "id_file": "forget_ids.json",
    },
    "10": {
        "caption": "forget10",
        "caption_perturbed": "forget10_perturbed",
        "vqa": "forget10+tofu",
        "output": "forget_10",
        "id_file": "forget_ids.json",
    },
    "90": {
        "caption": "retain90",
        "vqa": "retain90+tofu",
        "output": "retain_90",
        "id_file": "retain_ids.json",
    },
    "95": {
        "caption": "retain95",
        "vqa": "retain95+tofu",
        "output": "retain_95",
        "id_file": "retain_ids.json",
    },
    "99": {
        "caption": "retain99",
        "vqa": "retain99+tofu",
        "output": "retain_99",
        "id_file": "retain_ids.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split CLEAR into caption and VQA parquet directories."
    )
    parser.add_argument("--clear-root", type=Path, default=DEFAULT_CLEAR_ROOT)
    parser.add_argument(
        "--caption-output-root", type=Path, default=DEFAULT_CAPTION_OUTPUT_ROOT
    )
    parser.add_argument("--vqa-output-root", type=Path, default=DEFAULT_VQA_OUTPUT_ROOT)
    parser.add_argument(
        "--caption-question",
        default="Describe this image.",
        help="Prompt used for image-caption rows.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["01", "05", "10", "90", "95", "99"],
        choices=sorted(SPLIT_MAP),
        help="CLEAR forget/retain splits to export.",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Do not export the full/full+tofu splits.",
    )
    parser.add_argument(
        "--skip-perturbed",
        action="store_true",
        help="Do not export CLEAR perturbed caption eval splits.",
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


def normalize_image(image: dict[str, Any]) -> dict[str, Any]:
    if image.get("bytes") is not None:
        return {"bytes": image["bytes"], "path": None}
    return {"bytes": None, "path": image.get("path")}


def has_image(image: Any) -> bool:
    return isinstance(image, dict) and (
        image.get("bytes") is not None or image.get("path") is not None
    )


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def name_tokens(name: str) -> list[str]:
    return [token for token in normalize_text(name).split() if len(token) >= 3]


def unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def person_id_map(clear_root: Path) -> dict[str, str]:
    full_rows = read_parquet_dir(clear_root / "full")
    names = unique_in_order([row["name"] for row in full_rows if row.get("name")])
    width = max(3, len(str(len(names) - 1)))
    return {name: f"{idx:0{width}d}" for idx, name in enumerate(names)}


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
            candidates.append((len(hits), sum(len(token) for token in hits), name))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    best = candidates[0]
    tied = [candidate for candidate in candidates if candidate[:2] == best[:2]]
    if len(tied) == 1:
        return best[2]
    return None


def group_images_by_name(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image = row.get("image")
        name = row.get("name")
        caption = row.get("caption")
        if name and caption and has_image(image):
            grouped[name].append(
                {
                    "image": normalize_image(image),
                    "caption": caption,
                }
            )
    return dict(grouped)


def extract_qa_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    qas = []
    for row in rows:
        question = row.get("question")
        answer = row.get("answer")
        if question and answer:
            qas.append({"question": question, "answer": answer})
    return qas


def match_qa_by_person(
    qas: list[dict[str, str]], names: list[str]
) -> tuple[dict[str, list[dict[str, str]]], int]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    unmatched = 0
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


def write_rows(rows: list[dict[str, Any]], output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_dir / "train-00000-of-00001.parquet")


def write_ids(output_dir: Path, filename: str, ids: list[str]) -> None:
    with (output_dir / filename).open("w", encoding="utf-8") as handle:
        json.dump(ids, handle, ensure_ascii=False, indent=2)


def build_caption_rows(
    rows: list[dict[str, Any]],
    id_by_name: dict[str, str],
    source_split: str,
    caption_question: str,
) -> list[dict[str, Any]]:
    counters: dict[str, int] = defaultdict(int)
    output_rows = []
    for source_index, row in enumerate(rows):
        image = row.get("image")
        caption = row.get("caption")
        name = row.get("name")
        if not caption or not has_image(image):
            continue

        person_id = id_by_name.get(name, "unknown")
        image_index = counters[person_id]
        counters[person_id] += 1
        image_id = f"{person_id}_{image_index:02d}"

        output_row = {
            "image": normalize_image(image),
            "ID": person_id,
            "image_id": image_id,
            "name": name,
            "question": caption_question,
            "answer": caption,
            "source_split": source_split,
            "source_index": source_index,
        }
        if row.get("paraphrased_caption"):
            output_row["paraphrased_answer"] = row["paraphrased_caption"]
        if row.get("perturbed_captions"):
            output_row["perturbed_answers"] = row["perturbed_captions"]
        if row.get("perturbed_names"):
            output_row["perturbed_names"] = row["perturbed_names"]
        output_rows.append(output_row)
    return output_rows


def build_vqa_rows(
    caption_rows: list[dict[str, Any]],
    mixed_rows: list[dict[str, Any]],
    id_by_name: dict[str, str],
    source_split: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    images_by_name = group_images_by_name(caption_rows)
    names = list(images_by_name.keys())
    qas = extract_qa_rows(mixed_rows)
    qa_by_name, unmatched_qa = match_qa_by_person(qas, names)

    output_rows = []
    skipped_persons_without_qa = 0
    for name, images in images_by_name.items():
        qa_pairs = qa_by_name.get(name, [])
        if not qa_pairs:
            skipped_persons_without_qa += 1
            continue
        person_id = id_by_name.get(name, "unknown")
        metadata = [
            {
                "ID": person_id,
                "Question": qa_pair["Question"],
                "Answer": qa_pair["Answer"],
            }
            for qa_pair in qa_pairs
        ]
        output_rows.append(
            {
                "image": images[0]["image"],
                "ID": person_id,
                "name": name,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "source_split": source_split,
                "qa_count": len(metadata),
            }
        )

    stats = {
        "source_caption_rows": len(caption_rows),
        "source_qa_rows": len(qas),
        "output_person_rows": len(output_rows),
        "output_qa_pairs": sum(row["qa_count"] for row in output_rows),
        "unmatched_qa_rows": unmatched_qa,
        "skipped_persons_without_qa": skipped_persons_without_qa,
    }
    return output_rows, stats


def export_caption_split(
    clear_root: Path,
    split_name: str,
    output_name: str,
    output_root: Path,
    id_by_name: dict[str, str],
    caption_question: str,
    overwrite: bool,
    id_file: str | None = None,
) -> dict[str, Any]:
    rows = read_parquet_dir(clear_root / split_name)
    output_rows = build_caption_rows(rows, id_by_name, split_name, caption_question)
    output_dir = output_root / output_name
    write_rows(output_rows, output_dir, overwrite)

    ids = unique_in_order([row["ID"] for row in output_rows])
    if id_file:
        write_ids(output_dir, id_file, ids)
    else:
        write_ids(output_dir, "ids.json", ids)

    return {
        "source_split": split_name,
        "output_split": output_name,
        "rows": len(output_rows),
        "persons": len(ids),
    }


def export_vqa_split(
    clear_root: Path,
    caption_split: str,
    mixed_split: str,
    output_name: str,
    output_root: Path,
    id_by_name: dict[str, str],
    overwrite: bool,
    id_file: str | None = None,
) -> dict[str, Any]:
    caption_rows = read_parquet_dir(clear_root / caption_split)
    mixed_rows = read_parquet_dir(clear_root / mixed_split)
    output_rows, stats = build_vqa_rows(
        caption_rows, mixed_rows, id_by_name, mixed_split
    )
    output_dir = output_root / output_name
    write_rows(output_rows, output_dir, overwrite)

    ids = unique_in_order([row["ID"] for row in output_rows])
    if id_file:
        write_ids(output_dir, id_file, ids)
    else:
        write_ids(output_dir, "ids.json", ids)

    return {
        "source_caption_split": caption_split,
        "source_vqa_split": mixed_split,
        "output_split": output_name,
        **stats,
    }


def write_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    id_by_name = person_id_map(args.clear_root)

    caption_manifest: dict[str, Any] = {
        "task": "caption",
        "format": "one row per image with image, question, answer",
        "caption_question": args.caption_question,
        "splits": {},
    }
    vqa_manifest: dict[str, Any] = {
        "task": "vqa",
        "format": "one row per person with image, ID, metadata",
        "splits": {},
    }

    if not args.skip_full:
        caption_manifest["splits"]["full"] = export_caption_split(
            args.clear_root,
            "full",
            "full",
            args.caption_output_root,
            id_by_name,
            args.caption_question,
            args.overwrite,
        )
        vqa_manifest["splits"]["full"] = export_vqa_split(
            args.clear_root,
            "full",
            "full+tofu",
            "full",
            args.vqa_output_root,
            id_by_name,
            args.overwrite,
        )

    for key in args.splits:
        split = SPLIT_MAP[key]
        output_name = split["output"]
        caption_manifest["splits"][output_name] = export_caption_split(
            args.clear_root,
            split["caption"],
            output_name,
            args.caption_output_root,
            id_by_name,
            args.caption_question,
            args.overwrite,
            split["id_file"],
        )
        vqa_manifest["splits"][output_name] = export_vqa_split(
            args.clear_root,
            split["caption"],
            split["vqa"],
            output_name,
            args.vqa_output_root,
            id_by_name,
            args.overwrite,
            split["id_file"],
        )

        perturbed_split = split.get("caption_perturbed")
        if perturbed_split and not args.skip_perturbed:
            perturbed_output = f"{output_name}_perturbed"
            caption_manifest["splits"][perturbed_output] = export_caption_split(
                args.clear_root,
                perturbed_split,
                perturbed_output,
                args.caption_output_root,
                id_by_name,
                args.caption_question,
                args.overwrite,
                split["id_file"],
            )

    if not args.skip_perturbed:
        caption_manifest["splits"]["retain_perturbed"] = export_caption_split(
            args.clear_root,
            "retain_perturbed",
            "retain_perturbed",
            args.caption_output_root,
            id_by_name,
            args.caption_question,
            args.overwrite,
            "retain_ids.json",
        )

    write_manifest(args.caption_output_root, caption_manifest)
    write_manifest(args.vqa_output_root, vqa_manifest)

    print("Caption output:", args.caption_output_root)
    for name, stats in caption_manifest["splits"].items():
        print(f"  {name}: rows={stats['rows']} persons={stats['persons']}")

    print("VQA output:", args.vqa_output_root)
    for name, stats in vqa_manifest["splits"].items():
        print(
            f"  {name}: rows={stats['output_person_rows']} "
            f"qa={stats['output_qa_pairs']} "
            f"unmatched_qa={stats['unmatched_qa_rows']}"
        )


if __name__ == "__main__":
    main()

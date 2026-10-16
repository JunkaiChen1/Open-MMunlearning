#!/usr/bin/env python3
"""Prebuild CoVUBench parquet caches before launching parallel training."""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "unlearn" / "covubench"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    splits = [
        "forget_5",
        "forget_10",
        "forget_15",
        "forget_20",
        "retain_95",
        "retain_90",
        "retain_85",
        "retain_80",
    ]
    for split in splits:
        pattern = str(args.data_root / split / "train-*.parquet")
        dataset = datasets.load_dataset(
            "parquet",
            data_files=pattern,
            split="train",
            cache_dir=str(args.cache_dir),
        )
        print(f"{split}: {len(dataset)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def download_idk_data():
    data_dir = Path(os.environ.get("OPEN_UNLEARNING_DATA_DIR", "data"))
    snapshot_download(
        repo_id="open-unlearning/idk",
        allow_patterns="*.jsonl",
        repo_type="dataset",
        local_dir=str(data_dir),
    )


def main():
    parser = argparse.ArgumentParser(description="Download shared auxiliary data.")
    parser.add_argument(
        "--idk",
        action="store_true",
        help="Download IDK auxiliary data into OPEN_UNLEARNING_DATA_DIR (or ./data)",
    )
    args = parser.parse_args()

    if args.idk:
        download_idk_data()


if __name__ == "__main__":
    main()

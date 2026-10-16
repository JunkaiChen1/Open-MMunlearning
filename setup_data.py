"""Download benchmark sources and prepare the configured local data layouts."""

import argparse
import filecmp
import glob
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs/data/download.yaml"


def extract_archive(archive, destination, overwrite=False):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as handle:
        for entry in handle.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive member escapes its destination: {entry.filename}")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not overwrite and target.is_file() and target.stat().st_size == entry.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def link_path(source_pattern, destination, parent=False, overwrite=False):
    matches = sorted(glob.glob(source_pattern, recursive=True))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source for {source_pattern}, found {len(matches)}")
    source = Path(matches[0]).resolve()
    if parent:
        source = source.parent
    target = Path(destination)
    if target.resolve() == source:
        return
    if target.is_symlink() and overwrite:
        target.unlink()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists and differs from {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def download_http(source):
    output = Path(source["output"])
    marker = output.with_name(output.name + ".download.json")
    expected_size = source.get("size")

    def validate(path, size):
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if size is not None and path.stat().st_size != size:
            return False
        if output.suffix == ".zip" and not zipfile.is_zipfile(path):
            return False
        return True

    previous = json.loads(marker.read_text()) if marker.is_file() else {}
    if expected_size is None and previous.get("url") == source["url"]:
        expected_size = previous.get("size")
    if expected_size is not None and validate(output, expected_size):
        print(f"Reusing download: {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    print(f"Downloading {source['url']} -> {output}", flush=True)
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            if expected_size is None or offset != expected_size or not validate(partial, expected_size):
                request = urllib.request.Request(source["url"], headers={
                    "User-Agent": "open-mmunlearning/0.1", "Accept-Encoding": "identity",
                    **({"Range": f"bytes={offset}-"} if offset else {}),
                })
                with urllib.request.urlopen(request, timeout=60) as response:
                    if "text/html" in response.headers.get("Content-Type", "").lower():
                        raise ValueError(f"Expected a data file, received HTML: {source['url']}")
                    if response.status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not match or int(match[1]) != offset:
                            raise ValueError(f"Invalid download range for {output}")
                        total = int(match[3])
                    else:
                        offset = 0
                        length = response.headers.get("Content-Length")
                        total = int(length) if length is not None else None
                    if expected_size is not None and total is not None and total != expected_size:
                        raise ValueError(f"Unexpected download size for {output}: {total} != {expected_size}")
                    expected_size = expected_size if expected_size is not None else total
                    with partial.open("ab" if offset else "wb") as handle:
                        shutil.copyfileobj(response, handle, length=1024 * 1024)
            if not validate(partial, expected_size):
                raise ValueError(f"Incomplete or invalid download: {partial}")
            partial.replace(output)
            pending_marker = marker.with_suffix(".tmp")
            pending_marker.write_text(json.dumps({"url": source["url"], "size": output.stat().st_size}) + "\n")
            pending_marker.replace(marker)
            return
        except (OSError, urllib.error.URLError, ValueError) as error:
            if attempt == 2:
                raise
            print(f"Retrying download ({attempt + 1}/2): {error}", flush=True)
            if partial.is_file() and expected_size is not None and partial.stat().st_size >= expected_size:
                partial.unlink()
            time.sleep(attempt + 1)


def download_source(source, max_workers):
    if source["kind"] == "huggingface":
        snapshot_download(
            repo_id=source["repo_id"],
            repo_type=source.get("repo_type", "dataset"),
            revision=source.get("revision", "main"),
            allow_patterns=source["allow_patterns"],
            ignore_patterns=source.get("ignore_patterns"),
            local_dir=source["local_dir"],
            max_workers=max_workers,
        )
    elif source["kind"] == "http":
        download_http(source)
    elif source["kind"] == "gdrive":
        import gdown

        output = Path(source["output"])
        if output.is_file() and zipfile.is_zipfile(output):
            print(f"Reusing archive: {output}", flush=True)
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".part")
        result = gdown.download(id=source["file_id"], output=str(partial), resume=True)
        if result is None or not zipfile.is_zipfile(partial):
            raise RuntimeError(f"Could not download a valid ZIP archive: {output}")
        partial.replace(output)
    else:
        raise ValueError(f"Unknown download kind: {source['kind']}")


def require_files(patterns):
    missing = [pattern for pattern in patterns if not glob.glob(pattern)]
    if missing:
        raise FileNotFoundError("Missing data files: " + ", ".join(missing))


def prepare_dataset(name, config, args, data_root):
    if args.dry_run:
        print(json.dumps({name: config}, indent=2))
        return

    if not args.prepare_only:
        for source in config.get("sources", []):
            download_source(source, args.max_workers)
    for archive in config.get("archives", []):
        if args.prepare_only and not Path(archive["path"]).is_file():
            continue
        extract_archive(archive["path"], archive["destination"], args.overwrite)
    for link in config.get("links", []):
        link_path(**link, overwrite=args.overwrite)
    require_files(config.get("required_sources", []))

    command = config.get("prepare", [])
    if command and not args.download_only:
        script = REPO_ROOT / command[0]
        preparation_record = {"config": config}
        marker = data_root / ".setup" / f"{name}.json"
        previous = json.loads(marker.read_text()) if marker.is_file() else None
        ready = False
        if previous == preparation_record and not args.overwrite:
            try:
                require_files(config["required_outputs"])
                ready = True
            except FileNotFoundError:
                pass
        if ready:
            print(f"Reusing prepared {name} data.", flush=True)
        else:
            invocation = [sys.executable, str(script), *command[1:]]
            if args.overwrite:
                invocation.append("--overwrite")
            print(shlex.join(invocation), flush=True)
            subprocess.run(invocation, cwd=REPO_ROOT, check=True)
            require_files(config["required_outputs"])
            marker.parent.mkdir(parents=True, exist_ok=True)
            partial = marker.with_suffix(".tmp")
            partial.write_text(json.dumps(preparation_record, indent=2) + "\n")
            partial.replace(marker)

    if name == "mllmu" and args.mllmu_truth_ratio:
        target = data_root / "eval/mllmu/truth_ratio/full_set_generation_variants.jsonl"
        source = args.mllmu_truth_ratio.resolve()
        if source != target.resolve():
            if target.exists() and not args.overwrite:
                if not filecmp.cmp(source, target, shallow=False):
                    raise FileExistsError(f"{target} exists; use --overwrite to replace it")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    print(f"{name}: {'downloaded' if args.download_only else 'ready'}", flush=True)
    for optional in config.get("optional_files", []):
        if not Path(optional["path"]).is_file():
            print(f"Additional input needed: {optional['description']}", flush=True)


def main():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = preliminary.parse_known_args()
    config = OmegaConf.load(known.config)
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    for name in config.datasets:
        parser.add_argument(f"--{name}", action="store_true", help=f"Set up {name} data")
    parser.add_argument("--all", action="store_true", help="Set up every benchmark (excludes auxiliary IDK)")
    parser.add_argument("--eval-data", action="store_true", help="Set up all seven general multimodal evaluation datasets")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without network or filesystem writes")
    parser.add_argument("--overwrite", action="store_true", help="Replace converted outputs and managed links")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true", help="Download, extract, and link sources without conversion")
    mode.add_argument("--prepare-only", action="store_true", help="Use already downloaded sources without network access")
    parser.add_argument("--mllmu-truth-ratio", type=Path, help="Existing MLLMU Truth Ratio JSONL sidecar to install")
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    names = [
        name for name, dataset in config.datasets.items()
        if getattr(args, name) or (args.all and name != "idk")
        or (args.eval_data and dataset.get("group") == "evaluation")
    ]
    if not names:
        parser.error("Select at least one dataset, for example --mllmu or --fiubench")
    if args.mllmu_truth_ratio and "mllmu" not in names:
        parser.error("--mllmu-truth-ratio requires --mllmu or --all")
    if args.mllmu_truth_ratio and not args.mllmu_truth_ratio.is_file():
        parser.error(f"Truth Ratio sidecar not found: {args.mllmu_truth_ratio}")
    config.data_root = str(args.data_root.resolve())
    resolved = OmegaConf.to_container(config, resolve=True)
    for name in names:
        prepare_dataset(name, resolved["datasets"][name], args, args.data_root.resolve())


if __name__ == "__main__":
    main()

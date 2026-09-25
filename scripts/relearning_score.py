#!/usr/bin/env python3
"""Compare evaluator summaries from before and after a single-checkpoint ReLearn run.

This script reports raw per-metric changes only.  It does not calculate the
OpenUnlearning ReLearn R score because that score requires a retained-reference
model, which the one-checkpoint runner intentionally does not use.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before",
        required=True,
        type=Path,
        help="Evaluator *_SUMMARY.json file or directory for the unlearned checkpoint.",
    )
    parser.add_argument(
        "--after",
        required=True,
        type=Path,
        help="Evaluator *_SUMMARY.json file or directory for the relearned checkpoint.",
    )
    parser.add_argument(
        "--metric-key",
        action="append",
        required=True,
        metavar="SUMMARY_FILE:PATH",
        help=(
            "Numeric metric to compare; may be repeated. Example: "
            "FIUBENCH_SUMMARY.json:eval_forget_log.json/rougeL_recall/mean"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for the comparison JSON. The result is always printed.",
    )
    return parser.parse_args()


def _read_summaries(source: Path) -> Dict[str, Any]:
    source = source.expanduser().resolve()
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.rglob("*_SUMMARY.json"))
    else:
        raise FileNotFoundError(f"Summary file or directory does not exist: {source}")

    if not paths:
        raise FileNotFoundError(f"No *_SUMMARY.json files found under: {source}")

    summaries: Dict[str, Any] = {}
    for path in paths:
        name = path.name if source.is_file() else str(path.relative_to(source))
        with path.open(encoding="utf-8") as handle:
            summaries[name] = json.load(handle)
    return summaries


def _find_summary(summaries: Dict[str, Any], requested_name: str) -> Tuple[str, Any]:
    matches = [
        (name, value)
        for name, value in summaries.items()
        if name == requested_name or Path(name).name == requested_name
    ]
    if len(matches) != 1:
        available = ", ".join(sorted(summaries)) or "(none)"
        raise KeyError(
            f"Expected exactly one summary named '{requested_name}', found "
            f"{len(matches)}. Available: {available}"
        )
    return matches[0]


def _lookup_path(value: Any, path: str) -> Any:
    # Multimodal evaluators flatten some keys with '/', whereas other summary
    # writers produce nested dictionaries. Support both representations.
    if isinstance(value, dict) and path in value:
        return value[path]
    current = value
    for part in filter(None, path.replace(".", "/").split("/")):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(part)
        current = current[part]
    return current


def _metric_changes(
    before_summaries: Dict[str, Any],
    after_summaries: Dict[str, Any],
    metric_keys: Iterable[str],
) -> List[Dict[str, Any]]:
    changes = []
    for metric_key in metric_keys:
        if ":" not in metric_key:
            raise ValueError(
                "--metric-key must use SUMMARY_FILE:PATH, for example "
                "FIUBENCH_SUMMARY.json:eval_forget_log.json/rougeL_recall/mean."
            )
        summary_name, metric_path = metric_key.split(":", 1)
        before_name, before_summary = _find_summary(before_summaries, summary_name)
        after_name, after_summary = _find_summary(after_summaries, summary_name)
        try:
            before = _lookup_path(before_summary, metric_path)
            after = _lookup_path(after_summary, metric_path)
        except KeyError as error:
            raise KeyError(f"Metric '{metric_key}' was not found: {error}") from error
        if (
            isinstance(before, bool)
            or isinstance(after, bool)
            or not isinstance(before, (int, float))
            or not isinstance(after, (int, float))
        ):
            raise ValueError(f"Metric '{metric_key}' must resolve to numeric scalars.")

        absolute_change = after - before
        relative_change = absolute_change / abs(before) if before != 0 else None
        changes.append(
            {
                "metric_key": metric_key,
                "before_summary": before_name,
                "after_summary": after_name,
                "before": before,
                "after": after,
                "absolute_change": absolute_change,
                "relative_change": relative_change,
            }
        )
    return changes


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing score file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = _parse_args()
    before_summaries = _read_summaries(args.before)
    after_summaries = _read_summaries(args.after)
    result = {
        "protocol": "single-checkpoint ReLearn metric changes",
        "paper_relearn_score_available": False,
        "paper_relearn_score_note": (
            "OpenUnlearning R requires the retained-reference model's metric. "
            "This comparison intentionally reports unlearned-to-relearned changes only."
        ),
        "before": str(args.before.expanduser().resolve()),
        "after": str(args.after.expanduser().resolve()),
        "metric_changes": _metric_changes(
            before_summaries, after_summaries, args.metric_key
        ),
    }
    if args.output_json is not None:
        _write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

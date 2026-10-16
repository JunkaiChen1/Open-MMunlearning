#!/usr/bin/env python3
"""Run OpenUnlearning evaluations before and after bitsandbytes quantization.

The OpenUnlearning paper treats 4-bit quantization as a post-unlearning
stress-test intervention. This script therefore evaluates the same checkpoint
twice instead of producing a separately serialized quantized checkpoint.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        help="Hydra experiment, for example eval/fiubench/default.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model config name from configs/model, without .yaml.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory where both evaluations and the comparison are written.",
    )
    parser.add_argument(
        "--model-path",
        help="Optional checkpoint path overriding model.model_args.pretrained_model_name_or_path.",
    )
    parser.add_argument(
        "--adapter-path",
        help="Optional LoRA adapter path. The base model remains the model config path.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run src/eval.py (default: this interpreter).",
    )
    parser.add_argument(
        "--gpu",
        help="CUDA_VISIBLE_DEVICES value for both runs, for example 0 or 0,1.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map for both runs (default: auto).",
    )
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--double-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable nested/double quantization for 4-bit loading (default: enabled).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Only run the quantized evaluation. No robustness ratio is calculated.",
    )
    parser.add_argument(
        "--score-key",
        action="append",
        default=[],
        metavar="SUMMARY_FILE:PATH",
        help=(
            "A higher-is-better scalar used for Q=min(quantized/baseline, 1). "
            "May be repeated, e.g. FIUBENCH_SUMMARY.json:retain/ROUGE."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override applied to both evaluations; may be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest and print commands without starting evaluation.",
    )
    return parser.parse_args()


def _quantization_overrides(args: argparse.Namespace, enabled: bool) -> List[str]:
    overrides = [f"+model.quantization.enabled={'true' if enabled else 'false'}"]
    if not enabled:
        return overrides
    overrides.extend(
        [
            f"+model.quantization.bits={args.bits}",
            f"+model.quantization.device_map={args.device_map}",
        ]
    )
    if args.bits == 4:
        overrides.extend(
            [
                f"+model.quantization.quant_type={args.quant_type}",
                f"+model.quantization.compute_dtype={args.compute_dtype}",
                "+model.quantization.use_double_quant="
                + ("true" if args.double_quant else "false"),
            ]
        )
    return overrides


def _command(
    args: argparse.Namespace, output_dir: Path, task_name: str, quantized: bool
) -> List[str]:
    command = [
        args.python,
        "src/eval.py",
        "--config-name=eval.yaml",
        f"experiment={args.experiment}",
        f"model={args.model}",
        f"task_name={task_name}",
        f"paths.output_dir={output_dir}",
        f"model.model_args.device_map={args.device_map}",
    ]
    if args.model_path:
        command.append(
            "model.model_args.pretrained_model_name_or_path=" + args.model_path
        )
    if args.adapter_path:
        command.append("+model.adapter_path=" + args.adapter_path)
    command.extend(_quantization_overrides(args, enabled=quantized))
    command.extend(args.override)
    return command


def _read_summaries(output_dir: Path) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for summary_path in sorted(output_dir.rglob("*_SUMMARY.json")):
        relative_path = str(summary_path.relative_to(output_dir))
        with summary_path.open(encoding="utf-8") as handle:
            summaries[relative_path] = json.load(handle)
    return summaries


def _lookup_path(value: Any, path: str) -> Any:
    # The generic evaluator can emit nested dictionaries, while the
    # multimodal evaluators intentionally flatten summary keys with '/'.
    if isinstance(value, dict) and path in value:
        return value[path]
    current = value
    for part in filter(None, path.replace(".", "/").split("/")):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(part)
        current = current[part]
    return current


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


def _robustness(
    baseline: Dict[str, Any], quantized: Dict[str, Any], score_keys: Iterable[str]
) -> List[Dict[str, Any]]:
    results = []
    for score_key in score_keys:
        if ":" not in score_key:
            raise ValueError(
                "--score-key must use SUMMARY_FILE:PATH, for example "
                "MLLMU_SUMMARY.json:retain_shared/generation/ROUGE."
            )
        summary_name, metric_path = score_key.split(":", 1)
        baseline_name, baseline_summary = _find_summary(baseline, summary_name)
        quantized_name, quantized_summary = _find_summary(quantized, summary_name)
        try:
            before = _lookup_path(baseline_summary, metric_path)
            after = _lookup_path(quantized_summary, metric_path)
        except KeyError as error:
            raise KeyError(f"Metric '{score_key}' was not found: {error}") from error
        if isinstance(before, bool) or isinstance(after, bool):
            raise ValueError(f"Metric '{score_key}' must be numeric, not boolean.")
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            raise ValueError(f"Metric '{score_key}' must resolve to numeric scalars.")
        if before <= 0:
            raise ValueError(
                f"Metric '{score_key}' has baseline={before}; Q=min(after/before, 1) "
                "requires a positive baseline."
            )
        results.append(
            {
                "score_key": score_key,
                "baseline_summary": baseline_name,
                "quantized_summary": quantized_name,
                "baseline": before,
                "quantized": after,
                "quantization_robustness": min(after / before, 1.0),
            }
        )
    return results


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _run(command: List[str], log_path: Path, env: Dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix results into non-empty directory: {output_dir}. "
            "Choose a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = output_dir / "baseline"
    quantized_dir = output_dir / f"bnb_{args.bits}bit"
    commands: Dict[str, List[str]] = {}
    if not args.skip_baseline:
        commands["baseline"] = _command(
            args, baseline_dir, "quantization_stress_baseline", quantized=False
        )
    commands["quantized"] = _command(
        args, quantized_dir, f"quantization_stress_bnb_{args.bits}bit", quantized=True
    )

    manifest = {
        "protocol": "OpenUnlearning quantization stress test",
        "paper_formula": "Q = min(metric_after_quantization / metric_before_quantization, 1)",
        "quantization": {
            "backend": "bitsandbytes",
            "bits": args.bits,
            "quant_type": args.quant_type if args.bits == 4 else None,
            "compute_dtype": args.compute_dtype if args.bits == 4 else None,
            "use_double_quant": args.double_quant if args.bits == 4 else None,
            "device_map": args.device_map,
        },
        "commands": commands,
        "score_keys": args.score_key,
    }
    _write_json(output_dir / "manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for run_name, command in commands.items():
        print(f"Running {run_name}: {' '.join(command)}", flush=True)
        _run(command, output_dir / "logs" / f"{run_name}.log", env)

    quantized_summaries = _read_summaries(quantized_dir)
    result: Dict[str, Any] = {
        **manifest,
        "baseline_summaries": _read_summaries(baseline_dir)
        if not args.skip_baseline
        else None,
        "quantized_summaries": quantized_summaries,
        "robustness": [],
    }
    if not args.skip_baseline and args.score_key:
        result["robustness"] = _robustness(
            result["baseline_summaries"], quantized_summaries, args.score_key
        )
    _write_json(output_dir / "quantization_comparison.json", result)
    print(f"Results written to {output_dir / 'quantization_comparison.json'}")


if __name__ == "__main__":
    main()

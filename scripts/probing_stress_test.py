#!/usr/bin/env python3
"""Train and evaluate decoder probes for one unlearned multimodal checkpoint.

For every requested decoder depth, this runner truncates the language decoder,
freezes the retained model (including the visual path), trains only a fresh
output head on the benchmark forget set, and optionally evaluates the resulting
probe with an existing OpenUnlearning evaluation suite.
"""

import argparse
from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset_config: str
    default_split: str
    forget_pattern: str
    eval_experiment: Optional[str]


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "fiubench": BenchmarkSpec(
        "FIUBENCH_QA_forget",
        "forget_10",
        "data/unlearn/fiubench/{split}/train-00000-of-00001.parquet",
        "eval/fiubench/default",
    ),
    "mllmu": BenchmarkSpec(
        "MLLMU_QA_forget",
        "forget_10",
        "data/unlearn/mllmu/{split}/train-00000-of-00001.parquet",
        "eval/mllmubench/default",
    ),
    "clear": BenchmarkSpec(
        "CLEAR_FULL_QA_ft",
        "forget_10",
        "data/unlearn/clear/{split}/train-00000-of-00001.parquet",
        "eval/clear/default",
    ),
    "covubench": BenchmarkSpec(
        "COVUBENCH_QA_forget",
        "forget_5",
        "data/unlearn/covubench/{split}/train-*.parquet",
        "eval/covubench/default",
    ),
    "umubench": BenchmarkSpec(
        "UMUBENCH_QA_forget",
        "forget_10",
        "data/unlearn/umu/{split}/train-00000-of-00001.parquet",
        None,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=tuple(BENCHMARKS))
    parser.add_argument(
        "--model",
        required=True,
        help="A probed-* model config name from configs/model, without .yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="A full unlearned model or a local LoRA adapter directory.",
    )
    parser.add_argument(
        "--layers",
        required=True,
        nargs="+",
        type=int,
        help="Decoder depths to probe, for example --layers 8 16 24.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--forget-split")
    parser.add_argument(
        "--forget-data",
        help="Explicit forget parquet path or glob; overrides --forget-split.",
    )
    parser.add_argument(
        "--metadata-source",
        action="append",
        default=[],
        help="Restrict metadata QA sources, for example MM_QA; may be repeated.",
    )
    parser.add_argument(
        "--head-model-path",
        help="Matching model checkpoint used to initialize each output head.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-experiment",
        help=(
            "Hydra eval experiment; defaults to the benchmark evaluator when "
            "available."
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train and save probes without invoking src/eval.py.",
    )
    parser.add_argument(
        "--score-key",
        action="append",
        default=[],
        metavar="SUMMARY_FILE:PATH",
        help="Numeric summary metric to collect for every layer; may be repeated.",
    )
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value, for example 0.")
    parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="Extra Hydra override appended only to probe training.",
    )
    parser.add_argument(
        "--eval-override",
        action="append",
        default=[],
        help="Extra Hydra override appended only to probe evaluation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if any(layer < 1 for layer in args.layers):
        raise ValueError("Every --layers value must be a positive integer.")
    if len(set(args.layers)) != len(args.layers):
        raise ValueError("--layers must not contain duplicates.")
    model_config = REPO_ROOT / "configs" / "model" / f"{args.model}.yaml"
    if not model_config.is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_config}")
    if "model_handler: Probed" not in model_config.read_text(encoding="utf-8"):
        raise ValueError(
            f"Model config '{args.model}' is not a decoder probe. "
            "Use a probed-* config."
        )
    if args.skip_eval and args.score_key:
        raise ValueError("--score-key cannot be used together with --skip-eval.")


def _resolve_forget_data(args: argparse.Namespace) -> Tuple[str, str]:
    spec = BENCHMARKS[args.benchmark]
    split = args.forget_split or spec.default_split
    forget_data = args.forget_data or str(
        REPO_ROOT / spec.forget_pattern.format(split=split)
    )
    if not glob.glob(forget_data):
        raise FileNotFoundError(
            f"No forget parquet matches '{forget_data}'. Use --forget-data to "
            "provide an explicit path or glob."
        )
    return split, forget_data


def _checkpoint_overrides(checkpoint: str) -> Tuple[str, List[str]]:
    checkpoint_path = Path(checkpoint).expanduser()
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if not adapter_config_path.is_file():
        return "full_model", [
            "model.model_args.pretrained_model_name_or_path=" + checkpoint
        ]

    with adapter_config_path.open(encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            f"LoRA adapter config has no base_model_name_or_path: {adapter_config_path}"
        )
    return "adapter", [
        "model.model_args.pretrained_model_name_or_path=" + base_model,
        "+model.adapter_path=" + str(checkpoint_path.resolve()),
        "+model.merge_adapter=true",
    ]


def _train_command(
    args: argparse.Namespace,
    n_layers: int,
    probe_dir: Path,
    forget_data: str,
    checkpoint_overrides: Iterable[str],
) -> List[str]:
    dataset_config = BENCHMARKS[args.benchmark].dataset_config
    command = [
        args.python,
        "src/train.py",
        "--config-name=train.yaml",
        "mode=finetune",
        f"model={args.model}",
        "peft=none",
        "~eval",
        "collator=DataCollatorForMultimodalQADataset",
        f"data/datasets@data.train={dataset_config}",
        f"data.train.{dataset_config}.args.hf_args.data_files={forget_data}",
        "collator.DataCollatorForMultimodalQADataset.args.processor_path="
        "${model.tokenizer_args.pretrained_model_name_or_path}",
        "collator.DataCollatorForMultimodalQADataset.args.max_length="
        "${oc.select:model.tokenizer_args.model_max_length,2048}",
        f"model.model_args.n_layers={n_layers}",
        "model.model_args.freeze_base_model=true",
        f"task_name=probing_{args.benchmark}_layer_{n_layers}",
        f"paths.output_dir={probe_dir}",
        f"trainer.args.learning_rate={args.learning_rate}",
        f"trainer.args.num_train_epochs={args.epochs}",
        f"trainer.args.per_device_train_batch_size={args.batch_size}",
        "trainer.args.gradient_accumulation_steps="
        + str(args.gradient_accumulation_steps),
        f"trainer.args.seed={args.seed}",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
        "trainer.args.save_strategy=no",
        "trainer.args.save_only_model=true",
        "trainer.args.report_to=none",
        "trainer.args.gradient_checkpointing=false",
        "trainer.args.optim=adamw_torch",
        "+trainer.args.remove_unused_columns=false",
    ]
    command.extend(checkpoint_overrides)
    if args.head_model_path:
        command.append(
            "model.model_args.head_pretrained_model_name_or_path="
            + args.head_model_path
        )
    if args.metadata_source:
        values = ",".join(args.metadata_source)
        command.append(
            f"++data.train.{dataset_config}.args.metadata_sources=[{values}]"
        )
    if args.max_steps is not None:
        command.append(f"+trainer.args.max_steps={args.max_steps}")
    if args.max_length is not None:
        command.append(
            "collator.DataCollatorForMultimodalQADataset.args.max_length="
            + str(args.max_length)
        )
    command.extend(args.train_override)
    return command


def _eval_command(
    args: argparse.Namespace,
    eval_experiment: str,
    n_layers: int,
    probe_dir: Path,
    eval_dir: Path,
) -> List[str]:
    command = [
        args.python,
        "src/eval.py",
        "--config-name=eval.yaml",
        f"experiment={eval_experiment}",
        f"model={args.model}",
        f"model.model_args.pretrained_model_name_or_path={probe_dir}",
        f"model.model_args.n_layers={n_layers}",
        "+model.model_args.reinitialize_output_head=false",
        f"model.model_args.device_map={args.device_map}",
        f"task_name=probing_{args.benchmark}_layer_{n_layers}_eval",
        f"paths.output_dir={eval_dir}",
    ]
    command.extend(args.eval_override)
    return command


def _read_summaries(output_dir: Path) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for path in sorted(output_dir.rglob("*_SUMMARY.json")):
        with path.open(encoding="utf-8") as handle:
            summaries[str(path.relative_to(output_dir))] = json.load(handle)
    return summaries


def _lookup_path(value: Any, path: str) -> Any:
    if isinstance(value, dict) and path in value:
        return value[path]
    current = value
    for part in filter(None, path.replace(".", "/").split("/")):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(part)
        current = current[part]
    return current


def _find_summary(summaries: Dict[str, Any], requested: str) -> Tuple[str, Any]:
    matches = [
        (name, value)
        for name, value in summaries.items()
        if name == requested or Path(name).name == requested
    ]
    if len(matches) != 1:
        available = ", ".join(sorted(summaries)) or "(none)"
        raise KeyError(
            f"Expected one summary named '{requested}', found {len(matches)}. "
            f"Available: {available}"
        )
    return matches[0]


def _collect_scores(
    summaries: Dict[str, Any], score_keys: Iterable[str]
) -> List[Dict[str, Any]]:
    scores = []
    for score_key in score_keys:
        if ":" not in score_key:
            raise ValueError("--score-key must use SUMMARY_FILE:PATH.")
        summary_name, metric_path = score_key.split(":", 1)
        resolved_name, summary = _find_summary(summaries, summary_name)
        try:
            value = _lookup_path(summary, metric_path)
        except KeyError as error:
            raise KeyError(f"Metric '{score_key}' was not found: {error}") from error
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Metric '{score_key}' must resolve to a numeric scalar.")
        scores.append(
            {"score_key": score_key, "summary": resolved_name, "value": value}
        )
    return scores


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
    _validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix probing results into non-empty directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    forget_split, forget_data = _resolve_forget_data(args)
    checkpoint_kind, checkpoint_overrides = _checkpoint_overrides(args.checkpoint)
    eval_experiment = args.eval_experiment or BENCHMARKS[args.benchmark].eval_experiment
    run_eval = not args.skip_eval and eval_experiment is not None
    if args.score_key and not run_eval:
        raise ValueError(
            "--score-key requires an evaluator. Supply --eval-experiment or "
            "remove --score-key."
        )

    commands: Dict[str, Dict[str, List[str]]] = {}
    for n_layers in args.layers:
        layer_dir = output_dir / f"layer_{n_layers}"
        probe_dir = layer_dir / "checkpoint"
        layer_commands = {
            "train": _train_command(
                args,
                n_layers,
                probe_dir,
                forget_data,
                checkpoint_overrides,
            )
        }
        if run_eval:
            layer_commands["eval"] = _eval_command(
                args,
                eval_experiment,
                n_layers,
                probe_dir,
                layer_dir / "eval",
            )
        commands[str(n_layers)] = layer_commands

    manifest: Dict[str, Any] = {
        "protocol": "head-only truncated-decoder probing stress test",
        "benchmark": args.benchmark,
        "input_checkpoint": args.checkpoint,
        "input_checkpoint_kind": checkpoint_kind,
        "forget_split": forget_split,
        "forget_data": forget_data,
        "layers": args.layers,
        "head_initialization": args.head_model_path or "random",
        "eval_experiment": eval_experiment,
        "evaluation_enabled": run_eval,
        "evaluation_note": (
            None
            if run_eval or args.skip_eval
            else f"No default evaluator is registered for {args.benchmark}; use "
            "--eval-experiment after adding one."
        ),
        "score_keys": args.score_key,
        "commands": commands,
    }
    _write_json(output_dir / "probing_manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    layer_results = []
    for n_layers in args.layers:
        layer_commands = commands[str(n_layers)]
        print(f"Training probe at layer {n_layers}", flush=True)
        _run(
            layer_commands["train"],
            output_dir / "logs" / f"layer_{n_layers}_train.log",
            env,
        )

        summaries: Optional[Dict[str, Any]] = None
        scores: List[Dict[str, Any]] = []
        if run_eval:
            print(f"Evaluating probe at layer {n_layers}", flush=True)
            _run(
                layer_commands["eval"],
                output_dir / "logs" / f"layer_{n_layers}_eval.log",
                env,
            )
            summaries = _read_summaries(output_dir / f"layer_{n_layers}" / "eval")
            scores = _collect_scores(summaries, args.score_key)
        layer_results.append(
            {
                "n_layers": n_layers,
                "checkpoint": str(output_dir / f"layer_{n_layers}" / "checkpoint"),
                "summaries": summaries,
                "scores": scores,
            }
        )

    result = {**manifest, "status": "completed", "layer_results": layer_results}
    result_path = output_dir / "probing_results.json"
    _write_json(result_path, result)
    print(f"Results written to {result_path}")


if __name__ == "__main__":
    main()

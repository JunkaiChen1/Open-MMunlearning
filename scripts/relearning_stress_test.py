#!/usr/bin/env python3
"""Relearn one multimodal unlearned checkpoint on a benchmark forget split.

This runner intentionally takes one checkpoint.  It supports either a full
Hugging Face checkpoint or a trainable LoRA adapter checkpoint and writes the
relearned checkpoint to ``--output-dir``.  Evaluation and score calculation
are deliberately left to ``relearning_score.py``.
"""

import argparse
from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset_config: str
    default_split: str
    forget_pattern: str


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "fiubench": BenchmarkSpec(
        dataset_config="FIUBENCH_QA_forget",
        default_split="forget_10",
        forget_pattern="data/unlearn/fiubench/{split}/train-00000-of-00001.parquet",
    ),
    "mllmu": BenchmarkSpec(
        dataset_config="MLLMU_QA_forget",
        default_split="forget_10",
        forget_pattern="data/unlearn/mllmu/{split}/train-00000-of-00001.parquet",
    ),
    "clear": BenchmarkSpec(
        dataset_config="CLEAR_FULL_QA_ft",
        default_split="forget_10",
        forget_pattern="data/unlearn/clear/{split}/train-00000-of-00001.parquet",
    ),
    "covubench": BenchmarkSpec(
        dataset_config="COVUBENCH_QA_forget",
        default_split="forget_5",
        forget_pattern="data/unlearn/covubench/{split}/train-*.parquet",
    ),
    "umubench": BenchmarkSpec(
        dataset_config="UMUBENCH_QA_forget",
        default_split="forget_10",
        forget_pattern="data/unlearn/umu/{split}/train-00000-of-00001.parquet",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=tuple(BENCHMARKS),
        help="Forget-set schema to use.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model config name from configs/model, without .yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "One unlearned checkpoint: a full model checkpoint/Hugging Face ID, "
            "or a directory containing adapter_config.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory where the relearned checkpoint and run manifest are written.",
    )
    parser.add_argument(
        "--forget-split",
        help="Forget split name. Defaults to the benchmark's canonical split.",
    )
    parser.add_argument(
        "--forget-data",
        help="Explicit forget parquet path or glob. Overrides --forget-split.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="ReLearn learning rate (paper default: 2e-5).",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Number of ReLearn epochs (paper default: one).",
    )
    parser.add_argument("--max-steps", type=int, help="Optional cap for smoke tests.")
    parser.add_argument("--batch-size", type=int, help="Per-device train batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Override the multimodal collator's sequence length.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing by default to reduce memory use.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run src/train.py (default: this interpreter).",
    )
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value, for example 0.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override appended to the train command; may be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest and print the command without starting training.",
    )
    return parser.parse_args()


def _checkpoint_kind(checkpoint: str) -> str:
    path = Path(checkpoint).expanduser()
    if path.is_dir() and (path / "adapter_config.json").is_file():
        return "adapter"
    return "full_model"


def _resolve_forget_data(args: argparse.Namespace) -> tuple[str, str]:
    spec = BENCHMARKS[args.benchmark]
    split = args.forget_split or spec.default_split
    if args.forget_data:
        forget_data = args.forget_data
    else:
        forget_data = str(REPO_ROOT / spec.forget_pattern.format(split=split))

    # A dry-run only needs to render the command and manifest. The external
    # benchmark file is required once training is actually started.
    if not glob.glob(forget_data) and not args.dry_run:
        raise FileNotFoundError(
            f"No forget parquet matches '{forget_data}'. Use --forget-data to provide "
            "an explicit data path or glob."
        )
    return split, forget_data


def _checkpoint_overrides(checkpoint: str, checkpoint_kind: str) -> List[str]:
    if checkpoint_kind == "adapter":
        return ["+model.adapter_path=" + checkpoint, "peft=none"]
    return ["model.model_args.pretrained_model_name_or_path=" + checkpoint, "peft=none"]


def _build_command(
    args: argparse.Namespace,
    output_dir: Path,
    forget_data: str,
    checkpoint_kind: str,
) -> List[str]:
    spec = BENCHMARKS[args.benchmark]
    task_name = f"relearning_{args.benchmark}"
    command = [
        args.python,
        "src/train.py",
        "--config-name=train.yaml",
        "mode=finetune",
        f"model={args.model}",
        "collator=DataCollatorForMultimodalQADataset",
        f"data/datasets@data.train={spec.dataset_config}",
        f"data.train.{spec.dataset_config}.args.hf_args.data_files={forget_data}",
        "collator.DataCollatorForMultimodalQADataset.args.processor_path="
        "${model.tokenizer_args.pretrained_model_name_or_path}",
        "collator.DataCollatorForMultimodalQADataset.args.max_length="
        "${oc.select:model.tokenizer_args.model_max_length,2048}",
        "task_name=" + task_name,
        f"paths.output_dir={output_dir}",
        f"trainer.args.learning_rate={args.learning_rate}",
        f"trainer.args.num_train_epochs={args.epochs}",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
        "trainer.args.save_strategy=no",
        "trainer.args.save_only_model=true",
        "trainer.args.report_to=none",
        "+trainer.args.remove_unused_columns=false",
        "trainer.args.gradient_checkpointing="
        + ("true" if args.gradient_checkpointing else "false"),
    ]
    command.extend(_checkpoint_overrides(args.checkpoint, checkpoint_kind))
    if args.max_steps is not None:
        command.append(f"+trainer.args.max_steps={args.max_steps}")
    if args.batch_size is not None:
        command.append(f"trainer.args.per_device_train_batch_size={args.batch_size}")
    if args.gradient_accumulation_steps is not None:
        command.append(
            "trainer.args.gradient_accumulation_steps="
            + str(args.gradient_accumulation_steps)
        )
    if args.max_length is not None:
        command.append(
            "collator.DataCollatorForMultimodalQADataset.args.max_length="
            + str(args.max_length)
        )
    command.extend(args.override)
    return command


def _write_json(path: Path, value: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _run(command: List[str], log_path: Path, env: Dict[str, str]) -> None:
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
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a ReLearn run into non-empty directory: {output_dir}. "
            "Choose a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    split, forget_data = _resolve_forget_data(args)
    checkpoint_kind = _checkpoint_kind(args.checkpoint)
    command = _build_command(args, output_dir, forget_data, checkpoint_kind)
    manifest: Dict[str, object] = {
        "protocol": "single-checkpoint ReLearn stress test",
        "paper_defaults": {"learning_rate": 2e-5, "epochs": 1},
        "paper_relearn_score_available": False,
        "paper_relearn_score_note": (
            "OpenUnlearning R needs a retained reference model in addition to "
            "the unlearned model. This one-checkpoint runner deliberately does not "
            "supply one; evaluate checkpoints separately and use relearning_score.py "
            "for raw metric changes."
        ),
        "benchmark": args.benchmark,
        "forget_split": split,
        "forget_data": forget_data,
        "input_checkpoint": args.checkpoint,
        "input_checkpoint_kind": checkpoint_kind,
        "output_checkpoint": str(output_dir),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "command": command,
    }
    _write_json(output_dir / "relearning_manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    log_path = output_dir / "relearning.log"
    print(f"Running ReLearn: {' '.join(command)}", flush=True)
    _run(command, log_path, env)

    result = {
        **manifest,
        "status": "completed",
        "log": str(log_path),
    }
    _write_json(output_dir / "relearning_result.json", result)
    print(f"Relearned checkpoint written to {output_dir}")


if __name__ == "__main__":
    main()

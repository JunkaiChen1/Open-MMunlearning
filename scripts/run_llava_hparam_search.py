#!/usr/bin/env python3
"""Run the paper-derived LoRA hyperparameter search for multimodal unlearning.

Every trial has a parameter-qualified output directory and the launcher
persists a manifest/state file so interrupted runs can be resumed without
overwriting existing benchmark results.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "python"

BENCHES = {
    "mllmu": {"experiment": "unlearn/mllmubench/default", "ratios": ("5", "10", "15")},
    "fiubench": {
        "experiment": "unlearn/fiubench/default",
        "ratios": ("1", "5", "10"),
    },
    "clear": {"experiment": "unlearn/clear/default", "ratios": ("1", "5", "10")},
}

MODELS = {
    "llava1_5-7b": "llava-1.5-7b-hf",
    "llava1_5-13b": "llava-1.5-13b-hf",
}

# MLLMU-Bench's original baseline set is GA, GA-Diff, KL-Min, and NPO.
# RMU/UNDIAL remain available as estimated single-modal extensions, but are
# intentionally not included when the MLLMU paper baseline set is requested.
METHOD_ORDER = (
    "GA",
    "GD",
    "KL",
    "NPO",
    "RMU",
    "UNDIAL",
    "MMUNLEARNER",
    "MANU",
)
TRAINER_BY_METHOD = {
    "GA": "GradAscent",
    "GD": "GradDiff",
    "KL": "kl",
    "NPO": "NPO",
    "RMU": "RMU",
    "UNDIAL": "UNDIAL",
    "MMUNLEARNER": "MMUnlearner",
    "MANU": "MANU",
}

# MLLMU anchor: lr=2e-5, batch size 4, one epoch, max_length=384. The
# three-point values below are the closest practical neighborhood around that
# anchor. NPO's paper implementation is forget-only; the repository trainer
# exposes a retain-loss alpha, so its paper anchor is 0 and we include two
# one-sided neighboring values to satisfy the same three-value search rule.
MLLMU_LRS = (1e-5, 2e-5, 3e-5)
RETAIN_ALPHAS = (0.5, 1.0, 2.0)
MLLMU_NPO_ALPHAS = (0.0, 0.5, 1.0)
MLLMU_NPO_BETAS = (0.6, 0.9, 1.2)
MMU_SALIENCY_THRESHOLDS = (0.5, 1.0, 2.0)
MANU_PRUNE_PERCENTS = (2.0, 5.0, 10.0)
# Estimated spaces for methods without an MLLMU-Bench paper setting.
PREFERENCE_LRS = (1e-5, 2e-5, 5e-5)
PREFERENCE_BETAS = (0.05, 0.1, 0.5)
RMU_STEERING_COEFFS = (1.0, 10.0, 100.0)
RMU_TARGET_LAYERS = (6, 11, 16)
UNDIAL_LRS = (1e-5, 1e-4, 3e-4)
UNDIAL_ALPHAS = (1.0, 2.0, 5.0)
UNDIAL_BETAS = (3.0, 10.0, 30.0)
# Peak allocations observed for 7B GA/GD are about 19-22 GiB. Methods with a
# reference model need substantially more headroom. The scheduler uses these
# conservative reservations to fill a GPU without admitting an OOM-prone mix.
METHOD_MEMORY_ESTIMATES_MIB = {
    "GA": 22000,
    "GD": 24000,
    "KL": 38000,
    "NPO": 38000,
    "RMU": 38000,
    "UNDIAL": 38000,
    "MMUNLEARNER": 50000,
    "MANU": 30000,
}


@dataclass(frozen=True)
class Task:
    bench: str
    ratio: str
    model_key: str
    model_config: str
    method: str
    trainer: str
    params: tuple[tuple[str, str], ...]
    overrides: tuple[str, ...]
    adapter_path: str
    base_model_path: str
    output_dir: str
    log_path: str

    @property
    def name(self) -> str:
        return safe_name(
            "_".join(
                (
                    self.bench,
                    f"ratio{self.ratio}",
                    self.model_key,
                    self.method,
                    *[f"{key}{value}" for key, value in self.params],
                )
            )
        )


@dataclass(frozen=True)
class Slot:
    id: str
    gpu: str


@dataclass
class RunningTask:
    task: Task
    slot_id: str
    gpu: str
    process: subprocess.Popen[str]
    log_file: Any


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def format_value(value: float | int | str) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("e-0", "e-").replace(".", "p")


def parse_filter(value: str | None, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return allowed
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown selection {unknown}; allowed: {', '.join(allowed)}")
    if not selected:
        raise ValueError("Selection cannot be empty")
    return selected


def rmu_overrides(target_layer: int) -> tuple[str, ...]:
    """Target one LLaVA language layer and train LoRA weights in it plus l-2/l-1."""
    train_layers = "|".join(str(layer) for layer in range(target_layer - 2, target_layer + 1))
    module_regex = (
        "base_model\\.model\\.language_model\\.model\\.layers\\."
        f"{target_layer}"
    )
    trainable_regex = (
        "base_model\\.model\\.language_model\\.model\\.layers\\."
        f"({train_layers})\\..*\\.lora_(A|B)\\..*\\.weight"
    )
    return (
        f"trainer.method_args.module_regex=['{module_regex}']",
        f"trainer.method_args.trainable_params_regex=['{trainable_regex}']",
        "+trainer.method_args.restore_all_grads_after_optimizer=false",
    )


def method_variants(method: str) -> Iterator[tuple[tuple[tuple[str, str], ...], tuple[str, ...]]]:
    """Yield the exact hyperparameter grids used by this search.

    GA/GD/KL/NPO are centered on the MLLMU-Bench LLaVA setting. RMU and UNDIAL
    use conservative estimated spaces because MLLMU-Bench does not report
    those single-modal methods.
    """
    if method == "GA":
        for lr in MLLMU_LRS:
            yield (("lr", format_value(lr)),), (f"trainer.args.learning_rate={lr:g}",)
        return

    if method in {"GD", "KL"}:
        for lr, alpha in itertools.product(MLLMU_LRS, RETAIN_ALPHAS):
            params = (("lr", format_value(lr)), ("alpha", format_value(alpha)))
            overrides = (
                f"trainer.args.learning_rate={lr:g}",
                f"trainer.method_args.gamma=1.0",
                f"trainer.method_args.alpha={alpha:g}",
            )
            yield params, overrides
        return

    if method == "NPO":
        for lr, alpha, beta in itertools.product(
            MLLMU_LRS, MLLMU_NPO_ALPHAS, MLLMU_NPO_BETAS
        ):
            params = (
                ("lr", format_value(lr)),
                ("alpha", format_value(alpha)),
                ("beta", format_value(beta)),
            )
            overrides = (
                f"trainer.args.learning_rate={lr:g}",
                "trainer.method_args.gamma=1.0",
                f"trainer.method_args.alpha={alpha:g}",
                f"trainer.method_args.beta={beta:g}",
                # MLLMU has a handler-specific beta default; ++ also creates
                # this node for the other benchmarks so the search wins.
                f"++trainer.method_args_by_handler.NPO.beta={beta:g}",
            )
            yield params, overrides
        return

    if method == "RMU":
        for lr, steering_coeff, target_layer in itertools.product(
            PREFERENCE_LRS, RMU_STEERING_COEFFS, RMU_TARGET_LAYERS
        ):
            params = (
                ("lr", format_value(lr)),
                ("steer", format_value(steering_coeff)),
                ("layer", str(target_layer)),
            )
            overrides = (
                f"trainer.args.learning_rate={lr:g}",
                "trainer.method_args.gamma=1.0",
                "trainer.method_args.alpha=1.0",
                f"trainer.method_args.steering_coeff={steering_coeff:g}",
                *rmu_overrides(target_layer),
            )
            yield params, overrides
        return

    if method == "UNDIAL":
        for lr, alpha, beta in itertools.product(
            UNDIAL_LRS, UNDIAL_ALPHAS, UNDIAL_BETAS
        ):
            params = (
                ("lr", format_value(lr)),
                ("alpha", format_value(alpha)),
                ("beta", format_value(beta)),
            )
            overrides = (
                f"trainer.args.learning_rate={lr:g}",
                "trainer.method_args.gamma=1.0",
                f"trainer.method_args.alpha={alpha:g}",
                f"trainer.method_args.beta={beta:g}",
            )
            yield params, overrides
        return

    if method == "MMUNLEARNER":
        for lr, threshold in itertools.product(
            MLLMU_LRS, MMU_SALIENCY_THRESHOLDS
        ):
            params = (
                ("lr", format_value(lr)),
                ("threshold", format_value(threshold)),
            )
            overrides = (
                f"trainer.args.learning_rate={lr:g}",
                "trainer.method_args.gamma=1.0",
                "trainer.method_args.alpha=1.0",
                f"trainer.method_args.saliency_threshold={threshold:g}",
            )
            yield params, overrides
        return

    if method == "MANU":
        for prune_percent in MANU_PRUNE_PERCENTS:
            params = (("prune", format_value(prune_percent)),)
            overrides = (
                f"trainer.method_args.prune_percent={prune_percent:g}",
                "trainer.method_args.num_iterations=1",
            )
            yield params, overrides
        return

    raise ValueError(f"No search space defined for {method}")


def build_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    for bench in args.benches:
        spec = BENCHES[bench]
        ratios = args.ratios_by_bench[bench]
        for ratio, model_key, method in itertools.product(ratios, args.models, args.methods):
            adapter_path = args.adapter_root / bench / model_key
            adapter_config_path = adapter_path / "adapter_config.json"
            if not adapter_config_path.is_file():
                raise FileNotFoundError(f"Missing adapter config: {adapter_path}")
            if not any((adapter_path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
                raise FileNotFoundError(f"Missing adapter weights: {adapter_path}")

            with adapter_config_path.open(encoding="utf-8") as handle:
                adapter_config = json.load(handle)
            base_model_path = adapter_config.get("base_model_name_or_path")
            if not base_model_path:
                raise ValueError(f"Adapter config has no base_model_name_or_path: {adapter_config_path}")
            if not Path(base_model_path).exists():
                raise FileNotFoundError(
                    f"Base model from adapter config does not exist: {base_model_path}"
                )

            for params, overrides in method_variants(method):
                params_dir = "_".join(f"{key}_{value}" for key, value in params)
                output_dir = (
                    args.output_root
                    / bench
                    / f"forget_ratio_{ratio}"
                    / model_key
                    / method
                    / params_dir
                )
                task_stub = "_".join(
                    (bench, f"ratio{ratio}", model_key, method, *[f"{key}{value}" for key, value in params])
                )
                tasks.append(
                    Task(
                        bench=bench,
                        ratio=ratio,
                        model_key=model_key,
                        model_config=MODELS[model_key],
                        method=method,
                        trainer=TRAINER_BY_METHOD[method],
                        params=params,
                        overrides=overrides,
                        adapter_path=str(adapter_path),
                        base_model_path=str(base_model_path),
                        output_dir=str(output_dir),
                        log_path=str(args.log_dir / f"{safe_name(task_stub)}.log"),
                    )
                )

    return sorted(
        tasks,
        key=lambda task: (
            args.benches.index(task.bench),
            int(task.ratio),
            METHOD_ORDER.index(task.method),
            args.models.index(task.model_key),
            task.params,
        ),
    )


def task_complete(task: Task) -> bool:
    checkpoint_dir = Path(task.output_dir) / "checkpoint"
    if task.method == "MANU":
        return (checkpoint_dir / "unlearning_artifact.json").is_file() and (
            checkpoint_dir / "manu_masks.pt"
        ).is_file()
    return (checkpoint_dir / "adapter_config.json").is_file() and any(
        (checkpoint_dir / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def estimated_memory_mib(task: Task) -> int:
    return METHOD_MEMORY_ESTIMATES_MIB[task.method]


def build_slots(gpus: tuple[str, ...], slots_per_gpu: int) -> tuple[Slot, ...]:
    return tuple(
        Slot(id=f"{gpu}:{slot_index}", gpu=gpu)
        for slot_index in range(slots_per_gpu)
        for gpu in gpus
    )


def can_schedule(task: Task, slot: Slot, running: dict[str, RunningTask], args: argparse.Namespace) -> bool:
    reserved = sum(
        estimated_memory_mib(item.task)
        for item in running.values()
        if item.gpu == slot.gpu
    )
    return reserved + estimated_memory_mib(task) <= args.memory_cap_mib


def pop_next_task(slot: Slot, pending: list[Task], running: dict[str, RunningTask], args: argparse.Namespace) -> Task | None:
    for index, task in enumerate(pending):
        if can_schedule(task, slot, running, args):
            return pending.pop(index)
    return None


def command_for_task(task: Task, args: argparse.Namespace) -> list[str]:
    retain_ratio = 100 - int(task.ratio)
    checkpoint_dir = Path(task.output_dir) / "checkpoint"
    if task.method == "MMUNLEARNER":
        experiment = "unlearn/mllmubench/mmunlearner_llava7b"
    elif task.method == "MANU":
        experiment = "unlearn/mllmubench/manu_llava7b"
    else:
        experiment = BENCHES[task.bench]["experiment"]
    peft_name = "none" if task.method == "MANU" else "lora"
    epochs = 2 if task.method == "MMUNLEARNER" else 1
    gradient_checkpointing = "false" if task.method == "MANU" else "true"
    return [
        args.python_bin,
        "-u",
        "src/train.py",
        "--config-name=unlearn.yaml",
        f"experiment={experiment}",
        f"model={task.model_config}",
        f"trainer={task.trainer}",
        f"peft={peft_name}",
        f"+model.adapter_path={task.adapter_path}",
        *(["+model.merge_adapter=true"] if task.method == "MANU" else []),
        f"model.model_args.pretrained_model_name_or_path={task.base_model_path}",
        f"model.tokenizer_args.pretrained_model_name_or_path={task.base_model_path}",
        f"task_name={task.name}",
        f"forget_split=forget_{task.ratio}",
        f"retain_split=retain_{retain_ratio}",
        f"paths.output_dir={checkpoint_dir}",
        f"trainer.args.output_dir={checkpoint_dir}",
        f"trainer.args.logging_dir={checkpoint_dir / 'logs'}",
        f"trainer.args.num_train_epochs={epochs}",
        f"trainer.args.gradient_checkpointing={gradient_checkpointing}",
        "trainer.args.remove_unused_columns=false",
        "trainer.args.do_eval=false",
        "trainer.args.eval_strategy=no",
        "trainer.args.eval_on_start=false",
        "trainer.args.save_strategy=no",
        "trainer.args.report_to=none",
        f"trainer.args.logging_steps={args.logging_steps}",
        f"collator.DataCollatorForMultimodalQADataset.args.max_length={args.max_length}",
        *task.overrides,
    ]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def update_state(
    state: dict[str, dict[str, Any]], task: Task, status: str, **details: Any
) -> None:
    item = state.setdefault(task.name, {})
    item.update(
        {
            "bench": task.bench,
            "ratio": task.ratio,
            "model": task.model_key,
            "method": task.method,
            "trainer": task.trainer,
            "params": dict(task.params),
            "output_dir": task.output_dir,
            "log_path": task.log_path,
            "status": status,
            **details,
        }
    )
    if status == "RUNNING":
        item["started_at"] = now_text()
        item.pop("ended_at", None)
    if status in {"DONE", "FAIL"}:
        item["ended_at"] = now_text()


def write_progress(
    path: Path,
    tasks: list[Task],
    state: dict[str, dict[str, Any]],
    running: dict[str, RunningTask],
    args: argparse.Namespace,
) -> None:
    counts: dict[str, int] = {}
    for task in tasks:
        status = state.get(task.name, {}).get("status", "PENDING")
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# LLaVA Unlearning Hyperparameter Search",
        "",
        f"- Updated: `{now_text()}`",
        f"- GPUs: `{','.join(args.gpus)}`; slots per GPU: `{args.slots_per_gpu}`",
        f"- Reserved-memory cap per GPU: `{args.memory_cap_mib} MiB`",
        f"- Benchmarks: `{','.join(args.benches)}`",
        "- Forget ratios: `"
        + "; ".join(
            f"{bench}={','.join(args.ratios_by_bench[bench])}"
            for bench in args.benches
        )
        + "`",
        f"- Models: `{','.join(args.models)}`",
        f"- Methods: `{','.join(args.methods)}`",
        f"- Total trials: `{len(tasks)}`",
        "- Grid: GA=3, GD=9, KL=9, NPO=27, RMU=27, UNDIAL=27, "
        "MMUNLEARNER=9, MANU=3.",
        f"- Status: DONE `{counts.get('DONE', 0)}`, RUNNING `{len(running)}`, "
        f"PENDING `{counts.get('PENDING', 0)}`, FAIL `{counts.get('FAIL', 0)}`",
        "",
        "## Running",
        "",
        "| slot | GPU | task | reserved MiB | pid | log |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for slot_id, item in sorted(running.items()):
        lines.append(
            f"| {slot_id} | {item.gpu} | `{item.task.name}` | "
            f"{estimated_memory_mib(item.task)} | {item.process.pid} | `{item.task.log_path}` |"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def classify_failure(log_path: Path) -> str:
    if not log_path.is_file():
        return "missing_log"
    content = log_path.read_text(encoding="utf-8", errors="replace")
    for pattern, label in (
        (r"CUDA out of memory|OutOfMemoryError", "OOM"),
        (r"No module matched", "RMU_MODULE_MISMATCH"),
        (r"No space left on device", "NO_SPACE"),
        (r"Traceback", "TRACEBACK"),
    ):
        if re.search(pattern, content, re.IGNORECASE):
            return label
    return "exit_nonzero"


def launch(task: Task, slot: Slot, args: argparse.Namespace) -> RunningTask:
    output_dir = Path(task.output_dir)
    log_path = Path(task.log_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # All trials use the same local datasets. Sharing this cache avoids
    # materializing one identical Arrow copy per hyperparameter trial.
    args.hf_datasets_cache.mkdir(parents=True, exist_ok=True)

    command = command_for_task(task, args)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": slot.gpu,
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": str(args.hf_home),
            "HF_DATASETS_CACHE": str(args.hf_datasets_cache),
            "OMP_NUM_THREADS": "4",
        }
    )
    environment.pop("PYTHONPATH", None)
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    log_file.write("$ " + shlex.join(command) + "\n")
    log_file.write(f"# CUDA_VISIBLE_DEVICES={slot.gpu}\n")
    log_file.write(f"# SLOT={slot.id}\n")
    log_file.flush()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningTask(
        task=task,
        slot_id=slot.id,
        gpu=slot.gpu,
        process=process,
        log_file=log_file,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--memory-cap-mib", type=int, default=90000)
    # This run is scoped to MLLMU. Other benchmarks remain available through
    # an explicit --benches argument when a separate search is requested.
    parser.add_argument("--benches", default="mllmu")
    parser.add_argument(
        "--ratios",
        default=None,
        help=(
            "Comma-separated forget ratios. The selected ratios must exist for "
            "every selected benchmark; defaults to each benchmark's full set."
        ),
    )
    parser.add_argument("--models", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=REPO_ROOT / "vanilla_model",
        help="Root containing <benchmark>/<model>/adapter_config.json directories.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "results" / "hparam_search_20260805_llava_paper2506",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sample-interval", type=int, default=30)
    parser.add_argument("--launch-delay-seconds", type=float, default=10.0)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    args.benches = parse_filter(args.benches, tuple(BENCHES))
    requested_ratios = (
        tuple(item.strip() for item in args.ratios.split(",") if item.strip())
        if args.ratios is not None
        else None
    )
    if requested_ratios is not None and not requested_ratios:
        raise ValueError("Ratio selection cannot be empty")
    args.ratios_by_bench = {}
    for bench in args.benches:
        supported_ratios = BENCHES[bench]["ratios"]
        selected_ratios = requested_ratios or supported_ratios
        unknown_ratios = sorted(set(selected_ratios) - set(supported_ratios))
        if unknown_ratios:
            raise ValueError(
                f"Unsupported ratios {unknown_ratios} for {bench}; "
                f"allowed: {', '.join(supported_ratios)}"
            )
        args.ratios_by_bench[bench] = selected_ratios
    args.models = parse_filter(args.models, tuple(MODELS))
    args.methods = parse_filter(args.methods, METHOD_ORDER)
    args.gpus = tuple(dict.fromkeys(item.strip() for item in args.gpus.split(",") if item.strip()))
    if not args.gpus:
        raise ValueError("No GPUs selected")
    if args.slots_per_gpu < 1:
        raise ValueError("--slots-per-gpu must be at least 1")
    if args.memory_cap_mib < 1:
        raise ValueError("--memory-cap-mib must be positive")
    args.slots = build_slots(args.gpus, args.slots_per_gpu)
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.output_root = args.output_root or args.run_root / "outputs"
    args.log_dir = args.run_root / "logs"
    args.hf_home = args.run_root / "hf_home"
    args.hf_datasets_cache = args.run_root / "hf_datasets_cache"
    for path in (args.output_root, args.log_dir, args.hf_home, args.hf_datasets_cache):
        path.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args)
    manifest_path = args.run_root / "manifest.jsonl"
    state_path = args.run_root / "progress_state.json"
    progress_path = args.run_root / "progress.md"
    events_path = args.run_root / "events.jsonl"
    if not manifest_path.exists():
        with manifest_path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                record = asdict(task)
                record["command"] = command_for_task(task, args)
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    if args.dry_run:
        by_method = {method: sum(task.method == method for task in tasks) for method in args.methods}
        print(json.dumps({"run_root": str(args.run_root), "total_trials": len(tasks), "by_method": by_method}, indent=2))
        return 0

    state = {} if args.no_resume else load_state(state_path)
    pending: list[Task] = []
    for task in tasks:
        if task_complete(task):
            update_state(state, task, "DONE", message="adapter output exists")
        else:
            pending.append(task)
    write_json(state_path, state)

    running: dict[str, RunningTask] = {}
    print(f"[{now_text()}] run_root={args.run_root}")
    print(f"[{now_text()}] total={len(tasks)} pending={len(pending)}")
    while pending or running:
        for slot in args.slots:
            if slot.id in running or not pending:
                continue
            task = pop_next_task(slot, pending, running, args)
            if task is None:
                continue
            item = launch(task, slot, args)
            running[slot.id] = item
            update_state(
                state,
                task,
                "RUNNING",
                gpu=slot.gpu,
                slot=slot.id,
                pid=item.process.pid,
            )
            append_event(
                events_path,
                {
                    "time": now_text(),
                    "status": "RUNNING",
                    "gpu": slot.gpu,
                    "slot": slot.id,
                    "pid": item.process.pid,
                    "task": task.name,
                },
            )
            print(
                f"[{now_text()}] START gpu={slot.gpu} slot={slot.id} "
                f"pid={item.process.pid} task={task.name} reserved_mib={estimated_memory_mib(task)}"
            )
            if args.launch_delay_seconds:
                time.sleep(args.launch_delay_seconds)

        finished: list[str] = []
        for slot_id, item in list(running.items()):
            exit_code = item.process.poll()
            if exit_code is None:
                continue
            item.log_file.close()
            status = "DONE" if exit_code == 0 and task_complete(item.task) else "FAIL"
            message = "completed" if status == "DONE" else classify_failure(Path(item.task.log_path))
            update_state(
                state,
                item.task,
                status,
                gpu=item.gpu,
                slot=item.slot_id,
                pid=item.process.pid,
                exit_code=exit_code,
                message=message,
            )
            append_event(
                events_path,
                {
                    "time": now_text(),
                    "status": status,
                    "gpu": item.gpu,
                    "slot": item.slot_id,
                    "pid": item.process.pid,
                    "task": item.task.name,
                    "exit_code": exit_code,
                    "message": message,
                },
            )
            print(
                f"[{now_text()}] {status} gpu={item.gpu} slot={item.slot_id} "
                f"exit={exit_code} task={item.task.name} message={message}"
            )
            finished.append(slot_id)
        for slot_id in finished:
            running.pop(slot_id)

        write_json(state_path, state)
        write_progress(progress_path, tasks, state, running, args)
        if pending or running:
            time.sleep(args.sample_interval)

    write_progress(progress_path, tasks, state, running, args)
    failures = sum(1 for task in tasks if state.get(task.name, {}).get("status") == "FAIL")
    print(f"[{now_text()}] complete failures={failures} progress={progress_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Unified resumable launcher for the framework's unlearning grid.

The launcher owns scheduling only.  Each trial still runs through ``src/train.py``
and therefore uses the repository's Hydra configs and trainers.  Trial outputs
always end in a ``checkpoint`` directory so they can be consumed by the existing
evaluation scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "python"

BENCHES: dict[str, dict[str, Any]] = {
    "mllmu": {
        "ratios": ("5", "10", "15"),
        "experiment": "unlearn/mllmubench/default",
    },
    "fiubench": {
        "ratios": ("1", "5", "10"),
        "experiment": "unlearn/fiubench/default",
    },
    "clear": {
        "ratios": ("1", "5", "10"),
        "experiment": "unlearn/clear/default",
    },
}

MODELS = {
    "llava1_5-7b": "llava-1.5-7b-hf",
    "llava1_5-13b": "llava-1.5-13b-hf",
}

METHOD_ORDER = (
    "GA",
    "GD",
    "KL",
    "NPO",
    "RMU",
    "UNDIAL",
    "MMUNLEARNER",
    "MANU",
    "MIP_EDITOR",
    "SMFA",
)
DEFAULT_METHODS = METHOD_ORDER[:8]
TRAINERS = {
    "GA": "GradAscent",
    "GD": "GradDiff",
    "KL": "kl",
    "NPO": "NPO",
    "RMU": "RMU",
    "UNDIAL": "UNDIAL",
    "MMUNLEARNER": "MMUnlearner",
    "MANU": "MANU",
    "MIP_EDITOR": "MIPEditor",
    "SMFA": "SMFA",
}

# The first four spaces are centered on the MLLMU-Bench setting.  RMU and
# UNDIAL are the three-by-three-by-three estimates documented in docs/.
MLLMU_LRS = (1e-5, 2e-5, 3e-5)
RETAIN_ALPHAS = (0.5, 1.0, 2.0)
NPO_ALPHAS = (0.0, 0.5, 1.0)
NPO_BETAS = (0.6, 0.9, 1.2)
RMU_STEERING = (1.0, 10.0, 100.0)
RMU_LAYERS = (6, 11, 16)
UNDIAL_LRS = (1e-5, 1e-4, 3e-4)
UNDIAL_ALPHAS = (1.0, 2.0, 5.0)
UNDIAL_BETAS = (3.0, 10.0, 30.0)
MMU_THRESHOLDS = (0.5, 1.0, 2.0)
MANU_PRUNES = (2.0, 5.0, 10.0)
MIP_LRS = (1e-4, 5e-4, 1e-3)
MIP_TOPKS = (3, 5, 10)
MIP_STEERING = (0.5, 1.0, 2.0)
SMFA_LRS = (5e-5, 1e-4, 2e-4)
SMFA_MULTI_K = (2.5, 5.0, 10.0)

MEMORY_MIB = {
    "GA": 22000,
    "GD": 24000,
    "KL": 38000,
    "NPO": 38000,
    "RMU": 38000,
    "UNDIAL": 38000,
    "MMUNLEARNER": 50000,
    "MANU": 30000,
    "MIP_EDITOR": 38000,
    "SMFA": 50000,
}


@dataclass(frozen=True)
class Trial:
    bench: str
    ratio: str
    model: str
    method: str
    params: tuple[tuple[str, str], ...]
    overrides: tuple[str, ...]
    adapter_path: str
    base_model_path: str
    checkpoint_dir: str
    log_path: str

    @property
    def name(self) -> str:
        values = [self.bench, f"ratio{self.ratio}", self.model, self.method]
        values.extend(f"{key}{value}" for key, value in self.params)
        return safe_name("_".join(values))


@dataclass
class Running:
    trial: Trial
    gpu: str
    process: subprocess.Popen[str]
    log_file: Any


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def value_text(value: float | int | str) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("e-0", "e-").replace(".", "p")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_csv(value: str, allowed: Iterable[str], name: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise ValueError(f"{name} cannot be empty")
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {name} {unknown}; allowed: {', '.join(allowed)}")
    return selected


def rmu_overrides(layer: int) -> tuple[str, ...]:
    train_layers = "|".join(str(item) for item in range(layer - 2, layer + 1))
    module = (
        "base_model\\.model\\.language_model\\.model\\.layers\\."
        f"{layer}"
    )
    trainable = (
        "base_model\\.model\\.language_model\\.model\\.layers\\."
        f"({train_layers})\\..*\\.lora_(A|B)\\..*\\.weight"
    )
    return (
        f"trainer.method_args.module_regex=['{module}']",
        f"trainer.method_args.trainable_params_regex=['{trainable}']",
        "+trainer.method_args.restore_all_grads_after_optimizer=false",
    )


def variants(method: str) -> list[tuple[tuple[tuple[str, str], ...], tuple[str, ...]]]:
    result: list[tuple[tuple[tuple[str, str], ...], tuple[str, ...]]] = []
    if method == "GA":
        for lr in MLLMU_LRS:
            result.append(
                (
                    (("lr", value_text(lr)),),
                    (f"trainer.args.learning_rate={lr:g}",),
                )
            )
    elif method in {"GD", "KL"}:
        for lr, alpha in itertools.product(MLLMU_LRS, RETAIN_ALPHAS):
            result.append(
                (
                    (("lr", value_text(lr)), ("alpha", value_text(alpha))),
                    (
                        f"trainer.args.learning_rate={lr:g}",
                        "trainer.method_args.gamma=1.0",
                        f"trainer.method_args.alpha={alpha:g}",
                    ),
                )
            )
    elif method == "NPO":
        for lr, alpha, beta in itertools.product(MLLMU_LRS, NPO_ALPHAS, NPO_BETAS):
            result.append(
                (
                    (
                        ("lr", value_text(lr)),
                        ("alpha", value_text(alpha)),
                        ("beta", value_text(beta)),
                    ),
                    (
                        f"trainer.args.learning_rate={lr:g}",
                        "trainer.method_args.gamma=1.0",
                        f"trainer.method_args.alpha={alpha:g}",
                        f"trainer.method_args.beta={beta:g}",
                        f"++trainer.method_args_by_handler.NPO.beta={beta:g}",
                    ),
                )
            )
    elif method == "RMU":
        for lr, steering, layer in itertools.product(
            MLLMU_LRS, RMU_STEERING, RMU_LAYERS
        ):
            result.append(
                (
                    (
                        ("lr", value_text(lr)),
                        ("steer", value_text(steering)),
                        ("layer", str(layer)),
                    ),
                    (
                        f"trainer.args.learning_rate={lr:g}",
                        "trainer.method_args.gamma=1.0",
                        "trainer.method_args.alpha=1.0",
                        f"trainer.method_args.steering_coeff={steering:g}",
                        *rmu_overrides(layer),
                    ),
                )
            )
    elif method == "UNDIAL":
        for lr, alpha, beta in itertools.product(
            UNDIAL_LRS, UNDIAL_ALPHAS, UNDIAL_BETAS
        ):
            result.append(
                (
                    (
                        ("lr", value_text(lr)),
                        ("alpha", value_text(alpha)),
                        ("beta", value_text(beta)),
                    ),
                    (
                        f"trainer.args.learning_rate={lr:g}",
                        "trainer.method_args.gamma=1.0",
                        f"trainer.method_args.alpha={alpha:g}",
                        f"trainer.method_args.beta={beta:g}",
                    ),
                )
            )
    elif method == "MMUNLEARNER":
        for lr, threshold in itertools.product(MLLMU_LRS, MMU_THRESHOLDS):
            result.append(
                (
                    (("lr", value_text(lr)), ("threshold", value_text(threshold))),
                    (
                        f"trainer.args.learning_rate={lr:g}",
                        "trainer.method_args.gamma=1.0",
                        "trainer.method_args.alpha=1.0",
                        f"trainer.method_args.saliency_threshold={threshold:g}",
                    ),
                )
            )
    elif method == "MANU":
        for prune in MANU_PRUNES:
            result.append(
                (
                    (("prune", value_text(prune)),),
                    (
                        f"trainer.method_args.prune_percent={prune:g}",
                        "trainer.method_args.num_iterations=1",
                    ),
                )
            )
    elif method == "MIP_EDITOR":
        for lr, topk, steering in itertools.product(MIP_LRS, MIP_TOPKS, MIP_STEERING):
            result.append(
                (
                    (("lr", value_text(lr)), ("topk", str(topk)), ("steer", value_text(steering))),
                    (
                        f"trainer.args.learning_rate={lr:g}",
                        f"trainer.method_args.path_topk={topk}",
                        f"trainer.method_args.steering_coeff={steering:g}",
                    ),
                )
            )
    elif method == "SMFA":
        for lr, multi_k in itertools.product(SMFA_LRS, SMFA_MULTI_K):
            result.append(
                (
                    (("lr", value_text(lr)), ("multi_k", value_text(multi_k))),
                    (
                        f"trainer.args.learning_rate={lr:g}",
                        f"trainer.method_args.multi_k={multi_k:g}",
                        "trainer.method_args.text_k=10.0",
                    ),
                )
            )
    else:
        raise ValueError(f"No grid for method {method}")
    return result


def checkpoint_complete(trial: Trial) -> bool:
    path = Path(trial.checkpoint_dir)
    artifact_path = path / "unlearning_artifact.json"
    if trial.method == "MANU":
        return artifact_path.is_file() and (path / "manu_masks.pt").is_file()
    if trial.method == "SMFA":
        return artifact_path.is_file() and all(
            (path / adapter / "adapter_config.json").is_file()
            and any(
                (path / adapter / filename).is_file()
                for filename in ("adapter_model.safetensors", "adapter_model.bin")
            )
            for adapter in ("MFA_multi", "MFA_text", "RA")
        )
    return (path / "adapter_config.json").is_file() and any(
        (path / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    )


def load_adapter(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    if not any(
        (adapter_dir / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(f"Missing adapter weights: {adapter_dir}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    base = config.get("base_model_name_or_path")
    if not base:
        raise ValueError(f"Adapter config has no base model: {config_path}")
    if not Path(base).exists():
        raise FileNotFoundError(f"Base model does not exist: {base}")
    return str(base)


def build_trials(args: argparse.Namespace) -> list[Trial]:
    trials: list[Trial] = []
    for bench in args.benches:
        for ratio, model, method in itertools.product(
            args.ratios_by_bench[bench], args.models, args.methods
        ):
            adapter_dir = args.adapter_root / bench / model
            base_model = load_adapter(adapter_dir)
            for params, overrides in variants(method):
                param_dir = "_".join(f"{key}_{value}" for key, value in params)
                checkpoint_dir = (
                    args.run_root
                    / "checkpoints"
                    / bench
                    / f"forget_ratio_{ratio}"
                    / model
                    / method
                    / param_dir
                    / "checkpoint"
                )
                trial_stub = "_".join(
                    [bench, f"ratio{ratio}", model, method]
                    + [f"{key}{value}" for key, value in params]
                )
                trials.append(
                    Trial(
                        bench=bench,
                        ratio=ratio,
                        model=model,
                        method=method,
                        params=params,
                        overrides=overrides,
                        adapter_path=str(adapter_dir),
                        base_model_path=base_model,
                        checkpoint_dir=str(checkpoint_dir),
                        log_path=str(args.run_root / "logs" / f"{safe_name(trial_stub)}.log"),
                    )
                )
    return sorted(
        trials,
        key=lambda item: (
            args.benches.index(item.bench),
            int(item.ratio),
            args.models.index(item.model),
            METHOD_ORDER.index(item.method),
            item.params,
        ),
    )


def command_for(trial: Trial, args: argparse.Namespace) -> list[str]:
    retain = 100 - int(trial.ratio)
    if trial.method == "MMUNLEARNER":
        experiment = "unlearn/mllmubench/mmunlearner_llava7b"
    elif trial.method == "MANU":
        experiment = "unlearn/mllmubench/manu_llava7b"
    elif trial.method == "MIP_EDITOR":
        experiment = "unlearn/mllmubench/mip_editor_llava7b"
    elif trial.method == "SMFA":
        experiment = "unlearn/mllmubench/smfa_llava7b"
    else:
        experiment = BENCHES[trial.bench]["experiment"]
    # MANU edits the merged fine-tuned model directly and saves a compact mask
    # artifact. Other trainable methods use LoRA-backed checkpoints.
    peft = "none" if trial.method == "MANU" else "lora"
    default_epochs = 2 if trial.method == "MMUNLEARNER" else (3 if trial.method == "SMFA" else 1)
    epochs = args.epochs if args.epochs is not None else default_epochs
    gradient_checkpointing = "false" if trial.method == "MANU" else "true"
    checkpoint = trial.checkpoint_dir
    return [
        args.python_bin,
        "-u",
        "src/train.py",
        "--config-name=unlearn.yaml",
        f"experiment={experiment}",
        f"model={MODELS[trial.model]}",
        f"trainer={TRAINERS[trial.method]}",
        f"peft={peft}",
        f"+model.adapter_path={trial.adapter_path}",
        *(["+model.merge_adapter=true"] if trial.method == "MANU" else []),
        f"model.model_args.pretrained_model_name_or_path={trial.base_model_path}",
        f"model.tokenizer_args.pretrained_model_name_or_path={trial.base_model_path}",
        f"task_name={trial.name}",
        f"forget_split=forget_{trial.ratio}",
        f"retain_split=retain_{retain}",
        f"paths.output_dir={checkpoint}",
        f"trainer.args.output_dir={checkpoint}",
        f"trainer.args.logging_dir={checkpoint}/logs",
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
        *trial.overrides,
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def update_state(state: dict[str, Any], trial: Trial, status: str, **details: Any) -> None:
    item = state.setdefault(trial.name, {})
    item.update(
        {
            "trial": trial.name,
            "bench": trial.bench,
            "ratio": trial.ratio,
            "model": trial.model,
            "method": trial.method,
            "params": dict(trial.params),
            "checkpoint_dir": trial.checkpoint_dir,
            "log_path": trial.log_path,
            "status": status,
            **details,
        }
    )
    if status == "RUNNING":
        item["started_at"] = now()
        item.pop("ended_at", None)
    elif status in {"DONE", "FAIL"}:
        item["ended_at"] = now()


def summary(trials: list[Trial], state: dict[str, Any], running: dict[str, Running]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in trials:
        status = state.get(trial.name, {}).get("status", "PENDING")
        counts[status] = counts.get(status, 0) + 1
    counts["RUNNING"] = len(running)
    return counts


def write_progress(path: Path, trials: list[Trial], state: dict[str, Any], running: dict[str, Running], args: argparse.Namespace) -> None:
    counts = summary(trials, state, running)
    by_method = {method: sum(item.method == method for item in trials) for method in args.methods}
    lines = [
        "# Unified Unlearning Hyperparameter Search",
        "",
        f"- Updated: `{now()}`",
        f"- Total trials: `{len(trials)}`",
        f"- Benchmarks: `{','.join(args.benches)}`",
        f"- Ratios: `" + "; ".join(f"{name}={','.join(args.ratios_by_bench[name])}" for name in args.benches) + "`",
        f"- Models: `{','.join(args.models)}`",
        f"- Methods: `{','.join(args.methods)}`",
        "- Grid: " + ", ".join(f"{key}={value}" for key, value in by_method.items()) + ".",
        f"- GPUs: `{','.join(args.gpus)}`; max concurrent: `{args.max_concurrent}`",
        f"- Status: DONE `{counts.get('DONE', 0)}`, RUNNING `{counts.get('RUNNING', 0)}`, PENDING `{counts.get('PENDING', 0)}`, FAIL `{counts.get('FAIL', 0)}`",
        "",
        "## Running",
        "",
        "| GPU | PID | Trial | Log |",
        "| --- | ---: | --- | --- |",
    ]
    for item in running.values():
        lines.append(f"| {item.gpu} | {item.process.pid} | `{item.trial.name}` | `{item.trial.log_path}` |")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def failure_label(log_path: Path) -> str:
    if not log_path.is_file():
        return "missing_log"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for pattern, label in (
        (r"CUDA out of memory|OutOfMemoryError", "OOM"),
        (r"No module matched", "RMU_MODULE_MISMATCH"),
        (r"No space left on device", "NO_SPACE"),
        (r"Traceback", "TRACEBACK"),
    ):
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return "exit_nonzero"


def launch(trial: Trial, gpu: str, args: argparse.Namespace) -> Running:
    Path(trial.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(trial.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = command_for(trial, args)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": str(args.run_root / "hf_home"),
            # All trials use the same local datasets. Sharing this cache avoids
            # materializing one identical Arrow copy per hyperparameter trial.
            "HF_DATASETS_CACHE": str(args.hf_datasets_cache),
            "OMP_NUM_THREADS": str(args.omp_threads),
        }
    )
    Path(environment["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    environment.pop("PYTHONPATH", None)
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    log_file.write("$ " + shlex.join(command) + "\n")
    log_file.write(f"# CUDA_VISIBLE_DEVICES={gpu}\n")
    log_file.flush()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return Running(trial=trial, gpu=gpu, process=process, log_file=log_file)


def terminate(running: dict[str, Running]) -> None:
    for item in running.values():
        try:
            os.killpg(item.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 10
    while running and time.time() < deadline:
        for key, item in list(running.items()):
            if item.process.poll() is not None:
                item.log_file.close()
                running.pop(key)
        time.sleep(0.2)
    for item in running.values():
        try:
            os.killpg(item.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--benches", default="mllmu")
    parser.add_argument("--ratios", default=None)
    parser.add_argument("--models", default="llava1_5-7b")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--adapter-root", type=Path, default=ROOT / "vanilla_model")
    parser.add_argument("--run-root", type=Path, default=ROOT / "results" / "hparam_search_all_methods_checkpoint")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--sample-interval", type=float, default=10.0)
    parser.add_argument("--launch-delay-seconds", type=float, default=0.0)
    parser.add_argument("--omp-threads", type=int, default=4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the default epoch count for every selected method.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    args.benches = parse_csv(args.benches, BENCHES, "benchmarks")
    args.models = parse_csv(args.models, MODELS, "models")
    args.methods = parse_csv(args.methods, METHOD_ORDER, "methods")
    args.gpus = parse_csv(args.gpus, tuple(str(item) for item in range(16)), "GPUs")
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be positive")
    if args.max_length < 1:
        raise ValueError("--max-length must be positive")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    requested_ratios = (
        tuple(item.strip() for item in args.ratios.split(",") if item.strip())
        if args.ratios is not None
        else None
    )
    args.ratios_by_bench: dict[str, tuple[str, ...]] = {}
    for bench in args.benches:
        selected = requested_ratios or tuple(BENCHES[bench]["ratios"])
        unknown = sorted(set(selected) - set(BENCHES[bench]["ratios"]))
        if unknown:
            raise ValueError(f"Unsupported ratios for {bench}: {unknown}")
        args.ratios_by_bench[bench] = selected

    args.run_root.mkdir(parents=True, exist_ok=True)
    args.hf_datasets_cache = args.run_root / "hf_datasets_cache"
    for directory in (
        args.run_root / "checkpoints",
        args.run_root / "logs",
        args.run_root / "hf_home",
        args.hf_datasets_cache,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    trials = build_trials(args)
    by_method = {method: sum(item.method == method for item in trials) for method in args.methods}
    print(json.dumps({"run_root": str(args.run_root), "total_trials": len(trials), "by_method": by_method}, indent=2))
    manifest = args.run_root / "manifest.jsonl"
    fingerprint = hashlib.sha256(
        json.dumps([asdict(trial) for trial in trials], sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_payload = {
        "version": 1,
        "created_at": now(),
        "fingerprint": fingerprint,
        "args": {key: str(value) for key, value in vars(args).items() if key != "ratios_by_bench"},
        "trials": [dict(asdict(trial), command=command_for(trial, args)) for trial in trials],
    }
    if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")).get("fingerprint") != fingerprint:
        write_json(manifest, manifest_payload)
    if args.dry_run:
        return 0

    state_path = args.run_root / "state.json"
    progress_path = args.run_root / "progress.md"
    events_path = args.run_root / "events.jsonl"
    state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    pending: list[Trial] = []
    for trial in trials:
        if checkpoint_complete(trial):
            update_state(state, trial, "DONE", message="checkpoint exists")
        elif state.get(trial.name, {}).get("status") == "FAIL" and not args.retry_failed:
            continue
        else:
            pending.append(trial)
    write_json(state_path, state)
    running: dict[str, Running] = {}
    interrupted = False

    def on_signal(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    print(f"[{now()}] total={len(trials)} pending={len(pending)} run_root={args.run_root}")
    try:
        while pending or running:
            if interrupted:
                raise KeyboardInterrupt
            while pending and len(running) < args.max_concurrent:
                trial = pending.pop(0)
                gpu = args.gpus[len(running) % len(args.gpus)]
                item = launch(trial, gpu, args)
                running[trial.name] = item
                update_state(state, trial, "RUNNING", pid=item.process.pid, gpu=gpu)
                append_event(events_path, {"time": now(), "status": "RUNNING", "trial": trial.name, "pid": item.process.pid, "gpu": gpu})
                print(f"[{now()}] START gpu={gpu} pid={item.process.pid} trial={trial.name}")
                if args.launch_delay_seconds:
                    time.sleep(args.launch_delay_seconds)

            for name, item in list(running.items()):
                exit_code = item.process.poll()
                if exit_code is None:
                    continue
                item.log_file.close()
                status = "DONE" if exit_code == 0 and checkpoint_complete(item.trial) else "FAIL"
                message = "completed" if status == "DONE" else failure_label(Path(item.trial.log_path))
                update_state(state, item.trial, status, pid=item.process.pid, gpu=item.gpu, exit_code=exit_code, message=message)
                append_event(events_path, {"time": now(), "status": status, "trial": name, "pid": item.process.pid, "gpu": item.gpu, "exit_code": exit_code, "message": message})
                print(f"[{now()}] {status} gpu={item.gpu} exit={exit_code} trial={name} message={message}")
                running.pop(name)

            write_json(state_path, state)
            write_progress(progress_path, trials, state, running, args)
            if pending or running:
                time.sleep(args.sample_interval)
    except KeyboardInterrupt:
        print(f"[{now()}] stopping {len(running)} running trial(s)")
        terminate(running)
        for item in running.values():
            update_state(state, item.trial, "PENDING", message="interrupted")
        write_json(state_path, state)
        write_progress(progress_path, trials, state, {}, args)
        return 130

    write_progress(progress_path, trials, state, running, args)
    failures = sum(state.get(trial.name, {}).get("status") == "FAIL" for trial in trials)
    print(f"[{now()}] complete failures={failures} progress={progress_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

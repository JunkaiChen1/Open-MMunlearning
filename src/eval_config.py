"""Configuration resolvers for evaluation output names."""

from pathlib import Path
import re

from omegaconf import OmegaConf


def checkpoint_name(model_path, adapter_path=None):
    source = str(adapter_path or model_path).rstrip("/")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source).name).strip("_") or "checkpoint"
    kind = "adapter" if adapter_path else "model"
    return f"{kind}_{label}"


OmegaConf.register_new_resolver("eval_checkpoint_name", checkpoint_name, replace=True)

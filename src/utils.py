"""Small helpers shared by every training and evaluation script."""

import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    """Make runs reproducible (same subsample, same initialisation)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def model_slug(model_name: str) -> str:
    """'facebook/bart-base' -> 'facebook__bart-base' (safe for folder names)."""
    return model_name.replace("/", "__")


def model_path(cfg: dict, task: str, model_name: str) -> Path:
    """Where a fine-tuned model lives (does not create the folder)."""
    return ROOT / cfg["paths"]["outputs_dir"] / task / model_slug(model_name)


def output_dir(cfg: dict, task: str, model_name: str) -> Path:
    """Same as model_path but creates the folder, for use when training."""
    path = model_path(cfg, task, model_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prefix_for(model_name: str) -> str:
    """T5 was pre-trained with task prefixes; BART needs none."""
    return "summarize: " if "t5" in model_name.lower() else ""


def save_results(cfg: dict, task: str, model_name: str, payload: dict) -> Path:
    """Write one metrics JSON file; src/evaluation/compare.py reads these."""
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{task}__{model_slug(model_name)}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {path}")
    return path


def has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def build_training_args(cls, **kwargs):
    """Create TrainingArguments / Seq2SeqTrainingArguments across transformers versions.

    transformers 4.x has `warmup_ratio`; transformers 5.x removed it and accepts a
    float in `warmup_steps` instead. Unknown arguments are dropped with a notice
    rather than crashing a long Colab run.
    """
    import inspect
    params = inspect.signature(cls.__init__).parameters
    ratio = kwargs.pop("warmup_ratio", None)
    if ratio is not None:
        if "warmup_ratio" in params:
            kwargs["warmup_ratio"] = ratio
        elif "warmup_steps" in params:
            kwargs["warmup_steps"] = ratio
    for key in [k for k in kwargs if k not in params]:
        print(f"[note] this transformers version has no '{key}' argument; ignoring it")
        kwargs.pop(key)
    return cls(**kwargs)

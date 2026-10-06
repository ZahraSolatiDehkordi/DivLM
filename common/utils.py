import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

try:
    from transformers import set_seed as transformers_set_seed
except Exception:
    transformers_set_seed = None


def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def validate_grpo_generation_batch_size(batch_size, world_size, group_size, per_device_batch_size):
    multiple = world_size * math.lcm(group_size, per_device_batch_size)
    if batch_size % multiple:
        raise ValueError(
            f"GRPO generation batch size {batch_size} must be a multiple of {multiple} "
            f"for {world_size} process(es), groups of {group_size}, and per-device batch size "
            f"{per_device_batch_size}. Each process must receive complete generation groups."
        )


def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def set_all_seeds(
    seed,
    deterministic=True,
    use_transformers=False,
    deterministic_algorithms=False,
    set_cublas_workspace=False,
    disable_tf32=False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if use_transformers and transformers_set_seed is not None:
        transformers_set_seed(seed)
    if set_cublas_workspace:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if deterministic_algorithms and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def safe_wandb_call(fn, description):
    try:
        return fn()
    except Exception as exc:
        print(f"W&B {description} failed: {exc}")
        return None


def compact_number(value):
    text = f"{float(value):g}"
    text = text.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
    return text


def slugify(value, default="run"):
    text = str(value).strip().replace("\\", "/").rstrip("/")
    if "/" in text:
        text = text.split("/")[-1]
    text = re.sub(r"[^A-Za-z0-9._+-]+", "-", text).strip("-._")
    return text or default


def make_run_name(*parts):
    clean_parts = [slugify(part) for part in parts if part is not None and str(part).strip()]
    return "_".join(clean_parts) if clean_parts else "run"


def source_tag(source):
    normalized = str(source).replace("\\", "/").rstrip("/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return "run"
    index = len(parts) - 1
    leaf = parts[index]
    if re.match(r"(?:checkpoint|milestone)-\d+$", leaf) and index > 0:
        index -= 1
        leaf = parts[index]
    if leaf in {"adapter", "model", "checkpoints", "milestones"} and index > 0:
        index -= 1
        leaf = parts[index]
    return slugify(leaf)


def adapter_parent_tag(adapter_path):
    return source_tag(adapter_path)


def divlm_source_tag(cpt_base):
    return source_tag(cpt_base).replace("_cpt_delta", "_delta")


def is_complete_adapter_dir(path):
    if not path or not os.path.isdir(path):
        return False
    try:
        with open(os.path.join(path, "adapter_config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict) or not config:
            return False
        weights_path = os.path.join(path, "adapter_model.safetensors")
        if os.path.isfile(weights_path):
            from safetensors import safe_open
            with safe_open(weights_path, framework="pt", device="cpu") as weights:
                return bool(weights.keys())
        weights_path = os.path.join(path, "adapter_model.bin")
        if os.path.isfile(weights_path):
            weights = torch.load(weights_path, map_location="cpu", weights_only=True)
            return isinstance(weights, dict) and bool(weights) and all(
                isinstance(value, torch.Tensor) for value in weights.values()
            )
    except (OSError, ValueError, RuntimeError, EOFError):
        return False
    except Exception as exc:
        print(f"Could not validate adapter at {path}: {exc}")
        return False
    return False


def load_adapter_training_config(adapter_target):
    target = Path(adapter_target).resolve()
    candidates = [target]
    if re.fullmatch(r"(?:checkpoint|milestone)-\d+", target.name):
        candidates.append(target.parent)
        if target.parent.name in {"checkpoints", "milestones"}:
            candidates.append(target.parent.parent)
    elif target.name == "adapter":
        candidates.append(target.parent)
    for directory in candidates:
        path = directory / "full_config.json"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                config = json.load(handle)
            if not isinstance(config, dict):
                raise ValueError(f"Training configuration must be a JSON object: {path}")
            return config
    return {}


def validate_adapter_base(cpt_base, adapter_target, training_config=None):
    if training_config is None:
        training_config = load_adapter_training_config(adapter_target)
    recorded_base = training_config.get("base_model")
    if not recorded_base:
        with open(os.path.join(adapter_target, "adapter_config.json"), encoding="utf-8") as handle:
            recorded_base = json.load(handle).get("base_model_name_or_path")
    if not recorded_base:
        print("Warning: no saved adapter base is available for a consistency check.")
        return
    expected = os.path.normcase(os.path.realpath(os.path.expanduser(str(recorded_base))))
    supplied = os.path.normcase(os.path.realpath(os.path.expanduser(str(cpt_base))))
    if expected != supplied:
        raise ValueError(
            f"Adapter base mismatch: training used {recorded_base!r}, but --cpt-base is {cpt_base!r}. "
            "Use the training base model. If it was moved, update its saved path in full_config.json "
            "(or adapter_config.json when no training configuration exists)."
        )


def resolve_entity_h(override, training_config, default):
    value = override
    if value is None:
        value = training_config.get("h", training_config.get("entity_harshness", default))
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Named entity sensitivity h must be finite and non-negative.")
    return value

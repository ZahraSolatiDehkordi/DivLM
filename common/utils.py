import json
import os
import random
import re

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


def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def set_all_seeds(
    seed,
    deterministic=True,
    use_transformers=False,
    deterministic_algorithms=False,
    set_cublas_workspace=False,
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
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        return False
    adapter_weights = (
        "adapter_model.safetensors",
        "adapter_model.bin",
    )
    return any(os.path.exists(os.path.join(path, name)) for name in adapter_weights)

import gc
import json
import os
import re
from collections import Counter

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import config as cfg
from common.utils import is_complete_adapter_dir


METRICS = [
    "coherence",
    "prompt_adherence",
    "no_meta_commentary",
    "completeness",
    "quality_avg",
    "genre_coverage",
    "tone_coverage",
    "style_coverage",
    "coverage_avg",
    "R_genre",
    "R_tone",
    "R_style",
    "R_ne",
    "diversity_reward_avg",
    "cosine_distance",
    "avg_unique_entities_per_group",
    "entity_repetition_ratio",
    "avg_repeated_entities_per_group",
]


def normalize_prompt_records(items, source):
    records = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            item = {"prompt": item}
        if not isinstance(item, dict) or "prompt" not in item:
            raise ValueError(f"Record {idx} in {source} is missing a prompt field.")
        prompt = str(item["prompt"]).strip()
        if not prompt:
            raise ValueError(f"Record {idx} in {source} has an empty prompt.")
        record = dict(item)
        record["prompt"] = prompt
        record["plain_prompt"] = prompt
        record["test_index"] = idx
        records.append(record)
    return records


def load_json_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "prompts", "examples", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of prompt records.")
    return normalize_prompt_records(data, path)


def load_hf_records(path, split):
    dataset = load_from_disk(path)
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            available = ", ".join(dataset.keys())
            raise ValueError(f"Split '{split}' not found in {path}. Available splits: {available}.")
        dataset = dataset[split]
    if "prompt" not in dataset.column_names:
        raise ValueError(f"The test dataset at {path} must contain a 'prompt' column.")
    return normalize_prompt_records(dataset, path)


def load_test_records(path, split):
    if os.path.isdir(path):
        return load_hf_records(path, split)
    if os.path.isfile(path):
        return load_json_records(path)
    raise FileNotFoundError(f"Test set path not found: {path}")


def load_tokenizer(source):
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def model_kwargs():
    return {"torch_dtype": torch.bfloat16}


def load_causal_lm(source, tokenizer, device):
    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs())
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if model.generation_config.eos_token_id is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.to(device)
    model.eval()
    return model


def adapter_is_complete(path):
    return is_complete_adapter_dir(path)


def checkpoint_step(path):
    match = re.search(r"(?:checkpoint|milestone)-(\d+)$", os.path.basename(path.rstrip(os.sep)))
    return int(match.group(1)) if match else None


def latest_adapter_target(adapter_dir):
    if adapter_is_complete(adapter_dir):
        return adapter_dir
    final_adapter_dir = os.path.join(adapter_dir, "adapter")
    if adapter_is_complete(final_adapter_dir):
        return final_adapter_dir
    candidates = []
    search_dirs = [
        adapter_dir,
        os.path.join(adapter_dir, "checkpoints"),
        os.path.join(adapter_dir, "milestones"),
    ]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for name in os.listdir(search_dir):
            path = os.path.join(search_dir, name)
            step = checkpoint_step(path)
            if step is not None and os.path.isdir(path) and adapter_is_complete(path):
                candidates.append((step, path))
    if not candidates:
        raise FileNotFoundError(
            f"No complete adapter found in {adapter_dir}, its adapter subdirectory, or numeric checkpoint/milestone children."
        )
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def tokenizer_source_for_adapter(adapter_target, cpt_base):
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    if any(os.path.exists(os.path.join(adapter_target, name)) for name in tokenizer_files):
        return adapter_target
    return cpt_base


def load_adapter_model(cpt_base, adapter_target, tokenizer, device):
    base = load_causal_lm(cpt_base, tokenizer, device)
    model = PeftModel.from_pretrained(base, adapter_target)
    model.eval()
    return model


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean_value(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def average_present(values):
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(np.mean(present))


def coverage_by_group(values, group_size, possible):
    coverage_values = []
    for group_start in range(0, len(values), group_size):
        group = values[group_start:group_start + group_size]
        if len(group) != group_size:
            continue
        coverage_values.append(len(set(group)) / possible)
    return mean_value(coverage_values)


def entity_group_metrics(entity_sets, group_size):
    unique_per_group = []
    repeated_per_group = []
    repetition_ratios = []
    for group_start in range(0, len(entity_sets), group_size):
        group_sets = entity_sets[group_start:group_start + group_size]
        if len(group_sets) != group_size:
            continue
        freq = Counter()
        for entity_set in group_sets:
            for entity in entity_set:
                freq[entity] += 1
        total_group_entities = len(freq)
        unique_group_entities = sum(1 for count in freq.values() if count == 1)
        repeated_group_entities = sum(1 for count in freq.values() if count > 1)
        unique_per_group.append(unique_group_entities)
        repeated_per_group.append(repeated_group_entities)
        repetition_ratios.append(repeated_group_entities / total_group_entities if total_group_entities else 0.0)
    return {
        "avg_unique_entities_per_group": mean_value(unique_per_group),
        "entity_repetition_ratio": mean_value(repetition_ratios),
        "avg_repeated_entities_per_group": mean_value(repeated_per_group),
    }


def cosine_distance_mean(embeddings, group_size):
    if embeddings is None:
        return None
    distances = []
    for group_start in range(0, len(embeddings), group_size):
        group = embeddings[group_start:group_start + group_size]
        if len(group) < 2:
            continue
        sim = group @ group.T
        sim = (sim + 1.0) / 2.0
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        distances.append(1.0 - float(np.mean(sim[mask])))
    return mean_value(distances)


def model_table_rows(model_results, model_keys):
    rows = []
    for metric in METRICS:
        row = [metric]
        for model_key in model_keys:
            value = model_results.get(model_key, {}).get(metric)
            row.append(value if value is not None else "N/A")
        rows.append(row)
    return rows


def evaluate_outputs(outputs, reward_evaluator, encoder, group_size, num_generations):
    prompts = [item["prompt"] for item in outputs]
    completions = [item["generated"] for item in outputs]
    if len(completions) % group_size != 0:
        raise ValueError("Number of outputs must be divisible by group size.")
    reward_evaluator.compute_rewards(completions, prompts)
    components = reward_evaluator.last_components
    embeddings = None
    if encoder is not None:
        embeddings = encoder.encode(completions, convert_to_numpy=True, normalize_embeddings=True)
    quality_metrics = {
        "coherence": mean_value(components["coherence"]),
        "prompt_adherence": mean_value(components["adherence"]),
        "no_meta_commentary": mean_value(components["no_meta_commentary"]),
        "completeness": mean_value(components["completeness"]),
    }
    quality_metrics["quality_avg"] = average_present(quality_metrics.values())
    coverage_metrics = {
        "genre_coverage": coverage_by_group(components["genres"], group_size, cfg.GENRE_COUNT),
        "tone_coverage": coverage_by_group(components["tones"], group_size, cfg.TONE_COUNT),
        "style_coverage": coverage_by_group(components["styles"], group_size, cfg.STYLE_COUNT),
    }
    coverage_metrics["coverage_avg"] = average_present(coverage_metrics.values())
    diversity_reward_metrics = {
        "R_genre": mean_value(components["R_genre"]),
        "R_tone": mean_value(components["R_tone"]),
        "R_style": mean_value(components["R_style"]),
        "R_ne": mean_value(components["R_ne"]),
    }
    diversity_reward_metrics["diversity_reward_avg"] = average_present(diversity_reward_metrics.values())
    model_results = {
        "num_prompts": len(outputs) // num_generations,
        "num_outputs": len(outputs),
    }
    model_results.update(quality_metrics)
    model_results.update(coverage_metrics)
    model_results.update(diversity_reward_metrics)
    model_results["cosine_distance"] = cosine_distance_mean(embeddings, group_size)
    model_results.update(entity_group_metrics(components["normalized_entity_sets"], group_size))
    return model_results

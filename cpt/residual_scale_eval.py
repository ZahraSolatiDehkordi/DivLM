import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import wandb

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import config as cfg
from common.utils import env_int, make_run_name, safe_wandb_call, save_json, set_all_seeds, source_tag
from grpo.grpo_eval_core import evaluate_outputs, load_causal_lm, load_test_records, load_tokenizer, model_table_rows, unload_model
from grpo.grpo_generation import GenerationSettings, format_prompt, generate_outputs, make_system_prompt
from grpo.grpo_reward import FinalReward


parser = argparse.ArgumentParser(description="Evaluate one instruction-residual scale before GRPO.")
parser.add_argument("--instruct-model", type=str, required=True)
parser.add_argument("--residual-model-dir", type=str, required=True)
parser.add_argument("--test-set-path", type=str, required=True)
parser.add_argument("--test-split", type=str, default="test")
parser.add_argument("--output-dir", type=str, required=True, help="Root directory for evaluation outputs.")
parser.add_argument("--cpt-model", type=str, default=None)
parser.add_argument("--wandb-project", type=str, default=cfg.DIVLM_TEST_PROJECT)
parser.add_argument("--wandb", type=str, choices=("true", "false"), default="true")
parser.add_argument("--run-name", type=str, default=None, help="Optional run folder name.")
parser.add_argument("--seed", type=int, default=env_int("GRPO_EVAL_SEED", cfg.DEFAULT_SEED))
parser.add_argument("--num-generations", type=int, default=cfg.GROUP_SIZE)
parser.add_argument(
    "--generation-batch-size",
    "--gen-batch-size",
    dest="gen_batch_size",
    type=int,
    default=cfg.EVAL_GENERATION_BATCH_SIZE,
)
parser.add_argument("--judge-batch-size", type=int, default=cfg.JUDGE_BATCH_SIZE)
parser.add_argument("--h", type=float, default=cfg.ENTITY_H)
parser.add_argument("--length-penalty", type=str, choices=("true", "false"), default="true")
parser.add_argument("--quality-gate", type=str, choices=("true", "false"), default="true")
parser.add_argument("--group-quality-tau", type=float, default=cfg.GROUP_QUALITY_TAU)
parser.add_argument("--quality-tau", type=float, default=cfg.QUALITY_TAU)
parser.add_argument("--group-size", type=int, default=cfg.GROUP_SIZE)
parser.add_argument("--entity-clip-u", type=float, default=cfg.ENTITY_CLIP_U)
parser.add_argument("--max-new-tokens", type=int, default=cfg.MAX_NEW_TOKENS)
parser.add_argument("--target-words", type=int, default=cfg.TARGET_WORDS)
parser.add_argument("--temperature", type=float, default=cfg.GEN_TEMPERATURE)
parser.add_argument("--top-p", type=float, default=cfg.GEN_TOP_P)
parser.add_argument("--top-k", type=int, default=cfg.GEN_TOP_K)
parser.add_argument("--encoder-model", type=str, default=cfg.SENTENCE_ENCODER_MODEL)
args = parser.parse_args()
args.wandb = args.wandb == "true"
args.length_penalty = args.length_penalty == "true"
args.quality_gate = args.quality_gate == "true"


def validate_args():
    if args.h < 0:
        raise ValueError("--h must be non-negative.")
    if args.num_generations != args.group_size:
        raise ValueError("--num-generations must match --group-size for grouped evaluation.")
    if not 0.0 <= args.group_quality_tau <= 1.0:
        raise ValueError("--group-quality-tau must be between 0 and 1.")
    if not os.path.isdir(args.residual_model_dir):
        raise FileNotFoundError(f"Residual model directory not found: {args.residual_model_dir}")


def residual_label(path, existing_labels):
    tag = source_tag(path)
    match = re.search(r"(delta[0-9]+(?:\.[0-9]+)?)", tag)
    label = match.group(1) if match else tag
    if label not in existing_labels:
        return label
    suffix = 2
    while f"{label}_{suffix}" in existing_labels:
        suffix += 1
    return f"{label}_{suffix}"


def make_generation_settings():
    return GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=cfg.GEN_REPETITION_PENALTY,
        max_prompt_length=cfg.MAX_PROMPT_LENGTH,
        num_generations=args.num_generations,
        batch_size=args.gen_batch_size,
    )


def make_reward_evaluator(device):
    return FinalReward(
        device=device,
        group_size=args.group_size,
        entity_clip_u=args.entity_clip_u,
        h=args.h,
        quality_tau=args.quality_tau,
        use_group_quality_tau=args.quality_gate,
        group_quality_tau=args.group_quality_tau,
        use_length_penalty=args.length_penalty,
        judge_batch_size=args.judge_batch_size,
    )


def generate_model_outputs(model_key, source, plain_prompts, generation_settings, device, system_prompt):
    print(f"Loading {model_key} from {source}")
    tokenizer = load_tokenizer(source)
    formatted_prompts = [format_prompt(prompt, tokenizer, system_prompt=system_prompt) for prompt in plain_prompts]
    model = load_causal_lm(source, tokenizer, device)
    outputs = generate_outputs(
        model,
        tokenizer,
        formatted_prompts,
        plain_prompts,
        model_key,
        generation_settings,
        seed_fn=lambda: set_all_seeds(
            args.seed,
            use_transformers=True,
            deterministic_algorithms=True,
            set_cublas_workspace=True,
        ),
    )
    tokenizer_source = getattr(tokenizer, "name_or_path", source)
    unload_model(model)
    return outputs, tokenizer_source


def main():
    validate_args()
    set_all_seeds(
        args.seed,
        use_transformers=True,
        deterministic_algorithms=True,
        set_cublas_workspace=True,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_records = load_test_records(args.test_set_path, args.test_split)
    plain_prompts = [record["plain_prompt"] for record in test_records]
    generation_settings = make_generation_settings()
    system_prompt = make_system_prompt(args.target_words)

    model_sources = [("INS", args.instruct_model)]
    if args.cpt_model is not None:
        model_sources.append(("CPT", args.cpt_model))
    labels = {label for label, _ in model_sources}
    residual_model_label = residual_label(args.residual_model_dir, labels)
    model_sources.append((residual_model_label, args.residual_model_dir))

    run_name = args.run_name or make_run_name(
        "residual_scale_eval",
        residual_model_label,
        source_tag(args.test_set_path),
    )
    run_dir = os.path.join(args.output_dir, run_name)
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = os.path.join(run_dir, "wandb")
    os.environ["TMPDIR"] = os.path.join(run_dir, "tmp")
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    wandb_run = None
    if args.wandb:
        wandb_run = safe_wandb_call(
            lambda: wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "output_root": args.output_dir,
                    "run_dir": run_dir,
                    "instruct_model": args.instruct_model,
                    "cpt_model": args.cpt_model,
                    "residual_model_dir": args.residual_model_dir,
                    "residual_model_label": residual_model_label,
                    "test_set_path": args.test_set_path,
                    "test_split": args.test_split,
                    "num_test_prompts": len(test_records),
                    "seed": args.seed,
                    "G": args.group_size,
                    "h": args.h,
                    "u": args.entity_clip_u,
                    "tau": args.quality_tau,
                    "quality_gate": args.quality_gate,
                    "group_quality_tau": args.group_quality_tau,
                    "length_penalty": args.length_penalty,
                    "target_words": args.target_words,
                    "system_prompt": system_prompt,
                    "max_new_tokens": args.max_new_tokens,
                    "generation_batch_size": args.gen_batch_size,
                    "judge_batch_size": args.judge_batch_size,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                },
            ),
            "init",
        )

    outputs_by_model = {}
    tokenizers = {}
    for model_key, source in model_sources:
        outputs, tokenizer_source = generate_model_outputs(
            model_key,
            source,
            plain_prompts,
            generation_settings,
            device,
            system_prompt,
        )
        outputs_by_model[model_key] = outputs
        tokenizers[model_key] = tokenizer_source

    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(args.encoder_model, device=device)
    except Exception as exc:
        print(f"Sentence encoder unavailable; cosine distance will be skipped: {exc}")

    reward_evaluator = make_reward_evaluator(device)
    results_by_model = {}
    model_keys = [label for label, _ in model_sources]
    for model_key in model_keys:
        print(f"Evaluating {model_key}")
        results_by_model[model_key] = evaluate_outputs(
            outputs_by_model[model_key],
            reward_evaluator,
            encoder,
            args.group_size,
            args.num_generations,
        )

    results = {
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "configuration": {
            "output_root": args.output_dir,
            "run_dir": run_dir,
            "instruct_model": args.instruct_model,
            "cpt_model": args.cpt_model,
            "residual_model_dir": args.residual_model_dir,
            "residual_model_label": residual_model_label,
            "model_order": model_keys,
            "test_set_path": args.test_set_path,
            "test_split": args.test_split,
            "num_test_prompts": len(test_records),
            "seed": args.seed,
            "G": args.group_size,
            "h": args.h,
            "u": args.entity_clip_u,
            "tau": args.quality_tau,
            "quality_gate": args.quality_gate,
            "group_quality_tau": args.group_quality_tau,
            "length_penalty": args.length_penalty,
            "target_words": args.target_words,
            "system_prompt": system_prompt,
            "max_new_tokens": args.max_new_tokens,
            "generation_batch_size": args.gen_batch_size,
            "judge_batch_size": args.judge_batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "tokenizers": tokenizers,
        },
        "results": results_by_model,
    }
    results_path = os.path.join(metrics_dir, "eval_metrics.json")
    save_json(results_path, results)
    print(f"Saved metrics to {results_path}")

    if wandb_run is not None:
        metrics_table = wandb.Table(
            columns=["Metric", *model_keys],
            data=model_table_rows(results_by_model, model_keys),
        )
        safe_wandb_call(lambda: wandb.log({"metrics_table": metrics_table}), "table log")
        safe_wandb_call(lambda: wandb.finish(), "finish")


if __name__ == "__main__":
    main()

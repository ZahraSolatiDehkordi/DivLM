import argparse
import json
import logging
import os
import warnings
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import config as cfg
from common import utils as common_utils
from common.utils import divlm_source_tag, is_complete_adapter_dir, make_run_name, safe_wandb_call, set_all_seeds
from grpo import grpo_callbacks, grpo_generation, grpo_judge, grpo_reward, grpo_monitor
from grpo.grpo_callbacks import MilestoneCheckpointCallback
from grpo.grpo_generation import format_prompt, get_generation_eos_token_id, make_system_prompt
from grpo.grpo_reward import FinalReward
from grpo.grpo_monitor import (
    CollapseMonitorConfig,
    create_monitor_callback,
    resolve_resume_checkpoint,
    run_training_with_monitor,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("accelerate.utils.modeling").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The following layers were not sharded")
warnings.filterwarnings("ignore", message=".*layers were not sharded.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='GRPO Training Script')
parser.add_argument('--output-dir', type=str, required=True,
    help='Root directory for saving the GRPO run')
parser.add_argument('--run-name', type=str, default=None,
    help='Optional run folder name. If omitted, a name is generated from the inputs.')
parser.add_argument('--cpt-base', type=str, required=True,
    help='CPT+delta_theta base')
parser.add_argument('--train-dataset-path', type=str, required=True,
    help='Path to the training dataset saved with datasets.save_to_disk')
parser.add_argument('--h',
                    type=float, default=cfg.ENTITY_H,
                    help=f'Sensitivity parameter h for named entity reward. Default: {cfg.ENTITY_H}.')
parser.add_argument('--length-penalty', type=str, choices=("true", "false"), default="true",
                    help='Enable or disable the length penalty. Default: true.')
parser.add_argument('--quality-gate', type=str, choices=("true", "false"), default="true",
                    help='Enable or disable the group-level quality gate. Default: true.')
parser.add_argument('--target-words', type=int, default=cfg.TARGET_WORDS,
                    help=f'Target word count used in the generation prompt. Use 0 to omit the explicit word count. Default: {cfg.TARGET_WORDS}.')
parser.add_argument('--group-quality-tau', type=float, default=cfg.GROUP_QUALITY_TAU,
                    help=f'Group-level quality threshold used when the group quality gate is enabled. Default: {cfg.GROUP_QUALITY_TAU}.')
parser.add_argument('--epochs', type=float, default=cfg.GRPO_EPOCHS,
                    help=f'Number of training epochs. Default: {cfg.GRPO_EPOCHS}.')
parser.add_argument('--generation-batch-size', '--gen-batch-size', dest='generation_batch_size',
                    type=int, default=None,
                    help='Generation batch size used by GRPO. Default: 200 with multiple GPUs, otherwise 50.')
parser.add_argument('--judge-batch-size', type=int, default=cfg.JUDGE_BATCH_SIZE,
                    help=f'Batch size used by the LLM judge. Default: {cfg.JUDGE_BATCH_SIZE}.')
parser.add_argument('--max-monitor-restarts', type=int, default=cfg.MONITOR_MAX_RESTARTS,
                    help=f'Maximum number of collapse-monitor restarts. Default: {cfg.MONITOR_MAX_RESTARTS}.')
parser.add_argument('--accept-after-fraction', type=float, default=cfg.MONITOR_ACCEPT_AFTER_FRACTION,
                    help=f'Finalize a rollback checkpoint instead of restarting if collapse occurs after this training fraction. Default: {cfg.MONITOR_ACCEPT_AFTER_FRACTION}.')
parser.add_argument('--wandb', type=str, choices=("true", "false"), default="true",
                    help='Enable or disable W&B logging. Default: true.')
args = parser.parse_args()
args.length_penalty = args.length_penalty == "true"
args.quality_gate = args.quality_gate == "true"
args.wandb = args.wandb == "true"

QUALITY_TAU = cfg.QUALITY_TAU
USE_GROUP_QUALITY_TAU = args.quality_gate
GROUP_QUALITY_TAU = float(args.group_quality_tau)
if not 0.0 <= GROUP_QUALITY_TAU <= 1.0:
    raise ValueError("--group-quality-tau must be between 0 and 1.")
INVALID_R_QUAL = cfg.INVALID_R_QUAL
R_DIV_NORMALIZER = cfg.R_DIV_NORMALIZER
ENTITY_CLIP_U = cfg.ENTITY_CLIP_U
H = float(args.h)
if H < 0:
    raise ValueError("--h must be non-negative.")
LENGTH_PENALTY_TARGET_WORDS = cfg.LENGTH_PENALTY_TARGET_WORDS
USE_LENGTH_PENALTY = args.length_penalty
TARGET_WORDS = int(args.target_words)
SYSTEM_PROMPT = make_system_prompt(TARGET_WORDS)
GRPO_LEARNING_RATE = cfg.GRPO_LEARNING_RATE
GRPO_BETA = cfg.GRPO_BETA
GRPO_MAX_GRAD_NORM = cfg.GRPO_MAX_GRAD_NORM
GEN_TEMPERATURE = cfg.GEN_TEMPERATURE
GEN_TOP_P = cfg.GEN_TOP_P
GEN_TOP_K = cfg.GEN_TOP_K
NUM_TRAIN_EPOCHS = float(args.epochs)
if NUM_TRAIN_EPOCHS <= 0:
    raise ValueError("--epochs must be greater than 0.")
if args.generation_batch_size is not None and args.generation_batch_size <= 0:
    raise ValueError("--generation-batch-size must be greater than 0.")
JUDGE_BATCH_SIZE = int(args.judge_batch_size)
if JUDGE_BATCH_SIZE <= 0:
    raise ValueError("--judge-batch-size must be greater than 0.")
MAX_MONITOR_RESTARTS = int(args.max_monitor_restarts)
if MAX_MONITOR_RESTARTS < 0:
    raise ValueError("--max-monitor-restarts must be non-negative.")
ACCEPT_AFTER_FRACTION = float(args.accept_after_fraction)
if not 0.0 <= ACCEPT_AFTER_FRACTION <= 1.0:
    raise ValueError("--accept-after-fraction must be between 0 and 1.")

MONITOR_CONFIG = CollapseMonitorConfig(
    max_restarts=MAX_MONITOR_RESTARTS,
    accept_after_fraction=ACCEPT_AFTER_FRACTION,
)

base_model = args.cpt_base
ds_path = args.train_dataset_path
run_name = args.run_name or make_run_name(
    "DivLM",
    divlm_source_tag(base_model),
)
run_dir = os.path.join(args.output_dir, run_name)
adapter_dir = run_dir
checkpoint_dir = os.path.join(run_dir, "checkpoints")
milestone_dir = os.path.join(run_dir, "milestones")
code_dir = os.path.join(run_dir, "code")
wandb_dir = os.path.join(run_dir, "wandb")
tmp_dir = os.path.join(run_dir, "tmp")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(milestone_dir, exist_ok=True)
os.makedirs(code_dir, exist_ok=True)
os.makedirs(wandb_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
os.environ["WANDB_DIR"] = wandb_dir
os.environ["TMPDIR"] = tmp_dir

print(f"Run directory: {run_dir}")
print(f"Final adapter directory: {adapter_dir}")
print(f"Checkpoint directory: {checkpoint_dir}")
print(f"Milestone directory: {milestone_dir}")
print(f"Beta set to {GRPO_BETA} (run: {os.path.basename(run_dir.rstrip('/'))})")
print(f"Named entity sensitivity h set to {H}")
print(f"Length penalty enabled: {USE_LENGTH_PENALTY}")
print(f"Target words in prompt: {TARGET_WORDS}")
print(f"Quality gate enabled: {USE_GROUP_QUALITY_TAU} (group tau={GROUP_QUALITY_TAU})")
print(f"W&B logging enabled: {args.wandb}")
print(f"Training epochs: {NUM_TRAIN_EPOCHS}")
print(f"Judge batch size: {JUDGE_BATCH_SIZE}")
print(f"Monitor max restarts: {MAX_MONITOR_RESTARTS}")
print(f"Monitor accept-after fraction: {ACCEPT_AFTER_FRACTION}")

SEED = cfg.DEFAULT_SEED
set_all_seeds(SEED)

GROUP_SIZE = cfg.GROUP_SIZE
MAX_NEW_TOKENS = cfg.MAX_NEW_TOKENS

TRAIN_SIZE = cfg.TRAIN_SIZE
PER_DEVICE_TRAIN_BATCH_SIZE = cfg.GRPO_PER_DEVICE_TRAIN_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS = cfg.GRPO_GRADIENT_ACCUMULATION_STEPS
REWARD_PRINT_EVERY = cfg.REWARD_PRINT_EVERY

final_adapter_exists = is_complete_adapter_dir(adapter_dir)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_main_process = global_rank == 0

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

print(f"Target device: {device} | local_rank={local_rank} | global_rank={global_rank} | world_size={world_size}")

if final_adapter_exists:
    print("\n" + "=" * 80)
    print(f"ADAPTER FOUND IN '{adapter_dir}'")
    exit(0)

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

project_name = os.path.basename(os.path.normpath(run_dir))
if not project_name:
    project_name = "grpo_diversity_run"

wandb_run_name = project_name

if args.wandb and is_main_process:
    safe_wandb_call(
        lambda: wandb.init(
            project=cfg.DIVLM_TRAIN_PROJECT,
            name=wandb_run_name,
            resume="allow",
            id=wandb_run_name,
            config={
                "model": base_model,
                "output_root": args.output_dir,
                "run_dir": run_dir,
                "adapter_dir": adapter_dir,
                "checkpoint_dir": checkpoint_dir,
                "milestone_dir": milestone_dir,
                "wandb_dir": wandb_dir,
                "tmp_dir": tmp_dir,
                "G": GROUP_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "learning_rate": GRPO_LEARNING_RATE,
                "beta": GRPO_BETA,
                "epochs": NUM_TRAIN_EPOCHS,
                "requested_generation_batch_size": args.generation_batch_size,
                "judge_batch_size": JUDGE_BATCH_SIZE,
                "tau": QUALITY_TAU,
                "quality_gate_enabled": USE_GROUP_QUALITY_TAU,
                "group_tau_enabled": USE_GROUP_QUALITY_TAU,
                "group_tau": GROUP_QUALITY_TAU,
                "invalid_R_qual": INVALID_R_QUAL,
                "R_div_normalizer": R_DIV_NORMALIZER,
                "u": ENTITY_CLIP_U,
                "h": H,
                "use_length_penalty": USE_LENGTH_PENALTY,
                "target_words": TARGET_WORDS,
                "system_prompt": SYSTEM_PROMPT,
                "length_penalty_target_words": LENGTH_PENALTY_TARGET_WORDS,
                "max_grad_norm": GRPO_MAX_GRAD_NORM,
                "generation_temperature": GEN_TEMPERATURE,
                "generation_top_p": GEN_TOP_P,
                "generation_top_k": GEN_TOP_K,
                "monitor": MONITOR_CONFIG.as_dict(),
                "batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "world_size": world_size,
            },
        ),
        "init",
    )

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
)
model = model.to(device)

print(f"\n{'=' * 60}")
print(f"ATTEMPTING TO MOVE MODEL TO {device}")
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
print(f"{'=' * 60}")

print(f"\n{'=' * 60}")
print(f"DEVICE CHECK:")
print(f"Using device: {device}")
print(f"Model is on: {next(model.parameters()).device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
print(f"{'=' * 60}\n")

if len(tokenizer) > model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
if model.generation_config.eos_token_id is None:
    model.generation_config.eos_token_id = tokenizer.eos_token_id
generation_eos_token_id = get_generation_eos_token_id(model, tokenizer)

train_ds = load_from_disk(ds_path)

print("SYSTEM PROMPT:", SYSTEM_PROMPT)

print("Cleaning prompts in datasets...")
train_ds = train_ds.map(lambda x: {'plain_prompt': x['prompt']}, batched=False)

def format_prompts_batch(examples):
    examples['prompt'] = [format_prompt(p, tokenizer, system_prompt=SYSTEM_PROMPT) for p in examples['prompt']]
    return examples

train_ds = train_ds.map(format_prompts_batch, batched=True, desc="Formatting train")
print("Prompt cleaning complete!")

train_ds_select = train_ds.select(range(min(TRAIN_SIZE, len(train_ds))))

print(train_ds_select[0].keys())

print("[dataset] Train size:", len(train_ds_select))

if is_main_process:
    module_paths = [
        grpo_judge.__file__,
        grpo_reward.__file__,
        grpo_callbacks.__file__,
        cfg.__file__,
        grpo_generation.__file__,
        common_utils.__file__,
        grpo_monitor.__file__,
    ]
    for module_path in module_paths:
        copied_path = os.path.join(code_dir, os.path.basename(module_path))
        shutil.copy2(module_path, copied_path)

reward_evaluator = FinalReward(
    device=device,
    group_size=GROUP_SIZE,
    entity_clip_u=ENTITY_CLIP_U,
    h=H,
    quality_tau=QUALITY_TAU,
    use_group_quality_tau=USE_GROUP_QUALITY_TAU,
    group_quality_tau=GROUP_QUALITY_TAU,
    invalid_r_qual=INVALID_R_QUAL,
    r_div_normalizer=R_DIV_NORMALIZER,
    use_length_penalty=USE_LENGTH_PENALTY,
    length_penalty_target_words=LENGTH_PENALTY_TARGET_WORDS,
    judge_batch_size=JUDGE_BATCH_SIZE,
)

training_rewards_history = []
reward_batch_means = []

def reward_func(completions, **kwargs):
    all_rewards = []
    batch_prompts = kwargs["prompts"]

    G = GROUP_SIZE
    M = len(completions)

    if M != len(batch_prompts):
        raise ValueError(
            f"Lengths mismatch: completions={M}, prompts={len(batch_prompts)}"
        )
    if M % G != 0:
        raise ValueError(
            f"Completions are not divisible by group size: M={M}, G={G}"
        )

    batch_plain_prompts = kwargs["plain_prompt"]
    num_groups = M // G
    plain_prompt_count = (
        1 if isinstance(batch_plain_prompts, str) else len(batch_plain_prompts)
    )
    if plain_prompt_count not in (M, num_groups):
        raise ValueError(
            f"Unexpected plain_prompt length: {plain_prompt_count}; expected {M} "
            f"repeated prompts or {num_groups} unique prompts."
        )

    all_completions = list(completions)
    all_prompts = []
    for g_start in range(0, M, G):
        if plain_prompt_count == M:
            prompt_block = batch_plain_prompts[g_start:g_start + G]
            prompt = prompt_block[0]
            if any(block_prompt != prompt for block_prompt in prompt_block):
                raise ValueError(
                    "Expected each contiguous generation group to contain one repeated prompt, "
                    f"but group starting at index {g_start} contains mixed prompts."
                )
        elif isinstance(batch_plain_prompts, str):
            prompt = batch_plain_prompts
        else:
            prompt = batch_plain_prompts[g_start // G]
        all_prompts.extend([prompt] * G)

    all_group_rewards = reward_evaluator.compute_rewards(all_completions, all_prompts)

    training_rewards_history.extend(all_group_rewards.tolist())
    all_rewards.extend(all_group_rewards.tolist())

    if is_main_process:
        reward_batch_means.append(float(np.mean(all_group_rewards)))
        if len(reward_batch_means) % REWARD_PRINT_EVERY == 0:
            recent_reward = float(np.mean(reward_batch_means[-REWARD_PRINT_EVERY:]))
            print(f"[reward] batches={len(reward_batch_means)} mean_last_{REWARD_PRINT_EVERY}={recent_reward:.4f}")

    if args.wandb and is_main_process and wandb.run is not None and hasattr(reward_evaluator, "component_stats"):
        safe_wandb_call(lambda: wandb.log(reward_evaluator.component_stats), "reward log")

    return torch.tensor(all_rewards, dtype=torch.float32, device=device)

lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

)

model_with_lora = get_peft_model(model, lora_config)
model_with_lora.print_trainable_parameters()

num_gpus = torch.cuda.device_count()
GENERATION_BATCH_SIZE = (
    int(args.generation_batch_size)
    if args.generation_batch_size is not None
    else (
        cfg.TRAIN_GENERATION_BATCH_SIZE_MULTI_GPU
        if num_gpus > 1
        else cfg.TRAIN_GENERATION_BATCH_SIZE_SINGLE_GPU
    )
)
print(f"\n{'=' * 80}")
print(f"GPU Configuration:")
print(f"  Available GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"  Generation batch size: {GENERATION_BATCH_SIZE}")
print(f"{'=' * 80}\n")

model_with_lora.enable_input_require_grads()
model_with_lora.gradient_checkpointing_enable()

cpu_count = os.cpu_count() or 1
dataloader_workers = max(1, min(8, cpu_count // max(1, world_size)))

training_args = GRPOConfig(

    num_train_epochs=NUM_TRAIN_EPOCHS,
    output_dir=checkpoint_dir,

    num_generations=GROUP_SIZE,

    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=GRPO_LEARNING_RATE,
    lr_scheduler_type="cosine",
    logging_steps=100,

    generation_kwargs={
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": GEN_TEMPERATURE,
        "top_p": GEN_TOP_P,
        "top_k": GEN_TOP_K,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": generation_eos_token_id,
        "repetition_penalty": cfg.GEN_REPETITION_PENALTY,

    },

    use_vllm=False,
    bf16=True,

    generation_batch_size=GENERATION_BATCH_SIZE,

    max_grad_norm=GRPO_MAX_GRAD_NORM,
    warmup_ratio=0.1,

    report_to="none",

    save_strategy="steps",
    save_steps=200,
    save_total_limit=50,

    dataloader_num_workers=dataloader_workers,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,

    optim="adamw_torch_fused",
    gradient_checkpointing_kwargs={"use_reentrant": False},

    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=50,

    disable_tqdm=not is_main_process,
    logging_first_step=True,
    logging_nan_inf_filter=False,

    beta=GRPO_BETA

)

effective_batch_size = (training_args.per_device_train_batch_size *
                        training_args.gradient_accumulation_steps *
                        max(1, world_size))
print(f"Effective batch size: {effective_batch_size}")
print(f"  = {training_args.per_device_train_batch_size} (per device)")
print(f"  x {training_args.gradient_accumulation_steps} (grad accum)")
print(f"  x {max(1, world_size)} (world size)")

if args.wandb and is_main_process and wandb.run is not None:
    safe_wandb_call(lambda: wandb.config.update({
        "effective_batch_size": effective_batch_size,
        "num_gpus": num_gpus,
        "world_size": world_size,
        "per_device_batch_size": training_args.per_device_train_batch_size,
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "judge_batch_size": JUDGE_BATCH_SIZE,
        "dataloader_workers_per_rank": dataloader_workers,
    }), "config update")

milestone_callback = MilestoneCheckpointCallback(
    save_steps=500,
    adapter_dir=milestone_dir
)

monitor_callback = create_monitor_callback(
    adapter_dir=checkpoint_dir,
    config=MONITOR_CONFIG,
    is_main_process=is_main_process,
    reward_history=training_rewards_history,
)

if not hasattr(model_with_lora, "warnings_issued"):
    model_with_lora.warnings_issued = {}

trainer = GRPOTrainer(
    model=model_with_lora,
    args=training_args,
    train_dataset=train_ds_select,
    reward_funcs=reward_func,
    processing_class=tokenizer,
    callbacks=[milestone_callback, monitor_callback]
)

config_dict = training_args.to_dict()
config_dict.update({
    "output_root": args.output_dir,
    "run_dir": run_dir,
    "adapter_dir": adapter_dir,
    "checkpoint_dir": checkpoint_dir,
    "milestone_dir": milestone_dir,
    "wandb_dir": wandb_dir,
    "tmp_dir": tmp_dir,
    "base_model": base_model,
    "train_size": TRAIN_SIZE,
    "epochs": NUM_TRAIN_EPOCHS,
    "world_size": world_size,
    "generation_batch_size": GENERATION_BATCH_SIZE,
    "judge_batch_size": JUDGE_BATCH_SIZE,
    "dataloader_workers_per_rank": dataloader_workers,
    "G": GROUP_SIZE,
    "max_new_tokens": MAX_NEW_TOKENS,
    "tau": QUALITY_TAU,
    "quality_gate_enabled": USE_GROUP_QUALITY_TAU,
    "group_tau_enabled": USE_GROUP_QUALITY_TAU,
    "group_tau": GROUP_QUALITY_TAU,
    "invalid_R_qual": INVALID_R_QUAL,
    "R_div_normalizer": R_DIV_NORMALIZER,
    "u": ENTITY_CLIP_U,
    "h": H,
    "use_length_penalty": USE_LENGTH_PENALTY,
    "target_words": TARGET_WORDS,
    "system_prompt": SYSTEM_PROMPT,
    "length_penalty_target_words": LENGTH_PENALTY_TARGET_WORDS,
    "grpo_beta": GRPO_BETA,
    "learning_rate": GRPO_LEARNING_RATE,
    "max_grad_norm": GRPO_MAX_GRAD_NORM,
    "generation_temperature": GEN_TEMPERATURE,
    "generation_top_p": GEN_TOP_P,
    "generation_top_k": GEN_TOP_K,
    "monitor": MONITOR_CONFIG.as_dict(),
    "seed": SEED,
})

if is_main_process:
    with open(os.path.join(run_dir, "full_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

start = time.time()


def finalize_from_monitor_checkpoint(checkpoint_path):
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Monitor final checkpoint does not exist: {checkpoint_path}")
    if not is_main_process:
        return

    print("\n" + "=" * 80)
    print(f"FINALIZING MONITOR CHECKPOINT: {checkpoint_path}")
    print("=" * 80)

    excluded = {
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "rng_state.pth",
        "scaler.pt",
    }
    for name in os.listdir(checkpoint_path):
        if name in excluded:
            continue
        src = os.path.join(checkpoint_path, name)
        dst = os.path.join(adapter_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    tokenizer.save_pretrained(adapter_dir)
    print(f"Final adapter files copied to {adapter_dir}")


def finish_wandb(exit_code):
    if args.wandb and wandb.run is not None:
        safe_wandb_call(lambda: wandb.finish(exit_code=exit_code), "finish")

resume_step, resume_from, resume_reason, found_steps = resolve_resume_checkpoint(
    adapter_dir=checkpoint_dir,
    config=MONITOR_CONFIG,
    is_main_process=is_main_process,
)

if resume_from:
    print(f"Found checkpoint steps: {found_steps}")
    print(f"Resuming from {resume_reason}: step {resume_step} -> {resume_from}")
else:
    print("No checkpoint found, starting from scratch.")

run_training_with_monitor(
    trainer=trainer,
    resume_from=resume_from,
    monitor_callback=monitor_callback,
    config=MONITOR_CONFIG,
    is_main_process=is_main_process,
    finish_on_exit=finish_wandb,
    finalize_on_collapse=finalize_from_monitor_checkpoint,
)

end = time.time()
length = end - start
print("Training took", length, "seconds (", length / 60, "minutes )")

if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    reserved = torch.cuda.memory_reserved(local_rank) / 1e9
    allocated = torch.cuda.memory_allocated(local_rank) / 1e9
    peak = torch.cuda.max_memory_allocated(local_rank) / 1e9
    print(f"Rank {global_rank} GPU {local_rank}: Total={total:.1f}GB, "
          f"Reserved={reserved:.1f}GB, Allocated={allocated:.1f}GB, "
          f"Peak={peak:.1f}GB, Peak%={peak / total * 100:.1f}%")

if args.wandb and is_main_process and wandb.run is not None:
    safe_wandb_call(lambda: wandb.finish(), "finish")

print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

if torch.cuda.is_available() and torch.distributed.is_initialized():
    torch.distributed.barrier()

if is_main_process:
    print(f"Rank {global_rank}: Saving adapter...")

    model_to_save = trainer.model
    if hasattr(model_to_save, 'module'):
        model_to_save = model_to_save.module

    model_to_save.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    model_to_save.config.to_json_file(os.path.join(adapter_dir, "config.json"))
    if hasattr(model_to_save, 'generation_config'):
        model_to_save.generation_config.to_json_file(
            os.path.join(adapter_dir, "generation_config.json")
        )

    print(f"Rank {global_rank}: Model saved to {adapter_dir}")
else:
    print(f"Rank {global_rank}: Waiting for rank 0 to save...")

if torch.cuda.is_available() and torch.distributed.is_initialized():
    torch.distributed.barrier()
    print(f"Rank {global_rank}: Save completed, synchronized")

time.sleep(5)

if is_main_process:
    print("\nModel saved!")

if is_main_process and training_rewards_history:
    print(f"\n{'=' * 60}")
    print("TRAINING REWARDS SUMMARY:")
    print(f"Total rewards computed: {len(training_rewards_history)}")
    print(f"Mean reward: {np.mean(training_rewards_history):.3f}")
    print(f"Std reward: {np.std(training_rewards_history):.3f}")
    print(f"Min reward: {np.min(training_rewards_history):.3f}")
    print(f"Max reward: {np.max(training_rewards_history):.3f}")
    print(f"{'=' * 60}")

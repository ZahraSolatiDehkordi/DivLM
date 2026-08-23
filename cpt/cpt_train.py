import argparse
import os
import sys
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import config as cfg
from common.utils import is_main_process, make_run_name, safe_wandb_call, set_all_seeds, source_tag


parser = argparse.ArgumentParser(description="Continued pre-training with LoRA.")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True, help="Root directory for CPT outputs.")
parser.add_argument("--run-name", type=str, default=None, help="Optional run folder name.")
parser.add_argument("--wandb-project", type=str, default=cfg.DIVLM_TRAIN_PROJECT)
parser.add_argument("--wandb", type=str, choices=("true", "false"), default="true")
parser.add_argument("--epochs", type=float, default=cfg.CPT_EPOCHS)
parser.add_argument("--max-seq-length", type=int, default=cfg.CPT_MAX_SEQ_LENGTH)
parser.add_argument("--learning-rate", type=float, default=cfg.CPT_LEARNING_RATE)
parser.add_argument("--warmup-ratio", type=float, default=cfg.CPT_WARMUP_RATIO)
parser.add_argument("--per-device-train-batch-size", type=int, default=cfg.CPT_PER_DEVICE_TRAIN_BATCH_SIZE)
parser.add_argument("--gradient-accumulation-steps", type=int, default=cfg.CPT_GRADIENT_ACCUMULATION_STEPS)
parser.add_argument("--lora-r", type=int, default=cfg.CPT_LORA_R)
parser.add_argument("--lora-alpha", type=int, default=cfg.CPT_LORA_ALPHA)
parser.add_argument("--lora-dropout", type=float, default=cfg.CPT_LORA_DROPOUT)
parser.add_argument("--weight-decay", type=float, default=cfg.CPT_WEIGHT_DECAY)
parser.add_argument("--max-grad-norm", type=float, default=cfg.CPT_MAX_GRAD_NORM)
parser.add_argument("--logging-steps", type=int, default=cfg.CPT_LOGGING_STEPS)
parser.add_argument("--save-steps", type=int, default=cfg.CPT_SAVE_STEPS)
parser.add_argument("--save-total-limit", type=int, default=cfg.CPT_SAVE_TOTAL_LIMIT)
parser.add_argument("--seed", type=int, default=cfg.CPT_SEED)
args = parser.parse_args()
args.wandb = args.wandb == "true"

if args.epochs <= 0:
    raise ValueError("--epochs must be greater than 0.")
if args.per_device_train_batch_size <= 0:
    raise ValueError("--per-device-train-batch-size must be greater than 0.")
if args.gradient_accumulation_steps <= 0:
    raise ValueError("--gradient-accumulation-steps must be greater than 0.")

set_all_seeds(args.seed)
output_root = args.output_dir
model_tag = source_tag(args.model)
run_name = args.run_name or make_run_name(model_tag, "cpt")
run_dir = os.path.join(output_root, run_name)
checkpoint_dir = os.path.join(run_dir, "checkpoints")
adapter_output_dir = run_dir
os.makedirs(run_dir, exist_ok=True)
os.environ["WANDB_DIR"] = os.path.join(run_dir, "wandb")
os.environ["TMPDIR"] = os.path.join(run_dir, "tmp")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
print(f"CPT run directory: {run_dir}", flush=True)
print(f"CPT adapter directory: {adapter_output_dir}", flush=True)
print(f"CPT checkpoint directory: {checkpoint_dir}", flush=True)

wandb_enabled = False
if args.wandb and is_main_process():
    wandb_enabled = safe_wandb_call(
        lambda: wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "base_model": args.model,
                "dataset": args.dataset,
                "output_root": output_root,
                "run_dir": run_dir,
                "checkpoint_dir": checkpoint_dir,
                "adapter_output_dir": adapter_output_dir,
                "max_seq_length": args.max_seq_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "optimizer": "adamw_8bit",
                "seed": args.seed,
            },
        ),
        "init",
    ) is not None

print(f"Loading tokenizer for {args.model}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading base model {args.model}", flush=True)
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
base_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    attn_implementation="flash_attention_2",
)
base_model.config.use_cache = False
base_model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

print("Loading dataset", flush=True)
dataset = load_from_disk(args.dataset)
if "text" not in dataset.column_names:
    raise ValueError("The training dataset must contain a 'text' column.")

eos_token = tokenizer.eos_token or ""

def formatting_prompts_func(examples):
    return {
        "text": [
            text if text.endswith(eos_token) else text + eos_token
            for text in examples["text"]
        ]
    }


dataset = dataset.map(formatting_prompts_func, batched=True)
dataset = dataset.shuffle(seed=args.seed)

if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Total memory = {total_memory} GB.", flush=True)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=SFTConfig(
        output_dir=checkpoint_dir,
        report_to="wandb" if wandb_enabled else "none",
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=True,
        disable_tqdm=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
    ),
)

print("Starting training", flush=True)
trainer.train()
print("Training complete", flush=True)

if trainer.is_world_process_zero():
    model.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)
    print(f"Adapter saved to {adapter_output_dir}", flush=True)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
        peak = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1e9
        total = gpu.total_memory / 1e9
        print(f"Peak memory: {peak:.1f}GB / {total:.1f}GB ({peak / total * 100:.1f}%)")
        if wandb_enabled and wandb.run is not None:
            safe_wandb_call(lambda: wandb.log({"peak_memory_gb": peak, "peak_memory_pct": peak / total * 100}), "memory log")

    if wandb_enabled and wandb.run is not None:
        safe_wandb_call(lambda: wandb.finish(), "finish")

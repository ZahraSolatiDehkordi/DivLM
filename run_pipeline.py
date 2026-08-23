import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import config as cfg
from common.utils import compact_number, divlm_source_tag, make_run_name, source_tag


def run_command(command):
    print("\n" + "=" * 80, flush=True)
    print(" ".join(command), flush=True)
    print("=" * 80, flush=True)
    subprocess.run(command, check=True)


def python_command(script_path, script_args, nproc_per_node=1):
    if nproc_per_node > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(nproc_per_node),
            str(script_path),
            *script_args,
        ]
    return [sys.executable, str(script_path), *script_args]


parser = argparse.ArgumentParser(description="Run the full DivLM pipeline.")
parser.add_argument("--base-model", type=str, required=True)
parser.add_argument("--instruct-model", type=str, required=True)
parser.add_argument("--cpt-train-set", type=str, required=True)
parser.add_argument("--grpo-train-set", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--scale", type=float, required=True)
parser.add_argument("--test-set-path", type=str, default=None)
parser.add_argument("--length-penalty", type=str, choices=("true", "false"), default="true")
parser.add_argument("--quality-gate", type=str, choices=("true", "false"), default="true")
parser.add_argument("--target-words", type=int, default=cfg.TARGET_WORDS)
parser.add_argument("--h", type=float, default=None)
parser.add_argument("--grpo-epochs", type=float, default=cfg.GRPO_EPOCHS)
parser.add_argument("--max-monitor-restarts", type=int, default=cfg.MONITOR_MAX_RESTARTS)
parser.add_argument("--accept-after-fraction", type=float, default=cfg.MONITOR_ACCEPT_AFTER_FRACTION)
parser.add_argument("--grpo-nproc-per-node", type=int, default=1)
parser.add_argument("--skip-cpt", action="store_true")
parser.add_argument("--skip-residual", action="store_true")
parser.add_argument("--skip-grpo", action="store_true")
parser.add_argument("--skip-eval", action="store_true")
parser.add_argument("--wandb", type=str, choices=("true", "false"), default="true")
args = parser.parse_args()

if args.grpo_nproc_per_node <= 0:
    raise ValueError("--grpo-nproc-per-node must be greater than 0.")
if args.grpo_epochs <= 0:
    raise ValueError("--grpo-epochs must be greater than 0.")
if args.max_monitor_restarts < 0:
    raise ValueError("--max-monitor-restarts must be non-negative.")
if not 0.0 <= args.accept_after_fraction <= 1.0:
    raise ValueError("--accept-after-fraction must be between 0 and 1.")

scale_tag = f"delta{compact_number(args.scale)}"
model_tag = source_tag(args.base_model)
cpt_dir = os.path.join(args.output_dir, make_run_name(model_tag, "cpt"))
residual_model_dir = os.path.join(args.output_dir, make_run_name(source_tag(cpt_dir), scale_tag))
divlm_dir = os.path.join(args.output_dir, make_run_name("DivLM", divlm_source_tag(residual_model_dir)))

cpt_script = ROOT_DIR / "cpt" / "cpt_train.py"
residual_script = ROOT_DIR / "cpt" / "instruction_residuals.py"
grpo_script = ROOT_DIR / "grpo" / "grpo_train.py"
eval_script = ROOT_DIR / "grpo" / "grpo_eval.py"

print("Planned output paths:", flush=True)
print(f"  CPT adapter: {cpt_dir}", flush=True)
print(f"  CPT+residual model: {residual_model_dir}", flush=True)
print(f"  DivLM adapter: {divlm_dir}", flush=True)

if not args.skip_cpt:
    run_command(
        python_command(
            cpt_script,
            [
                "--model",
                args.base_model,
                "--dataset",
                args.cpt_train_set,
                "--output-dir",
                args.output_dir,
                "--wandb",
                args.wandb,
            ],
        )
    )

if not args.skip_residual:
    run_command(
        python_command(
            residual_script,
            [
                "--base-model",
                args.base_model,
                "--instruct-model",
                args.instruct_model,
                "--cpt-adapter",
                cpt_dir,
                "--output-dir",
                args.output_dir,
                "--scale",
                str(args.scale),
            ],
        )
    )

if not args.skip_grpo:
    grpo_args = [
        "--output-dir",
        args.output_dir,
        "--cpt-base",
        residual_model_dir,
        "--train-dataset-path",
        args.grpo_train_set,
        "--length-penalty",
        args.length_penalty,
        "--quality-gate",
        args.quality_gate,
        "--target-words",
        str(args.target_words),
        "--epochs",
        str(args.grpo_epochs),
        "--max-monitor-restarts",
        str(args.max_monitor_restarts),
        "--accept-after-fraction",
        str(args.accept_after_fraction),
        "--wandb",
        args.wandb,
    ]
    if args.h is not None:
        grpo_args.extend(["--h", str(args.h)])
    run_command(python_command(grpo_script, grpo_args, nproc_per_node=args.grpo_nproc_per_node))

if args.test_set_path is not None and not args.skip_eval:
    eval_args = [
        "--adapter-dir",
        divlm_dir,
        "--cpt-base",
        residual_model_dir,
        "--instruct-model",
        args.instruct_model,
        "--test-set-path",
        args.test_set_path,
        "--output-dir",
        args.output_dir,
        "--length-penalty",
        args.length_penalty,
        "--quality-gate",
        args.quality_gate,
        "--target-words",
        str(args.target_words),
        "--wandb",
        args.wandb,
    ]
    if args.h is not None:
        eval_args.extend(["--h", str(args.h)])
    run_command(python_command(eval_script, eval_args))

print("\nPipeline complete.", flush=True)

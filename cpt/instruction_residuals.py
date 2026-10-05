import argparse
import gc
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import config as cfg
from common.utils import adapter_parent_tag, compact_number, make_run_name, set_all_seeds


def compute_instruction_residual(base_model, instruct_model, residual_path):
    print(f"Computing instruction residual: {instruct_model} - {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": "cpu"},
        torch_dtype=torch.bfloat16,
    )
    instruct = AutoModelForCausalLM.from_pretrained(
        instruct_model,
        device_map={"": "cpu"},
        torch_dtype=torch.bfloat16,
    )
    base_state = base.state_dict()
    instruct_state = instruct.state_dict()
    residual = OrderedDict()
    for key, base_tensor in base_state.items():
        instruct_tensor = instruct_state.get(key)
        if instruct_tensor is None:
            continue
        if base_tensor.shape != instruct_tensor.shape:
            continue
        if not torch.is_floating_point(base_tensor):
            continue
        residual[key] = (instruct_tensor - base_tensor).cpu()
    torch.save(residual, residual_path)
    del base, instruct, base_state, instruct_state
    gc.collect()
    print(f"Saved instruction residual to {residual_path}")
    return residual


def load_instruction_residual(base_model, instruct_model, residual_path):
    if os.path.exists(residual_path):
        print(f"Loading instruction residual from {residual_path}")
        return torch.load(residual_path, map_location="cpu")
    return compute_instruction_residual(base_model, instruct_model, residual_path)


def apply_instruction_residual(model, residual, scale):
    state = model.state_dict()
    updated_tensors = set()
    with torch.no_grad():
        for key, tensor in state.items():
            if key not in residual:
                continue
            tensor_key = (tensor.device, tensor.data_ptr(), tensor.shape, tensor.stride())
            if tensor_key in updated_tensors:
                continue
            tensor.copy_(
                tensor + scale * residual[key].to(
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
            )
            updated_tensors.add(tensor_key)


parser = argparse.ArgumentParser(description="Apply an instruction residual to a CPT adapter.")
parser.add_argument("--base-model", type=str, required=True)
parser.add_argument("--instruct-model", type=str, required=True)
parser.add_argument("--cpt-adapter", type=str, required=True)
parser.add_argument("--residual-path", type=str, default=None)
parser.add_argument("--output-dir", type=str, required=True, help="Root directory for residual outputs.")
parser.add_argument("--run-name", type=str, default=None, help="Optional run folder name.")
parser.add_argument("--scale", type=float, required=True)
parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
args = parser.parse_args()

set_all_seeds(args.seed)
output_root = args.output_dir
cpt_tag = adapter_parent_tag(args.cpt_adapter)
scale_tag = f"delta{compact_number(args.scale)}"
run_name = args.run_name or make_run_name(cpt_tag, scale_tag)
run_dir = os.path.join(output_root, run_name)
model_output_dir = run_dir
residual_dir = os.path.join(run_dir, "residuals")
os.makedirs(run_dir, exist_ok=True)
os.environ["TMPDIR"] = os.path.join(run_dir, "tmp")
os.makedirs(residual_dir, exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
if args.residual_path is None:
    args.residual_path = os.path.join(residual_dir, "instruction_residual.pt")
os.makedirs(os.path.dirname(os.path.abspath(args.residual_path)), exist_ok=True)
print(f"Residual run directory: {run_dir}")
print(f"Residual-adjusted model directory: {model_output_dir}")

instruction_residual = load_instruction_residual(
    base_model=args.base_model,
    instruct_model=args.instruct_model,
    residual_path=args.residual_path,
)

print(f"Loading base model from {args.base_model}")
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else {"": "cpu"}
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
)

print(f"Loading tokenizer from {args.instruct_model}")
tokenizer = AutoTokenizer.from_pretrained(args.instruct_model)

print(f"Loading CPT adapter from {args.cpt_adapter}")
model = PeftModel.from_pretrained(model, args.cpt_adapter)
model = model.merge_and_unload()

print(f"Applying instruction residual with scale {args.scale}")
apply_instruction_residual(model, instruction_residual, args.scale)
del instruction_residual
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model.eval()

print(f"Saving model to {model_output_dir}")
model.save_pretrained(model_output_dir, safe_serialization=True)
tokenizer.save_pretrained(model_output_dir)
print("Model and tokenizer saved successfully")
print(f"Use this path as --cpt-base for GRPO: {model_output_dir}")

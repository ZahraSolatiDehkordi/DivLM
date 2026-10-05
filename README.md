# Improving Diversity in LLM Short Story Generation (DivLM)

This repository contains the implementation of the paper "Improving Diversity in LLM Short Story Generation".

## Structure

```text
common/  Shared configuration and utility functions
cpt/     Continued pre-training, instruction residuals, and evaluation
grpo/    Reinforcement learning (GRPO) and evaluation
```

## Setup

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## Data Formats

The CPT and GRPO training datasets must be HuggingFace datasets saved with `save_to_disk` containing a "text" field and a "prompt" field respectively.

Evaluation data can be either a JSON file or a HuggingFace dataset. It must contain a "prompt" field.

## Main Commands

All `--output-dir` arguments are output roots. Each script creates a named run folder inside that root. Use `--run-name` only if you want to override the generated name. 

Continued pre-training:

```bash
python cpt/cpt_train.py \
  --model MODEL \
  --dataset CPT_TRAIN_SET \
  --output-dir OUTPUT_ROOT
```

This writes the CPT LoRA adapter to:

```text
OUTPUT_ROOT/{MODEL}_cpt
```

Instruction residuals:

```bash
python cpt/instruction_residuals.py \
  --base-model BASE_MODEL \
  --instruct-model INSTRUCT_MODEL \
  --cpt-adapter OUTPUT_ROOT/{MODEL}_cpt \
  --output-dir OUTPUT_ROOT \
  --scale SCALE
```

Choose `--scale` explicitly for each model you want to create. Recommended range: `0.5` to `0.9`.

This writes the model to:

```text
OUTPUT_ROOT/{MODEL}_cpt_delta{SCALE}
```

Residual scale evaluation:

```bash
python cpt/residual_scale_eval.py \
  --instruct-model INSTRUCT_MODEL \
  --residual-model-dir OUTPUT_ROOT/{MODEL}_cpt_delta{SCALE} \
  --test-set-path TEST_SET \
  --output-dir OUTPUT_ROOT \
  --target-words 500
```

Run this once for each residual scale you want to evaluate.

Residual scale evaluation writes metrics to:

```text
OUTPUT_ROOT/residual_scale_eval_delta{SCALE}_{TEST_SET}/metrics/eval_metrics.json
```

GRPO training:

```bash
python grpo/grpo_train.py \
  --output-dir OUTPUT_ROOT \
  --cpt-base OUTPUT_ROOT/{MODEL}_cpt_delta{SCALE} \
  --train-dataset-path GRPO_TRAIN_SET \
  --epochs 2 \
  --length-penalty true \
  --target-words 500
```

For multi-GPU GRPO training, launch the script with `torch.distributed.run`:

```bash
python -m torch.distributed.run --nproc_per_node NUM_GPUS grpo/grpo_train.py \
  --output-dir OUTPUT_ROOT \
  --cpt-base OUTPUT_ROOT/{MODEL}_cpt_delta{SCALE} \
  --train-dataset-path GRPO_TRAIN_SET \
  --epochs 2 \
  --length-penalty true \
  --target-words 500
```

This writes the final DivLM adapter to:

```text
OUTPUT_ROOT/DivLM_{MODEL}_delta{SCALE}
```

To ask for a different approximate output length in the prompt, set `--target-words`. Use `--target-words 0` to omit an explicit word count from the prompt.

To penalize short outputs during reward calculation, keep `--length-penalty true`. The target length can be set in `common/config.py`:

```python
LENGTH_PENALTY_TARGET_WORDS = 500
```

Evaluation:

```bash
python grpo/grpo_eval.py \
  --adapter-dir OUTPUT_ROOT/DivLM_{MODEL}_delta{SCALE} \
  --cpt-base OUTPUT_ROOT/{MODEL}_cpt_delta{SCALE} \
  --instruct-model INSTRUCT_MODEL \
  --test-set-path TEST_SET \
  --output-dir OUTPUT_ROOT \
  --target-words 500
```

Evaluation writes metrics to:

```text
OUTPUT_ROOT/DivLM_{MODEL}_delta{SCALE}_{TEST_SET}_eval/metrics/eval_metrics.json
```

Full pipeline:

```bash
python run_pipeline.py \
  --base-model BASE_MODEL \
  --instruct-model INSTRUCT_MODEL \
  --cpt-train-set CPT_TRAIN_SET \
  --grpo-train-set GRPO_TRAIN_SET \
  --output-dir OUTPUT_ROOT \
  --scale SCALE \
  --test-set-path TEST_SET \
  --grpo-epochs 2 \
  --length-penalty true \
  --target-words 500
```

For multi-GPU GRPO training through the full pipeline, add `--grpo-nproc-per-node NUM_GPUS`.

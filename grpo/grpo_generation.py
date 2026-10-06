from dataclasses import dataclass

import torch
from jinja2 import TemplateError

from common import config as cfg


def make_system_prompt(target_words=cfg.TARGET_WORDS):
    target_words = int(target_words)
    if target_words <= 0:
        return "You are a creative writing assistant. Write a short story for the following prompt."
    return f"You are a creative writing assistant. Write a short story in {target_words} words for the following prompt."


SYSTEM_PROMPT_STORY = make_system_prompt(cfg.TARGET_WORDS)


@dataclass
class GenerationSettings:
    max_new_tokens: int = cfg.MAX_NEW_TOKENS
    temperature: float = cfg.GEN_TEMPERATURE
    top_p: float = cfg.GEN_TOP_P
    top_k: int = cfg.GEN_TOP_K
    repetition_penalty: float = cfg.GEN_REPETITION_PENALTY
    max_prompt_length: int = cfg.MAX_PROMPT_LENGTH
    num_generations: int = cfg.GROUP_SIZE
    batch_size: int = cfg.EVAL_GENERATION_BATCH_SIZE


def format_prompt(plain_prompt, tokenizer, system_prompt=SYSTEM_PROMPT_STORY):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": plain_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (TemplateError, ValueError) as exc:
            message = str(exc).lower()
            if not any(text in message for text in ("system role", "system message", "roles must alternate")):
                raise
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{plain_prompt}"}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\nPrompt:\n{plain_prompt}\n\nStory:\n"


def get_generation_eos_token_id(model, tokenizer):
    generation_config = getattr(model, "generation_config", None)
    configured_eos = getattr(generation_config, "eos_token_id", None)
    if configured_eos is None:
        configured_eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    eos_token_ids = []
    for value in (configured_eos, getattr(tokenizer, "eos_token_id", None)):
        values = value if isinstance(value, (list, tuple)) else [value]
        for token_id in values:
            if token_id is not None and token_id not in eos_token_ids:
                eos_token_ids.append(token_id)
    if not eos_token_ids:
        return None
    return eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids


def generate_outputs(model, tokenizer, formatted_prompts, plain_prompts, label, settings, seed_fn=None):
    if seed_fn is not None:
        seed_fn()
    model.eval()
    all_formatted = []
    metadata = []
    for prompt_idx, formatted_prompt in enumerate(formatted_prompts):
        for generation_idx in range(settings.num_generations):
            all_formatted.append(formatted_prompt)
            metadata.append((prompt_idx, generation_idx, plain_prompts[prompt_idx]))
    outputs = []
    batch_size = max(1, min(settings.batch_size, len(all_formatted)))
    for batch_start in range(0, len(all_formatted), batch_size):
        batch_end = min(batch_start + batch_size, len(all_formatted))
        batch_prompts = all_formatted[batch_start:batch_end]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_prompt_length,
            padding=True,
        ).to(model.device)
        eos_token_id = get_generation_eos_token_id(model, tokenizer)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                top_k=settings.top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=settings.repetition_penalty,
            )
        input_length = inputs["input_ids"].shape[1]
        for local_idx, token_ids in enumerate(generated):
            prompt_idx, generation_idx, plain_prompt = metadata[batch_start + local_idx]
            text = tokenizer.decode(token_ids[input_length:], skip_special_tokens=True).strip()
            outputs.append({
                "model": label,
                "prompt_index": prompt_idx,
                "generation_index": generation_idx,
                "prompt": plain_prompt,
                "generated": text,
                "length_chars": len(text),
                "length_words": len(text.split()),
            })
        print(f"{label}: generated {len(outputs)}/{len(all_formatted)}")
    return outputs

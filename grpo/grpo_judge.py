import json
import os
import re
import unicodedata
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import config as cfg

class LLMJudge:

    def __init__(self, max_batch_size=cfg.JUDGE_BATCH_SIZE):
        self.model_name = cfg.JUDGE_MODEL_NAME
        self.max_batch_size = max_batch_size
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))

        print(f"\n{'=' * 80}")
        print(f"INITIALIZING LLM JUDGE (Rank {global_rank})")
        print(f"{'=' * 80}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"
        print("Tokenizer loaded")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device = f"cuda:{local_rank}"
        print(f"  Local rank: {local_rank}, Global rank: {global_rank}")
        print(f"  Judge will use single GPU: {device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        )

        print(f"Judge loaded on {device}")
        print(f"{'=' * 80}\n")

    def _apply_chat_template(self, messages: list) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def _tokenize_batch(self, texts: list, max_length: int):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.model.device)

    def _generation_kwargs(self, max_new_tokens: int) -> dict:
        return dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
        )

    def _decode_output(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _default_result(self):
        return {
            "adheres_to_prompt": True,
            "has_meta_commentary": False,
            "is_complete": True,
            "is_coherent": True,
            "is_grammatical": True,
        }

    def _clean_json_response(self, response: str) -> str:
        if "</think>" in response:
            response = response[response.rfind("</think>") + len("</think>"):].strip()
        elif "<think>" in response:
            response = ""
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        if not (cleaned.startswith("{") and cleaned.endswith("}")):
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0).strip()
        return cleaned

    def judge_structural_batch(self, prompts, outputs: List[str]) -> List[dict]:
        all_results = []
        for i in range(0, len(outputs), self.max_batch_size):
            batch_outputs = outputs[i:i + self.max_batch_size]
            batch_prompts = (
                prompts[i:i + self.max_batch_size]
                if isinstance(prompts, list)
                else prompts
            )
            batch_results = self._judge_structural_batch_internal(batch_prompts, batch_outputs)
            all_results.extend(batch_results)
        return all_results

    def _judge_structural_batch_internal(self, prompts, outputs: List[str]) -> List[dict]:
        system_prompt = """You are a strict writing evaluator.

Your job is to read a PROMPT and a model-generated OUTPUT and answer these questions:

1) Does the story in OUTPUT follow the essential requirements of the PROMPT
   (characters, setting, constraints, required events, point of view, and style if specified)?

2) Does the OUTPUT contain any meta-commentary?
   Meta-commentary includes, but is not limited to:
   - The model talking about the story (e.g. "Here is your story", "In this story I will...")
   - Explanations or analysis about the story
   - Instructions or comments about what to do next
   - Imagined dialogue or turns such as "User:", "Assistant:", "System:", or lines like
     "user: now please continue the story with these elements"

3) Is the story complete?
   A complete story has a beginning and development and reaches some kind of ending.
   It is not obviously cut off mid-sentence or mid-scene, and does not
   rely on an unfinished "to be continued" unless the PROMPT explicitly allows that.

4) Is the story coherent?
   A coherent story has consistent internal logic: events follow causally from one another,
   characters behave consistently with their established traits, there are no unexplained
   contradictions, and the narrative does not abruptly shift in ways that make no sense.

5) Is the story grammatical?
   The text is free of grammatical errors that impede readability, such as
   broken sentence structure or missing verbs. Intentional stylistic
   choices (dialect, unconventional punctuation for effect, stream-of-consciousness) should
   NOT be penalized.

Respond only with a single valid JSON object. No explanation or extra text.
"""
        all_messages = []
        for idx, output in enumerate(outputs):
            clean_prompt = (
                prompts[idx] if isinstance(prompts, list) else prompts
            ).strip()
            user_prompt = f"""Evaluate the following story:

PROMPT:
{clean_prompt}

OUTPUT:
{output}

Respond only with a JSON object in this exact format, using lowercase booleans:

{{
  "adheres_to_prompt": true/false,
  "has_meta_commentary": true/false,
  "is_complete": true/false,
  "is_coherent": true/false,
  "is_grammatical": true/false
}}"""
            all_messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

        texts = [self._apply_chat_template(msgs) for msgs in all_messages]
        inputs = self._tokenize_batch(texts, max_length=4096)

        with torch.no_grad():
            batch_outputs = self.model.generate(
                **inputs,
                **self._generation_kwargs(max_new_tokens=400),
            )

        results = []
        input_length = inputs["input_ids"].shape[1]

        for i, output in enumerate(batch_outputs):
            response = self._decode_output(output[input_length:])
            try:
                cleaned_response = self._clean_json_response(response)
                result = json.loads(cleaned_response)
                required_keys = [
                    "adheres_to_prompt", "has_meta_commentary", "is_complete",
                    "is_coherent", "is_grammatical",
                ]
                if all(key in result for key in required_keys):
                    results.append(result)
                else:
                    print(f"Warning: Judge response missing keys. Got: {result.keys()}")
                    results.append(self._default_result())
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse judge response: {e}")
                results.append(self._default_result())

        return results

    def judge_style_genre_batch(self, prompts, outputs: List[str]) -> List[dict]:
        all_results = []
        for i in range(0, len(outputs), self.max_batch_size):
            batch_outputs = outputs[i:i + self.max_batch_size]
            batch_prompts = (
                prompts[i:i + self.max_batch_size]
                if isinstance(prompts, list)
                else prompts
            )
            batch_results = self._judge_style_genre_batch_internal(batch_prompts, batch_outputs)
            all_results.extend(batch_results)
        return all_results

    def _judge_style_genre_batch_internal(self, prompts, outputs: List[str]) -> List[dict]:
        allowed_tones = ["somber", "neutral", "humorous"]
        allowed_styles = ["dialogue_heavy", "descriptive", "action_focused"]
        allowed_genres = [
            "fantasy", "science_fiction", "horror", "romance",
            "mystery_thriller", "drama_literary", "historical",
            "action_adventure", "other",
        ]

        system_prompt = f"""You are a literary analyst evaluating creative writing.

    TASK 1 - GENRE (select exactly ONE):
    {', '.join(allowed_genres)}

    TASK 2 - TONE (select exactly ONE):
    {', '.join(allowed_tones)}

    TASK 3 - STYLE (select exactly ONE):
    {', '.join(allowed_styles)}

    For Tasks 1-3: You MUST choose exactly one option from each list. If multiple options seem to apply, choose the most dominant one.

    TASK 4 - NAMED ENTITIES (list ALL named entities that appear):
    Extract ONLY specific proper nouns that uniquely identify something:
    - Character names: specific names of characters (e.g. "Emily", "Shadowfang")
        * If a character has both a first and last name, list them as SEPARATE entities (e.g. "John Smith" -> ["John", "Smith"])
    - Specific place names: named cities or locations (e.g. "Tokyo", "Hogwarts")
      * Do NOT include generic references like "the city", "the village"
    - Organization names: specific named groups (e.g. "Stark Industries", "The Order")
    - Other specific proper nouns: named events, works, etc.
      * Do NOT include generic nouns used as references: "Mom", "Dad", "City", "Town", "Earth", "Humans", "World".

    Respond only with valid JSON. No explanation."""

        all_messages = []
        for idx, output in enumerate(outputs):
            clean_prompt = (
                prompts[idx] if isinstance(prompts, list) else prompts
            ).strip()
            user_prompt = f"""Analyze this story:

    PROMPT:
    {clean_prompt}

    OUTPUT:
    {output}

    Respond with JSON in this exact format:

    {{
      "genre": "one_genre_from_list",
      "tone":  "one_tone_from_list",
      "style": "one_style_from_list",
      "named_entities": ["Entity1", "Entity2", "Entity3"]
    }}

    Remember:
    - genre must be EXACTLY ONE of: {', '.join(allowed_genres)}
    - tone must be EXACTLY ONE of: {', '.join(allowed_tones)}
    - style must be EXACTLY ONE of: {', '.join(allowed_styles)}
    - named_entities must be a list of strings (can be empty if no named entities found)"""

            all_messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

        texts = [self._apply_chat_template(msgs) for msgs in all_messages]
        inputs = self._tokenize_batch(texts, max_length=4096)

        with torch.no_grad():
            batch_outputs = self.model.generate(
                **inputs,
                **self._generation_kwargs(max_new_tokens=300),
            )

        results = []
        input_length = inputs["input_ids"].shape[1]

        for i, output in enumerate(batch_outputs):
            response = self._decode_output(output[input_length:])
            try:
                cleaned_response = self._clean_json_response(response)
                result = json.loads(cleaned_response)

                genre = result.get("genre", "other")
                if genre not in allowed_genres:
                    genre = "other"

                tone = result.get("tone", "neutral")
                if tone not in allowed_tones:
                    tone = "neutral"

                style = result.get("style", "descriptive")
                if style not in allowed_styles:
                    style = "descriptive"

                named_entities = result.get("named_entities", [])
                if not isinstance(named_entities, list):
                    named_entities = []

                results.append({
                    "genre": genre,
                    "tone": tone,
                    "style": style,
                    "named_entities": named_entities,
                })

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse style/genre response: {e}")
                results.append({
                    "genre": "other",
                    "tone": "neutral",
                    "style": "descriptive",
                    "named_entities": [],
                })

        return results

    def extract_prompt_q_batch(self, prompts: List[str]) -> List[set]:
        system_prompt = """You are a named entity extractor.

    Extract ONLY specific proper nouns from the given text:
    - Character names (e.g. "John", "Shadowfang")
      * If a character has both a first and last name, list them as SEPARATE entities (e.g. "John Smith" -> ["John", "Smith"])
    - Specific place names (e.g. "Tokyo", "Hogwarts")
    - Organisation names (e.g. "Stark Industries")
    - Other specific proper nouns (named events, objects, etc.)

    Do NOT include generic words like "the city", "the town", "humans", "world", "earth", "mom", "dad".

    Respond only with valid JSON. No explanation."""

        all_messages = []
        for prompt in prompts:
            user_prompt = f"""Extract named entities from this text:

    {prompt}

    Respond with JSON in this exact format:
    {{"named_entities": ["Entity1", "Entity2"]}}"""

            all_messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

        texts = [self._apply_chat_template(msgs) for msgs in all_messages]
        inputs = self._tokenize_batch(texts, max_length=1024)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self._generation_kwargs(max_new_tokens=300),
            )

        input_length = inputs["input_ids"].shape[1]
        results = []
        for output in outputs:
            response = self._decode_output(output[input_length:])
            try:
                cleaned = self._clean_json_response(response)
                parsed = json.loads(cleaned)
                entities = parsed.get("named_entities", [])
                if not isinstance(entities, list):
                    entities = []
                results.append(set(normalize_entity_list(entities)))
            except json.JSONDecodeError:
                results.append(set())

        return results

    def judge_batch(self, prompts: str, outputs: List[str]) -> List[dict]:
        structural_results = self.judge_structural_batch(prompts, outputs)
        style_genre_results = self.judge_style_genre_batch(prompts, outputs)

        combined_results = []
        for structural, style_genre in zip(structural_results, style_genre_results):
            combined_results.append({**structural, **style_genre})

        return combined_results

def normalize_entity(entity: str) -> str:
    if not entity or not isinstance(entity, str):
        return ""

    normalized = unicodedata.normalize('NFKC', entity)

    normalized = re.sub(r"(?:'s|\u2019s)$", "", normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"^the\s+", "", normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"[^\w\s\-]", "", normalized)

    normalized = re.sub(r"\s+", " ", normalized).strip()

    normalized = normalized.lower()

    return normalized

def normalize_entity_list(entities: list) -> set:
    normalized = set()
    for entity in entities:
        cleaned = normalize_entity(entity)
        if cleaned:
            normalized.add(cleaned)
    return normalized


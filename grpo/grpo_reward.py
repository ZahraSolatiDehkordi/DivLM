import re
from collections import Counter

import numpy as np
import torch

from common import config as cfg
from grpo.grpo_judge import LLMJudge, normalize_entity_list

STORY_FINAL_PUNCTUATION = ('.', '!', '?', '"', "'", ')', '...', "\u2026", "\u201d", "\u2019")


def clean_leaked_tokens(text):
    if not text:
        return text

    special_tokens = [
        '<|im_end|>', '<|im_start|>', '<|endoftext|>',
        '<|im_start|>user', '<|im_start|>assistant',
        '<|preamble_by_system_1|>', '</s>', '<s>',
        '|<|endoftext|>|', '|<|'
    ]

    cleaned = text
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    for token in special_tokens:
        cleaned = cleaned.replace(token, '')

    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    cleaned = re.sub(r'style="[^"]*"', '', cleaned)
    cleaned = re.sub(r'font-[a-z]+:\s*[^;]+;?', '', cleaned)

    cleaned = re.sub(r'\[endif\]', '', cleaned)
    cleaned = re.sub(r'<!--.*?-->', '', cleaned)

    cleaned = re.sub(r'[ \t\f\v]+', ' ', cleaned)
    cleaned = re.sub(r' *\n *', '\n', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()

    return cleaned

def is_degenerate_output(text):
    if not text or not text.strip():
        return True

    tokens = text.split()
    if len(tokens) < 50:
        return True

    leaked_patterns = [
        '<|im_end|>', '<|im_start|>', '|<|', '<|im_start|>user',
        '<|im_start|>assistant', '<|preamble_by_system_1|>',
        '|<|endoftext|>|', '</s>', '<s>'
    ]
    if any(pattern in text for pattern in leaked_patterns):
        return True

    if re.search(r'(font-size|font-weight|style="|<[a-z]+>)', text):
        return True

    if len(set(tokens)) / len(tokens) < 0.3:
        return True

    if not text.strip().endswith(STORY_FINAL_PUNCTUATION):
        return True

    return False

def is_incomplete(text):

    if not text.strip().endswith(STORY_FINAL_PUNCTUATION):
        return True

    return False

def has_meta_commentary(text):
    meta_patterns = [
        r'(?i)I hope you liked',
        r'(?i)I hope you enjoyed',
        r'(?i)Please (reply|let me know)',
        r'(?i)Your feedback',
        r'<\|im_end\|>',
        r'<\|im_start\|>',
        r'\(Note:',
        r'\|\s*-\s*.*Assistant',
        r'(?i)Please provide feedback',
        r'(?i)Would love to hear your thoughts',
        r'(?i)Is there anything else I can help you with',
        r'(?i)Can I help you with something else',
        r"(?i)I can['\u2019]t help you with that",
        r"(?i)I can['\u2019]t provide information"
    ]

    for pattern in meta_patterns:
        if re.search(pattern, text):
            return True
    return False

def compute_diversity_score(
    c_i: int,
    G: int,
    K: int,
    all_counts: Counter
) -> float:
    c_floor = G // K
    c_ceil = c_floor + 1
    xi = G % K

    if K > G:
        c_star = 1
        d_i = abs(c_i - c_star)
        d_max = G - 1
        S_i = 1.0 - (d_i / d_max)

    elif xi == 0:
        c_star = G // K
        d_i = abs(c_i - c_star)
        d_max = max(G - c_star, c_star - 1)
        S_i = 1.0 - (d_i / d_max)

    else:

        c_star_values = {c_floor, c_ceil}
        d_i = min(abs(c_i - c_star) for c_star in c_star_values)

        d_max = max(c_floor - 1, G - c_ceil)
        S_i = 1.0 - (d_i / d_max)

        P_i = 0.0
        if c_i == c_ceil:
            num_at_ceil = sum(1 for c in all_counts.values() if c == c_ceil)
            excess = max(0, num_at_ceil - xi)
            P_i = excess / K
        elif c_i == c_floor:
            num_at_floor = sum(1 for c in all_counts.values() if c == c_floor)
            excess = max(0, num_at_floor - (K - xi))
            P_i = excess / K
        S_i = max(0.0, S_i - P_i)

    return S_i

def compute_length_penalty(
    completions,
    target_words=500,
):
    if target_words <= 0:
        raise ValueError("target_words must be greater than 0.")

    word_counts = np.array([len(c.split()) for c in completions], dtype=np.float32)
    shortfall = np.maximum(0.0, target_words - word_counts)
    return np.minimum(1.0, shortfall / target_words).astype(np.float32)

class FinalReward:
    def __init__(
        self,
        device=None,
        group_size=cfg.GROUP_SIZE,
        entity_clip_u=cfg.ENTITY_CLIP_U,
        h=cfg.ENTITY_H,
        quality_tau=cfg.QUALITY_TAU,
        use_group_quality_tau=True,
        group_quality_tau=cfg.GROUP_QUALITY_TAU,
        invalid_r_qual=cfg.INVALID_R_QUAL,
        r_div_normalizer=cfg.R_DIV_NORMALIZER,
        use_length_penalty=True,
        length_penalty_target_words=cfg.LENGTH_PENALTY_TARGET_WORDS,
        judge_batch_size=cfg.JUDGE_BATCH_SIZE,
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.group_size = group_size
        self.entity_clip_u = entity_clip_u
        self.h = h
        self.quality_tau = quality_tau
        self.use_group_quality_tau = use_group_quality_tau
        self.group_quality_tau = group_quality_tau
        self.invalid_r_qual = invalid_r_qual
        self.r_div_normalizer = r_div_normalizer
        self.use_length_penalty = use_length_penalty
        self.length_penalty_target_words = length_penalty_target_words
        self.judge_batch_size = judge_batch_size
        self.llm_judge = None
        print(f"FinalReward initialized on device: {device}")

    def compute_rewards(self, completions, prompts):

        if self.llm_judge is None:
            self.llm_judge = LLMJudge(max_batch_size=self.judge_batch_size)

        n = len(completions)

        batch_prompts = prompts

        degenerate_mask = np.array([is_degenerate_output(c) for c in completions])
        completeness_scores_heuristic = np.array([0.0 if is_incomplete(c) else 1.0 for c in completions], dtype=np.float32)
        meta_score = np.array([0.0 if has_meta_commentary(c) else 1.0 for c in completions], dtype=np.float32)
        cleaned_completions = np.array([clean_leaked_tokens(c) for c in completions])

        judge_results = self.llm_judge.judge_batch(batch_prompts, cleaned_completions)

        prompt_adherence_scores = np.array([1.0 if r['adheres_to_prompt'] else 0.0 for r in judge_results],
                                           dtype=np.float32)
        meta_commentary_scores = np.array([0.0 if r['has_meta_commentary'] else 1.0 for r in judge_results],
                                           dtype=np.float32)
        completeness_scores = np.array([1.0 if r['is_complete'] else 0.0 for r in judge_results], dtype=np.float32)

        coherence_scores = np.array([1.0 if r['is_coherent'] else 0.0 for r in judge_results], dtype=np.float32)
        grammar_scores = np.array([1.0 if r['is_grammatical'] else 0.0 for r in judge_results], dtype=np.float32)

        meta_commentary_scores = np.minimum(meta_commentary_scores, meta_score)
        completeness_scores = np.minimum(completeness_scores, completeness_scores_heuristic)

        genres = [r.get('genre', 'other') for r in judge_results]
        tones = [r.get('tone', 'neutral') for r in judge_results]
        styles = [r.get('style', 'descriptive') for r in judge_results]

        q_raw_list = [r.get('named_entities', []) for r in judge_results]

        prompt_adherence_weight = 1
        meta_commentary_weight = 1
        completeness_weight = 1

        r_qual_rewards = (
                prompt_adherence_scores * prompt_adherence_weight +
                meta_commentary_scores * meta_commentary_weight +
                completeness_scores * completeness_weight

        )
        r_qual_rewards = r_qual_rewards / 3

        p_len = (
            compute_length_penalty(
                completions,
                target_words=self.length_penalty_target_words,
            )
            if self.use_length_penalty
            else np.zeros(n, dtype=np.float32)
        )
        malformed_mask = (
            degenerate_mask
            | (coherence_scores == 0.0)
            | (grammar_scores == 0.0)
        )
        r_qual_rewards = np.where(malformed_mask, self.invalid_r_qual, r_qual_rewards)

        G = self.group_size
        num_groups = n // G

        r_gts_rewards = np.zeros(n, dtype=np.float32)

        r_genre_rewards = np.zeros(n, dtype=np.float32)
        r_style_rewards = np.zeros(n, dtype=np.float32)
        r_tone_rewards = np.zeros(n, dtype=np.float32)

        K_GENRE = 9
        K_TONE = 3
        K_STYLE = 3

        r_ne_rewards = np.zeros(n, dtype=np.float32)

        unique_group_prompts = [
            batch_prompts if isinstance(batch_prompts, str)
            else batch_prompts[group_idx * G]
            for group_idx in range(num_groups)
        ]
        prompt_q_list = self.llm_judge.extract_prompt_q_batch(unique_group_prompts)
        normalized_entity_sets = [set() for _ in range(n)]

        for group_idx in range(num_groups):
            start_idx = group_idx * G
            end_idx = start_idx + G

            prompt_q = prompt_q_list[group_idx]

            q_sets = []
            for i in range(start_idx, end_idx):
                q_i = normalize_entity_list(q_raw_list[i]) - prompt_q
                q_sets.append(q_i)
                normalized_entity_sets[i] = q_i

            g_nu_counts = Counter()
            for entities in q_sets:
                for entity in entities:
                    g_nu_counts[entity] += 1

            no_entity_indices = []
            r_ne_scores_in_group = []
            for i in range(G):

                local_idx = start_idx + i
                q_i = q_sets[i]

                if len(q_i) > 0:
                    q_i_size = len(q_i)
                    g_nu_repetition_sum = sum(
                        (g_nu_counts[e] - 1) / (G - 1)
                        for e in q_i
                    )
                    clipped_g_nu_repetition_sum = min(g_nu_repetition_sum, self.entity_clip_u)
                    clipped_q_i_size = min(q_i_size, self.entity_clip_u)
                    mean_g_nu_norm = clipped_g_nu_repetition_sum / clipped_q_i_size
                    mean_g_nu_norm = float(np.clip(mean_g_nu_norm, 0.0, 1.0))

                    r_ne_score = (1 - mean_g_nu_norm) ** self.h

                    r_ne_rewards[local_idx] = r_ne_score
                    r_ne_scores_in_group.append(float(r_ne_score))
                else:
                    no_entity_indices.append(local_idx)

            neutral_r_ne_score = float(np.mean(r_ne_scores_in_group)) if r_ne_scores_in_group else 0.5
            for local_idx in no_entity_indices:
                r_ne_rewards[local_idx] = neutral_r_ne_score

        for group_idx in range(num_groups):
            start_idx = group_idx * G
            end_idx = start_idx + G
            group_genres = genres[start_idx:end_idx]
            group_styles = styles[start_idx:end_idx]
            group_tones = tones[start_idx:end_idx]
            genre_counts = Counter(group_genres)
            style_counts = Counter(group_styles)
            tone_counts = Counter(group_tones)

            for i in range(start_idx, end_idx):
                genre = genres[i]
                tone = tones[i]
                style = styles[i]
                genre_score = compute_diversity_score(genre_counts[genre], self.group_size, K_GENRE, genre_counts)
                tone_score = compute_diversity_score(tone_counts[tone], self.group_size, K_TONE, tone_counts)
                style_score = compute_diversity_score(style_counts[style], self.group_size, K_STYLE, style_counts)
                r_genre_rewards[i] = genre_score
                r_style_rewards[i] = style_score
                r_tone_rewards[i] = tone_score
                r_gts_rewards[i] = genre_score + tone_score + style_score

        tau = self.quality_tau
        use_group_tau = self.use_group_quality_tau
        group_tau = self.group_quality_tau

        individual_tau_pass = r_qual_rewards >= tau
        group_tau_pass_diagnostic = np.zeros_like(individual_tau_pass, dtype=bool)
        group_r_qual_mean_values = []
        group_tau_pass_values = []

        for group_start in range(0, n, self.group_size):
            group_end = min(group_start + self.group_size, n)
            group_mean = float(np.mean(r_qual_rewards[group_start:group_end]))
            group_pass_value = group_mean >= group_tau
            group_tau_pass_diagnostic[group_start:group_end] = group_pass_value
            group_r_qual_mean_values.append(group_mean)
            group_tau_pass_values.append(group_pass_value)

        group_r_qual_mean_values = np.asarray(group_r_qual_mean_values, dtype=np.float32)
        group_tau_pass_values = np.asarray(group_tau_pass_values, dtype=bool)
        group_tau_pass = (
            group_tau_pass_diagnostic
            if use_group_tau
            else np.ones_like(individual_tau_pass, dtype=bool)
        )
        r_div_gate = individual_tau_pass & group_tau_pass

        r_div_rewards = (r_ne_rewards + r_gts_rewards) / self.r_div_normalizer
        r_total_rewards = np.where(
            r_div_gate,
            r_qual_rewards + r_div_rewards,
            r_qual_rewards
        )
        r_total_rewards = np.where(malformed_mask, self.invalid_r_qual, r_total_rewards - p_len)

        self.component_stats = {
            "train/quality_reward": float(r_qual_rewards.mean()),
            "train/named_entity_reward": float(r_ne_rewards.mean()),
            "train/genre_reward": float(r_genre_rewards.mean()),
            "train/tone_reward": float(r_tone_rewards.mean()),
            "train/style_reward": float(r_style_rewards.mean()),
            "train/diversity_reward": float(r_div_rewards.mean()),
            "train/total_reward": float(r_total_rewards.mean()),
        }

        self.last_components = {
            "R_total": r_total_rewards,
            "R_qual": r_qual_rewards,
            "R_div": r_div_rewards,
            "R_ne": r_ne_rewards,
            "R_genre": r_genre_rewards,
            "R_tone": r_tone_rewards,
            "R_style": r_style_rewards,
            "length_penalty": p_len,
            "adherence": prompt_adherence_scores,
            "no_meta_commentary": meta_commentary_scores,
            "completeness": completeness_scores,
            "coherence": coherence_scores,
            "grammar": grammar_scores,
            "degenerate": degenerate_mask.astype(np.float32),
            "valid_quality": (~malformed_mask).astype(np.float32),
            "individual_quality_gate": individual_tau_pass.astype(np.float32),
            "group_quality_gate": group_tau_pass.astype(np.float32),
            "diversity_gate": r_div_gate.astype(np.float32),
            "group_R_qual_mean": group_r_qual_mean_values,
            "group_quality_gate_values": group_tau_pass_values.astype(np.float32),
            "genres": genres,
            "tones": tones,
            "styles": styles,
            "named_entities": q_raw_list,
            "normalized_entity_sets": normalized_entity_sets,
            "prompt_entity_sets": prompt_q_list,
            "judge_results": judge_results,
            "cleaned_completions": cleaned_completions.tolist(),
        }

        return r_total_rewards


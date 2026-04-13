"""Evaluation metrics for reasoning quality in VLM scene descriptions.

Metrics:
    - Contentfulness: fraction of thought stream that is scene content vs meta-commentary
    - Thought Coverage (TC): how much of the thought stream made it into the output
    - Output Grounding (OG): how much of the output was grounded in the thought stream
    - F1: harmonic mean of TC and OG
    - Dominant Entity: most prominent subject, action, setting per scene
"""

import json
import logging
import re
from typing import Any

import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from thefuzz import fuzz

from src.judge import LLMJudge
from src.schemas import (
    ContentfulnessResult,
    CoverageExtraction,
    DominantExtraction,
    ExtractedItems,
    FinalOnlyExtraction,
    BatchCoverageExtraction,
    BatchDominantExtraction,
    BatchFinalOnlyExtraction,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# -- Regex patterns to filter meta/process sentences ---------------------------

META_PATTERNS = [
    re.compile(r"\b(I|we)\s+(will|am|need|should|can|must|have to)\b", re.IGNORECASE),
    re.compile(r"\b(let me|let's|okay|now|time to|first|next|finally)\b", re.IGNORECASE),
    re.compile(r"\b(step[- ]by[- ]step|here'?s my|my approach|I'll|I think)\b", re.IGNORECASE),
    re.compile(r"\b(json|format|output|schema|structure|template|field)\b", re.IGNORECASE),
    re.compile(r"\b(analyzing|extracting|summarizing|processing|distilling)\b", re.IGNORECASE),
    re.compile(r"\b(get started|break down|pay attention|remember to)\b", re.IGNORECASE),
]

# POS tags for noun phrases and verb phrases
_NP_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "DT", "PRP", "PRP$"}
_VP_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}

# -- LLM Judge System Prompts -------------------------------------------------

COVERAGE_PROMPT = """You are an expert AI extraction system.
You will be provided with TWO pieces of text: a 'Thought Stream' and a 'Final Structured Output'.

Your job is to identify all distinct entities/objects, actions/verb phrases, and settings/locations mentioned in EACH text separately.

IMPORTANT RULES:
- Return minimal, atomic noun phrases and verb phrases ONLY.
- Do NOT return full descriptive sentences.
- Each item should be 1-4 words maximum.
- Examples of GOOD items: "microphone", "hearing room", "speaking", "gesturing with hands"
- Examples of BAD items: "A man in a suit is speaking into a microphone at a podium"
- Deduplicate: do not return the same item twice in a list.
"""

DOMINANT_PROMPT = """You are an AI tasked with analyzing a vision-model's scene processing to determine its focal points.
Given a list of entities, actions, and settings extracted from a scene, identify the overall TOP 3 of each, as well as the SINGLE dominant subject, action, and setting.

Explicit tie-break rule for choosing the 'main' item:
1. Frequency of appearance/mention
2. Confidence/certainty described
3. Centrality or visual prominence in the description.
"""

BATCH_COVERAGE_PROMPT = """You are an expert AI extraction system.
You will receive MULTIPLE SCENES, each numbered (Scene 1, Scene 2, ...).
Each scene has a 'Thought Stream' and a 'Final Structured Output'.

For EACH scene INDEPENDENTLY, identify all distinct entities/objects, actions/verb phrases, and settings/locations.
Return a 'scenes' list with EXACTLY one CoverageExtraction per scene, preserving input order.

RULES:
- Atomic noun/verb phrases ONLY (1-4 words max).
- Examples of GOOD: "microphone", "hearing room", "speaking", "gesturing with hands"
- Examples of BAD: "A man in a suit is speaking into a microphone"
- Deduplicate within each scene.
"""

BATCH_DOMINANT_PROMPT = """You are an AI tasked with analyzing a vision-model's scene processing to determine focal points.
You will receive MULTIPLE SCENES, each numbered (Scene 1, Scene 2, ...).
Each scene has pre-extracted entities, actions, and settings.

For EACH scene INDEPENDENTLY, identify the TOP 3 of each category and the SINGLE dominant subject, action, and setting.
Return a 'scenes' list with EXACTLY one DominantExtraction per scene, preserving input order.

Tie-break rule for 'main' item:
1. Frequency of mention
2. Confidence/certainty described
3. Visual prominence in the description.
"""

BATCH_FINAL_ONLY_PROMPT = """You are an expert AI extraction system.
You will receive MULTIPLE SCENES, each numbered (Scene 1, Scene 2, ...).
Each scene has ONLY a 'Final Structured Output' (no thought stream).

For EACH scene INDEPENDENTLY:
1. Extract all distinct entities/objects, actions/verb phrases, and settings/locations (atomic, 1-4 words max).
2. Identify the SINGLE dominant subject, action, and setting.

Return a 'scenes' list with EXACTLY one FinalOnlyExtraction per scene, preserving input order.

RULES:
- Atomic noun/verb phrases ONLY (1-4 words max).
- Deduplicate within each scene.
- Leave main_subject/action/setting as empty string only if truly indeterminate.
"""


# -- Fuzzy Matching Utilities --------------------------------------------------

def canonicalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deduplicate(items: list[str]) -> list[str]:
    """Canonicalize, deduplicate, and remove empty strings."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        canon = canonicalize(item)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def fuzzy_intersect_count(list_a: list[str], list_b: list[str], threshold: int = 75) -> int:
    """Count how many items in list_a have a fuzzy match in list_b.

    Uses cascaded matching: exact -> token_sort_ratio -> partial_ratio.
    """
    matches = 0
    for a_item in list_a:
        if not a_item:
            continue
        for b_item in list_b:
            if not b_item:
                continue
            if a_item == b_item:
                matches += 1
                break
            if fuzz.token_sort_ratio(a_item, b_item) >= threshold:
                matches += 1
                break
            if fuzz.partial_ratio(a_item, b_item) >= threshold:
                matches += 1
                break
    return matches


# -- Contentfulness Helpers ----------------------------------------------------

def _is_meta_sentence(sentence: str) -> bool:
    """Return True if the sentence is meta/process language."""
    for pattern in META_PATTERNS:
        if pattern.search(sentence):
            return True
    return False


def _count_np_vp_words(text: str) -> int:
    """POS-tag text and count words in noun phrases or verb phrases."""
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return sum(1 for _, tag in tagged if tag in _NP_TAGS or tag in _VP_TAGS)


# -- MetricsCalculator --------------------------------------------------------

class MetricsCalculator:
    """Computes all evaluation metrics for VLM scene descriptions."""

    def __init__(self, judge: LLMJudge, fuzzy_threshold: int = 75):
        self.judge = judge
        self.fuzzy_threshold = fuzzy_threshold
        self.none_count = 0

    # -- Contentfulness (deterministic, no LLM) --------------------------------

    def compute_contentfulness(self, thought_stream: str) -> ContentfulnessResult:
        """Compute contentfulness: content_words / total_words.

        Content words are noun-phrase and verb-phrase words from non-meta sentences.
        """
        if not thought_stream or not thought_stream.strip():
            return ContentfulnessResult(content_words=0, total_words=0, contentfulness=0.0)

        words = thought_stream.split()
        total_words = len(words)
        if total_words == 0:
            return ContentfulnessResult(content_words=0, total_words=0, contentfulness=0.0)

        sentences = nltk.sent_tokenize(thought_stream)
        content_sentences = [s for s in sentences if s.strip() and not _is_meta_sentence(s)]

        if not content_sentences:
            return ContentfulnessResult(content_words=0, total_words=total_words, contentfulness=0.0)

        content_text = " ".join(content_sentences)
        content_word_count = _count_np_vp_words(content_text)
        contentfulness = max(0.0, min(1.0, content_word_count / total_words))

        return ContentfulnessResult(
            content_words=content_word_count,
            total_words=total_words,
            contentfulness=contentfulness,
        )

    # -- Coverage (Thought Coverage + Output Grounding) ------------------------

    def _compute_recall_precision(
        self, extraction: CoverageExtraction,
    ) -> tuple[float | None, float | None]:
        """Compute TC (recall) and OG (precision) from a CoverageExtraction."""
        thoughts_all = deduplicate(
            extraction.thought_items.entities
            + extraction.thought_items.actions
            + extraction.thought_items.settings
        )
        final_all = deduplicate(
            extraction.final_items.entities
            + extraction.final_items.actions
            + extraction.final_items.settings
        )

        recall = (
            None
            if not thoughts_all
            else max(0.0, min(
                1.0,
                fuzzy_intersect_count(thoughts_all, final_all, self.fuzzy_threshold) / len(thoughts_all),
            ))
        )
        precision = (
            None
            if not final_all
            else max(0.0, min(
                1.0,
                fuzzy_intersect_count(final_all, thoughts_all, self.fuzzy_threshold) / len(final_all),
            ))
        )
        return recall, precision

    async def compute_coverage(
        self, thought_stream: str, final_output: dict[str, Any],
    ) -> tuple[CoverageExtraction, float | None, float | None, TokenUsage]:
        """Compute coverage for a single scene.

        Returns (extraction, thought_coverage, output_grounding, token_usage).
        """
        final_str = json.dumps(final_output)
        user_prompt = f"Thought Stream:\n{thought_stream}\n\nFinal Output:\n{final_str}"

        extraction, usage = await self.judge.extract_structured(
            system_prompt=COVERAGE_PROMPT,
            user_prompt=user_prompt,
            response_model=CoverageExtraction,
        )

        if extraction is None:
            logger.warning("LLM extraction returned None for coverage.")
            self.none_count += 1
            empty = CoverageExtraction(thought_items=ExtractedItems(), final_items=ExtractedItems())
            return empty, None, None, usage

        recall, precision = self._compute_recall_precision(extraction)
        return extraction, recall, precision, usage

    async def compute_coverage_batch(
        self, scenes_data: list[dict[str, Any]],
    ) -> tuple[list[tuple[CoverageExtraction, float | None, float | None]], TokenUsage]:
        """Compute coverage for N scenes in a single LLM call."""
        n = len(scenes_data)
        _empty = CoverageExtraction(thought_items=ExtractedItems(), final_items=ExtractedItems())

        parts = []
        for i, s in enumerate(scenes_data, 1):
            final_str = json.dumps(s["final_output"])
            parts.append(
                f"Scene {i}:\nThought Stream:\n{s['thought_stream']}\n\nFinal Output:\n{final_str}"
            )
        user_prompt = "\n\n---\n\n".join(parts)

        batch_result, usage = await self.judge.extract_structured(
            system_prompt=BATCH_COVERAGE_PROMPT,
            user_prompt=user_prompt,
            response_model=BatchCoverageExtraction,
        )

        if batch_result is None or not batch_result.scenes:
            logger.warning(f"Batch coverage returned None for {n} scenes.")
            self.none_count += n
            return [(_empty, None, None)] * n, usage

        extractions = list(batch_result.scenes)
        if len(extractions) < n:
            logger.warning(f"Batch coverage returned {len(extractions)}/{n} scenes. Padding.")
            extractions += [_empty] * (n - len(extractions))

        results = []
        for ext in extractions[:n]:
            recall, precision = self._compute_recall_precision(ext)
            results.append((ext, recall, precision))
        return results, usage

    # -- Dominant Entity Analysis -----------------------------------------------

    async def compute_dominant(
        self, extraction: CoverageExtraction,
    ) -> tuple[DominantExtraction, TokenUsage]:
        """Compute dominant entities for a single scene."""
        all_e = extraction.thought_items.entities + extraction.final_items.entities
        all_a = extraction.thought_items.actions + extraction.final_items.actions
        all_s = extraction.thought_items.settings + extraction.final_items.settings

        _empty = DominantExtraction(
            top_entities=[], top_actions=[], top_settings=[],
            main_subject="", main_action="", main_setting="",
        )

        if not all_e and not all_a and not all_s:
            return _empty, TokenUsage()

        user_prompt = (
            f"Extracted Entities:\n{all_e}\n\n"
            f"Extracted Actions:\n{all_a}\n\n"
            f"Extracted Settings:\n{all_s}"
        )

        dominant, usage = await self.judge.extract_structured(
            system_prompt=DOMINANT_PROMPT,
            user_prompt=user_prompt,
            response_model=DominantExtraction,
        )

        if dominant is None:
            logger.warning("LLM extraction returned None for dominant.")
            self.none_count += 1
            return _empty, usage

        return dominant, usage

    async def compute_dominant_batch(
        self, extractions: list[CoverageExtraction],
    ) -> tuple[list[DominantExtraction], TokenUsage]:
        """Compute dominant entities for N scenes in a single LLM call."""
        n = len(extractions)
        _empty = DominantExtraction(
            top_entities=[], top_actions=[], top_settings=[],
            main_subject="", main_action="", main_setting="",
        )

        parts = []
        for i, ext in enumerate(extractions, 1):
            all_e = ext.thought_items.entities + ext.final_items.entities
            all_a = ext.thought_items.actions + ext.final_items.actions
            all_s = ext.thought_items.settings + ext.final_items.settings
            parts.append(
                f"Scene {i}:\nEntities: {all_e}\nActions: {all_a}\nSettings: {all_s}"
            )
        user_prompt = "\n\n---\n\n".join(parts)

        batch_result, usage = await self.judge.extract_structured(
            system_prompt=BATCH_DOMINANT_PROMPT,
            user_prompt=user_prompt,
            response_model=BatchDominantExtraction,
        )

        if batch_result is None or not batch_result.scenes:
            logger.warning(f"Batch dominant returned None for {n} scenes.")
            self.none_count += n
            return [_empty] * n, usage

        dominants = list(batch_result.scenes)
        if len(dominants) < n:
            logger.warning(f"Batch dominant returned {len(dominants)}/{n} scenes. Padding.")
            dominants += [_empty] * (n - len(dominants))

        return dominants[:n], usage

    # -- Final-Only (scenes without thought stream) ----------------------------

    async def compute_final_only_batch(
        self, scenes_data: list[dict[str, Any]],
    ) -> tuple[list[FinalOnlyExtraction], TokenUsage]:
        """Extract entities + dominant for scenes without a thought stream."""
        n = len(scenes_data)
        _empty = FinalOnlyExtraction()

        parts = []
        for i, s in enumerate(scenes_data, 1):
            final_str = json.dumps(s["final_output"])
            parts.append(f"Scene {i}:\nFinal Output:\n{final_str}")
        user_prompt = "\n\n---\n\n".join(parts)

        batch_result, usage = await self.judge.extract_structured(
            system_prompt=BATCH_FINAL_ONLY_PROMPT,
            user_prompt=user_prompt,
            response_model=BatchFinalOnlyExtraction,
        )

        if batch_result is None or not batch_result.scenes:
            logger.warning(f"Batch final-only returned None for {n} scenes.")
            self.none_count += n
            return [_empty] * n, usage

        results = list(batch_result.scenes)
        if len(results) < n:
            logger.warning(f"Batch final-only returned {len(results)}/{n} scenes. Padding.")
            results += [_empty] * (n - len(results))

        return results[:n], usage

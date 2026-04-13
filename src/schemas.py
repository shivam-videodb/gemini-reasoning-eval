"""Pydantic models for the gemini-reasoning-eval pipeline."""

from pydantic import BaseModel, Field
from typing import Any


# -- Token Usage (LLM Judge) --------------------------------------------------

class TokenUsage(BaseModel):
    """Token counts from the OpenAI LLM judge API."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


# -- Token Usage (Gemini VLM) -------------------------------------------------

class GeminiTokenUsage(BaseModel):
    """Token counts from the Gemini API response."""

    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    text_tokens: int = 0
    image_tokens: int = 0
    total_tokens: int = 0


# -- Scene Data ----------------------------------------------------------------

class SceneMetadata(BaseModel):
    """Metadata for a single video scene."""

    media_id: str
    scene_idx: int
    scene_start: float = 0.0
    scene_end: float = 0.0
    num_frames: int = 0


class SceneResult(BaseModel):
    """Raw VLM output for a single scene."""

    key: str
    model: str
    thinking_budget: int
    metadata: SceneMetadata
    thought: str | None = None
    response: dict[str, Any] = Field(default_factory=dict)
    tokens: GeminiTokenUsage = Field(default_factory=GeminiTokenUsage)
    error: str | None = None


# -- Extraction Models (LLM Judge Output) -------------------------------------

class ExtractedItems(BaseModel):
    """Atomic noun/verb phrases extracted from text."""

    entities: list[str] = Field(
        default_factory=list,
        description="Atomic noun phrases (entities/objects), 1-4 words each.",
    )
    actions: list[str] = Field(
        default_factory=list,
        description="Atomic verb phrases (actions), 1-4 words each.",
    )
    settings: list[str] = Field(
        default_factory=list,
        description="Atomic setting/location noun phrases, 1-4 words each.",
    )


class CoverageExtraction(BaseModel):
    """Entities/actions/settings extracted from thought stream and final output."""

    thought_items: ExtractedItems = Field(
        description="Items extracted from the thought stream.",
    )
    final_items: ExtractedItems = Field(
        description="Items extracted from the final output.",
    )


class DominantExtraction(BaseModel):
    """Top entities/actions/settings and single dominant per category."""

    top_entities: list[str] = Field(
        default_factory=list, description="Top 3 entities in the scene.",
    )
    top_actions: list[str] = Field(
        default_factory=list, description="Top 3 actions in the scene.",
    )
    top_settings: list[str] = Field(
        default_factory=list, description="Top 3 settings in the scene.",
    )
    main_subject: str = Field(default="", description="Single dominant subject/entity.")
    main_action: str = Field(default="", description="Single dominant action.")
    main_setting: str = Field(default="", description="Single dominant setting.")


class FinalOnlyExtraction(BaseModel):
    """Combined extraction + dominant for scenes without a thought stream."""

    entities: list[str] = Field(
        default_factory=list, description="Atomic noun phrases from final output.",
    )
    actions: list[str] = Field(
        default_factory=list, description="Atomic verb phrases from final output.",
    )
    settings: list[str] = Field(
        default_factory=list, description="Atomic setting/location phrases from final output.",
    )
    main_subject: str = Field(default="", description="Single dominant subject/entity.")
    main_action: str = Field(default="", description="Single dominant action.")
    main_setting: str = Field(default="", description="Single dominant setting.")


# -- Batch Wrappers (for multi-scene LLM calls) --------------------------------

class BatchCoverageExtraction(BaseModel):
    scenes: list[CoverageExtraction] = Field(
        description="One CoverageExtraction per scene, preserving input order.",
    )


class BatchDominantExtraction(BaseModel):
    scenes: list[DominantExtraction] = Field(
        description="One DominantExtraction per scene, preserving input order.",
    )


class BatchFinalOnlyExtraction(BaseModel):
    scenes: list[FinalOnlyExtraction] = Field(
        description="One FinalOnlyExtraction per scene, preserving input order.",
    )


# -- Metric Results ------------------------------------------------------------

class ContentfulnessResult(BaseModel):
    """Result of the contentfulness metric computation."""

    content_words: int
    total_words: int
    contentfulness: float


class SceneMetrics(BaseModel):
    """Evaluation metrics for a single scene."""

    key: str
    model: str
    thinking_budget: int
    contentfulness: ContentfulnessResult | None = None
    thought_coverage: float | None = None
    output_grounding: float | None = None
    f1: float | None = None
    dominant_subject: str = ""
    dominant_action: str = ""
    dominant_setting: str = ""
    thought_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    image_tokens: int = 0
    text_tokens: int = 0
    total_tokens: int = 0


class VariantMetrics(BaseModel):
    """Aggregated metrics for a single model+budget variant."""

    model: str
    thinking_budget: int
    scene_count: int = 0
    contentfulness_mean: float = 0.0
    contentfulness_std: float = 0.0
    thought_coverage_mean: float | None = None
    output_grounding_mean: float | None = None
    f1_mean: float | None = None
    f1_std: float | None = None
    f1_cv: float | None = None       # Coefficient of variation = f1_std / f1_mean (paper Table 3)
    perfect_f1_pct: float | None = None
    low_f1_pct: float | None = None
    thought_tokens_mean: float = 0.0
    image_tokens_mean: float = 0.0
    text_tokens_mean: float = 0.0
    input_tokens_mean: float = 0.0
    output_tokens_mean: float = 0.0
    total_tokens_mean: float = 0.0
    error_count: int = 0
    error_pct: float | None = None   # error_count / scene_count * 100 (paper Table 2)


class AggregatedResults(BaseModel):
    """Full aggregated output across all variants."""

    variants: list[VariantMetrics] = Field(default_factory=list)
    video_count: int = 0
    total_scenes: int = 0

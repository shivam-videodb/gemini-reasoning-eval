"""Gemini VLM caller for video scene understanding with configurable thinking budgets."""

import asyncio
import json
import logging
from typing import Any

import requests as http_requests
from google import genai
from google.genai import types
from tenacity import AsyncRetrying, wait_exponential, stop_after_attempt

from src.schemas import SceneResult, SceneMetadata, GeminiTokenUsage

logger = logging.getLogger(__name__)


class GeminiVLM:
    """Async wrapper around the Google GenAI SDK for scene description with extended thinking."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        thinking_budget: int = -1,
        temperature: float = 0.0,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.thinking_budget = thinking_budget
        self.temperature = temperature

        label = "dynamic" if thinking_budget == -1 else str(thinking_budget)
        logger.info(f"GeminiVLM initialized: {model}, thinking_budget={label}")

    def _variant_key(self) -> str:
        """Return a string like 'gemini-2.5-flash_128' for file naming."""
        budget_str = str(self.thinking_budget) if self.thinking_budget >= 0 else "-1"
        return f"{self.model_name}_{budget_str}"

    def _build_config(self) -> types.GenerateContentConfig:
        """Build generation config with thinking budget and thought text enabled."""
        config_kwargs: dict[str, Any] = {"temperature": self.temperature}

        if self.thinking_budget >= 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                includeThoughts=True,
            )
        else:
            # Dynamic (unlimited) budget — no token cap, expose thought text
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                includeThoughts=True,
            )

        return types.GenerateContentConfig(**config_kwargs)

    async def describe_scene(
        self,
        frames: list[Any],
        prompt: str,
        scene_metadata: SceneMetadata,
    ) -> SceneResult:
        """Send scene frames + prompt to Gemini and return structured result.

        Args:
            frames: List of PIL images for the scene.
            prompt: The VLM prompt text.
            scene_metadata: Metadata about the scene.

        Returns:
            SceneResult with thought stream, structured response, and token counts.
        """
        key = f"{scene_metadata.media_id}-scene-{scene_metadata.scene_idx}"

        try:
            response = await self._call_gemini(frames, prompt)
        except Exception as e:
            logger.error(f"Gemini call failed for {key} after retries: {e}")
            return SceneResult(
                key=key,
                model=self.model_name,
                thinking_budget=self.thinking_budget,
                metadata=scene_metadata,
                error=str(e),
            )

        # Extract thought stream and response text
        thought = ""
        response_text = ""

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "thought", None) is True:
                        thought += part.text or ""
                    else:
                        response_text += part.text or ""

        # Parse the JSON response
        response_dict: dict[str, Any] = {}
        if response_text:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            try:
                response_dict = json.loads(text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON for {key}")
                response_dict = {"raw_text": response_text}

        # Extract token counts
        tokens = GeminiTokenUsage()
        if response.usage_metadata:
            um = response.usage_metadata
            tokens = GeminiTokenUsage(
                input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                thought_tokens=getattr(um, "thoughts_token_count", 0) or 0,
                image_tokens=getattr(um, "prompt_image_token_count", 0) or 0,
                text_tokens=getattr(um, "prompt_text_token_count", 0) or 0,
                total_tokens=getattr(um, "total_token_count", 0) or 0,
            )

        return SceneResult(
            key=key,
            model=self.model_name,
            thinking_budget=self.thinking_budget,
            metadata=scene_metadata,
            thought=thought if thought else None,
            response=response_dict,
            tokens=tokens,
        )

    async def _call_gemini(self, frames: list[Any], prompt: str) -> Any:
        """Make the async Gemini API call with retry logic."""
        contents = [prompt] + frames
        config = self._build_config()

        async for attempt in AsyncRetrying(
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(3),
            before_sleep=lambda rs: logger.warning(
                f"Retrying Gemini call... Attempt {rs.attempt_number}"
            ),
        ):
            with attempt:
                return await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

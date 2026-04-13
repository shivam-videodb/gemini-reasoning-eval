"""OpenAI LLM judge for structured output extraction with caching and rate limiting."""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import TypeVar, Type

import diskcache as dc
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, InternalServerError
from tenacity import (
    AsyncRetrying,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

from src.schemas import TokenUsage

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TokenRateLimiter:
    """Token-bucket rate limiter that enforces a tokens-per-minute ceiling."""

    def __init__(self, tokens_per_minute: int):
        self._rate = tokens_per_minute / 60.0
        self._capacity = float(tokens_per_minute)
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 700) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                wait = (tokens - self._tokens) / self._rate

            await asyncio.sleep(wait)


class LLMJudge:
    """Async OpenAI client with structured output parsing, caching, and rate limiting.

    Used as the evaluation judge for coverage and dominant entity metrics.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 5,
        retry_min_wait: int = 2,
        retry_max_wait: int = 60,
        tokens_per_minute: int = 2_000_000,
        cache_dir: str = "cache",
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.rate_limiter = TokenRateLimiter(tokens_per_minute)
        self.call_count = 0

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self._cache = dc.Cache(str(cache_path))

        logger.info(f"LLMJudge initialized: model={model}, TPM={tokens_per_minute:,}")

    def _hash_prompt(self, system_prompt: str, user_prompt: str) -> str:
        content = f"{system_prompt}|{user_prompt}|{self.model}"
        return hashlib.md5(content.encode()).hexdigest()

    async def extract_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> tuple[T | None, TokenUsage]:
        """Call the LLM with structured output parsing.

        Returns (parsed_result, token_usage). Token usage is zero on cache hit.
        """
        cache_key = self._hash_prompt(system_prompt, user_prompt)

        # Check cache
        cached_data = self._cache.get(cache_key)
        if cached_data is not None:
            try:
                return response_model.model_validate_json(cached_data), TokenUsage()
            except Exception as e:
                logger.warning(f"Cache corruption for {cache_key}: {e}")

        # Make API call with retries
        return await self._call_api(system_prompt, user_prompt, response_model, cache_key)

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        cache_key: str,
    ) -> tuple[T | None, TokenUsage]:
        async for attempt in AsyncRetrying(
            wait=wait_exponential(multiplier=1, min=self.retry_min_wait, max=self.retry_max_wait),
            stop=stop_after_attempt(self.max_retries),
            retry=retry_if_exception_type((RateLimitError, APITimeoutError, InternalServerError)),
            before_sleep=lambda rs: logger.warning(
                f"Retrying LLM call... Attempt {rs.attempt_number}"
            ),
        ):
            with attempt:
                self.call_count += 1
                estimated_tokens = len(system_prompt) // 4 + len(user_prompt) // 4 + 300
                await self.rate_limiter.acquire(estimated_tokens)

                completion = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=response_model,
                )

                result = completion.choices[0].message

                if not result.parsed:
                    logger.error("Failed to parse structured output from LLM")
                    return None, TokenUsage()

                # Extract token counts
                usage = TokenUsage()
                if completion.usage:
                    details = getattr(completion.usage, "completion_tokens_details", None)
                    reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
                    usage = TokenUsage(
                        input_tokens=completion.usage.prompt_tokens,
                        output_tokens=completion.usage.completion_tokens,
                        reasoning_tokens=reasoning,
                        total_tokens=completion.usage.total_tokens,
                    )

                # Cache the result
                self._cache.set(cache_key, result.parsed.model_dump_json())

                return result.parsed, usage

"""
API wrappers for the evaluated models.

- OpenAIClient      → GPT-5.4 via OpenAI API
- OpenRouterClient  → Claude Sonnet 4.6 / GLM-5.1 via OpenRouter (OpenAI-compatible)
- MiniMaxClient     → MiniMax M2.7 via direct MiniMax API
"""

import os
from openai import AsyncOpenAI

TEMPERATURE = 0.0
MAX_TOKENS = 4096


class OpenAIClient:
    def __init__(self):
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-5.4"

    async def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # GPT-5.4 is a reasoning model: uses max_completion_tokens, not max_tokens.
        # Budget 16384 to give ample room for internal thinking + response.
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=16384,
        )
        return resp.choices[0].message.content.strip()


class OpenRouterClient:
    """Generic OpenRouter client. Set model via constructor or OPENROUTER_MODEL env."""

    # Models that use internal reasoning tokens (like o1/GPT-5.4).
    # These need max_completion_tokens instead of max_tokens to budget
    # for both thinking and response tokens.
    REASONING_MODELS = {"z-ai/glm-5.1"}

    def __init__(self, model: str | None = None):
        self._client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            timeout=300.0,
        )
        self.model = model or os.getenv(
            "OPENROUTER_MODEL", "anthropic/claude-sonnet-4-6"
        )

    async def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        is_reasoning = self.model in self.REASONING_MODELS

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": TEMPERATURE,
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = 32768
            # Stream reasoning models so we capture partial output even if
            # the token budget runs out before content is produced.
            kwargs["stream"] = True
            # Ask OpenRouter to include the reasoning/thinking text.
            kwargs["extra_body"] = {"include_reasoning": True}
        else:
            kwargs["max_tokens"] = 8192

        if not is_reasoning:
            resp = await self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()

        # Streaming path for reasoning models — accumulate both content and
        # reasoning so we never lose data even if content is empty.
        content_parts = []
        reasoning_parts = []
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            if delta.content:
                content_parts.append(delta.content)
            # OpenRouter streams reasoning in delta.reasoning or
            # delta.reasoning_content depending on the provider.
            reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)

        content = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts).strip()

        if content:
            return content

        # Content was empty — the model spent all tokens on reasoning.
        # Return the reasoning text so we still have a response to grade.
        # This is suboptimal (reasoning isn't formatted as an answer) but
        # better than an empty string.
        if reasoning:
            return reasoning

        return ""


class MiniMaxClient:
    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=os.getenv("MINIMAX_API_KEY"),
            base_url="https://api.minimax.io/v1",
            timeout=300.0,
        )
        self.model = "MiniMax-M2.7"

    async def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=16384,
        )
        return resp.choices[0].message.content.strip()

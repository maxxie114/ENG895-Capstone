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
    def __init__(self, model: str | None = None):
        self._client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model or os.getenv(
            "OPENROUTER_MODEL", "anthropic/claude-sonnet-4-6"
        )

    async def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=8192,
        )
        return resp.choices[0].message.content.strip()


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

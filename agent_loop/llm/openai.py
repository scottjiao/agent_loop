"""OpenAI-compatible LLM client implementation."""

from __future__ import annotations

import json
from typing import Any

from agent_loop.core.types import Message, Role, ToolCall, ToolSpec
from agent_loop.llm.base import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """LLM client using the ``openai`` Python package.

    Parameters
    ----------
    model : str
        Model name (e.g. ``"gpt-4o"``).
    api_key : str | None
        API key.  Falls back to the ``OPENAI_API_KEY`` env var.
    base_url : str | None
        Custom base URL (for Azure, local proxies, etc.).
    **default_kwargs
        Extra keyword arguments forwarded to every ``chat.completions.create`` call
        (e.g. ``temperature``, ``max_tokens``).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        **default_kwargs: Any,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIClient. "
                "Install it with: pip install openai"
            ) from exc

        self._model = model
        self._default_kwargs = default_kwargs
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            **self._default_kwargs,
        }
        if tools:
            kwargs["tools"] = [t.to_openai_schema() for t in tools]

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        # Parse tool calls if present.
        tool_calls: list[ToolCall] | None = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc.function.arguments}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        message = Message(
            role=Role.ASSISTANT,
            content=msg.content,
            tool_calls=tool_calls,
        )

        usage_dict: dict[str, int] = {}
        if response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        finish_reason = choice.finish_reason or "stop"

        return LLMResponse(
            message=message,
            finish_reason=finish_reason,
            usage=usage_dict,
            raw=response,
        )

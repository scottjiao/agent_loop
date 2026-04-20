"""OpenAI-compatible LLM client implementation."""

from __future__ import annotations

from typing import Any

from agent_loop.core.chat_format import ChatFormat, OpenAIChatFormat
from agent_loop.core.types import Message, Role, ToolSpec
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
    chat_format : ChatFormat | None
        Message format strategy.  Defaults to ``OpenAIChatFormat``.
    **default_kwargs
        Extra keyword arguments forwarded to every ``chat.completions.create`` call
        (e.g. ``temperature``, ``max_tokens``).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        chat_format: ChatFormat | None = None,
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
        self.chat_format: ChatFormat = chat_format or OpenAIChatFormat()

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        # Serialize via chat_format — the only place format knowledge is used.
        serialized_messages = self.chat_format.serialize_messages(messages, tools)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": serialized_messages,
            **self._default_kwargs,
        }
        if tools:
            kwargs["tools"] = [t.to_openai_schema() for t in tools]

        raw = await self._client.chat.completions.create(**kwargs)

        # Extract structured data via chat_format.
        tool_calls = self.chat_format.extract_tool_calls(raw) or None
        content = self.chat_format.extract_content(raw)
        finish_reason = self.chat_format.extract_finish_reason(raw)
        usage_dict = self.chat_format.extract_usage(raw)

        message = Message(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )

        return LLMResponse(
            message=message,
            finish_reason=finish_reason,
            usage=usage_dict,
            raw=raw,
        )

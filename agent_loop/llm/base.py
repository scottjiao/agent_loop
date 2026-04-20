"""Abstract LLM client protocol."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from agent_loop.core.chat_format import ChatFormat
from agent_loop.core.types import Message, ToolCall, ToolSpec


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    message: Message
    finish_reason: str = "stop"  # "stop", "tool_calls", "length"
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None  # Original provider-specific response object.


class LLMClient(abc.ABC):
    """Abstract interface for calling an LLM.

    Each concrete client owns a ``ChatFormat`` that handles all
    format-specific serialization and parsing.
    """

    chat_format: ChatFormat

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and return a response.

        Parameters
        ----------
        messages : list[Message]
            Internal Message objects.  The concrete client uses its
            ``chat_format`` to serialize them into the provider's format.
        tools : list[ToolSpec] | None
            Available tools (serialized by ``chat_format``).
        """

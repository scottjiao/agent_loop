"""ChatFormat — pluggable protocol for LLM message/tool serialization and parsing.

A ChatFormat encapsulates all format-specific knowledge for one model family:
- How to serialize internal Message/ToolSpec objects for the LLM
- How to parse tool calls from raw LLM responses
- How to represent tool results as conversation messages

This is owned by the LLMClient, NOT the Agent.  The Agent only ever
operates on internal types (Message, ToolSpec, ToolCall, ToolResult).
"""

from __future__ import annotations

import abc
import json
from typing import Any

from agent_loop.core.types import Message, Role, ToolCall, ToolResult, ToolSpec


class ChatFormat(abc.ABC):
    """Abstract protocol for one LLM message format."""

    @abc.abstractmethod
    def serialize_messages(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Any:
        """Convert internal Messages + ToolSpecs into the format the LLM expects.

        Returns
        -------
        Any
            OpenAI  → list[dict]
            Local   → prompt string or token ids
            Anthropic → Anthropic-format message list
        """

    @abc.abstractmethod
    def extract_tool_calls(self, raw_response: Any) -> list[ToolCall]:
        """Parse tool calls from the LLM's raw response object.

        Parameters
        ----------
        raw_response : Any
            Provider-specific raw response (e.g. OpenAI ChatCompletion,
            raw text for Hermes/Qwen3, etc.).

        Returns
        -------
        list[ToolCall]
            Parsed tool calls in internal representation.  Empty list if none.
        """

    @abc.abstractmethod
    def extract_content(self, raw_response: Any) -> str | None:
        """Extract the text content from the raw LLM response."""

    @abc.abstractmethod
    def extract_finish_reason(self, raw_response: Any) -> str:
        """Extract the finish reason from the raw LLM response."""

    @abc.abstractmethod
    def extract_usage(self, raw_response: Any) -> dict[str, int]:
        """Extract token usage stats from the raw LLM response."""

    def format_tool_result(self, result: ToolResult) -> Message:
        """Convert a ToolResult into a Message for the conversation.

        Default: OpenAI-style tool message.  Override for formats that
        represent tool results differently (e.g. Hermes wraps in
        ``<tool_response>`` inside a user message).
        """
        return Message(
            role=Role.TOOL,
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
        )

    def to_training_messages(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Any:
        """Produce a representation suitable for ``tokenizer.apply_chat_template()``.

        By default this delegates to ``serialize_messages()``.  Override when
        the training-time format differs from the inference-time API payload
        (e.g. for local models where inference goes through an OpenAI-compat
        server but training uses ``apply_chat_template`` directly).
        """
        return self.serialize_messages(messages, tools)


class OpenAIChatFormat(ChatFormat):
    """Default format — OpenAI function-calling protocol.

    Equivalent to the previously hard-coded behavior, zero behavioral change.
    """

    # -- Serialization ------------------------------------------------------

    def serialize_messages(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> list[dict[str, Any]]:
        result = [msg.to_openai_dict() for msg in messages]
        return result

    def serialize_tool_schemas(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Convenience: serialize tool specs to OpenAI function schemas."""
        return [t.to_openai_schema() for t in tools]

    # -- Extraction from raw OpenAI response --------------------------------

    def extract_tool_calls(self, raw_response: Any) -> list[ToolCall]:
        choice = raw_response.choices[0]
        msg = choice.message
        if not msg.tool_calls:
            return []
        result: list[ToolCall] = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": tc.function.arguments}
            result.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )
        return result

    def extract_content(self, raw_response: Any) -> str | None:
        return raw_response.choices[0].message.content

    def extract_finish_reason(self, raw_response: Any) -> str:
        return raw_response.choices[0].finish_reason or "stop"

    def extract_usage(self, raw_response: Any) -> dict[str, int]:
        usage = raw_response.usage
        if not usage:
            return {}
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

"""Generic conversation-history compaction for long-running agents."""

from __future__ import annotations

from agent_loop.core.context import Context
from agent_loop.core.types import CompactionConfig, Message, Role
from agent_loop.llm.base import LLMClient


class ContextCompactor:
    """Summarize older context into one compact memory message."""

    def __init__(self, llm: LLMClient, config: CompactionConfig) -> None:
        if config.keep_recent_messages < 1:
            raise ValueError("keep_recent_messages must be at least 1")
        if config.trigger_message_count <= config.keep_recent_messages:
            raise ValueError("trigger_message_count must be greater than keep_recent_messages")
        self.llm = llm
        self.config = config

    async def compact_if_needed(self, context: Context) -> bool:
        """Compact old messages when the configured threshold is exceeded."""
        messages = context.messages
        if len(messages) < self.config.trigger_message_count:
            return False

        split_at = len(messages) - self.config.keep_recent_messages
        older = messages[:split_at]
        recent = messages[split_at:]
        response = await self.llm.chat(
            [
                Message(role=Role.SYSTEM, content=self.config.summary_prompt),
                Message(role=Role.USER, content=self._format_history(older)),
            ],
            tools=None,
        )
        summary = (response.message.content or "").strip()
        if not summary:
            raise RuntimeError("Context compaction LLM returned no summary text")

        context.replace_messages(
            [
                Message(
                    role=Role.USER,
                    content=f"[Compacted conversation memory]\n{summary}",
                ),
                *recent,
            ]
        )
        return True

    def _format_history(self, messages: list[Message]) -> str:
        lines = ["Older conversation history to compact:"]
        for index, message in enumerate(messages, start=1):
            role = message.role.value
            name = f" {message.name}" if message.name else ""
            content = message.content or ""
            lines.append(f"\n[{index}] {role}{name}:\n{content}")
            if message.tool_calls:
                calls = [
                    {"name": call.name, "arguments": call.arguments}
                    for call in message.tool_calls
                ]
                lines.append(f"Tool calls: {calls}")
        return "\n".join(lines)

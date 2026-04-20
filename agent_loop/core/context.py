"""Conversation context and message history management."""

from __future__ import annotations

from agent_loop.core.types import AgentConfig, Message, Role, ToolResult


class Context:
    """Manages the conversation message list and system prompt assembly."""

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._messages: list[Message] = []
        self._skill_instructions: list[str] = []

    # -- Message management -------------------------------------------------

    def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def add_user(self, content: str) -> None:
        self.add_message(Message(role=Role.USER, content=content))

    def add_assistant(self, message: Message) -> None:
        self.add_message(message)

    def add_tool_result(self, result: ToolResult) -> None:
        self.add_message(
            Message(
                role=Role.TOOL,
                content=result.content,
                tool_call_id=result.tool_call_id,
                name=result.name,
            )
        )

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def last_assistant_message(self) -> Message | None:
        for msg in reversed(self._messages):
            if msg.role == Role.ASSISTANT:
                return msg
        return None

    # -- System prompt assembly ---------------------------------------------

    def set_skill_instructions(self, instructions: list[str]) -> None:
        self._skill_instructions = instructions

    def build_system_prompt(self) -> str:
        parts = [self._config.system_prompt]
        for instr in self._skill_instructions:
            parts.append(f"\n\n{instr}")
        return "".join(parts)

    # -- Serialization for LLM calls ----------------------------------------

    def to_openai_messages(self) -> list[dict]:
        """Build the messages list for an OpenAI-compatible API call."""
        result: list[dict] = [
            {"role": "system", "content": self.build_system_prompt()}
        ]
        # Apply sliding window truncation.
        msgs = self._messages
        max_msgs = self._config.max_context_messages
        if len(msgs) > max_msgs:
            msgs = msgs[-max_msgs:]
        for msg in msgs:
            result.append(msg.to_openai_dict())
        return result

    # -- Utilities ----------------------------------------------------------

    def clear(self) -> None:
        self._messages.clear()
        self._skill_instructions.clear()

    @property
    def message_count(self) -> int:
        return len(self._messages)

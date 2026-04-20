"""Shared types for the agent loop."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class State(Enum):
    """Agent loop states."""

    INIT = "init"
    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    FINISHED = "finished"
    ERROR = "error"


class Role(Enum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolSpec:
    """Unified tool description regardless of provider."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    provider_id: str = ""

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]

    @staticmethod
    def generate_id() -> str:
        return f"call_{uuid.uuid4().hex[:12]}"


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """A single message in the conversation."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _json_dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""

    max_iterations: int = 20
    max_context_messages: int = 100
    system_prompt: str = "You are a helpful assistant."
    dry_run: bool = False


@dataclass
class TransitionRecord:
    """Record of a state transition."""

    from_state: State
    to_state: State
    iteration: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)

"""Generic LLM-backed subagent tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_loop.core.types import Message, Role, ToolSpec
from agent_loop.llm.base import LLMClient
from agent_loop.tools.base import ToolProvider


@dataclass(frozen=True)
class SubagentSpec:
    """Configuration for one callable LLM-backed subagent."""

    name: str
    description: str
    llm: LLMClient
    system_prompt: str


class SubagentToolProvider(ToolProvider):
    """Expose focused LLM calls as normal tools for a supervising agent."""

    def __init__(
        self,
        subagents: list[SubagentSpec],
        *,
        provider_id: str = "subagents",
    ) -> None:
        self._id = provider_id
        self._subagents = {subagent.name: subagent for subagent in subagents}

    @property
    def provider_id(self) -> str:
        return self._id

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=subagent.name,
                description=subagent.description,
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The focused task for the subagent.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Relevant context the subagent should use.",
                            "default": "",
                        },
                        "constraints": {
                            "type": "string",
                            "description": "Requirements, style rules, or boundaries.",
                            "default": "",
                        },
                        "expected_output": {
                            "type": "string",
                            "description": "The requested output shape.",
                            "default": "",
                        },
                    },
                    "required": ["task"],
                },
                provider_id=self.provider_id,
            )
            for subagent in self._subagents.values()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        subagent = self._subagents.get(name)
        if subagent is None:
            raise ValueError(f"Unknown subagent tool: {name}")

        task = str(arguments["task"]).strip()
        context = str(arguments.get("context", "")).strip()
        constraints = str(arguments.get("constraints", "")).strip()
        expected_output = str(arguments.get("expected_output", "")).strip()
        response = await subagent.llm.chat(
            [
                Message(role=Role.SYSTEM, content=subagent.system_prompt),
                Message(
                    role=Role.USER,
                    content=self._build_prompt(
                        task=task,
                        context=context,
                        constraints=constraints,
                        expected_output=expected_output,
                    ),
                ),
            ],
            tools=None,
        )
        content = (response.message.content or "").strip()
        if not content:
            raise RuntimeError(f"Subagent '{name}' returned no text response")
        return content

    def _build_prompt(
        self,
        *,
        task: str,
        context: str,
        constraints: str,
        expected_output: str,
    ) -> str:
        sections = [f"Task:\n{task}"]
        if context:
            sections.append(f"Context:\n{context}")
        if constraints:
            sections.append(f"Constraints:\n{constraints}")
        if expected_output:
            sections.append(f"Expected output:\n{expected_output}")
        return "\n\n".join(sections)

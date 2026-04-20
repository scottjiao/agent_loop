"""Tool provider protocol and router."""

from __future__ import annotations

import abc
from typing import Any

from agent_loop.core.types import ToolCall, ToolResult, ToolSpec


class ToolProvider(abc.ABC):
    """Abstract base for all tool providers (local, MCP, etc.)."""

    @property
    @abc.abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider."""

    @abc.abstractmethod
    async def list_tools(self) -> list[ToolSpec]:
        """Return all tools this provider offers."""

    @abc.abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool and return the result string."""

    async def startup(self) -> None:
        """Optional lifecycle: called before first use."""

    async def shutdown(self) -> None:
        """Optional lifecycle: called on agent shutdown."""


class ToolRouter:
    """Aggregates multiple providers and routes tool calls."""

    def __init__(self, providers: list[ToolProvider] | None = None) -> None:
        self._providers: list[ToolProvider] = providers or []
        self._tool_map: dict[str, ToolProvider] = {}
        self._specs: list[ToolSpec] = []

    def add_provider(self, provider: ToolProvider) -> None:
        self._providers.append(provider)

    async def startup(self) -> None:
        """Initialize all providers and build the tool map."""
        self._tool_map.clear()
        self._specs.clear()
        for provider in self._providers:
            await provider.startup()
            tools = await provider.list_tools()
            for tool in tools:
                tool.provider_id = provider.provider_id
                if tool.name in self._tool_map:
                    raise ValueError(
                        f"Duplicate tool name '{tool.name}' from provider "
                        f"'{provider.provider_id}'. Tool names must be unique."
                    )
                self._tool_map[tool.name] = provider
                self._specs.append(tool)

    async def shutdown(self) -> None:
        for provider in self._providers:
            await provider.shutdown()

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return list(self._specs)

    async def execute(self, call: ToolCall) -> ToolResult:
        """Route a tool call to the right provider and return a unified result."""
        provider = self._tool_map.get(call.name)
        if provider is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Error: unknown tool '{call.name}'",
                is_error=True,
            )
        try:
            content = await provider.call_tool(call.name, call.arguments)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=content,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Error executing tool '{call.name}': {e}",
                is_error=True,
            )

    async def execute_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls concurrently."""
        import asyncio

        return list(await asyncio.gather(*(self.execute(c) for c in calls)))

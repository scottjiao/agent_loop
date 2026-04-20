"""A generic, state-machine-driven agent loop template."""

from agent_loop.core.types import State, ToolSpec, ToolResult, ToolCall, Message, Role
from agent_loop.core.hooks import HookPoint, HookContext, HookRegistry, HookHandler
from agent_loop.core.state_machine import StateMachine
from agent_loop.core.context import Context
from agent_loop.core.agent import Agent, AgentConfig
from agent_loop.tools.base import ToolProvider, ToolRouter
from agent_loop.tools.local import LocalToolProvider, load_tools_from_directory
from agent_loop.tools.mcp import MCPToolProvider, load_mcp_configs
from agent_loop.skills.base import Skill, SkillManager, HookBinding, load_skill, load_skills_from_directory
from agent_loop.llm.base import LLMClient, LLMResponse

__all__ = [
    "State",
    "ToolSpec",
    "ToolResult",
    "ToolCall",
    "Message",
    "Role",
    "HookPoint",
    "HookContext",
    "HookRegistry",
    "HookHandler",
    "HookBinding",
    "StateMachine",
    "Context",
    "Agent",
    "AgentConfig",
    "ToolProvider",
    "ToolRouter",
    "LocalToolProvider",
    "load_tools_from_directory",
    "MCPToolProvider",
    "load_mcp_configs",
    "Skill",
    "SkillManager",
    "load_skill",
    "load_skills_from_directory",
    "LLMClient",
    "LLMResponse",
]

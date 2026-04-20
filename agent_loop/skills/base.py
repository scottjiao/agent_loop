"""Skills system — behavior modules that register hooks into the agent loop.

A Skill packages:
    1. Instructions (prompt injection — soft control)
    2. Hook handlers (code at specific hook points — hard control)
    3. Associated tools (informational)

Folder layout for a skill::

    skills/
        trending_assistant/        # skill folder
            instructions.md        # prompt text (data, easy to edit)
            skill.py               # optional: hooks / code logic
        base.py                    # this module
"""

from __future__ import annotations

import importlib
import pathlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agent_loop.core.hooks import DEFAULT_PRIORITY, HandlerFn, HookPoint, HookRegistry
from agent_loop.core.types import State, ToolSpec


@dataclass
class HookBinding:
    """Declares that a skill wants to register a handler at a hook point."""

    hook_point: HookPoint
    handler: HandlerFn
    priority: int = DEFAULT_PRIORITY


@dataclass
class Skill:
    """A behavior module that shapes how the agent operates.

    Attributes
    ----------
    name : str
        Unique name for the skill.
    description : str
        Human-readable description.
    instructions : str
        Prompt text injected into the system prompt when the skill is active.
    hooks : list[HookBinding]
        Hook handlers this skill registers into the HookRegistry.
    tools : list[ToolSpec]
        Additional tools this skill exposes (informational — the actual
        provider is responsible for serving them).
    active_states : set[State] | None
        States in which this skill's *instructions* are active.
        ``None`` means always active.  Hook handlers are always registered
        (they control their own activation via the HookContext data).
    """

    name: str
    description: str = ""
    instructions: str = ""
    hooks: list[HookBinding] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    active_states: set[State] | None = None


class SkillManager:
    """Manages skill registration and wires skills into a HookRegistry."""

    def __init__(self, hook_registry: HookRegistry | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        self._registry = hook_registry or HookRegistry()

    @property
    def hook_registry(self) -> HookRegistry:
        return self._registry

    def register(self, skill: Skill) -> None:
        """Register a skill and wire its hook bindings into the registry."""
        self._skills[skill.name] = skill
        for binding in skill.hooks:
            self._registry.register(
                hook_point=binding.hook_point,
                handler=binding.handler,
                priority=binding.priority,
                name=f"{skill.name}.{binding.hook_point.value}",
                source=skill.name,
            )

    def unregister(self, name: str) -> None:
        """Unregister a skill and remove all its hook handlers."""
        if name in self._skills:
            del self._skills[name]
            self._registry.unregister_by_source(name)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    @property
    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def active_skills(self, state: State) -> list[Skill]:
        """Return skills whose *instructions* are active in the given state."""
        return [
            s
            for s in self._skills.values()
            if s.active_states is None or state in s.active_states
        ]

    def collect_instructions(self, state: State) -> list[str]:
        """Gather instruction strings from all skills active in the given state."""
        return [
            s.instructions
            for s in self.active_skills(state)
            if s.instructions
        ]


# ---------------------------------------------------------------------------
# Folder-based skill loading
# ---------------------------------------------------------------------------

def load_skill(skill_dir: str | pathlib.Path) -> Skill:
    """Load a single skill from a folder.

    Expected layout::

        skill_dir/
            instructions.md   — prompt text (required)
            skill.py          — optional Python module; may export:
                                  ``hooks``: list[HookBinding]
                                  ``tools``: list[ToolSpec]
                                  ``active_states``: set[State]
                                  ``description``: str
                                  ``name``: str  (overrides folder name)

    The folder name is used as the skill ``name`` unless ``skill.py`` exports
    a different one.
    """
    skill_dir = pathlib.Path(skill_dir)
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

    # --- instructions.md (data) -------------------------------------------
    instructions_file = skill_dir / "instructions.md"
    if instructions_file.exists():
        instructions = instructions_file.read_text(encoding="utf-8").strip()
    else:
        instructions = ""

    # --- defaults from folder name ----------------------------------------
    skill_name = skill_dir.name
    description = ""
    hooks: list[HookBinding] = []
    tools: list[ToolSpec] = []
    active_states: set[State] | None = None

    # --- skill.py (code, optional) ----------------------------------------
    skill_py = skill_dir / "skill.py"
    if skill_py.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            f"agent_loop.skills.{skill_name}.skill", skill_py
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            skill_name = getattr(mod, "name", skill_name)
            description = getattr(mod, "description", description)
            hooks = getattr(mod, "hooks", hooks)
            tools = getattr(mod, "tools", tools)
            active_states = getattr(mod, "active_states", active_states)

    return Skill(
        name=skill_name,
        description=description,
        instructions=instructions,
        hooks=hooks,
        tools=tools,
        active_states=active_states,
    )


def load_skills_from_directory(skills_root: str | pathlib.Path) -> list[Skill]:
    """Scan a directory and load every sub-folder that contains an ``instructions.md``.

    Folders without ``instructions.md`` are silently skipped (they may be
    ``__pycache__``, ``base.py``, etc.).
    """
    skills_root = pathlib.Path(skills_root)
    skills: list[Skill] = []
    for child in sorted(skills_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(("_", ".")):
            continue
        if (child / "instructions.md").exists():
            skills.append(load_skill(child))
    return skills

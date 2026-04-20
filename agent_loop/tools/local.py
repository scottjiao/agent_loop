"""Local tool provider — register Python functions as tools via decorators."""

from __future__ import annotations

import inspect
import json
import pathlib
from collections.abc import Callable
from typing import Any, get_type_hints

from agent_loop.core.types import ToolSpec
from agent_loop.tools.base import ToolProvider

# Mapping from Python types to JSON Schema types.
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(tp: type) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    json_type = _TYPE_MAP.get(tp)
    if json_type:
        return {"type": json_type}
    # Fallback for complex or missing types.
    return {"type": "string"}


def _build_parameters_schema(func: Callable) -> dict[str, Any]:
    """Inspect a function's signature to build a JSON Schema for its parameters."""
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        prop = _python_type_to_json_schema(hints.get(name, str))
        # Use parameter doc from the docstring if available (simple heuristic).
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


class LocalToolProvider(ToolProvider):
    """Register local async/sync Python functions as tools."""

    def __init__(self, provider_id: str = "local") -> None:
        self._id = provider_id
        self._tools: dict[str, ToolSpec] = {}
        self._funcs: dict[str, Callable] = {}

    @property
    def provider_id(self) -> str:
        return self._id

    # -- Registration -------------------------------------------------------

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable:
        """Decorator to register a function as a tool.

        Usage::

            local = LocalToolProvider()

            @local.tool()
            async def search_web(query: str) -> str:
                \"\"\"Search the web.\"\"\"
                ...
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or (inspect.getdoc(func) or "")
            schema = _build_parameters_schema(func)
            spec = ToolSpec(
                name=tool_name,
                description=tool_desc,
                parameters=schema,
                provider_id=self._id,
            )
            self._tools[tool_name] = spec
            self._funcs[tool_name] = func
            return func

        return decorator

    def register(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Programmatic registration (non-decorator)."""
        self.tool(name=name, description=description)(func)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    # -- ToolProvider interface ---------------------------------------------

    async def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        func = self._funcs.get(name)
        if func is None:
            raise ValueError(f"Tool '{name}' not registered in LocalToolProvider")
        result = func(**arguments)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Folder-based tool loading
# ---------------------------------------------------------------------------

def load_tool(tool_dir: str | pathlib.Path) -> LocalToolProvider:
    """Load a single local tool provider from a folder.

    Expected layout::

        tool_dir/
            tool.py   — must export ``provider: LocalToolProvider``

    The module-level ``provider`` object already has tools registered via the
    ``@provider.tool()`` decorator, so we just import and return it.
    """
    import importlib.util

    tool_dir = pathlib.Path(tool_dir)
    tool_py = tool_dir / "tool.py"
    if not tool_py.exists():
        raise FileNotFoundError(f"No tool.py found in {tool_dir}")

    spec = importlib.util.spec_from_file_location(
        f"agent_loop.tools.{tool_dir.name}.tool", tool_py
    )
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {tool_py}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    provider = getattr(mod, "provider", None)
    if provider is None:
        raise AttributeError(
            f"{tool_py} must export a module-level 'provider' (LocalToolProvider instance)"
        )
    return provider


def load_tools_from_directory(
    tools_root: str | pathlib.Path,
) -> list[LocalToolProvider]:
    """Scan a directory and load every sub-folder that contains a ``tool.py``.

    Folders without ``tool.py`` are silently skipped (e.g. ``__pycache__``).
    """
    tools_root = pathlib.Path(tools_root)
    providers: list[LocalToolProvider] = []
    for child in sorted(tools_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(("_", ".")):
            continue
        if (child / "tool.py").exists():
            providers.append(load_tool(child))
    return providers

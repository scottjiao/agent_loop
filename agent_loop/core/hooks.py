"""Event + Priority hook system for the agent loop.

Core concepts:
    HookPoint   — a named point in the agent loop where handlers can be registered
    HookContext — mutable data bag flowing through handlers at a hook point
    HookHandler — a callback registered at a hook point with a priority
    HookRegistry — manages registration and dispatching
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook points — the static set of places where the loop accepts intervention
# ---------------------------------------------------------------------------

class HookPoint(Enum):
    """All hook points exposed by the agent loop."""

    # PLANNING phase
    BEFORE_PLAN = "before_plan"          # Before LLM call; can inject/modify instructions
    AFTER_PLAN = "after_plan"            # After LLM call; can inspect/modify LLM response

    # ACTING phase
    BEFORE_ACT = "before_act"            # Before tool execution; can inspect/filter/block tool calls
    AFTER_ACT = "after_act"              # After tool execution; can inspect/modify tool results

    # OBSERVING phase
    ON_OBSERVE = "on_observe"            # When results are added to context; can annotate

    # REFLECTING phase
    ON_REFLECT = "on_reflect"            # Decide continue/stop; can override next state

    # Lifecycle
    ON_START = "on_start"                # Agent run begins
    ON_FINISH = "on_finish"              # Agent run ends
    ON_ERROR = "on_error"                # Error occurred


# ---------------------------------------------------------------------------
# HookContext — the data flowing through handlers
# ---------------------------------------------------------------------------

class HookContext:
    """Mutable data bag passed to every handler at a hook point.

    Handlers can read/write arbitrary keys via attribute access or dict-style.
    Two control signals are available:
        prevent_default() — skip the agent loop's default behavior for this hook point
        stop_propagation() — skip remaining handlers (lower priority won't run)
    """

    def __init__(self, hook_point: HookPoint, **initial_data: Any) -> None:
        self._hook_point = hook_point
        self._default_prevented = False
        self._propagation_stopped = False
        self._data: dict[str, Any] = initial_data

    @property
    def hook_point(self) -> HookPoint:
        return self._hook_point

    # -- Control signals ----------------------------------------------------

    def prevent_default(self) -> None:
        """Tell the agent loop to skip its default behavior at this hook point."""
        self._default_prevented = True

    def stop_propagation(self) -> None:
        """Stop calling remaining handlers (lower-priority ones are skipped)."""
        self._propagation_stopped = True

    @property
    def is_default_prevented(self) -> bool:
        return self._default_prevented

    @property
    def is_propagation_stopped(self) -> bool:
        return self._propagation_stopped

    # -- Data access (dict-style + attribute-style) -------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"HookContext has no attribute '{name}'") from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


# ---------------------------------------------------------------------------
# Handler type and registration entry
# ---------------------------------------------------------------------------

# Handler signature: async (ctx: HookContext) -> None
HandlerFn = Callable[[HookContext], Awaitable[None] | None]

DEFAULT_PRIORITY = 100


@dataclass
class HookHandler:
    """A registered handler entry."""

    fn: HandlerFn
    priority: int = DEFAULT_PRIORITY  # Lower = runs first
    name: str = ""                     # For debugging/logging
    source: str = ""                   # Which skill/plugin registered this


# ---------------------------------------------------------------------------
# HookRegistry — the "mod manager"
# ---------------------------------------------------------------------------

class HookRegistry:
    """Central registry that manages handler registration and dispatching."""

    def __init__(self) -> None:
        self._handlers: dict[HookPoint, list[HookHandler]] = {
            hp: [] for hp in HookPoint
        }

    # -- Registration -------------------------------------------------------

    def register(
        self,
        hook_point: HookPoint,
        handler: HandlerFn,
        priority: int = DEFAULT_PRIORITY,
        name: str = "",
        source: str = "",
    ) -> HookHandler:
        """Register a handler at a hook point with a given priority."""
        entry = HookHandler(fn=handler, priority=priority, name=name, source=source)
        handlers = self._handlers[hook_point]
        handlers.append(entry)
        # Keep sorted by priority (lower first).
        handlers.sort(key=lambda h: h.priority)
        logger.debug(
            "Registered hook: %s @ %s (priority=%d, source=%s)",
            name or handler.__name__,
            hook_point.value,
            priority,
            source,
        )
        return entry

    def unregister(self, hook_point: HookPoint, handler: HookHandler) -> None:
        """Remove a specific handler entry."""
        try:
            self._handlers[hook_point].remove(handler)
        except ValueError:
            pass

    def unregister_by_source(self, source: str) -> int:
        """Remove all handlers from a given source. Returns count removed."""
        count = 0
        for hp in HookPoint:
            before = len(self._handlers[hp])
            self._handlers[hp] = [h for h in self._handlers[hp] if h.source != source]
            count += before - len(self._handlers[hp])
        return count

    def clear(self, hook_point: HookPoint | None = None) -> None:
        """Clear handlers for a specific hook point, or all if None."""
        if hook_point:
            self._handlers[hook_point].clear()
        else:
            for hp in HookPoint:
                self._handlers[hp].clear()

    # -- Dispatching --------------------------------------------------------

    async def emit(self, hook_point: HookPoint, **data: Any) -> HookContext:
        """Emit a hook event: create a HookContext and run all handlers in priority order.

        Returns the HookContext so the caller can inspect results and control signals.
        """
        ctx = HookContext(hook_point, **data)
        handlers = self._handlers.get(hook_point, [])

        for handler in handlers:
            if ctx.is_propagation_stopped:
                logger.debug(
                    "Propagation stopped at hook %s, skipping %s",
                    hook_point.value,
                    handler.name or handler.fn.__name__,
                )
                break
            try:
                result = handler.fn(ctx)
                if result is not None:
                    await result
            except Exception:
                logger.exception(
                    "Error in hook handler '%s' at %s",
                    handler.name or handler.fn.__name__,
                    hook_point.value,
                )

        return ctx

    # -- Introspection ------------------------------------------------------

    def handlers_for(self, hook_point: HookPoint) -> list[HookHandler]:
        """Return all handlers for a hook point (sorted by priority)."""
        return list(self._handlers.get(hook_point, []))

    def summary(self) -> dict[str, int]:
        """Return a {hook_point_name: handler_count} summary."""
        return {hp.value: len(hs) for hp, hs in self._handlers.items() if hs}

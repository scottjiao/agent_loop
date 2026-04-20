"""State machine engine with declarative transitions and hooks."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent_loop.core.types import State, TransitionRecord

logger = logging.getLogger(__name__)

# Type aliases for hooks.
# Hooks receive (from_state, to_state, metadata) and may be sync or async.
Hook = Callable[[State, State, dict[str, Any]], Awaitable[None] | None]

# Default allowed transitions.
DEFAULT_TRANSITIONS: dict[State, set[State]] = {
    State.INIT: {State.PLANNING},
    State.PLANNING: {State.ACTING, State.FINISHED},
    State.ACTING: {State.OBSERVING, State.ERROR},
    State.OBSERVING: {State.REFLECTING},
    State.REFLECTING: {State.PLANNING, State.FINISHED},
    State.ERROR: {State.PLANNING, State.FINISHED},
}


class InvalidTransitionError(Exception):
    pass


class StateMachine:
    """Manages agent state transitions with hooks and history."""

    def __init__(
        self,
        transitions: dict[State, set[State]] | None = None,
        initial_state: State = State.INIT,
    ) -> None:
        self._transitions = transitions or DEFAULT_TRANSITIONS
        self._state = initial_state
        self._history: list[TransitionRecord] = []
        self._iteration = 0

        # Hook registries.
        self._on_enter: dict[State, list[Hook]] = {}
        self._on_exit: dict[State, list[Hook]] = {}
        self._before_transition: list[Hook] = []
        self._after_transition: list[Hook] = []

    # -- Properties ---------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def history(self) -> list[TransitionRecord]:
        return list(self._history)

    @property
    def iteration(self) -> int:
        return self._iteration

    # -- Transition ---------------------------------------------------------

    async def transition(self, to: State, **metadata: Any) -> None:
        """Transition to a new state, enforcing rules and firing hooks."""
        frm = self._state
        allowed = self._transitions.get(frm, set())
        if to not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {frm.value} to {to.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        # Fire before-transition hooks.
        for hook in self._before_transition:
            await _maybe_await(hook(frm, to, metadata))

        # Fire on-exit hooks for the current state.
        for hook in self._on_exit.get(frm, []):
            await _maybe_await(hook(frm, to, metadata))

        # Perform transition.
        self._state = to
        record = TransitionRecord(
            from_state=frm,
            to_state=to,
            iteration=self._iteration,
            metadata=metadata,
        )
        self._history.append(record)
        logger.debug("Transition: %s -> %s (iter=%d)", frm.value, to.value, self._iteration)

        # Fire on-enter hooks for the new state.
        for hook in self._on_enter.get(to, []):
            await _maybe_await(hook(frm, to, metadata))

        # Fire after-transition hooks.
        for hook in self._after_transition:
            await _maybe_await(hook(frm, to, metadata))

    def increment_iteration(self) -> None:
        self._iteration += 1

    def reset(self, initial_state: State = State.INIT) -> None:
        self._state = initial_state
        self._history.clear()
        self._iteration = 0

    # -- Hook registration --------------------------------------------------

    def on_enter(self, state: State, hook: Hook) -> None:
        self._on_enter.setdefault(state, []).append(hook)

    def on_exit(self, state: State, hook: Hook) -> None:
        self._on_exit.setdefault(state, []).append(hook)

    def before_transition(self, hook: Hook) -> None:
        self._before_transition.append(hook)

    def after_transition(self, hook: Hook) -> None:
        self._after_transition.append(hook)

    # -- Introspection ------------------------------------------------------

    def can_transition(self, to: State) -> bool:
        return to in self._transitions.get(self._state, set())

    def allowed_transitions(self) -> set[State]:
        return self._transitions.get(self._state, set())

    def add_transition(self, from_state: State, to_state: State) -> None:
        self._transitions.setdefault(from_state, set()).add(to_state)


async def _maybe_await(result: Awaitable[None] | None) -> None:
    """Await if the result is a coroutine, otherwise do nothing."""
    if result is not None:
        await result

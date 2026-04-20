"""Agent — the main async loop orchestrating state machine, LLM, tools, and skills."""

from __future__ import annotations

import logging
from typing import Any

from agent_loop.core.context import Context
from agent_loop.core.hooks import HookPoint, HookRegistry
from agent_loop.core.state_machine import StateMachine
from agent_loop.core.types import AgentConfig, Message, Role, State, ToolResult
from agent_loop.llm.base import LLMClient, LLMResponse
from agent_loop.skills.base import SkillManager
from agent_loop.tools.base import ToolProvider, ToolRouter

logger = logging.getLogger(__name__)


class Agent:
    """State-machine-driven agent loop with a hook system.

    Parameters
    ----------
    llm : LLMClient
        The language model backend.
    tools : list[ToolProvider] | None
        Tool providers (local, MCP, …).
    skills : list | None
        Skill instances to register.
    config : AgentConfig | None
        Agent configuration.
    state_machine : StateMachine | None
        Custom state machine (uses default if not provided).
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: list[ToolProvider] | None = None,
        skills: list | None = None,
        config: AgentConfig | None = None,
        state_machine: StateMachine | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or AgentConfig()
        self.sm = state_machine or StateMachine()
        self.context = Context(self.config)
        self.router = ToolRouter(tools)

        # Hook registry is shared between agent and skill manager.
        self.hooks = HookRegistry()
        self.skill_manager = SkillManager(hook_registry=self.hooks)

        if skills:
            for skill in skills:
                self.skill_manager.register(skill)

        # Accumulated token usage across the run.
        self.total_usage: dict[str, int] = {}

        # The last LLM response for the run.
        self._last_response: LLMResponse | None = None

        # dry_run 模式下，截停时存放完整的 LLM 调用 payload。
        self.dry_run_payload: dict[str, Any] | None = None

    # -- Public API ---------------------------------------------------------

    async def run(self, user_message: str) -> str:
        """Run the full agent loop for a single user turn. Returns the final text."""
        try:
            await self.router.startup()
            self.sm.reset()
            self.context.clear()
            self.total_usage = {}

            # ON_START hook.
            await self.hooks.emit(
                HookPoint.ON_START,
                agent=self,
                user_message=user_message,
            )

            # INIT → add user message, transition to PLANNING.
            self.context.add_user(user_message)
            await self.sm.transition(State.PLANNING)

            while self.sm.state not in (State.FINISHED, State.ERROR):
                self.sm.increment_iteration()
                if self.sm.iteration > self.config.max_iterations:
                    logger.warning("Max iterations (%d) reached.", self.config.max_iterations)
                    await self.sm.transition(State.FINISHED)
                    break

                await self._step()

            # ON_FINISH hook.
            final_answer = self._extract_final_answer()
            await self.hooks.emit(
                HookPoint.ON_FINISH,
                agent=self,
                answer=final_answer,
                usage=self.total_usage,
            )
            return final_answer

        finally:
            await self.router.shutdown()

    # -- State handlers -----------------------------------------------------

    async def _step(self) -> None:
        """Execute one step based on the current state."""
        handlers = {
            State.PLANNING: self._handle_planning,
            State.ACTING: self._handle_acting,
            State.OBSERVING: self._handle_observing,
            State.REFLECTING: self._handle_reflecting,
        }
        handler = handlers.get(self.sm.state)
        if handler is None:
            raise RuntimeError(f"No handler for state {self.sm.state.value}")
        try:
            await handler()
        except Exception as e:
            logger.exception("Error in state %s", self.sm.state.value)
            # ON_ERROR hook — plugins can inspect the error.
            await self.hooks.emit(
                HookPoint.ON_ERROR,
                agent=self,
                error=e,
                state=self.sm.state,
            )
            # Inject error into context so the LLM can recover.
            self.context.add_message(
                Message(role=Role.USER, content=f"[System error: {e}]")
            )
            if self.sm.can_transition(State.ERROR):
                await self.sm.transition(State.ERROR)
                if self.sm.can_transition(State.PLANNING):
                    await self.sm.transition(State.PLANNING)
                else:
                    await self.sm.transition(State.FINISHED)
            else:
                await self.sm.transition(State.FINISHED)

    async def _handle_planning(self) -> None:
        """Call the LLM to decide next action (or produce final answer)."""
        # Collect skill instructions.
        instructions = self.skill_manager.collect_instructions(State.PLANNING)
        self.context.set_skill_instructions(instructions)

        # BEFORE_PLAN hook — plugins can modify instructions, context, or skip the LLM call.
        ctx = await self.hooks.emit(
            HookPoint.BEFORE_PLAN,
            agent=self,
            context=self.context,
            tool_specs=self.router.tool_specs,
        )

        if ctx.is_default_prevented:
            # A plugin handled planning; it should have put a Message into context.
            last = self.context.last_assistant_message
            if last and last.tool_calls:
                await self.sm.transition(State.ACTING)
            else:
                await self.sm.transition(State.FINISHED)
            return

        # --- dry_run: 截停在 LLM 调用前，dump 完整 payload 并退出 ---
        if self.config.dry_run:
            import json as _json

            messages = self.context.messages_with_system_prompt()
            tools = self.router.tool_specs or []
            chat_fmt = self.llm.chat_format
            self.dry_run_payload = {
                "messages": chat_fmt.serialize_messages(messages, tools),
                "tools": [t.to_openai_schema() for t in tools],
            }
            print("\n" + "=" * 70)
            print("  DRY RUN — payload that would be sent to LLM")
            print("=" * 70)
            print(_json.dumps(self.dry_run_payload, indent=2, ensure_ascii=False))
            print("=" * 70 + "\n")
            await self.sm.transition(State.FINISHED)
            return

        # Call LLM (default behavior) — passes internal types; the LLMClient
        # uses its chat_format to serialize.
        response = await self.llm.chat(
            messages=self.context.messages_with_system_prompt(),
            tools=self.router.tool_specs or None,
        )
        self._last_response = response
        self._accumulate_usage(response.usage)
        self.context.add_assistant(response.message)

        # AFTER_PLAN hook — plugins can inspect/modify the LLM response.
        ctx = await self.hooks.emit(
            HookPoint.AFTER_PLAN,
            agent=self,
            response=response,
            message=response.message,
        )

        if response.message.tool_calls:
            await self.sm.transition(State.ACTING)
        else:
            await self.sm.transition(State.FINISHED)

    async def _handle_acting(self) -> None:
        """Execute tool calls from the last LLM response."""
        last = self.context.last_assistant_message
        if not last or not last.tool_calls:
            await self.sm.transition(State.OBSERVING)
            return

        tool_calls = last.tool_calls

        # BEFORE_ACT hook — plugins can inspect/filter/block tool calls.
        ctx = await self.hooks.emit(
            HookPoint.BEFORE_ACT,
            agent=self,
            tool_calls=tool_calls,
        )
        # Allow plugins to modify tool_calls via context.
        tool_calls = ctx.get("tool_calls", tool_calls)

        if ctx.is_default_prevented:
            # Plugin handled execution; check if it provided results.
            self._pending_results = ctx.get("results", [])
            await self.sm.transition(State.OBSERVING)
            return

        # Execute tools (default behavior).
        results = await self.router.execute_many(
            tool_calls, max_concurrency=self.config.max_parallel_tool_calls,
        )

        # AFTER_ACT hook — plugins can inspect/modify tool results.
        ctx = await self.hooks.emit(
            HookPoint.AFTER_ACT,
            agent=self,
            results=results,
            tool_calls=tool_calls,
        )
        results = ctx.get("results", results)

        self._pending_results = results
        await self.sm.transition(State.OBSERVING)

    async def _handle_observing(self) -> None:
        """Add tool results to context."""
        results: list[ToolResult] = getattr(self, "_pending_results", [])

        # ON_OBSERVE hook — plugins can annotate, filter, or transform results
        # before they are written to context.
        ctx = await self.hooks.emit(
            HookPoint.ON_OBSERVE,
            agent=self,
            results=results,
        )
        results = ctx.get("results", results)

        if not ctx.is_default_prevented:
            for r in results:
                r = ToolResult(
                    tool_call_id=r.tool_call_id,
                    name=r.name,
                    content=self._truncate_tool_response(r.content),
                    is_error=r.is_error,
                )
                msg = self.llm.chat_format.format_tool_result(r)
                self.context.add_message(msg)

        self._pending_results = []
        await self.sm.transition(State.REFLECTING)

    async def _handle_reflecting(self) -> None:
        """Decide whether to continue planning or finish.

        Default: loop back to PLANNING.  Plugins can override via ON_REFLECT.
        """
        ctx = await self.hooks.emit(
            HookPoint.ON_REFLECT,
            agent=self,
            iteration=self.sm.iteration,
            max_iterations=self.config.max_iterations,
        )

        if ctx.is_default_prevented:
            # Plugin decided the next state; it should set ctx.next_state.
            next_state = ctx.get("next_state", State.PLANNING)
            await self.sm.transition(next_state)
        else:
            await self.sm.transition(State.PLANNING)

    # -- Helpers ------------------------------------------------------------

    def _truncate_tool_response(self, content: str) -> str:
        """Truncate tool response content based on config. Final safeguard after hooks."""
        max_len = self.config.max_tool_response_length
        if max_len is None or len(content) <= max_len:
            return content
        side = self.config.tool_response_truncate_side
        if side == "left":
            return content[:max_len] + "...(truncated)"
        elif side == "right":
            return "(truncated)..." + content[-max_len:]
        else:  # middle
            half = max_len // 2
            return content[:half] + "...(truncated)..." + content[-half:]

    def _accumulate_usage(self, usage: dict[str, int]) -> None:
        for k, v in usage.items():
            self.total_usage[k] = self.total_usage.get(k, 0) + v

    def _extract_final_answer(self) -> str:
        """Pull the last assistant text content as the final answer."""
        last = self.context.last_assistant_message
        if last and last.content:
            return last.content
        return "(Agent finished with no text response.)"

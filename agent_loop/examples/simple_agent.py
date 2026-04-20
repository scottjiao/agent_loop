"""Minimal example — an agent with a local tool, skills with hooks, and OpenAI backend.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m agent_loop.examples.simple_agent
"""

from __future__ import annotations

import asyncio
import random

from agent_loop import Agent, AgentConfig, HookBinding, HookPoint, LocalToolProvider, Skill
from agent_loop.core.hooks import HookContext
from agent_loop.llm.openai import OpenAIClient


# 1. Create a local tool provider and register tools.
local = LocalToolProvider()


@local.tool()
async def roll_dice(sides: int = 6) -> str:
    """Roll a dice with the given number of sides."""
    result = random.randint(1, sides)
    return f"Rolled a {result} (d{sides})"


@local.tool()
async def calculate(expression: str) -> str:
    """Evaluate a simple math expression (e.g. '2 + 3 * 4')."""
    # Only allow safe characters.
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression)  # noqa: S307 — safe due to allowlist above
    except Exception as e:
        return f"Error: {e}"
    return str(result)


# 2. Define hook handlers.
async def log_before_act(ctx: HookContext) -> None:
    """Log every tool call before execution."""
    for tc in ctx.get("tool_calls", []):
        print(f"  [hook] About to call tool: {tc.name}({tc.arguments})")


async def log_after_act(ctx: HookContext) -> None:
    """Log every tool result after execution."""
    for r in ctx.get("results", []):
        status = "ERROR" if r.is_error else "OK"
        print(f"  [hook] Tool result ({status}): {r.name} -> {r.content[:80]}")


# 3. Define a skill with both instructions AND hooks.
math_skill = Skill(
    name="math_helper",
    description="Guides the agent to use calculation tools.",
    instructions=(
        "When the user asks a math question, prefer using the 'calculate' tool "
        "rather than computing in your head. Show your work."
    ),
    hooks=[
        HookBinding(HookPoint.BEFORE_ACT, log_before_act, priority=50),
        HookBinding(HookPoint.AFTER_ACT, log_after_act, priority=50),
    ],
)


async def main() -> None:
    llm = OpenAIClient(model="gpt-4o")

    agent = Agent(
        llm=llm,
        tools=[local],
        skills=[math_skill],
        config=AgentConfig(
            max_iterations=10,
            system_prompt="You are a helpful assistant with access to tools.",
        ),
    )

    user_input = "Roll two d20 dice and then calculate the sum of the results."
    print(f"User: {user_input}\n")

    answer = await agent.run(user_input)
    print(f"\nAgent: {answer}")
    print(f"\nToken usage: {agent.total_usage}")
    print(f"Iterations: {agent.sm.iteration}")
    print(f"Hooks registered: {agent.hooks.summary()}")


if __name__ == "__main__":
    asyncio.run(main())

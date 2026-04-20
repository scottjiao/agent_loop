"""测试脚本 — 通过主链路 dry_run 模式查看最终发给 LLM 的完整 prompt。

MCP 服务器会真实启动、工具会真实发现、skill 指令会真实注入，
仅在 llm.chat() 调用前截停并打印完整 payload。

用法:
    python -m agent_loop.examples.test_prompt_assembly
"""

from __future__ import annotations

import asyncio
import logging

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")

from agent_loop.main import create_agent


async def main() -> None:
    agent = await create_agent(dry_run=True)

    user_msg = "现在几点了？最近有什么热门话题？"
    print(f"User message: {user_msg}\n")
    await agent.run(user_msg)

    if agent.dry_run_payload:
        n_msgs = len(agent.dry_run_payload["messages"])
        n_tools = len(agent.dry_run_payload["tools"])
        print(f"\nMessages: {n_msgs}, Tools: {n_tools}")


if __name__ == "__main__":
    asyncio.run(main())

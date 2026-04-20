"""主 Agent 工厂 — 自动扫描 tools / skills / mcp，构建一个完整配置的 Agent。

目录约定::

    project_root/
        configs/
            agent.json          # 主 agent 配置
            mcp_servers.json    # MCP 服务器列表
        agent_loop/
            tools/              # 文件夹化 local tools (每个子目录含 tool.py)
            skills/             # 文件夹化 skills  (每个子目录含 instructions.md)
            main.py             # 本文件

用法::

    from agent_loop.main import create_agent

    agent = await create_agent()               # 正常模式
    agent = await create_agent(dry_run=True)    # dry_run 模式
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

from agent_loop.core.agent import Agent
from agent_loop.core.types import AgentConfig
from agent_loop.llm.base import LLMClient
from agent_loop.llm.openai import OpenAIClient
from agent_loop.skills.base import load_skills_from_directory
from agent_loop.tools.local import load_tools_from_directory
from agent_loop.tools.mcp import load_mcp_configs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径常量 — 基于本文件位置向上推导项目根目录
# ---------------------------------------------------------------------------
_PKG_DIR = pathlib.Path(__file__).resolve().parent          # agent_loop/
_PROJECT_ROOT = _PKG_DIR.parent                              # project root
_CONFIGS_DIR = _PROJECT_ROOT / "configs"

AGENT_CONFIG_PATH = _CONFIGS_DIR / "agent.json"
MCP_CONFIG_PATH = _CONFIGS_DIR / "mcp_servers.json"
TOOLS_DIR = _PKG_DIR / "tools"
SKILLS_DIR = _PKG_DIR / "skills"


def load_agent_config(
    path: pathlib.Path = AGENT_CONFIG_PATH,
    *,
    dry_run: bool = False,
) -> tuple[AgentConfig, dict[str, Any]]:
    """读取 agent.json，返回 (AgentConfig, raw_dict)。

    raw_dict 包含 model 等 LLM 层面的配置，AgentConfig 只管 agent loop 自身的参数。
    """
    raw: dict[str, Any] = {}
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))

    config = AgentConfig(
        max_iterations=raw.get("max_iterations", 20),
        max_context_messages=raw.get("max_context_messages", 100),
        system_prompt=raw.get("system_prompt", "You are a helpful assistant."),
        dry_run=dry_run,
    )
    return config, raw


def create_llm(raw_config: dict[str, Any], *, dry_run: bool = False) -> LLMClient:
    """根据配置创建 LLM 客户端。dry_run 时用占位 key 绕过初始化校验。"""
    kwargs: dict[str, Any] = {}
    if dry_run:
        kwargs["api_key"] = "dry-run-placeholder"
    return OpenAIClient(
        model=raw_config.get("model", "gpt-4o"),
        **kwargs,
    )


async def create_agent(
    *,
    dry_run: bool = False,
    config_path: pathlib.Path = AGENT_CONFIG_PATH,
    mcp_config_path: pathlib.Path = MCP_CONFIG_PATH,
    tools_dir: pathlib.Path = TOOLS_DIR,
    skills_dir: pathlib.Path = SKILLS_DIR,
) -> Agent:
    """一键构建主 Agent。

    自动扫描:
    - ``tools_dir`` 下所有含 ``tool.py`` 的子目录 → LocalToolProvider
    - ``skills_dir`` 下所有含 ``instructions.md`` 的子目录 → Skill
    - ``mcp_config_path`` → MCPToolProvider(s)

    Parameters
    ----------
    dry_run : bool
        True 时 LLM 调用前截停并 dump payload。
    config_path / mcp_config_path / tools_dir / skills_dir :
        可覆盖默认路径。
    """
    # 1. 读取主配置
    agent_config, raw_config = load_agent_config(config_path, dry_run=dry_run)
    logger.info("Agent config loaded: %s", config_path)

    # 2. 创建 LLM
    llm = create_llm(raw_config, dry_run=dry_run)
    logger.info("LLM created: model=%s", raw_config.get("model", "gpt-4o"))

    # 3. 自动发现 local tools
    local_providers = load_tools_from_directory(tools_dir)
    logger.info("Local tools loaded: %d providers from %s", len(local_providers), tools_dir)

    # 4. 自动发现 MCP 服务器
    mcp_providers = []
    if mcp_config_path.exists():
        mcp_providers = load_mcp_configs(mcp_config_path)
        logger.info("MCP providers loaded: %d from %s", len(mcp_providers), mcp_config_path)

    # 5. 自动发现 skills
    skills = load_skills_from_directory(skills_dir)
    logger.info("Skills loaded: %d from %s", len(skills), skills_dir)

    # 6. 组装 Agent
    all_tools = [*local_providers, *mcp_providers]
    agent = Agent(
        llm=llm,
        tools=all_tools,
        skills=skills,
        config=agent_config,
    )
    logger.info(
        "Agent assembled: %d tool providers, %d skills",
        len(all_tools),
        len(skills),
    )
    return agent

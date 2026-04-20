"""MCP (Model Context Protocol) tool provider — discovers and calls tools from MCP servers.

Config file format (JSON list)::

    [
        {
            "name": "Trends Hub",
            "organization": "baranwang",
            "description": "基于 MCP 协议的全网热点趋势聚合服务",
            "web": "https://github.com/baranwang/mcp-trends-hub",
            "config": {
                "mcpServers": {
                    "trends-hub": {
                        "command": "npx",
                        "args": ["-y", "mcp-trends-hub@1.6.2"],
                        "env": {}
                    }
                }
            },
            "category": "Discovery"
        }
    ]
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

from agent_loop.core.types import ToolSpec
from agent_loop.tools.base import ToolProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rich config dataclass matching the user-facing JSON template
# ---------------------------------------------------------------------------

@dataclass
class MCPServerEntry:
    """One entry in the MCP config file — may contain multiple servers."""

    name: str
    description: str = ""
    organization: str = ""
    web: str = ""
    category: str = ""
    config: dict[str, Any] = field(default_factory=dict)


class MCPToolProvider(ToolProvider):
    """Connects to an MCP server via stdio and exposes its tools.

    Parameters
    ----------
    command : str
        The executable to launch (e.g. ``"npx"``, ``"python"``).
    args : list[str]
        Arguments for the command.
    env : dict | None
        Extra environment variables for the subprocess.
    provider_id : str
        Unique id for this provider (defaults to ``"mcp"``).
    name : str
        Human-readable name (from config).
    description : str
        Description of the MCP service (from config).
    organization : str
        Organization or author (from config).
    web : str
        Homepage / repository URL (from config).
    category : str
        Category tag (from config).
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        provider_id: str = "mcp",
        *,
        name: str = "",
        description: str = "",
        organization: str = "",
        web: str = "",
        category: str = "",
    ) -> None:
        self._command = command
        self._args = args or []
        self._env = env
        self._id = provider_id

        # Rich metadata.
        self.name = name
        self.description = description
        self.organization = organization
        self.web = web
        self.category = category

        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._tools: dict[str, ToolSpec] = {}

        # Read buffer for the stdio JSON-RPC stream.
        self._read_lock = asyncio.Lock()
        self._buf = b""

    @property
    def provider_id(self) -> str:
        return self._id

    # -- Lifecycle ----------------------------------------------------------

    async def startup(self) -> None:
        """Launch the MCP server subprocess and perform initialization."""
        import shutil
        import sys

        command = self._command
        # Windows 下 npx/node 等需要用 .cmd 后缀
        if sys.platform == "win32" and not command.endswith(".cmd"):
            cmd_path = shutil.which(command) or shutil.which(command + ".cmd")
            if cmd_path:
                command = cmd_path

        self._process = await asyncio.create_subprocess_exec(
            command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )
        logger.info("MCP server started: %s %s (pid=%s)", self._command, self._args, self._process.pid)

        # Drain stderr in background so it doesn't block the pipe.
        async def _drain_stderr() -> None:
            assert self._process and self._process.stderr
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                logger.debug("MCP stderr: %s", line.decode(errors="replace").rstrip())

        self._stderr_task = asyncio.create_task(_drain_stderr())

        # Initialize handshake (with timeout).
        try:
            await asyncio.wait_for(
                self._send_request("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "agent_loop", "version": "0.1.0"},
                }),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"MCP server '{self._id}' did not respond to initialize within 30s"
            )

        # Send initialized notification.
        await self._send_notification("notifications/initialized", {})

        # Discover tools.
        result = await self._send_request("tools/list", {})
        for tool_data in result.get("tools", []):
            spec = ToolSpec(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                parameters=tool_data.get("inputSchema", {"type": "object", "properties": {}}),
                provider_id=self._id,
            )
            self._tools[spec.name] = spec
        logger.info("MCP provider '%s' discovered %d tools", self._id, len(self._tools))

    async def shutdown(self) -> None:
        stderr_task = getattr(self, "_stderr_task", None)
        if stderr_task and not stderr_task.done():
            stderr_task.cancel()
        if self._process:
            # Close pipe transports before terminating to avoid Windows GC warnings.
            # stdin is a StreamWriter (has .close), stdout/stderr are StreamReaders
            # (must close via the underlying transport).
            if self._process.stdin:
                self._process.stdin.close()
            transport = self._process._transport  # type: ignore[attr-defined]
            if transport is not None:
                for fd in (1, 2):  # stdout, stderr
                    pipe_transport = transport.get_pipe_transport(fd)
                    if pipe_transport:
                        pipe_transport.close()
            if self._process.returncode is None:
                try:
                    self._process.terminate()
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    self._process.kill()
            logger.info("MCP server shut down (pid=%s)", self._process.pid)
        self._process = None

    # -- ToolProvider interface ---------------------------------------------

    async def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in MCP provider '{self._id}'")
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        # MCP tool results contain a 'content' array.
        content_parts = result.get("content", [])
        texts = []
        for part in content_parts:
            if part.get("type") == "text":
                texts.append(part["text"])
            else:
                texts.append(json.dumps(part, ensure_ascii=False))
        return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)

    # -- JSON-RPC over stdio ------------------------------------------------

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        self._request_id += 1
        req_id = self._request_id
        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        await self._write(msg)
        return await self._read_response(req_id)

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._write(msg)

    async def _write(self, msg: dict[str, Any]) -> None:
        """Write a JSON-RPC message as a single newline-delimited line (MCP stdio)."""
        assert self._process and self._process.stdin
        body = json.dumps(msg, ensure_ascii=False)
        self._process.stdin.write((body + "\n").encode())
        await self._process.stdin.drain()

    async def _read_response(self, expected_id: int) -> dict[str, Any]:
        """Read JSON-RPC messages until we get the response matching expected_id."""
        assert self._process and self._process.stdout
        async with self._read_lock:
            while True:
                msg = await self._read_message()
                # Skip notifications (no "id" field).
                if "id" not in msg:
                    logger.debug("MCP notification: %s", msg.get("method"))
                    continue
                if msg["id"] == expected_id:
                    if "error" in msg:
                        err = msg["error"]
                        raise RuntimeError(
                            f"MCP error {err.get('code')}: {err.get('message')}"
                        )
                    return msg.get("result", {})

    async def _read_message(self) -> dict[str, Any]:
        """Read a single newline-delimited JSON-RPC message (MCP stdio)."""
        assert self._process and self._process.stdout
        while True:
            line = await self._process.stdout.readline()
            if not line:
                raise ConnectionError("MCP server closed stdout unexpectedly")
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            return json.loads(line_str)

    # -- Factory from config ------------------------------------------------

    @classmethod
    def from_config_entry(cls, entry: dict[str, Any]) -> list[MCPToolProvider]:
        """Create provider(s) from a single config-file entry.

        One entry may declare multiple servers under ``config.mcpServers``.
        Each server key becomes a separate :class:`MCPToolProvider`.
        """
        meta = {
            "name": entry.get("name", ""),
            "description": entry.get("description", ""),
            "organization": entry.get("organization", ""),
            "web": entry.get("web", ""),
            "category": entry.get("category", ""),
        }
        servers: dict[str, Any] = (
            entry.get("config", {}).get("mcpServers", {})
        )
        providers: list[MCPToolProvider] = []
        for server_key, server_cfg in servers.items():
            providers.append(
                cls(
                    command=server_cfg["command"],
                    args=server_cfg.get("args", []),
                    env=server_cfg.get("env"),
                    provider_id=server_key,
                    **meta,
                )
            )
        return providers


def load_mcp_configs(path: str | pathlib.Path) -> list[MCPToolProvider]:
    """Load a JSON config file and return a list of :class:`MCPToolProvider`.

    The file should be a JSON array matching the template documented at the
    top of this module.
    """
    path = pathlib.Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        data = [data]
    providers: list[MCPToolProvider] = []
    for entry in data:
        providers.extend(MCPToolProvider.from_config_entry(entry))
    return providers

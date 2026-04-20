# agent-loop

A minimal, state-machine-driven agent loop framework for building tool-using LLM agents.

## Design Principles

- **Declarative state machine** — transitions are explicit and validated; illegal state changes are rejected at runtime.
- **Hook-based extensibility** — 9 hook points with priority ordering, `prevent_default()` and `stop_propagation()` control signals. Extend behavior without modifying core code.
- **Skill system** — bundle prompt instructions + hook handlers + tool metadata into reusable behavior modules.
- **Provider-agnostic tools** — `ToolProvider` abstraction supports local Python functions, MCP servers, and custom backends through a unified `ToolRouter`.
- **Pluggable LLM backend** — abstract `LLMClient` interface; swap OpenAI for any compatible provider.
- **Zero heavy dependencies** — core has no required dependencies. `openai` is optional.

## Architecture

```
Agent
├── StateMachine        INIT → PLANNING → ACTING → OBSERVING → REFLECTING → FINISHED
│   └── Transition rules, on_enter/on_exit hooks, history tracking
├── Context             Message history, system prompt assembly, sliding-window truncation
├── HookRegistry        Priority-ordered handlers at 9 hook points
├── SkillManager        Register/activate skills by state, collect instructions
├── ToolRouter          Aggregate multiple ToolProviders, deduplicate, route calls
│   ├── LocalToolProvider   Python functions via decorators
│   └── MCPToolProvider     MCP servers via stdio JSON-RPC
└── LLMClient           Abstract chat interface (OpenAI implementation provided)
```

### State Machine

```
                ┌─────────────────────────────────────────┐
                │                                         │
                ▼                                         │
INIT ──→ PLANNING ──→ ACTING ──→ OBSERVING ──→ REFLECTING ┘
              │                                     │
              └──→ FINISHED ◄───────────────────────┘
              
         ERROR ──→ PLANNING / FINISHED
```

### Hook Points

| Phase | Hook | Purpose |
|-------|------|---------|
| Planning | `BEFORE_PLAN` | Modify instructions/context before LLM call |
| Planning | `AFTER_PLAN` | Inspect/modify LLM response |
| Acting | `BEFORE_ACT` | Filter/block tool calls |
| Acting | `AFTER_ACT` | Inspect/modify tool results |
| Observing | `ON_OBSERVE` | Annotate results before context write |
| Reflecting | `ON_REFLECT` | Override continue/stop decision |
| Lifecycle | `ON_START` / `ON_FINISH` / `ON_ERROR` | Run-level events |

## Project Structure

```
agent_loop/
├── core/
│   ├── agent.py           # Main Agent class — orchestrates the loop
│   ├── state_machine.py   # Declarative state machine with hooks
│   ├── context.py         # Message history and prompt assembly
│   ├── hooks.py           # Event + priority hook system
│   └── types.py           # Shared types (State, Message, ToolCall, ToolSpec, ...)
├── llm/
│   ├── base.py            # Abstract LLMClient protocol
│   └── openai.py          # OpenAI-compatible implementation
├── tools/
│   ├── base.py            # ToolProvider / ToolRouter abstractions
│   ├── local.py           # Register Python functions as tools via decorators
│   └── mcp.py             # MCP server tool provider (stdio JSON-RPC)
├── skills/
│   ├── base.py            # Skill dataclass and SkillManager
│   └── trending_assistant/  # Example skill (instructions.md + skill.py)
├── examples/
│   └── simple_agent.py    # Runnable example
└── main.py                # Factory: auto-discover tools/skills/MCP, build Agent
configs/
├── agent.json             # Agent configuration
└── mcp_servers.json       # MCP server definitions
```

## Quick Start

### Installation

```bash
pip install -e ".[openai]"
```

### Minimal Example

```python
import asyncio
from agent_loop import Agent, AgentConfig, LocalToolProvider
from agent_loop.llm.openai import OpenAIClient

# 1. Define tools
local = LocalToolProvider()

@local.tool()
async def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

# 2. Build agent
agent = Agent(
    llm=OpenAIClient(model="gpt-4o"),
    tools=[local],
    config=AgentConfig(system_prompt="You are a helpful assistant."),
)

# 3. Run
answer = asyncio.run(agent.run("What's trending today?"))
print(answer)
```

### Using Skills

```python
from agent_loop import Skill, HookBinding, HookPoint

my_skill = Skill(
    name="analyst",
    instructions="Always cite sources when presenting data.",
    hooks=[
        HookBinding(HookPoint.BEFORE_ACT, my_pre_tool_handler, priority=50),
    ],
)

agent = Agent(llm=llm, tools=[local], skills=[my_skill])
```

### Using the Factory

```python
from agent_loop.main import create_agent

# Auto-discovers tools/, skills/, and MCP configs
agent = await create_agent()
answer = await agent.run("Roll a d20")

# Dry-run mode — dumps the LLM payload without calling the API
agent = await create_agent(dry_run=True)
await agent.run("What time is it?")
```

### Adding a Local Tool

Create a folder under `agent_loop/tools/` with a `tool.py`:

```
agent_loop/tools/my_tool/
    tool.py     # must define `provider = LocalToolProvider(...)`
```

```python
# agent_loop/tools/my_tool/tool.py
from agent_loop.tools.local import LocalToolProvider

provider = LocalToolProvider(provider_id="my_tool")

@provider.tool()
async def my_function(param: str) -> str:
    """Description shown to the LLM."""
    return f"result: {param}"
```

### Adding a Skill

Create a folder under `agent_loop/skills/` with at least `instructions.md`:

```
agent_loop/skills/my_skill/
    instructions.md     # Prompt text injected into system prompt
    skill.py            # Optional: hooks, metadata
```

### Adding an MCP Server

Add an entry to `configs/mcp_servers.json`:

```json
[
    {
        "name": "My Service",
        "description": "...",
        "config": {
            "mcpServers": {
                "my-server": {
                    "command": "npx",
                    "args": ["-y", "my-mcp-package@1.0.0"]
                }
            }
        }
    }
]
```

## Configuration

`configs/agent.json`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | string | `"You are a helpful assistant."` | Base system prompt |
| `model` | string | `"gpt-4o"` | LLM model name |
| `max_iterations` | int | `20` | Max agent loop iterations before forced stop |
| `max_context_messages` | int | `100` | Sliding window for message history |

## Requirements

- Python ≥ 3.11
- `openai` (optional, for `OpenAIClient`)

## License

Apache 2.0

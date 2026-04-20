# Development Notes: verl agent_loop 对比分析与改进计划

> 来源：对 verl 项目 `verl/experimental/agent_loop/` 实现的逐文件对比分析。
> 日期：2026-04-20

---

## 1. verl 实现概要

verl 的 agent loop 是为 **RL 训练 rollout** 设计的，核心职责是：在训练过程中让模型与工具环境交互，收集 trajectory（prompt tokens + response tokens + tool observation tokens），用于 PPO / reward model 训练。

### 关键文件

| 文件 | 职责 |
|------|------|
| `agent_loop.py` | 基类 `AgentLoopBase`、`AgentLoopWorker`（batch 调度）、`AsyncLLMServerManager`（负载均衡）、`GlobalRequestLoadBalancer`、输出后处理（padding / position_ids / attention_mask） |
| `tool_agent_loop.py` | `ToolAgentLoop` — 多轮工具调用的完整状态机，`AgentData` 状态包，`AgentState` 枚举 |
| `single_turn_agent_loop.py` | `SingleTurnAgentLoop` — 单轮生成 |
| `tool_parser.py` | `ToolParser` 注册表 — 从原始 token 输出解析工具调用（hermes / gpt-oss / qwen3_coder 格式） |
| `utils.py` | 配置路径解析、gpt-oss 特殊格式处理 |
| `prometheus_utils.py` | Prometheus 监控配置（Ray 多节点分发） |

### 关键数据结构

- **`AgentLoopOutput`**：`prompt_ids` + `response_ids` + `response_mask` + `response_logprobs` + `routed_experts`
- **`AgentData`**：封装单次 agent loop 执行的所有可变状态（messages, image/video, metrics, turn 计数器, tool_calls, extra_fields）
- **`FunctionCall`**：pydantic model，`name` + `arguments`（JSON string）

---

## 2. verl 做得好的地方（值得借鉴）

### 2.1 ToolParser — 从原始 token 解析工具调用 ⭐⭐⭐

**问题**：我们目前完全依赖 LLM API 返回结构化 `tool_calls` 字段。这对 OpenAI API 没问题，但对本地 / 开源模型（Qwen、Hermes 格式、gpt-oss 等）不可用——这些模型的工具调用是嵌在纯文本输出中的特殊 token 序列。

**verl 的做法**：
```python
class ToolParser(ABC):
    _registry: dict[str, type["ToolParser"]] = {}
    
    @abstractmethod
    async def extract_tool_calls(self, responses_ids, tools) -> tuple[str, list[FunctionCall]]:
        ...
    
    @classmethod
    def register(cls, name: str): ...  # 装饰器注册

# 三种实现：
@ToolParser.register("hermes")    # <tool_call>JSON</tool_call>
@ToolParser.register("gpt-oss")   # <|start|>assistant<|channel|>... 
@ToolParser.register("qwen3_coder")  # <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
```

**建议方案**：在 `LLMClient` 层增加可插拔的 `ToolCallExtractor` 接口。默认实现依赖 API 结构化输出；可切换为 regex 解析器。这让框架同时支持 API 模式和 raw-token 模式。

### 2.2 Tool Response 截断策略 ⭐⭐⭐

**问题**：工具返回内容可能非常长（网页抓取、代码搜索结果等），直接塞入上下文会导致 token 超限或浪费预算。

**verl 的做法**：
```python
if len(tool_response_text) > self.max_tool_response_length:
    if self.tool_response_truncate_side == "left":
        tool_response_text = tool_response_text[:max_len] + "...(truncated)"
    elif self.tool_response_truncate_side == "right":
        tool_response_text = "(truncated)..." + tool_response_text[-max_len:]
    else:  # middle
        half = max_len // 2
        tool_response_text = tool_response_text[:half] + "...(truncated)..." + tool_response_text[-half:]
```

**实现位置**：应在 `_handle_observing()` 中，写入 context 之前执行截断。可作为 `AgentConfig` 的可选参数：
```python
@dataclass
class AgentConfig:
    max_tool_response_length: int | None = None  # None = 不截断
    tool_response_truncate_side: str = "middle"   # "left" | "right" | "middle"
```

### 2.3 并行工具执行上限 ⭐⭐

**问题**：模型可能一次请求调用大量工具，无限并发会给外部服务造成压力。

**verl 的做法**：
```python
for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
    tasks.append(self._call_tool(tool_call, ...))
responses = await asyncio.gather(*tasks)
```

**建议**：在 `AgentConfig` 中增加 `max_parallel_tool_calls: int = 10`，在 `_handle_acting()` 中对 `tool_calls` 切片后再传给 `router.execute_many()`。

### 2.4 AgentData — 显式状态包传递给 Tool ⭐⭐

**问题**：某些工具需要访问对话上下文（比如 SWE-agent 工具需要知道之前的操作历史）。我们当前的 `ToolProvider.call_tool(name, arguments)` 签名不允许传递额外上下文。

**verl 的做法**：`AgentData` 包含 messages、metrics、extra_fields 等，直接传给 `tool.execute(instance_id, tool_args, agent_data=agent_data)`。

**建议**：不需要引入完整的 `AgentData`，但可以：
1. 在 `Context` 上增加 `extra_fields: dict[str, Any]` 扩展字段
2. 在 `ToolRouter.execute()` 中支持可选的 `context` 参数传递

### 2.5 Token 级响应追踪 ⭐⭐

**verl 的做法**：
- `response_mask`：`1` = LLM 生成，`0` = tool response / padding → 用于 RL loss masking
- `response_logprobs`：每个 token 的 log probability

**对通用框架的价值**：
- 成本归因：分清哪些 token 是 LLM 生成的 vs. 工具注入的
- 调试/审计：检查模型在工具调用边界的行为
- 未来 RL fine-tuning 集成

**建议**：作为可选的 Tracing / Telemetry 层，不侵入核心 loop。可通过 hook 实现。

### 2.6 Per-request Tool Filtering ⭐

**verl 的做法**：
```python
tool_selection = extra_info.get("tool_selection")
if tool_selection:
    selected = {name: self.tools[name] for name in tool_selection}
```

**建议**：在 `Agent.run()` 中增加可选参数 `tool_filter: set[str] | None = None`，传入时只向 LLM 展示 filtered 的 tools。

### 2.7 Interaction 支持 ⭐

**verl 的做法**：`INTERACTING` 状态 — 除了 tool response，还支持在循环中注入外部用户/环境输入。

**建议**：未来可考虑让 `REFLECTING` 阶段通过 `ON_REFLECT` hook 注入外部输入，或增加 `run_interactive()` 生成器入口。优先级较低。

### 2.8 多模态支持 ⭐

**verl 的做法**：全链路 image/video 支持 — message 提取 → LLM 传入 → tool 可返回图片 → processor 处理。

**建议**：在 `Message`、`ToolResult` 中预留多模态字段。当前不急，但 type 设计时不要堵死。

---

## 3. verl 的问题（我们应避免的）

| 问题 | 描述 |
|------|------|
| **强耦合训练生态** | 依赖 `ray`, `hydra`, `torch`, `tensordict`, `transformers`, `verl.protocol` 等 ~20 个包，完全无法独立使用 |
| **无 hook / 插件系统** | 任何行为扩展都必须修改 `_handle_*` 方法源码 |
| **无 Skill 抽象** | 所有行为逻辑硬编码，无法组合复用 |
| **状态机无校验** | `while` + `if/elif` 链条，不校验转换合法性，无 history |
| **单体文件过大** | `agent_loop.py` ~900 行，混杂了 load balancer、server manager、worker、post-processing、scoring |
| **RL 逻辑侵入** | padding, attention_mask, position_ids, distillation, reward scoring 等与通用 agent 无关 |
| **动态属性注入** | `agent_data._active_tools` 用下划线前缀的动态属性注入，不在 `__init__` 中声明 |
| **Agent loop 每次 new** | 每个请求都 `hydra.utils.instantiate` 创建新的 agent loop 实例，无复用 |

---

## 4. 实施优先级

| 优先级 | 改进项 | 实现位置 | 复杂度 | 理由 |
|--------|--------|----------|--------|------|
| **P0** | **ChatFormat 协议 — 统一消息/schema/解析/结果格式** | 新建 `core/chat_format.py` + 重构 Agent | 中 | 框架通用性的根基；当前四处硬绑 OpenAI 格式，无法支持开源模型 |
| **P0** | Tool response 截断 | `agent.py` `_handle_observing()` + `AgentConfig` | 低 | 防止上下文爆炸，任何生产使用都需要 |
| **P0** | 并行 tool 执行上限 | `agent.py` `_handle_acting()` + `AgentConfig` | 低 | 防止对外部服务造成 DDoS |
| **P1** | Context extra_fields | `context.py` | 低 | 让 tool 可以访问/存储会话级状态 |
| **P2** | Token 级追踪 | 新建 `telemetry.py` 或通过 hook | 中 | 可观测性，未来 RL 集成基础 |
| **P2** | Per-request tool filtering | `agent.py` `run()` 参数 | 低 | 动态控制可用工具 |
| **P3** | Interactive / multi-turn | `agent.py` 新方法或 hook | 中 | 环境交互、多用户轮次 |
| **P3** | 多模态 Message | `types.py` 字段预留 | 低 | 未来 VLM agent 的基础 |

---

## 5. P0 实现草案

### 5.1 Tool Response 截断

```python
# core/types.py — AgentConfig 新增字段
@dataclass
class AgentConfig:
    max_tool_response_length: int | None = None
    tool_response_truncate_side: str = "middle"  # "left" | "right" | "middle"

# core/agent.py — _handle_observing() 中截断
def _truncate_tool_response(self, content: str) -> str:
    max_len = self.config.max_tool_response_length
    if max_len is None or len(content) <= max_len:
        return content
    side = self.config.tool_response_truncate_side
    if side == "left":
        return content[:max_len] + "...(truncated)"
    elif side == "right":
        return "(truncated)..." + content[-max_len:]
    else:
        half = max_len // 2
        return content[:half] + "...(truncated)..." + content[-half:]
```

### 5.2 并行 Tool 执行上限

```python
# core/types.py — AgentConfig 新增字段
@dataclass
class AgentConfig:
    max_parallel_tool_calls: int = 10

# core/agent.py — _handle_acting() 中切片
tool_calls = tool_calls[:self.config.max_parallel_tool_calls]
results = await self.router.execute_many(tool_calls)
```

### 5.3 ChatFormat 协议 — 统一消息/schema/解析/结果格式

#### 5.3.1 问题：四处硬绑 OpenAI 格式

当前代码中有四个耦合点把框架锁死在 OpenAI API 格式上：

| 位置 | 现状 | 问题 |
|------|------|------|
| `Message.to_openai_dict()` | 直接输出 OpenAI message 格式 | Anthropic、本地模型的 message 格式不同 |
| `ToolSpec.to_openai_schema()` | 只有 OpenAI function calling schema | Hermes 用 JSON-in-XML，Qwen3 用自定义 XML |
| `Context.to_openai_messages()` | 写死了序列化方式 | 不同模型的 system prompt 处理方式不同 |
| `Agent._handle_planning()` | 假设 `response.message.tool_calls` 已结构化 | 开源模型在纯文本中嵌入 tool call token |

verl 的 `ToolParser` 只解决了第四个问题（输出解析），其余三个散落各处。

#### 5.3.2 方案：ChatFormat 策略对象

把一种 LLM 格式的**所有**格式约定收拢到一个可替换的策略对象中：

```python
# core/chat_format.py

class ChatFormat(ABC):
    """定义一种 LLM 对话格式的完整协议。"""
    
    @abstractmethod
    def serialize_messages(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Any:
        """把内部 Message 列表序列化成 LLM 需要的输入格式。
        
        OpenAI    → list[dict]  (标准 message dicts)
        本地模型  → apply_chat_template 后的 token ids 或 prompt string
        Anthropic → Anthropic 格式的 message list
        """
    
    @abstractmethod
    def serialize_tool_schema(self, spec: ToolSpec) -> Any:
        """把 ToolSpec 转成该格式的 tool 描述。
        
        OpenAI → {"type": "function", "function": {...}}
        Hermes → 嵌入 system prompt 的文本描述
        """
    
    @abstractmethod
    def extract_tool_calls(self, response: LLMResponse) -> list[ToolCall]:
        """从 LLM 响应中提取工具调用。
        
        OpenAI → 读 response.message.tool_calls (结构化字段)
        Hermes → regex 解析 <tool_call>JSON</tool_call>
        Qwen3  → XML 解析 <function=name><parameter=key>value</parameter></function>
        """
    
    @abstractmethod
    def format_tool_result(self, result: ToolResult) -> Message:
        """把工具执行结果格式化为对话消息。
        
        OpenAI → Message(role=TOOL, tool_call_id=..., content=...)
        Hermes → Message(role=USER, content="<tool_response>...</tool_response>")
        gpt-oss → 特殊 token 手动拼接
        """
```

提供默认实现：

```python
class OpenAIChatFormat(ChatFormat):
    """默认实现 — 等价于当前硬编码的行为，零行为变更。"""
    
    def serialize_messages(self, messages, tools=None):
        result = []
        for msg in messages:
            result.append(msg.to_openai_dict())
        return result
    
    def serialize_tool_schema(self, spec):
        return spec.to_openai_schema()
    
    def extract_tool_calls(self, response):
        return response.message.tool_calls or []
    
    def format_tool_result(self, result):
        return Message(
            role=Role.TOOL,
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
        )


class HermesChatFormat(ChatFormat):
    """Hermes 格式 — <tool_call>JSON</tool_call>，对应 verl 的 HermesToolParser。"""
    ...

class Qwen3ChatFormat(ChatFormat):
    """Qwen3 格式 — <tool_call><function=...>...</function></tool_call>。"""
    ...
```

#### 5.3.3 Agent 中的使用

```python
class Agent:
    def __init__(self, llm, tools, chat_format: ChatFormat | None = None, ...):
        self.chat_format = chat_format or OpenAIChatFormat()
    
    async def _handle_planning(self):
        # 序列化用 chat_format，不再硬写 to_openai_messages()
        messages = self.chat_format.serialize_messages(
            self.context.messages_with_system_prompt(),
            self.router.tool_specs,
        )
        response = await self.llm.chat(messages=messages, ...)
        
        # 解析 tool calls 用 chat_format，不再假设结构化输出
        tool_calls = self.chat_format.extract_tool_calls(response)
    
    async def _handle_observing(self):
        for r in results:
            # 格式化 tool result 用 chat_format
            msg = self.chat_format.format_tool_result(r)
            self.context.add_message(msg)
```

#### 5.3.4 与 verl ToolParser 的对比

| 维度 | verl `ToolParser` | 我们的 `ChatFormat` |
|------|-------------------|---------------------|
| 覆盖范围 | 只管输出解析（extract_tool_calls） | 消息序列化 + schema 格式 + 输出解析 + 结果格式，完整闭环 |
| 内聚性 | 解析逻辑独立，其余散落各处 | 同一种格式的四个维度在一个类中定义 |
| Message 格式 | 硬编码 OpenAI dict | `serialize_messages()` 可适配任意格式 |
| Tool schema | 硬传 `list[dict]` | `serialize_tool_schema()` 按格式转换 |
| 注册机制 | `ToolParser._registry` 类级注册表 | 直接传入 Agent 构造函数，简单直接 |

#### 5.3.5 迁移策略

1. 新建 `core/chat_format.py`，实现 `ChatFormat` ABC + `OpenAIChatFormat`
2. `Agent.__init__` 增加 `chat_format` 参数，默认 `OpenAIChatFormat()`
3. `_handle_planning` / `_handle_observing` 改用 `self.chat_format.*`
4. `Message.to_openai_dict()` 和 `ToolSpec.to_openai_schema()` 保留但标记为 convenience method
5. `Context.to_openai_messages()` 改名为 `Context.messages_with_system_prompt()`，返回 `list[Message]` 而非 `list[dict]`
6. **现有代码行为完全不变**（`OpenAIChatFormat` 复现当前逻辑）

---

## 6. 架构思考：如何干净地支持 RL 训练

### 6.1 问题

verl 的训练相关逻辑（padding, response_mask, position_ids, attention_mask, reward scoring, distillation）非常有价值，
但它们全部硬编码在 `agent_loop.py` 的 `_agent_loop_postprocess` 中（200+ 行），与 agent 执行逻辑强耦合。
这导致 agent loop 无法脱离 `ray`, `torch`, `transformers` 等训练生态独立使用。

### 6.2 解耦方案：三层架构

```
┌─────────────────────────────────────────────┐
│  Layer 1: Agent Loop (核心，零训练依赖)        │
│  职责：状态机驱动、LLM 调用、工具执行           │
│  产出：Trajectory (结构化执行轨迹)             │
└──────────────────┬──────────────────────────┘
                   │  Trajectory
                   ▼
┌─────────────────────────────────────────────┐
│  Layer 2: TrajectoryCollector (Hook 插件)     │
│  职责：通过 hook 旁路采集 token 级训练信号      │
│  采集：logprobs, token 边界, tool/llm 标记    │
│  实现：一个 Skill，挂载到 AFTER_PLAN 等 hook   │
└──────────────────┬──────────────────────────┘
                   │  Trajectory + Signals
                   ▼
┌─────────────────────────────────────────────┐
│  Layer 3: TrainingAdapter (独立模块)           │
│  职责：padding, mask, position_ids, reward    │
│  产出：Training-ready tensor (TensorDict等)   │
│  位置：contrib/verl_adapter.py 或 verl 侧     │
└─────────────────────────────────────────────┘
```

### 6.3 Layer 1 — Agent Loop 产出 Trajectory

Agent loop 只诚实记录"发生了什么"，不关心训练 tensor：

```python
@dataclass
class TrajectoryStep:
    role: str                    # "llm" | "tool" | "user"
    token_ids: list[int] | None  # 可选，raw token 模式才有
    content: str
    logprobs: list[float] | None # 可选，LLM 返回的
    metadata: dict[str, Any]     # tool_name, call_id, timing, etc.

@dataclass
class Trajectory:
    prompt: list[Message]
    steps: list[TrajectoryStep]
    usage: dict[str, int]
    metrics: dict[str, Any]
```

### 6.4 Layer 2 — 通过 Hook 旁路采集训练信号

训练信号采集作为一个独立的 Skill/Plugin，通过 hook 系统挂载，不修改 agent loop 核心代码：

```python
class TrainingSignalCollector:
    """通过 hook 旁路采集 RL 训练所需的 token 级信号。
    
    挂载方式：作为 Skill 注册到 Agent，或直接在 HookRegistry 上注册。
    不挂载时 agent loop 完全不受影响。
    """
    
    hooks = [
        # AFTER_PLAN: LLM 刚返回，采集 logprobs 和 token_ids
        HookBinding(HookPoint.AFTER_PLAN, collect_llm_output, priority=90),
        
        # ON_OBSERVE: tool 结果写入 context 时，标记 tool token 边界
        HookBinding(HookPoint.ON_OBSERVE, mark_tool_boundaries, priority=90),
        
        # ON_FINISH: 整个 run 结束，组装完整 trajectory
        HookBinding(HookPoint.ON_FINISH, finalize_trajectory, priority=90),
    ]
```

这样：
- **通用用户**不挂载这个 collector，agent loop 无训练开销
- **训练团队**挂载后，hook 自动采集所有必要信号
- agent loop 代码**零侵入**

### 6.5 Layer 3 — TrainingAdapter 独立做 tensor 加工

这是纯粹的数据转换层，不在 agent_loop 包内：

```python
class TrainingAdapter:
    """把 Trajectory 转成训练所需的 padded tensor batch。
    
    对应 verl 的 _agent_loop_postprocess 逻辑，但完全解耦。
    可放在 agent_loop/contrib/verl_adapter.py 或 verl 侧。
    """
    
    def to_training_batch(self, trajectory: Trajectory, tokenizer, config) -> dict:
        # 1. 根据 step.role 构建 response_mask (1=llm, 0=tool)
        # 2. left-pad prompt_ids, right-pad response_ids
        # 3. 拼接 attention_mask, 计算 position_ids
        # 4. 处理多模态 rope (如有)
        # 5. 可选：调用 reward model 计算 rm_scores
        return {
            "prompt_ids": ...,
            "response_ids": ...,
            "response_mask": ...,
            "attention_mask": ...,
            "position_ids": ...,
            "logprobs": ...,
        }
```

### 6.6 对比：verl 现状 vs. 我们的方案

| 维度 | verl 现状 | 三层方案 |
|------|-----------|----------|
| 训练逻辑位置 | `_agent_loop_postprocess` 内联 200+ 行 | 独立 `TrainingAdapter` 模块 |
| logprobs 采集 | `agent_data.response_logprobs` 硬编码 | `AFTER_PLAN` hook 旁路采集 |
| response_mask 构建 | agent loop 内维护 `[1]*n + [0]*m` | `ON_OBSERVE` hook 标记边界，adapter 构建 mask |
| reward scoring | `_compute_score` 内联 | 独立 reward module 消费 Trajectory |
| 依赖 | agent loop 必须依赖 torch/ray | agent loop 零训练依赖，adapter 层才引入 torch |
| 不需要训练时 | 仍然承受所有训练代码的依赖和开销 | 不挂载 collector = 零开销 |

### 6.7 核心洞察

> **我们的 hook 系统恰好是解决这个耦合问题的钥匙。**
> verl 没有 hook，所以它只能把训练逻辑塞进 agent loop。
> 我们有 hook，就可以让训练逻辑作为"观察者"挂在旁路上，agent loop 本身保持干净。

这也验证了 hook 系统的设计价值——它不只是给 "插件开发者" 用的 fancy feature，
而是在架构层面实现关注点分离的核心机制。

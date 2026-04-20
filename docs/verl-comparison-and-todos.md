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

### 2.2 Tool Response 截断策略 ⭐⭐⭐ ✅ 已实现

**问题**：工具返回内容可能非常长（网页抓取、代码搜索结果等），直接塞入上下文会导致 token 超限或浪费预算。

**实现**：在 `_handle_observing()` 中，ON_OBSERVE hook 之后、写入 context 之前，作为最终保底截断。通过 `AgentConfig` 配置：

```python
AgentConfig(
    max_tool_response_length=4000,        # None = 不截断
    tool_response_truncate_side="middle",  # "left" | "right" | "middle"
)
```

截断策略支持三种模式：
- `left`：保留前 N 字符 + `...(truncated)`
- `right`：`(truncated)...` + 保留后 N 字符
- `middle`：保留前后各半 + `...(truncated)...`

### 2.3 并行工具执行上限 ⭐⭐ ✅ 已实现

**问题**：模型可能一次请求调用大量工具，无限并发会给外部服务造成压力。

**实现**：采用 `asyncio.Semaphore` 资源池模式（而非 verl 的简单 slice 丢弃），所有 tool call 都会执行，但同时并发不超过 N 个，超出的排队等待：

```python
# AgentConfig
AgentConfig(max_parallel_tool_calls=5)  # None = 无限制

# ToolRouter.execute_many() 内部
semaphore = asyncio.Semaphore(max_concurrency)
async def _limited(call):
    async with semaphore:
        return await self.execute(call)
return list(await asyncio.gather(*(_limited(c) for c in calls)))
```

与 verl 的关键差异：verl 用 slice 直接丢弃超出的 tool call，我们用 semaphore 让所有调用都执行，只是控制并发度。

### 2.4 AgentData — 显式状态包传递给 Tool ⭐⭐ ✅ 部分实现

**问题**：某些工具需要访问对话上下文（比如 SWE-agent 工具需要知道之前的操作历史）。我们当前的 `ToolProvider.call_tool(name, arguments)` 签名不允许传递额外上下文。

**已实现**：`Context.extra: dict[str, Any]` 扩展字段，`clear()` 时自动清空。hooks、tools、skills 可通过它传递任意上下文。

**待做**：在 `ToolRouter.execute()` 中支持可选的 `context` 参数传递，让工具能直接访问 `context.extra`。

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

| 优先级 | 改进项 | 实现位置 | 复杂度 | 状态 |
|--------|--------|----------|--------|------|
| **P0** | ~~ChatFormat 协议 — 统一消息/schema/解析/结果格式~~ | `core/chat_format.py` + Agent/LLMClient/Context 重构 | 中 | ✅ 已完成 |
| **P0** | ~~Tool response 截断~~ | `agent.py` `_handle_observing()` + `AgentConfig` | 低 | ✅ 已完成 |
| **P0** | ~~并行 tool 执行上限~~ | `agent.py` `_handle_acting()` + `ToolRouter` + `AgentConfig` | 低 | ✅ 已完成 |
| **P1** | ~~Context extra_fields~~ | `context.py` | 低 | ✅ 已完成 |
| **P2** | Token 级追踪 | 新建 `telemetry.py` 或通过 hook | 中 | 待实现 |
| **P2** | Per-request tool filtering | `agent.py` `run()` 参数 | 低 | 待实现 |
| **P3** | Interactive / multi-turn | `agent.py` 新方法或 hook | 中 | 待实现 |
| **P3** | 多模态 Message | `types.py` 字段预留 | 低 | 待实现 |

---

## 5. P0/P1 实现记录

### 5.1 Tool Response 截断 — ✅ 已实现

**位置**：`core/agent.py` `_truncate_tool_response()` + `_handle_observing()`

关键设计决策：截断放在 ON_OBSERVE hook **之后**、写入 context **之前**，作为最终保底。
这样 hook 可以做自定义处理（比如摘要、结构化提取），截断只防止极端情况。

```python
# AgentConfig 新增字段
max_tool_response_length: int | None = None  # None = 不截断
tool_response_truncate_side: str = "middle"  # "left" | "right" | "middle"

# _handle_observing() 中，对每个 result 截断后再写入 context
r = ToolResult(
    tool_call_id=r.tool_call_id, name=r.name,
    content=self._truncate_tool_response(r.content),
    is_error=r.is_error,
)
```

### 5.2 并行 Tool 执行上限 — ✅ 已实现

**位置**：`tools/base.py` `ToolRouter.execute_many()` + `core/agent.py` `_handle_acting()`

关键设计决策：使用 `asyncio.Semaphore` 资源池模式，而非 verl 的 slice 丢弃。
所有 tool call 都会执行，只是控制并发度，超出的排队等待。

```python
# AgentConfig 新增字段
max_parallel_tool_calls: int | None = None  # None = 无限制

# ToolRouter.execute_many(calls, max_concurrency=N)
semaphore = asyncio.Semaphore(max_concurrency)
async def _limited(call):
    async with semaphore:
        return await self.execute(call)
return list(await asyncio.gather(*(_limited(c) for c in calls)))
```

### 5.3 Context extra_fields — ✅ 已实现

**位置**：`core/context.py`

`Context.extra: dict[str, Any]`，`clear()` 时自动清空。
hooks、tools、skills 可通过它传递任意上下文数据。

### 5.4 ChatFormat 协议 — ✅ 已实现

> 详见 `docs/chat-format.md` 和 `core/chat_format.py`。
>
> 最终方案与原始草案的关键差异：**ChatFormat 由 LLMClient 持有**，而非 Agent 持有。
> `LLMClient.chat()` 签名改为接受 `list[Message]` 内部类型，Agent 完全不接触格式逻辑。

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

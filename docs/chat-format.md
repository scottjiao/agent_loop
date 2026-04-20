# ChatFormat 协议 — 实现记录

> 日期：2026-04-20

## 概述

引入 `ChatFormat` 抽象协议，将所有 LLM 格式相关的知识（消息序列化、tool schema 格式、tool call 解析、tool result 表示）收拢到一个可替换的策略对象中。该对象由 `LLMClient` 持有，Agent 核心循环完全不接触格式逻辑。

## 动机

之前框架在五个位置硬编码了 OpenAI 格式：

| # | 位置 | 硬编码内容 |
|---|------|-----------|
| 1 | `Message.to_openai_dict()` | 消息序列化为 OpenAI dict |
| 2 | `ToolSpec.to_openai_schema()` | tool 描述只有 function calling 格式 |
| 3 | `Context.to_openai_messages()` | system prompt 拼装 + 序列化 → `list[dict]` |
| 4 | `LLMClient.chat(messages: list[dict])` | 接口本身要求 pre-serialized dicts |
| 5 | `OpenAIClient.chat()` 内部 | JSON 解析 tool_calls、usage 提取 |

这使得框架无法支持 Hermes（`<tool_call>JSON</tool_call>`）、Qwen3（XML function call）、gpt-oss 等开源模型格式，也无法实现训推一致（训练时 `apply_chat_template` 与推理时 API 调用走不同路径）。

## 核心设计决策

### ChatFormat 归 LLMClient 持有，而非 Agent

原始草案（见 `verl-comparison-and-todos.md` 5.3）将 ChatFormat 作为 Agent 的参数。最终方案改为 LLMClient 持有，原因：

1. **消除配对风险** — ChatFormat 是模型级属性，与 LLMClient 天然配对。如果 Agent 同时持有 `llm` 和 `chat_format`，用户必须手动确保二者匹配（OpenAI LLM + Hermes ChatFormat = 运行时爆炸）。
2. **Agent 更干净** — Agent 只操作内部类型 (`Message`, `ToolSpec`, `ToolCall`)，不知道也不关心序列化细节。
3. **单一真相源** — `llm.chat_format` 就是获取格式信息的唯一入口。

### LLMClient.chat() 签名改为接受 list[Message]

```python
# Before
async def chat(self, messages: list[dict[str, Any]], ...) -> LLMResponse

# After
async def chat(self, messages: list[Message], ...) -> LLMResponse
```

这使得整个 Agent 内部完全以 `Message` 对象流转，序列化推迟到 LLMClient 内部（通过 `chat_format.serialize_messages()`）。

## ChatFormat ABC

```python
class ChatFormat(ABC):
    # 5 个抽象方法
    serialize_messages(messages, tools) -> Any       # Message → LLM 输入格式
    extract_tool_calls(raw_response) -> list[ToolCall]  # 原始响应 → 结构化 tool calls
    extract_content(raw_response) -> str | None      # 原始响应 → 文本内容
    extract_finish_reason(raw_response) -> str       # 原始响应 → finish reason
    extract_usage(raw_response) -> dict[str, int]    # 原始响应 → token 用量

    # 2 个可覆盖方法（有默认实现）
    format_tool_result(result) -> Message            # ToolResult → 对话消息
    to_training_messages(messages, tools) -> Any     # 训练时格式（默认委托 serialize_messages）
```

### format_tool_result 为什么在 ChatFormat 里

不同格式对 tool result 的表示完全不同：

| 格式 | tool result 表示 |
|------|----------------|
| OpenAI | `{"role": "tool", "tool_call_id": "xxx", "content": "..."}` |
| Hermes | `{"role": "user", "content": "<tool_response>...</tool_response>"}` |
| Qwen3 | `{"role": "tool", "content": "..."}` (无 tool_call_id) |

### to_training_messages 的训推一致路径

```
训练路径                           推理路径
────────                           ────────
list[Message] + list[ToolSpec]     list[Message] + list[ToolSpec]
        │                                  │
        ▼                                  ▼
chat_format.to_training_messages() chat_format.serialize_messages()
        │                                  │
        ▼                                  ▼
tokenizer.apply_chat_template()    API call / local inference
```

两条路径从同一个内部表示出发，经过同一个 ChatFormat 实例。只要这两个方法对同一组 messages 产出语义等价的格式，就保证训推一致。

## 改动文件清单

### 新建

| 文件 | 内容 |
|------|------|
| `core/chat_format.py` | `ChatFormat` ABC + `OpenAIChatFormat` 默认实现 |

### 修改

| 文件 | 改动 |
|------|------|
| `llm/base.py` | `LLMClient.chat()` 签名 `list[dict]` → `list[Message]`；声明 `chat_format: ChatFormat` 属性 |
| `llm/openai.py` | 构造函数新增可选 `chat_format` 参数（默认 `OpenAIChatFormat`）；`chat()` 内部通过 `self.chat_format` 做序列化和解析，移除硬编码的 JSON 解析 |
| `core/context.py` | 新增 `messages_with_system_prompt() -> list[Message]`（系统提示 + 滑动窗口截断）；`to_openai_messages()` 降级为兼容 convenience method |
| `core/agent.py` | `_handle_planning`: 用 `messages_with_system_prompt()` + `llm.chat(list[Message])`；dry_run 通过 `llm.chat_format` 序列化 |
| `core/agent.py` | `_handle_observing`: 用 `llm.chat_format.format_tool_result()` 替代 `context.add_tool_result()` |
| `__init__.py` | 导出 `ChatFormat`, `OpenAIChatFormat` |

### 保留未改

| 文件/方法 | 说明 |
|-----------|------|
| `Message.to_openai_dict()` | 保留为 convenience method，不再被核心 loop 调用 |
| `ToolSpec.to_openai_schema()` | 保留为 convenience method，仍被 `OpenAIClient` 的 `tools` 参数序列化使用 |
| `Context.to_openai_messages()` | 保留为兼容 convenience，内部委托新方法 |

## 行为变化

**零。** `OpenAIChatFormat` 精确复现了之前硬编码的逻辑。所有现有代码（examples、dry_run、MCP 集成）行为不变。

## 扩展示例

支持新模型格式只需实现 `ChatFormat` 子类：

```python
class HermesChatFormat(ChatFormat):
    def serialize_messages(self, messages, tools=None):
        # 把 tool schemas 嵌入 system prompt 的 <tools> 块
        # 把其余消息转为 Hermes chat template 格式
        ...

    def extract_tool_calls(self, raw_response):
        # regex 解析 <tool_call>{"name": ..., "arguments": ...}</tool_call>
        ...

    def format_tool_result(self, result):
        # Hermes: tool result 是 user role + <tool_response> 包裹
        return Message(role=Role.USER,
                       content=f"<tool_response>{result.content}</tool_response>")

# 使用
agent = Agent(
    llm=OpenAIClient(model="hermes-3", chat_format=HermesChatFormat()),
    tools=[local],
)
```

## 与 verl ToolParser 的对比

| 维度 | verl `ToolParser` | 我们的 `ChatFormat` |
|------|-------------------|---------------------|
| 覆盖范围 | 只管输出解析 | 消息序列化 + schema + 输出解析 + 结果格式，完整闭环 |
| 归属 | 独立对象 + 注册表 | LLMClient 持有，无配对风险 |
| Agent 感知？ | 是（agent loop 调用 parser） | 否（Agent 只操作内部类型） |
| 训推一致 | 未覆盖 | `to_training_messages()` 提供训练路径 |

# Interaction (多轮对话) 设计方案

> 日期：2026-04-20  
> 状态：设计调研，暂不实现

---

## 1. 需求

让 agent loop 支持多轮交互：LLM 回复纯文本（无 tool_calls）时，不直接 FINISHED，而是把回复交给用户，等用户回复后继续循环。

---

## 2. verl 的做法

verl 在状态机里加了 `INTERACTING` 状态——当 LLM 生成结果中没检测到工具调用时，进入该状态等待外部输入。

**我们不采用这个方案**，理由：
- Interaction 本质是 "输入来源"，不是 "处理阶段"。状态机描述的是 agent 的处理流程（想→做→看→反思），每个状态对应一段计算逻辑。"等待用户输入" 是 I/O 阻塞，语义不一致。
- verl 没有 hook 系统，只能往状态机硬塞。我们有 hook，不需要这样做。

---

## 3. 方案比选

### 3.1 纯 async generator

`run_interactive()` 作为 async generator，yield 出 LLM 回复，调用者 send() 用户消息。

- **优点**：调用者体验直觉，状态机不改
- **缺点**：yield/send 发生在 loop 外部，hook 系统看不到 interaction 过程。训练轨迹采集器（通过 hook 挂载）会漏掉 interaction 轮次

### 3.2 纯 hook 拦截

在 ON_REFLECT hook 中 await 一个外部 Future/Queue 来等待用户输入，所有事件都在 loop 内发生。

- **优点**：hook 全程可观测，训练轨迹完整
- **缺点**：调用者需要搞 Queue/Future，体验不直观

### 3.3 generator + hook emit（推荐）

两者结合——对外是 generator API，对内在 yield 前后 emit hook：

```python
async def run_interactive(self, user_message: str):
    # ... 正常 loop ...
    
    while ...:
        await self._step()
        
        if self._needs_user_input():
            # yield 之前：通知 hook 系统
            await self.hooks.emit(HookPoint.ON_INTERACT, 
                agent=self, 
                assistant_message=self.context.last_assistant_message)
            
            user_reply = yield self.context.last_assistant_message.content
            
            # yield 之后：用户回复到了，写入 context，通知 hook
            self.context.add_user(user_reply)
            await self.hooks.emit(HookPoint.ON_USER_INPUT,
                agent=self,
                user_message=user_reply)
            
            await self.sm.transition(State.PLANNING)
```

### 方案对比

| 维度 | 纯 generator | 纯 hook 拦截 | generator + hook emit |
|------|-------------|-------------|----------------------|
| 调用者体验 | 直觉 | 需要搞 Queue/Future | 直觉 |
| hook 可观测性 | 有盲区 | 完整 | 完整 |
| 训练轨迹采集 | 漏 interaction 轮 | 完整 | 完整 |
| 状态机改动 | 不改 | 不改 | 不改 |
| 复杂度 | 低 | 中 | 低-中 |

---

## 4. 关键设计决策

### 4.1 状态机不改

不加 `INTERACTING` 状态。interaction 的本质是 REFLECTING 阶段决定 "需要用户补充信息"，然后回到 PLANNING。`REFLECTING → PLANNING` 这条边已经存在。

需要加的是 hook point（`ON_INTERACT` / `ON_USER_INPUT`），而非 state。Hook point 是轻量可选的；state 是结构性全局的。

### 4.2 不用 `transfer_to_user` 工具

让 LLM "意识到" 有个工具叫 `transfer_to_user` 的问题：
- 浪费 tool schema token
- LLM 可能在不该调的时候调（幻觉），也可能该调的时候不调（遗忘）
- "要不要跟用户说话" 不应该依赖 LLM 的工具调用能力

更自然的信号：**LLM 回复纯文本、无 tool_calls = 它认为该轮到用户了**。

### 4.3 判断 "需要用户输入" vs. "真的结束了"

在 interactive 模式下需要区分。可选信号：
- 调用者显式 close generator
- finish_reason 差异（`stop` vs `end_turn`）
- 对话超时
- 调用者决定：yield 出去后，send 新消息 = 继续，不 send = 结束

### 4.4 `run()` 不动

`run()` 保持单轮语义不变。`run_interactive()` 是新增入口。

---

## 5. 训练兼容性

推荐方案（generator + hook emit）对训练最友好：

- `ON_INTERACT`：TrainingSignalCollector 记录 "LLM 在这里停下来等用户"，标记轨迹断点
- `ON_USER_INPUT`：记录用户输入，token mask=0

`Trajectory` 里每一步都有记录，response_mask 构建不缺信息。采集器只需挂这两个 hook，不需要改 agent loop 核心代码。

---

## 6. 实现清单（未来）

1. 在 `HookPoint` 中新增 `ON_INTERACT` 和 `ON_USER_INPUT`
2. 在 `Agent` 中新增 `run_interactive()` async generator 方法
3. 实现 `_needs_user_input()` 判断逻辑
4. `run()` 不动
5. 状态机不动

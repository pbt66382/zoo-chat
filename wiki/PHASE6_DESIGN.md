# Phase 6 设计文档：SSE 流式输出 + React TypeScript 前端

## 目标

将原有的同步问答接口（`/api/chat`）升级为服务端推送（SSE）流式接口，并配套构建全新的 React TypeScript 前端，实现逐 token 流式展示、Markdown 渲染、产品线识别等特性。

---

## 架构变化

```
Phase 5（同步）           Phase 6（流式）
─────────────────         ────────────────────────────────
POST /api/chat            POST /api/chat/stream
  │                         │
  ├─ Pipeline.run()          ├─ Pre-pipeline（5 steps）
  │                         │    └─ ProductDetection → Intent → Router
  └─ 等待全部生成完毕           │         → SlotFilling → Retrieval
       └─ 返回 JSON           │
                            ├─ GenerationStep.stream()   ← astream()
                            │    └─ 每个 token yield chunk 事件
                            └─ 完成后 yield done 事件（含元数据）
```

### Pipeline 拆分策略

流式输出的核心挑战：LangChain `astream()` 只能在生成阶段使用，而意图识别、槽位填充、检索等步骤需要先于生成完成。

解决方案：将 Pipeline 拆为两段：

| 段 | 步骤 | 执行方式 |
|----|------|----------|
| Pre-pipeline | ProductDetection, Intent, Router, SlotFilling, Retrieval | `await pipeline.run()` 阻塞完成 |
| 生成阶段 | GenerationStep | `async for chunk in gen_step.stream(ctx)` |

短路场景（Router 自动回复、槽位填充追问）由 `ctx.answer` 判断：若已有答案则直接将其作为单个 chunk yield，跳过 GenerationStep。

---

## 后端实现

### SSE 事件协议

```
data: {"type": "chunk", "content": "<增量文本>"}
data: {"type": "done",  "session_id": "...", "product_id": "...", "product_name": "...",
       "intent_id": "...", "needs_followup": false, "latency_ms": 1234.5}
data: {"type": "error", "message": "<错误描述>"}
```

每行以 `data: ` 开头，以 `\n\n` 结尾（标准 SSE 格式）。

### chat_service.py — stream()

```python
async def stream(self, question, session_id=None, ...) -> AsyncIterator[str]:
    # 1. 构建 ChatContext
    ctx = ChatContext(question=question, session_id=session_id or str(uuid4()))

    # 2. 运行 Pre-pipeline（5 步，不含生成）
    pre_pipeline = ChatPipeline(steps=[
        ProductDetectionStep(), IntentStep(...), RouterStep(),
        SlotFillingStep(), RetrievalStep(),
    ])
    ctx = await pre_pipeline.run(ctx)

    # 3. 生成阶段
    accumulated = ""
    if ctx.answer:
        # 短路：直接 yield 已有答案
        yield json.dumps({"type": "chunk", "content": ctx.answer})
        accumulated = ctx.answer
    else:
        gen_step = GenerationStep()
        async for chunk in gen_step.stream(ctx):
            accumulated += chunk
            yield json.dumps({"type": "chunk", "content": chunk})

    # 4. 更新会话历史
    ctx.answer = accumulated
    self._session_store.update(ctx)

    # 5. yield done 元数据
    yield json.dumps({
        "type": "done",
        "session_id": ctx.session_id,
        "product_id": ctx.product_id,
        "product_name": ctx.product_name,
        "intent_id": ctx.intent_id,
        "needs_followup": ctx.needs_followup,
        "latency_ms": (time.time() - start) * 1000,
    })
```

### chat.py — /api/chat/stream

```python
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        try:
            async for data in get_chat_service().stream(...):
                yield f"data: {data}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # 禁止 Nginx 缓冲
            "Connection": "keep-alive",
        },
    )
```

---

## 前端实现

### 技术选型

| 工具 | 版本 | 用途 |
|------|------|------|
| React | 18 | UI 框架 |
| TypeScript | 5 | 类型安全 |
| Vite | 5 | 构建工具 + 开发代理 |
| react-markdown | ^9 | Markdown 渲染 |
| rehype-highlight | ^7 | 代码语法高亮 |
| highlight.js | ^11 | 高亮样式（github-dark） |

### 组件结构

```
App.tsx
├── Header（标题 + ProductBadge）
├── ChatArea
│   └── ChatMessage × N
│       ├── 用户消息：plain text
│       └── Bot 消息：ReactMarkdown + 流式游标 / loading dots
├── ChatInput
│   ├── 快捷提问 chips × 5
│   └── Textarea + 发送按钮
└── StatusBar（阶段标签 + 延迟 / 生成状态 + 技术栈）
```

### SSE 消费循环（App.tsx）

```typescript
const res = await fetch('/api/chat/stream', { method: 'POST', ... })
const reader = res.body!.getReader()
const decoder = new TextDecoder()
let buf = ''

while (true) {
  const { done, value } = await reader.read()
  if (done) break
  buf += decoder.decode(value, { stream: true })
  const lines = buf.split('\n')
  buf = lines.pop() ?? ''          // 保留不完整的最后一行

  for (const line of lines) {
    if (!line.startsWith('data: ')) continue
    const event: SSEEvent = JSON.parse(line.slice(6))

    if (event.type === 'chunk') {
      // 追加到 bot 消息内容（状态更新 → React re-render → 逐字显示）
      setMessages(prev => prev.map(m =>
        m.id === botId ? { ...m, content: m.content + event.content } : m
      ))
    } else if (event.type === 'done') {
      setMeta({ sessionId: event.session_id, productId: event.product_id, ... })
      setLatency(event.latency_ms)
      // 关闭流式游标
      setMessages(prev => prev.map(m =>
        m.id === botId ? { ...m, isStreaming: false } : m
      ))
    }
  }
}
```

关键设计点：
- `buf` 保留跨 chunk 的不完整行，避免截断 JSON 解析出错
- `isStreaming: true` 期间显示 `▌` 游标；`content === ''` 时显示三点加载动画
- 所有消息状态用 `id` 精确定位更新，避免全量重渲染

### 类型定义（types.ts）

```typescript
export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

export interface ChatMeta {
  sessionId: string | null
  productId: string | null
  productName: string | null
  intentId: string | null
  needsFollowup: boolean
}

export type SSEEvent =
  | { type: 'chunk'; content: string }
  | { type: 'done'; session_id: string; product_id: string; product_name: string;
      intent_id: string; needs_followup: boolean; latency_ms: number }
  | { type: 'error'; message: string }
```

### ProductBadge 颜色编码

| product_id | 颜色 | 产品线 |
|------------|------|--------|
| meetings | #1a73e8 蓝 | 会议服务 |
| phone | #0f9d58 绿 | 话机 |
| earbuds | #7b1fa2 紫 | 耳机 |
| mouse | #f57c00 橙 | 鼠标 |
| screen | #00838f 青 | 会议大屏 |
| video_calls | #c62828 红 | 可视电话 |
| general | #546e7a 灰 | 通用 |

### Vite 开发代理

```typescript
// vite.config.ts
server: {
  port: 5173,
  proxy: { '/api': 'http://localhost:8000' }
}
```

开发时访问 `http://localhost:5173`，所有 `/api/*` 请求自动转发到后端 `:8000`，避免 CORS 问题。

### 生产部署

`npm run build` 产物输出到 `frontend/dist/`，FastAPI 自动检测并挂载：

```python
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
_frontend_legacy = Path(__file__).parent.parent / "frontend"
_static_dir = _frontend_dist if _frontend_dist.exists() else _frontend_legacy
app.mount("/static", StaticFiles(directory=str(_static_dir), html=True), name="static")
```

优先使用 `dist/`（React 构建产物），回退到 `frontend/`（旧版 HTML）。

---

## CSS 设计系统

深色主题，基于 CSS 变量：

```css
:root {
  --bg:      #0f1419;   /* 页面背景 */
  --surface: #161b22;   /* 卡片/气泡/输入框背景 */
  --border:  #2f3336;   /* 边框 */
  --muted:   #8899a6;   /* 次要文字 */
  --text:    #e7e9ea;   /* 主文字 */
  --accent:  #f0a500;   /* 强调色（用户气泡、标题、按钮） */
}
```

动画：
- `fadeUp`：消息出现时从下方淡入
- `dot`：加载三点缩放动画（3 个点错开 0.2s 延迟）
- `blink`：流式游标 `▌` 每秒闪烁

---

## 学习要点

| 概念 | 说明 |
|------|------|
| SSE vs WebSocket | SSE 单向（server→client），基于 HTTP，更简单；WS 双向，适合实时交互 |
| StreamingResponse | FastAPI 的异步生成器响应，`media_type="text/event-stream"` |
| ReadableStream | 浏览器原生 API，`getReader()` 逐块读取，无需第三方库 |
| 不完整行缓冲 | `buf = lines.pop()` 保留跨 TCP 包截断的不完整 SSE 行 |
| astream() | LangChain 链的异步流式迭代，底层调用 LLM 的 stream 接口 |
| React 状态精确更新 | 用 `id` 定位单条消息更新，避免触发整个列表重渲染 |
| Vite 代理 | 开发时解决前后端跨域，生产时合并到同一 origin |

---

## 与 Phase 5 的差异

| 维度 | Phase 5 | Phase 6 |
|------|---------|---------|
| 响应方式 | 同步 JSON | SSE 流式 |
| 前端技术 | 原生 HTML/JS | React 18 + TypeScript |
| 构建工具 | 无 | Vite 5 |
| Markdown | 无渲染 | react-markdown + highlight |
| 首字延迟 | 全部生成后 | 首 token 即显示 |
| 产品线展示 | 无 | 彩色徽章实时更新 |

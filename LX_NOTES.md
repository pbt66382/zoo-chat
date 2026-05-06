# LX 学习笔记

> 个人学习记录，每当我理解了一个知识点，就更新到这里。供复习和与搭档共享理解用。

---

## 2025-05-06：Memory（对话记忆）机制

### 问题：为什么 LLM 没有记忆？

LLM 每次 `POST /api/chat` 都是**独立请求**，LLM 本身不知道你之前问过什么。这和 Java 的 `HttpSession` 一样——每个请求都是全新的，必须手动把上下文带过来。

### Phase 1 的解决方案：前端维护 history 数组

```
前端：前端负责维护 history 数组
  history = [
    {role: "user", content: "如何创建会议"},
    {role: "assistant", content: "点击新建会议按钮..."}
  ]
  ↓ 第2轮请求时一起发到后端
POST /api/chat {
  message: "可以设密码吗",
  history: history,
  session_id: "xxx"
}

后端：收到后格式化，注入 system prompt
  history_text = "用户: 如何创建会议\n助手: 点击新建会议按钮..."
  → build_faq_chain(history=history_text)
  → Prompt 里多了【历史对话】字段
  → LLM 基于上下文理解"密码"="会议密码"
```

### 踩坑：浏览器缓存导致前端代码不生效

改完 JS 后浏览器可能用旧缓存，导致 `history` 字段发不出去。

**解决**：Mac 用 `Cmd + Shift + R` 强制刷新，Windows 用 `Ctrl + Shift + R`。

### 关键代码点

**前端**（必须带上 history）：
```javascript
body: JSON.stringify({
  message: message,
  history: history,        // ← 关键，不能漏
  session_id: sessionId
})
```

收到回答后存入 history：
```javascript
history.push({ role: 'user', content: message });
history.push({ role: 'assistant', content: data.answer });
```

**后端**（chat.py，把 history 格式化后传给 chain）：
```python
history_text = ""
if request.history:
    history_text = format_chat_history([
        {"role": msg.role, "content": msg.content}
        for msg in request.history
    ])
answer = invoke_faq_chain(request.message, history=history_text)
```

### 为什么前端维护 history 而不是后端？

Phase 1 是最简单的方案：
- 前端知道所有对话（用户自己加的、前端自己加的 loading 等）
- 后端只负责"问-答"，不维护状态（无状态服务）
- 适合快速上线，不需要 Redis 等外部存储

**Phase 3 会升级为**：后端用 `ConversationBufferMemory` 管理，Redis 持久化，支持刷新页面后继续对话。

### 和 Java 的类比

| 概念 | Python（前端） | Java |
|---|---|---|
| 存储历史 | `let history = []`（JS 变量） | `HttpSession.setAttribute("history", list)` |
| 发送历史 | `body: { history: history }` | `model.addAttribute("history", list)` |
| 注入上下文 | 塞进 system prompt | 拼进 ModelAndView 的 Model |

---

## 待理解清单

- [ ] LangChain ConversationBufferMemory（Phase 3）
- [ ] 向量检索 / Embedding（Phase 2）
- [ ] LCEL 的 Runnable 接口原理

---

## 疑问记录

（暂无）


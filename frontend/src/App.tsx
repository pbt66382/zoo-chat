import { useState, useRef, useEffect, useCallback } from 'react'
import ChatMessage from './components/ChatMessage'
import ChatInput from './components/ChatInput'
import ProductBadge from './components/ProductBadge'
import type { Message, ChatMeta, SSEEvent } from './types'

const INITIAL_META: ChatMeta = {
  sessionId: null,
  productId: null,
  productName: null,
  intentId: null,
  needsFollowup: false,
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [meta, setMeta] = useState<ChatMeta>(INITIAL_META)
  const [latency, setLatency] = useState<number | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isStreaming) return

    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: text }
    const botId = crypto.randomUUID()
    const botMsg: Message = { id: botId, role: 'assistant', content: '', isStreaming: true }

    setMessages(prev => [...prev, userMsg, botMsg])
    setIsStreaming(true)
    setLatency(null)

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: meta.sessionId }),
      })

      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const event: SSEEvent = JSON.parse(line.slice(6))

          if (event.type === 'chunk') {
            setMessages(prev =>
              prev.map(m => m.id === botId ? { ...m, content: m.content + event.content } : m)
            )
          } else if (event.type === 'done') {
            setMeta({
              sessionId: event.session_id,
              productId: event.product_id,
              productName: event.product_name,
              intentId: event.intent_id,
              needsFollowup: event.needs_followup,
            })
            setLatency(event.latency_ms)
            setMessages(prev =>
              prev.map(m => m.id === botId ? { ...m, isStreaming: false } : m)
            )
          } else if (event.type === 'error') {
            setMessages(prev =>
              prev.map(m =>
                m.id === botId
                  ? { ...m, content: `错误：${event.message}`, isStreaming: false }
                  : m
              )
            )
          }
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setMessages(prev =>
        prev.map(m =>
          m.id === botId ? { ...m, content: `请求失败：${msg}`, isStreaming: false } : m
        )
      )
    } finally {
      setIsStreaming(false)
    }
  }, [isStreaming, meta.sessionId])

  return (
    <div className="layout">
      {/* Header */}
      <header className="header">
        <div className="header-icon">Z</div>
        <div className="header-text">
          <h1>Zoo AI 智能客服</h1>
          <p>多产品线 · Phase 6 · SSE 流式输出</p>
        </div>
        <ProductBadge productId={meta.productId} productName={meta.productName} />
      </header>

      {/* Chat area */}
      <main className="chat-area">
        {messages.length === 0 && (
          <div className="welcome">
            <div className="welcome-icon">🤖</div>
            <h2>欢迎使用 Zoo 智能客服</h2>
            <p>我可以帮您解答会议服务、耳机、鼠标、可视电话、会议大屏等产品问题</p>
          </div>
        )}
        {messages.map(m => <ChatMessage key={m.id} message={m} />)}
        <div ref={bottomRef} />
      </main>

      {/* Input */}
      <ChatInput onSend={sendMessage} disabled={isStreaming} />

      {/* Status bar */}
      <footer className="status-bar">
        <span>Zoo AI Chat · Phase 6</span>
        <span>
          {isStreaming
            ? '⏳ 生成中…'
            : latency != null
            ? `✅ ${latency.toFixed(0)} ms`
            : ''}
        </span>
        <span>Powered by <strong>DeepSeek</strong></span>
      </footer>
    </div>
  )
}

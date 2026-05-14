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

// SSE 消息类型
export type SSEEvent =
  | { type: 'chunk'; content: string }
  | {
      type: 'done'
      session_id: string
      intent_id: string | null
      intent_confidence: number
      product_id: string | null
      product_name: string | null
      needs_followup: boolean
      pending_slot: string | null
      latency_ms: number
    }
  | { type: 'error'; message: string }

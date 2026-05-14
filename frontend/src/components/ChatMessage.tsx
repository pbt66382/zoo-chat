import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css'
import type { Message } from '../types'

interface Props {
  message: Message
}

export default function ChatMessage({ message }: Props) {
  const isUser = message.role === 'user'
  const showCursor = message.isStreaming && message.content.length > 0
  const showDots   = message.isStreaming && message.content.length === 0

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className={`avatar ${isUser ? 'avatar-user' : 'avatar-bot'}`}>
        {isUser ? '👤' : 'Z'}
      </div>

      <div className={`bubble ${isUser ? 'bubble-user' : 'bubble-bot'}`}>
        {isUser ? (
          <span style={{ whiteSpace: 'pre-wrap' }}>{message.content}</span>
        ) : showDots ? (
          <div className="loading-dots">
            <span /><span /><span />
          </div>
        ) : (
          <div className="markdown-body">
            <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
              {message.content}
            </ReactMarkdown>
            {showCursor && <span className="stream-cursor">▌</span>}
          </div>
        )}
      </div>
    </div>
  )
}

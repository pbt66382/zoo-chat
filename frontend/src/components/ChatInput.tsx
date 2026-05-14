import { useRef, KeyboardEvent } from 'react'

const SUGGESTIONS = [
  '如何创建会议',
  '如何共享屏幕',
  '会议中没有声音',
  '蓝牙耳机连不上',
  '如何升级话机固件',
]

interface Props {
  onSend: (text: string) => void
  disabled: boolean
}

export default function ChatInput({ onSend, disabled }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null)

  function autoResize() {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }

  function submit() {
    const text = ref.current?.value.trim() ?? ''
    if (!text || disabled) return
    onSend(text)
    if (ref.current) {
      ref.current.value = ''
      ref.current.style.height = 'auto'
    }
  }

  function handleKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="input-area">
      <div className="suggestions">
        {SUGGESTIONS.map(s => (
          <button
            key={s}
            className="chip"
            onClick={() => { if (ref.current) ref.current.value = s; submit() }}
            disabled={disabled}
          >
            {s}
          </button>
        ))}
      </div>

      <div className="input-row">
        <div className={`input-box ${disabled ? 'input-box-disabled' : ''}`}>
          <textarea
            ref={ref}
            placeholder="请输入您的问题… (Shift+Enter 换行)"
            rows={1}
            onInput={autoResize}
            onKeyDown={handleKey}
            disabled={disabled}
          />
          <button
            className="send-btn"
            onClick={submit}
            disabled={disabled}
            aria-label="发送"
          >
            &#10148;
          </button>
        </div>
      </div>
    </div>
  )
}

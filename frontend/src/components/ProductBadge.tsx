const PRODUCT_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  meetings: { bg: '#1a3a5c', text: '#60a5fa', label: '会议服务' },
  phone:    { bg: '#1a3d2e', text: '#4ade80', label: '可视电话' },
  earbuds:  { bg: '#2d1f4e', text: '#a78bfa', label: '耳机' },
  mouse:    { bg: '#3d2810', text: '#fb923c', label: '鼠标' },
  screen:   { bg: '#0f3030', text: '#2dd4bf', label: '会议大屏' },
  calls:    { bg: '#3d1a1a', text: '#f87171', label: '通话服务' },
  general:  { bg: '#2a2a2a', text: '#9ca3af', label: '通用' },
}

interface Props {
  productId: string | null
  productName: string | null
}

export default function ProductBadge({ productId, productName }: Props) {
  if (!productId) return null
  const color = PRODUCT_COLORS[productId] ?? PRODUCT_COLORS.general

  return (
    <span
      style={{
        marginLeft: 'auto',
        padding: '0.25rem 0.75rem',
        borderRadius: '9999px',
        fontSize: '0.75rem',
        fontWeight: 600,
        background: color.bg,
        color: color.text,
        border: `1px solid ${color.text}33`,
        whiteSpace: 'nowrap',
      }}
    >
      {productName ?? color.label}
    </span>
  )
}

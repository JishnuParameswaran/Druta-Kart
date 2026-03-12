const EMOTION_EMOJI = {
  anger: '😠', sadness: '😢', joy: '😊', fear: '😨',
  surprise: '😲', disgust: '🤢', neutral: '😐',
}

const INTENT_COLOR = {
  late_delivery: 'bg-yellow-500/20 text-yellow-300',
  damaged_product: 'bg-red-500/20 text-red-300',
  wrong_item: 'bg-orange-500/20 text-orange-300',
  missing_item: 'bg-orange-500/20 text-orange-300',
  payment_issue: 'bg-purple-500/20 text-purple-300',
  refund_request: 'bg-blue-500/20 text-blue-300',
  general: 'bg-slate-500/20 text-slate-300',
}

export default function MessageBubble({ msg }) {
  const isUser = msg.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[75%]">
          {msg.imageName && (
            <div className="flex justify-end mb-1">
              <span className="text-xs text-slate-400 bg-slate-700 px-2 py-1 rounded-full">
                📎 {msg.imageName}
              </span>
            </div>
          )}
          <div className="bg-indigo-600 text-white px-4 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed">
            {msg.text}
          </div>
        </div>
      </div>
    )
  }

  const intentClass = INTENT_COLOR[msg.intent] || INTENT_COLOR.general
  const emotionEmoji = EMOTION_EMOJI[msg.emotion] || '😐'

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[80%]">
        <div className="flex items-center gap-2 mb-1 px-1">
          <span className="text-xs font-semibold text-indigo-400">Druta Kart AI</span>
          {msg.agent_used && (
            <span className="text-xs text-slate-500">{msg.agent_used.replace('_agent','').replace('_',' ')}</span>
          )}
        </div>

        <div className="bg-slate-700 text-slate-100 px-4 py-3 rounded-2xl rounded-tl-sm text-sm leading-relaxed">
          {msg.text}
        </div>

        <div className="flex flex-wrap items-center gap-2 mt-2 px-1">
          {msg.intent && (
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${intentClass}`}>
              {msg.intent.replace(/_/g, ' ')}
            </span>
          )}
          <span className="text-xs text-slate-400">{emotionEmoji} {msg.emotion}</span>
          {msg.resolved && (
            <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full">✓ resolved</span>
          )}
          {msg.fraud_flagged && (
            <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded-full">⚠ fraud flag</span>
          )}
          {msg.rag_used && (
            <span className="text-xs bg-slate-600 text-slate-400 px-2 py-0.5 rounded-full">📚 RAG</span>
          )}
          {msg.offer_given && (
            <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">
              🎁 {msg.offer_given.offer_type || 'offer'}
            </span>
          )}
          {msg.latency_ms && (
            <span className="text-xs text-slate-500">{(msg.latency_ms / 1000).toFixed(1)}s</span>
          )}
        </div>

        {msg.tools_called?.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1 px-1">
            {msg.tools_called.map(t => (
              <span key={t} className="text-xs bg-slate-600/50 text-slate-400 px-2 py-0.5 rounded">
                🔧 {t.replace(/_tool/,'').replace(/_/g,' ')}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { sendMessage, uploadImage, sendVoice, healthCheck } from '../services/api'
import MessageBubble from '../components/MessageBubble'
import InputBar from '../components/InputBar'
import Sidebar from '../components/Sidebar'

function newSession() {
  return { userId: 'user_' + uuidv4().slice(0, 8), sessionId: uuidv4() }
}

export default function ChatPage() {
  const [session, setSession] = useState(() => newSession())
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [pendingImage, setPendingImage] = useState(null)   // { file, name, path }
  const [status, setStatus] = useState('checking')
  const [error, setError] = useState(null)
  const bottomRef = useRef()

  useEffect(() => {
    healthCheck()
      .then(() => setStatus('online'))
      .catch(() => setStatus('offline'))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function handleSend(text) {
    setError(null)
    const userMsg = {
      id: uuidv4(), role: 'user', text,
      imageName: pendingImage?.name || null,
    }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    try {
      let imagePath = pendingImage?.path || null

      if (pendingImage?.file && !imagePath) {
        const up = await uploadImage({ sessionId: session.sessionId, file: pendingImage.file })
        imagePath = up.image_path
      }
      setPendingImage(null)

      const res = await sendMessage({
        userId: session.userId,
        sessionId: session.sessionId,
        message: text,
        imagePath,
      })

      setMessages(prev => [...prev, {
        id: uuidv4(), role: 'ai',
        text: res.response,
        intent: res.intent,
        emotion: res.emotion,
        language: res.language,
        resolved: res.resolved,
        fraud_flagged: res.fraud_flagged,
        hallucination_flagged: res.hallucination_flagged,
        offer_given: res.offer_given,
        tools_called: res.tools_called,
        agent_used: res.agent_used,
        rag_used: res.rag_used,
        latency_ms: res.latency_ms,
      }])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleImageUpload(file) {
    setPendingImage({ file, name: file.name, path: null })
  }

  async function handleVoice(file) {
    setError(null)
    setLoading(true)
    const userMsg = { id: uuidv4(), role: 'user', text: `🎤 Voice message: ${file.name}` }
    setMessages(prev => [...prev, userMsg])

    try {
      const res = await sendVoice({ userId: session.userId, sessionId: session.sessionId, file })
      setMessages(prev => [...prev, {
        id: uuidv4(), role: 'ai',
        text: res.text_response,
        intent: res.intent,
        emotion: res.emotion,
        language: res.language,
        resolved: res.resolved,
        latency_ms: res.latency_ms,
        tools_called: [],
      }])

      if (res.audio_base64) {
        const audio = new Audio(`data:audio/wav;base64,${res.audio_base64}`)
        audio.play().catch(() => {})
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleNewChat() {
    setSession(newSession())
    setMessages([])
    setPendingImage(null)
    setError(null)
  }

  return (
    <div className="flex h-screen bg-slate-900 text-white overflow-hidden">
      <Sidebar
        userId={session.userId}
        sessionId={session.sessionId}
        onNewChat={handleNewChat}
        status={status}
      />

      <div className="flex flex-col flex-1 min-w-0">
        <div className="border-b border-slate-700 px-6 py-3 flex items-center gap-3 bg-slate-800">
          <div>
            <h1 className="font-semibold text-sm text-white">Customer Support</h1>
            <p className="text-xs text-slate-400">Multilingual · Voice · Image · Agentic AI</p>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center gap-4 text-slate-500">
              <div className="text-5xl">🛒</div>
              <div>
                <div className="text-white font-medium mb-1">Druta Kart AI Support</div>
                <div className="text-sm">Ask about your order, report a problem, or send a voice message.</div>
                <div className="text-xs mt-3 space-y-1">
                  <div>💬 "My order is late"</div>
                  <div>📦 "I received a damaged product"</div>
                  <div>💰 "I want a refund for ORD-12345"</div>
                </div>
              </div>
            </div>
          )}

          {messages.map(msg => <MessageBubble key={msg.id} msg={msg} />)}

          {loading && (
            <div className="flex justify-start mb-4">
              <div className="bg-slate-700 px-4 py-3 rounded-2xl rounded-tl-sm text-sm text-slate-400">
                <span className="animate-pulse">Thinking…</span>
              </div>
            </div>
          )}

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-sm px-4 py-3 rounded-lg mb-4">
              ⚠ {error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        <InputBar
          onSend={handleSend}
          onImageUpload={handleImageUpload}
          onVoice={handleVoice}
          disabled={loading}
          pendingImage={pendingImage}
          onClearImage={() => setPendingImage(null)}
        />
      </div>
    </div>
  )
}

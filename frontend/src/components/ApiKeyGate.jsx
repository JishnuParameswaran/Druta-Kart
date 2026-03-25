import { useState } from 'react'

export default function ApiKeyGate({ onKeySet }) {
  const [key, setKey] = useState('')
  const [error, setError] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = key.trim()
    if (!trimmed.startsWith('gsk_') || trimmed.length < 20) {
      setError('That does not look like a valid Groq key. It should start with gsk_')
      return
    }
    setError('')
    onKeySet(trimmed)
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white flex items-center justify-center px-4">
      <div className="w-full max-w-md">

        {/* Logo + title */}
        <div className="text-center mb-8">
          <div className="text-6xl mb-4">🛒</div>
          <h1 className="text-2xl font-bold text-white mb-1">Druta Kart AI Support</h1>
          <p className="text-slate-400 text-sm">Multilingual · Voice · Image · Agentic AI</p>
        </div>

        {/* Card */}
        <div className="bg-slate-800 rounded-2xl p-8 border border-slate-700">
          <h2 className="text-lg font-semibold mb-1">Enter your Groq API Key</h2>
          <p className="text-slate-400 text-sm mb-6">
            This demo runs on your own free Groq key — your usage, your account.{' '}
            <a
              href="https://console.groq.com/keys"
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-400 hover:text-indigo-300 underline"
            >
              Get a free key here
            </a>{' '}
            (takes 30 seconds).
          </p>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <input
              type="password"
              value={key}
              onChange={e => { setKey(e.target.value); setError('') }}
              placeholder="gsk_..."
              autoComplete="off"
              spellCheck={false}
              className="w-full bg-slate-700 text-slate-100 placeholder-slate-500 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />

            {error && (
              <p className="text-red-400 text-xs">{error}</p>
            )}

            <button
              type="submit"
              disabled={!key.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded-xl px-4 py-3 text-sm font-semibold transition-colors"
            >
              Start Chatting →
            </button>
          </form>
        </div>

        {/* Privacy note */}
        <p className="text-center text-slate-500 text-xs mt-6">
          Your key is stored only in your browser's local storage.<br />
          It is never saved on any server or shared with anyone.
        </p>
      </div>
    </div>
  )
}

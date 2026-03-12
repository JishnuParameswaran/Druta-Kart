import { useRef, useState } from 'react'

export default function InputBar({ onSend, onImageUpload, onVoice, disabled, pendingImage, onClearImage }) {
  const [text, setText] = useState('')
  const fileRef = useRef()
  const audioRef = useRef()

  function handleSend() {
    if (!text.trim() || disabled) return
    onSend(text.trim())
    setText('')
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t border-slate-700 bg-slate-800 p-4">
      {pendingImage && (
        <div className="flex items-center gap-2 mb-2 text-sm text-slate-300 bg-slate-700 px-3 py-2 rounded-lg">
          <span>📎 {pendingImage.name}</span>
          <button onClick={onClearImage} className="ml-auto text-slate-400 hover:text-red-400">✕</button>
        </div>
      )}

      <div className="flex items-end gap-2">
        <button
          title="Upload image"
          onClick={() => fileRef.current.click()}
          disabled={disabled}
          className="p-2 text-slate-400 hover:text-indigo-400 disabled:opacity-40 transition-colors"
        >
          📷
          <input
            ref={fileRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={e => { if (e.target.files[0]) onImageUpload(e.target.files[0]); e.target.value = '' }}
          />
        </button>

        <button
          title="Send voice"
          onClick={() => audioRef.current.click()}
          disabled={disabled}
          className="p-2 text-slate-400 hover:text-indigo-400 disabled:opacity-40 transition-colors"
        >
          🎤
          <input
            ref={audioRef}
            type="file"
            accept="audio/wav,audio/mp3,audio/ogg,audio/webm"
            className="hidden"
            onChange={e => { if (e.target.files[0]) onVoice(e.target.files[0]); e.target.value = '' }}
          />
        </button>

        <textarea
          rows={1}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKey}
          disabled={disabled}
          placeholder="Type your message… (Enter to send)"
          className="flex-1 resize-none bg-slate-700 text-slate-100 placeholder-slate-400 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
        />

        <button
          onClick={handleSend}
          disabled={disabled || !text.trim()}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded-xl px-4 py-2.5 text-sm font-medium transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  )
}

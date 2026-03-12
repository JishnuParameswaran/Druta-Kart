export default function Sidebar({ userId, sessionId, onNewChat, status }) {
  return (
    <div className="w-64 bg-slate-900 border-r border-slate-700 flex flex-col p-4 shrink-0">
      <div className="flex items-center gap-2 mb-8">
        <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">DK</div>
        <div>
          <div className="text-white font-semibold text-sm">Druta Kart</div>
          <div className="text-slate-400 text-xs">AI Support</div>
        </div>
      </div>

      <button
        onClick={onNewChat}
        className="w-full bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium px-3 py-2 rounded-lg transition-colors mb-6"
      >
        + New Chat
      </button>

      <div className="space-y-3 text-xs text-slate-400">
        <div>
          <div className="text-slate-500 uppercase tracking-wider text-[10px] mb-1">User ID</div>
          <div className="text-slate-300 font-mono truncate">{userId}</div>
        </div>
        <div>
          <div className="text-slate-500 uppercase tracking-wider text-[10px] mb-1">Session</div>
          <div className="text-slate-300 font-mono truncate">{sessionId.slice(0, 16)}…</div>
        </div>
        <div>
          <div className="text-slate-500 uppercase tracking-wider text-[10px] mb-1">Backend</div>
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${status === 'online' ? 'bg-green-400' : status === 'checking' ? 'bg-yellow-400' : 'bg-red-400'}`} />
            <span className="text-slate-300">{status}</span>
          </div>
        </div>
      </div>

      <div className="mt-auto text-[10px] text-slate-600 leading-relaxed">
        Powered by Groq · Sarvam · Supabase
      </div>
    </div>
  )
}

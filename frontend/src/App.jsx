import { useState } from 'react'
import ApiKeyGate from './components/ApiKeyGate'
import ChatPage from './pages/ChatPage'

export default function App() {
  const [groqApiKey, setGroqApiKey] = useState(
    () => localStorage.getItem('groq_api_key') || ''
  )

  function handleKeySet(key) {
    localStorage.setItem('groq_api_key', key)
    setGroqApiKey(key)
  }

  function handleChangeKey() {
    localStorage.removeItem('groq_api_key')
    setGroqApiKey('')
  }

  if (!groqApiKey) {
    return <ApiKeyGate onKeySet={handleKeySet} />
  }

  return <ChatPage groqApiKey={groqApiKey} onChangeKey={handleChangeKey} />
}

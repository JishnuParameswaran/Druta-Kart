const BASE_URL = import.meta.env.VITE_API_URL || 'https://druta-kart-production.up.railway.app'

export async function sendMessage({ userId, sessionId, message, imagePath }) {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      session_id: sessionId,
      message,
      image_path: imagePath || null,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function uploadImage({ sessionId, file }) {
  const form = new FormData()
  form.append('session_id', sessionId)
  form.append('file', file)

  const res = await fetch(`${BASE_URL}/upload-image?session_id=${sessionId}`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function sendVoice({ userId, sessionId, file }) {
  const form = new FormData()
  form.append('user_id', userId)
  form.append('session_id', sessionId)
  form.append('file', file)

  const res = await fetch(`${BASE_URL}/voice`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function healthCheck() {
  const res = await fetch(`${BASE_URL}/health`)
  return res.json()
}

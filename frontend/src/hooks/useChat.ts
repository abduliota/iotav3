'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { Message, Source } from '@/types'
import { streamQuery, submitFeedback, fetchSessionMessages } from '@/lib/api'
import { getUserId, getSessionId, newSession } from '@/lib/identity'

export function useChat() {
  const [messages,  setMessages]  = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error,     setError]     = useState<string | null>(null)

  // Use refs for userId + sessionId so callbacks always see the latest value
  // without needing to be re-created (fixes stale closure bugs)
  const userIdRef    = useRef<string>('')
  const sessionIdRef = useRef<string>('')

  // Also keep state versions for components that need to render them
  const [userId,    setUserId]    = useState<string>('')
  const [sessionId, setSessionId] = useState<string>('')

  // ── Init identity on mount ─────────────────────────────────────────────────
  useEffect(() => {
    const sid = getSessionId()
    sessionIdRef.current = sid
    setSessionId(sid)

    getUserId().then(uid => {
      userIdRef.current = uid
      setUserId(uid)

      // Restore previous messages for this session
      fetchSessionMessages(sid).then(history => {
        if (!history.length) return
        const restored: Message[] = []
        history.forEach(m => {
          restored.push({
            id:        m.message_id + '_user',
            role:      'user',
            content:   m.user_message,
            sources:   [],
            timestamp: new Date(m.timestamp),
          })
          restored.push({
            id:        m.message_id,
            role:      'assistant',
            content:   m.assistant_message,
            sources:   [],
            timestamp: new Date(m.timestamp),
            feedback:  null,
          })
        })
        setMessages(restored)
      }).catch(console.error)
    }).catch(console.error)
  }, [])

  // ── Send message ───────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (query: string) => {
    if (!query.trim() || isLoading) return
    setError(null)

    // Always read from refs — never stale
    const uid = userIdRef.current || 'anon'
    const sid = sessionIdRef.current || getSessionId()

    const userMsg: Message = {
      id:        uuidv4(),
      role:      'user',
      content:   query.trim(),
      sources:   [],
      timestamp: new Date(),
    }

    const assistantId = uuidv4()
    const assistantMsg: Message = {
      id:          assistantId,
      role:        'assistant',
      content:     '',
      sources:     [],
      timestamp:   new Date(),
      isStreaming: true,
    }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setIsLoading(true)

    try {
      let fullContent    = ''
      let finalSources:  Source[] = []
      let finalMessageId = assistantId
      let method         = ''
      let cached         = false

      for await (const chunk of streamQuery({ query: query.trim(), userId: uid, sessionId: sid })) {
        if (chunk.type === 'token' && chunk.text) {
          fullContent += chunk.text
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, content: fullContent, isStreaming: true } : m
          ))
        } else if (chunk.type === 'sources') {
          finalSources   = chunk.sources    ?? []
          finalMessageId = chunk.message_id ?? assistantId
          method         = chunk.method     ?? ''
          cached         = chunk.cached     ?? false
        } else if (chunk.type === 'done') {
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? { ...m, id: finalMessageId, content: fullContent, sources: finalSources,
                  isStreaming: false, method, cached, feedback: null }
              : m
          ))
        } else if (chunk.type === 'error') {
          throw new Error(chunk.message || 'Stream error')
        }
      }
    } catch (err: any) {
      setError(err.message || 'Something went wrong. Please try again.')
      setMessages(prev => prev.map(m =>
        m.id === assistantId
          ? { ...m, content: 'Sorry, something went wrong. Please try again.', isStreaming: false }
          : m
      ))
    } finally {
      setIsLoading(false)
    }
  }, [isLoading])   // ← no userId/sessionId deps needed — we use refs

  // ── Feedback ───────────────────────────────────────────────────────────────
  const giveFeedback = useCallback(async (
    messageId: string,
    feedback:  1 | 0,
    comment?:  string,
  ) => {
    // Read refs directly — always current values, never stale
    const uid = userIdRef.current
    const sid = sessionIdRef.current

    if (!uid || !sid) {
      console.warn('[feedback] userId or sessionId not ready yet')
      return
    }

    // Optimistic UI update
    setMessages(prev => prev.map(m =>
      m.id === messageId ? { ...m, feedback } : m
    ))

    // Get the message content using functional state update trick
    // (we can't read messages state here without stale closure)
    // Instead, pass the values we already have
    setMessages(prev => {
      const msgIndex = prev.findIndex(m => m.id === messageId)
      const msg      = prev[msgIndex]
      const userMsg  = msgIndex > 0 ? prev[msgIndex - 1] : null

      if (!msg) return prev

      // Fire-and-forget DB call
      submitFeedback({
        sessionId:        sid,
        userId:           uid,
        messageId,
        feedback,
        comments: comment,
        userMessage:      userMsg?.content,
        assistantMessage: msg.content,
      }).catch(e => console.error('[feedback] submit failed:', e))

      return prev   // no state change here — we already updated above
    })
  }, [])   // ← no deps needed — uses refs and functional state

  // ── New conversation ───────────────────────────────────────────────────────
  const startNewConversation = useCallback(() => {
    const sid = newSession()
    sessionIdRef.current = sid
    setSessionId(sid)
    setMessages([])
    setError(null)
  }, [])

  return {
    messages,
    isLoading,
    error,
    userId,
    sessionId,
    sendMessage,
    giveFeedback,
    startNewConversation,
  }
}
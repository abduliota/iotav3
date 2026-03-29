'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { Message, Source, Conversation } from '@/types'
import { streamQuery, submitFeedback, fetchSessionMessages, fetchConversations } from '@/lib/api'
import { getUserId, getSessionId, newSession } from '@/lib/identity'

export function useChat() {
  const [messages,       setMessages]       = useState<Message[]>([])
  const [isLoading,      setIsLoading]      = useState(false)
  const [error,          setError]          = useState<string | null>(null)
  const [conversations,  setConversations]  = useState<Conversation[]>([])

  const userIdRef    = useRef<string>('')
  const sessionIdRef = useRef<string>('')

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

      // Load conversation history for sidebar
      fetchConversations(uid).then(convs => {
        setConversations(convs)
      }).catch(console.error)
    }).catch(console.error)
  }, [])

  // ── Load a past conversation ───────────────────────────────────────────────
  const loadConversation = useCallback(async (targetSessionId: string) => {
    if (targetSessionId === sessionIdRef.current) return

    sessionIdRef.current = targetSessionId
    setSessionId(targetSessionId)
    setMessages([])
    setError(null)

    try {
      const history = await fetchSessionMessages(targetSessionId)
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
    } catch (err) {
      console.error('[loadConversation]', err)
    }
  }, [])

  // ── Send message ───────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (query: string) => {
    if (!query.trim() || isLoading) return
    setError(null)

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
          // Refresh conversations list after sending a message
          fetchConversations(uid).then(convs => setConversations(convs)).catch(() => {})
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
  }, [isLoading])

  // ── Feedback ───────────────────────────────────────────────────────────────
  const giveFeedback = useCallback(async (
    messageId: string,
    feedback:  1 | 0,
    comment?:  string,
  ) => {
    const uid = userIdRef.current
    const sid = sessionIdRef.current

    if (!uid || !sid) {
      console.warn('[feedback] userId or sessionId not ready yet')
      return
    }

    setMessages(prev => prev.map(m =>
      m.id === messageId ? { ...m, feedback } : m
    ))

    setMessages(prev => {
      const msgIndex = prev.findIndex(m => m.id === messageId)
      const msg      = prev[msgIndex]
      const userMsg  = msgIndex > 0 ? prev[msgIndex - 1] : null

      if (!msg) return prev

      submitFeedback({
        sessionId:        sid,
        userId:           uid,
        messageId,
        feedback,
        comments: comment,
        userMessage:      userMsg?.content,
        assistantMessage: msg.content,
      }).catch(e => console.error('[feedback] submit failed:', e))

      return prev
    })
  }, [])

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
    conversations,
    sendMessage,
    giveFeedback,
    startNewConversation,
    loadConversation,
  }
}
'use client'

import { useEffect, useRef, useState } from 'react'
import { useChat } from '@/hooks/useChat'
import { MessageBubble } from '@/components/MessageBubble'
import { SourcesPanel } from '@/components/SourcesPanel'
import { InputBar } from '@/components/InputBar'
import { Source } from '@/types'

export default function HomePage() {
  const {
    messages,
    isLoading,
    error,
    userId,
    sendMessage,
    giveFeedback,
    startNewConversation,
  } = useChat()

  const bottomRef       = useRef<HTMLDivElement>(null)
  const [dark, setDark] = useState(false)
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const [showIdentity, setShowIdentity] = useState(false)

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Dark mode
  useEffect(() => {
    const saved = localStorage.getItem('nora_dark')
    if (saved === 'true') setDark(true)
  }, [])
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('nora_dark', String(dark))
  }, [dark])

  // Current sources = last assistant message sources
  const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant')
  const currentSources: Source[] = lastAssistant?.sources ?? []
  const sourcesLoading = isLoading && currentSources.length === 0

  // Short display ID
  const shortId = userId ? userId.slice(0, 8) + '...' : 'identifying...'

  const handleResetIdentity = () => {
    localStorage.removeItem('nora_user_id')
    sessionStorage.removeItem('nora_session_id')
    setShowIdentity(false)
    window.location.reload()
  }

  return (
    <div className="h-screen flex flex-col bg-surface dark:bg-surface-dark font-sans overflow-hidden">

      {/* ── Top bar ──────────────────────────────────────────────────── */}
      <header className="shrink-0 flex items-center justify-between px-4 md:px-6 py-3 border-b border-border dark:border-border-dark bg-panel dark:bg-panel-dark">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl flex items-center justify-center shadow-sm border border-border bg-stone-100 text-accent dark:bg-accent dark:text-white dark:border-transparent">
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-semibold text-ink-DEFAULT dark:text-zinc-100 font-display leading-tight">
              IOTA AI
            </h1>
            <p className="text-[11px] text-ink-faint dark:text-zinc-500 leading-none mt-0.5">
              Ask questions about SAMA Regulations
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Sources chip — mobile */}
          {currentSources.length > 0 && (
            <button
              onClick={() => setSourcesOpen(o => !o)}
              className="lg:hidden flex items-center gap-1.5 text-[11px] px-2.5 py-1.5 rounded-full bg-accent-muted text-accent dark:bg-accent/20 dark:text-accent-light font-medium"
            >
              <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12" />
              </svg>
              {currentSources.length} sources
            </button>
          )}

          {/* ── Identity / fingerprint indicator ── */}
          <div className="relative">
            <button
              onClick={() => setShowIdentity(o => !o)}
              title="Your identity"
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-mono transition-colors ${
                userId
                  ? 'text-accent dark:text-accent-light bg-accent-muted/60 dark:bg-accent/10 hover:bg-accent-muted dark:hover:bg-accent/20'
                  : 'text-ink-faint dark:text-zinc-600 bg-stone-100 dark:bg-zinc-800 animate-pulse'
              }`}
            >
              {/* Fingerprint icon */}
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 11c0 3.517-1.009 6.799-2.753 9.571m-3.44-2.04.054-.09A13.916 13.916 0 0 0 8 11a4 4 0 1 1 8 0c0 1.017-.07 2.019-.203 3m-2.118 6.844A21.88 21.88 0 0 0 15.171 17m3.839 1.132c.645-2.266.99-4.659.99-7.132A8 8 0 0 0 8 4.07M3 15.364c.64-1.319 1-2.8 1-4.364 0-1.457.39-2.823 1.07-4" />
              </svg>
              {shortId}
            </button>

            {/* Identity dropdown */}
            {showIdentity && (
              <>
                <div
                  className="fixed inset-0 z-30"
                  onClick={() => setShowIdentity(false)}
                />
                <div className="absolute right-0 top-full mt-2 w-64 z-40 bg-panel dark:bg-panel-dark border border-border dark:border-border-dark rounded-xl shadow-lg p-3 animate-fade-in">
                  <div className="space-y-2.5">
                    {/* Status */}
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
                      <span className="text-xs font-medium text-ink-DEFAULT dark:text-zinc-100">
                        Recognized via Fingerprint
                      </span>
                    </div>

                    {/* User ID */}
                    <div className="bg-stone-50 dark:bg-zinc-800 rounded-lg px-3 py-2">
                      <p className="text-[10px] text-ink-faint dark:text-zinc-500 mb-0.5">Your device ID</p>
                      <p className="text-[11px] font-mono text-ink-muted dark:text-zinc-300 break-all leading-relaxed">
                        {userId || 'Loading...'}
                      </p>
                    </div>

                    {/* Explanation */}
                    <p className="text-[10px] text-ink-faint dark:text-zinc-600 leading-relaxed">
                      Your browser fingerprint is used to identify your device — no account or login required. Your conversations are linked to this ID.
                    </p>

                    {/* Reset button */}
                    <button
                      onClick={handleResetIdentity}
                      className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg border border-red-200 dark:border-red-800 text-red-500 dark:text-red-400 text-xs hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                    >
                      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
                      </svg>
                      Reset identity (new device ID)
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* New chat */}
          <button
            onClick={startNewConversation}
            title="New conversation"
            className="p-2 rounded-lg text-ink-muted dark:text-zinc-400 hover:bg-stone-100 dark:hover:bg-zinc-800 hover:text-ink-DEFAULT dark:hover:text-zinc-200 transition-colors"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
          </button>

          {/* Dark mode */}
          <button
            onClick={() => setDark(d => !d)}
            title="Toggle dark mode"
            className="p-2 rounded-lg text-ink-muted dark:text-zinc-400 hover:bg-stone-100 dark:hover:bg-zinc-800 hover:text-ink-DEFAULT dark:hover:text-zinc-200 transition-colors"
          >
            {dark ? (
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v2.25m6.364.386-1.591 1.591M21 12h-2.25m-.386 6.364-1.591-1.591M12 18.75V21m-4.773-4.227-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0Z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21.752 15.002A9.72 9.72 0 0 1 18 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 0 0 3 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 0 0 9.002-5.998Z" />
              </svg>
            )}
          </button>
        </div>
      </header>

      {/* ── Body ─────────────────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0">
        <div className="flex flex-col flex-1 min-w-0">
          <div className="flex-1 overflow-y-auto px-4 md:px-6 py-6 space-y-6">
            {messages.length === 0 ? (
              <WelcomeScreen />
            ) : (
              messages.map(msg => (
                <MessageBubble key={msg.id} message={msg} onFeedback={giveFeedback} />
              ))
            )}
            {error && (
              <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-sm text-red-600 dark:text-red-400 animate-fade-in">
                <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
                </svg>
                {error}
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="shrink-0 px-4 md:px-6 py-4 border-t border-border dark:border-border-dark bg-panel dark:bg-panel-dark">
            <InputBar onSend={sendMessage} isLoading={isLoading} />
          </div>
        </div>

        {/* Sources sidebar — desktop */}
        <aside className="hidden lg:flex flex-col w-80 xl:w-96 border-l border-border dark:border-border-dark bg-surface dark:bg-surface-dark shrink-0">
          <SourcesPanel sources={currentSources} isLoading={sourcesLoading} />
        </aside>
      </div>

      {/* Mobile sources drawer */}
      {sourcesOpen && (
        <>
          <div className="lg:hidden fixed inset-0 bg-black/30 z-40 animate-fade-in" onClick={() => setSourcesOpen(false)} />
          <div className="lg:hidden fixed bottom-0 left-0 right-0 z-50 bg-panel dark:bg-panel-dark rounded-t-2xl border-t border-border dark:border-border-dark max-h-[70vh] flex flex-col animate-slide-up shadow-2xl">
            <div className="flex items-center justify-between px-4 py-3 border-b border-border dark:border-border-dark">
              <h3 className="text-sm font-semibold text-ink-DEFAULT dark:text-zinc-100">Sources</h3>
              <button title="Close sources" onClick={() => setSourcesOpen(false)} className="p-1.5 rounded-lg hover:bg-stone-100 dark:hover:bg-zinc-800 text-ink-muted dark:text-zinc-400">
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              <SourcesPanel sources={currentSources} isLoading={sourcesLoading} />
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function WelcomeScreen() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4 py-16 animate-fade-in">
      <div className="w-14 h-14 rounded-2xl bg-accent/10 dark:bg-accent/20 border border-accent/20 flex items-center justify-center mb-5">
        <svg className="w-7 h-7 text-accent dark:text-accent-light" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      </div>
      <h2 className="text-lg font-semibold text-ink-DEFAULT dark:text-zinc-100 font-display mb-2">
        IOTA AI
      </h2>
      <p className="text-sm text-ink-muted dark:text-zinc-400 max-w-xs leading-relaxed">
        Ask questions about SAMA Regulations
      </p>
    </div>
  )
}
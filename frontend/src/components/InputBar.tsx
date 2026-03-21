'use client'

import { useState, useRef, KeyboardEvent } from 'react'
import clsx from 'clsx'

interface Props {
  onSend:    (query: string) => void
  isLoading: boolean
  disabled?: boolean
}

const SUGGESTIONS = [
  'What is the LCR requirement?',
  'Explain SAMA cybersecurity framework',
  'What are NCA ECC controls?',
  'PDPL data subject rights',
]

export function InputBar({ onSend, isLoading, disabled }: Props) {
  const [value, setVal] = useState('')
  const textRef = useRef<HTMLTextAreaElement>(null)

  const submit = () => {
    const q = value.trim()
    if (!q || isLoading || disabled) return
    onSend(q)
    setVal('')
    if (textRef.current) {
      textRef.current.style.height = 'auto'
    }
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const autoResize = () => {
    if (!textRef.current) return
    textRef.current.style.height = 'auto'
    textRef.current.style.height = Math.min(textRef.current.scrollHeight, 160) + 'px'
  }

  return (
    <div className="space-y-3">
      {/* Suggestion chips — shown when empty */}
      {!value && !isLoading && (
        <div className="flex gap-2 flex-wrap px-1">
          {SUGGESTIONS.map(s => (
            <button
              key={s}
              onClick={() => { setVal(s); textRef.current?.focus() }}
              className="text-[11px] px-3 py-1.5 rounded-full border border-border dark:border-border-dark text-ink-muted dark:text-zinc-400 hover:border-accent/50 hover:text-accent dark:hover:text-accent-light hover:bg-accent-muted/30 transition-all duration-150 whitespace-nowrap"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input container */}
      <div className={clsx(
        'flex items-end gap-3 px-4 py-3 rounded-2xl border transition-all duration-150',
        'bg-panel dark:bg-panel-dark',
        isLoading || disabled
          ? 'border-border dark:border-border-dark opacity-60'
          : 'border-border dark:border-border-dark focus-within:border-accent/60 dark:focus-within:border-accent-light/40 focus-within:shadow-[0_0_0_3px_rgba(26,107,74,0.08)]'
      )}>
        <textarea
          ref={textRef}
          value={value}
          onChange={e => { setVal(e.target.value); autoResize() }}
          onKeyDown={onKeyDown}
          placeholder="Ask a question about SAMA or NORA…"
          rows={1}
          disabled={isLoading || disabled}
          className="flex-1 bg-transparent text-sm text-ink-DEFAULT dark:text-zinc-200 placeholder:text-ink-faint dark:placeholder:text-zinc-600 resize-none focus:outline-none leading-relaxed min-h-[24px] max-h-40"
        />

        {/* Send button */}
        <button
          onClick={submit}
          disabled={!value.trim() || isLoading || disabled}
          className={clsx(
            'shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-150',
            value.trim() && !isLoading && !disabled
              ? 'bg-accent-muted text-accent border border-accent/30 hover:bg-accent-muted/80 dark:bg-accent dark:text-white dark:border-transparent dark:hover:bg-accent/90 shadow-sm hover:shadow'
              : 'bg-stone-100 dark:bg-zinc-800 text-ink-faint dark:text-zinc-600 cursor-not-allowed'
          )}
        >
          {isLoading ? (
            <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
          ) : (
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12zm0 0h7.5" />
            </svg>
          )}
        </button>
      </div>

      <p className="text-[10px] text-center text-ink-faint dark:text-zinc-700 px-2">
        Answers only based on banking and regulations!
      </p>
    </div>
  )
}

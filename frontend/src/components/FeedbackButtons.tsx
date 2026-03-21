'use client'

import { useState } from 'react'
import clsx from 'clsx'

interface Props {
  messageId: string
  current:   1 | 0 | null | undefined
  onFeedback: (messageId: string, vote: 1 | 0, comment?: string) => void
}

export function FeedbackButtons({ messageId, current, onFeedback }: Props) {
  const [showComment, setShowComment] = useState(false)
  const [comment,     setComment]     = useState('')

  const handleVote = (vote: 1 | 0) => {
    if (vote === 0 && current !== 0) {
      setShowComment(true)
    } else {
      setShowComment(false)
    }
    onFeedback(messageId, vote)
  }

  const submitComment = () => {
    onFeedback(messageId, 0, comment)
    setShowComment(false)
    setComment('')
  }

  return (
    <div className="mt-2">
      <div className="flex items-center gap-1.5">
        {/* Like */}
        <button
          onClick={() => handleVote(1)}
          title="Good answer"
          className={clsx(
            'group flex items-center gap-1 px-2 py-1 rounded-md text-xs transition-all duration-150',
            current === 1
              ? 'bg-accent-muted text-accent dark:bg-accent/20 dark:text-accent-light'
              : 'text-ink-faint hover:text-ink-DEFAULT hover:bg-stone-100 dark:text-zinc-500 dark:hover:bg-zinc-800 dark:hover:text-zinc-200'
          )}
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill={current === 1 ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
          </svg>
        </button>

        {/* Dislike */}
        <button
          onClick={() => handleVote(0)}
          title="Poor answer"
          className={clsx(
            'group flex items-center gap-1 px-2 py-1 rounded-md text-xs transition-all duration-150',
            current === 0
              ? 'bg-red-50 text-red-500 dark:bg-red-900/20 dark:text-red-400'
              : 'text-ink-faint hover:text-ink-DEFAULT hover:bg-stone-100 dark:text-zinc-500 dark:hover:bg-zinc-800 dark:hover:text-zinc-200'
          )}
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill={current === 0 ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
          </svg>
        </button>
      </div>

      {/* Comment box — shows after dislike */}
      {showComment && (
        <div className="mt-2 animate-fade-in">
          <textarea
            value={comment}
            onChange={e => setComment(e.target.value)}
            placeholder="What was wrong with this answer? (optional)"
            rows={2}
            className="w-full text-xs px-3 py-2 rounded-lg border border-border dark:border-border-dark bg-surface dark:bg-surface-dark text-ink-DEFAULT dark:text-zinc-200 placeholder:text-ink-faint dark:placeholder:text-zinc-600 resize-none focus:outline-none focus:ring-1 focus:ring-accent"
          />
          <div className="flex gap-2 mt-1.5">
            <button
              onClick={submitComment}
              className="text-xs px-3 py-1 rounded-md bg-accent text-white hover:bg-accent/90 transition-colors"
            >
              Submit
            </button>
            <button
              onClick={() => setShowComment(false)}
              className="text-xs px-3 py-1 rounded-md text-ink-muted hover:text-ink-DEFAULT dark:text-zinc-400 transition-colors"
            >
              Skip
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

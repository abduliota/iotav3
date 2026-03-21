'use client'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Message } from '@/types'
import { ThinkingAnimation } from './ThinkingAnimation'
import { FeedbackButtons } from './FeedbackButtons'
import clsx from 'clsx'

interface Props {
  message:    Message
  onFeedback: (messageId: string, vote: 1 | 0, comment?: string) => void
}

export function MessageBubble({ message, onFeedback }: Props) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end animate-slide-up">
        <div className="max-w-[75%] md:max-w-[60%]">
          <div className="bg-accent-DEFAULT text-white px-4 py-2.5 rounded-2xl rounded-tr-sm text-sm leading-relaxed shadow-sm">
            {message.content}
          </div>
          <p className="text-[10px] text-ink-faint dark:text-zinc-600 text-right mt-1 mr-1">
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </p>
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="flex gap-3 animate-slide-up group">
      {/* Avatar */}
      <div className="shrink-0 mt-0.5">
        <div className="w-7 h-7 rounded-full bg-accent-DEFAULT/10 dark:bg-accent-DEFAULT/20 border border-accent-DEFAULT/20 flex items-center justify-center">
          <svg className="w-3.5 h-3.5 text-accent-DEFAULT dark:text-accent-light" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>
        </div>
      </div>

      <div className="flex-1 min-w-0 pb-2">
        {/* Thinking animation or content */}
        {message.isStreaming && !message.content ? (
          <ThinkingAnimation />
        ) : (
          <>
            {/* Markdown content */}
            <div className={clsx(
              'prose prose-sm dark:prose-invert max-w-none',
              'prose-p:leading-relaxed prose-p:my-1.5',
              'prose-strong:text-ink-DEFAULT dark:prose-strong:text-zinc-100',
              'prose-code:text-accent-DEFAULT dark:prose-code:text-accent-light',
              'prose-code:bg-stone-100 dark:prose-code:bg-zinc-800',
              'prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs',
              'text-ink-DEFAULT dark:text-zinc-200 text-sm',
            )}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>

            {/* Cursor blink while streaming */}
            {message.isStreaming && (
              <span className="inline-block w-0.5 h-4 bg-accent-DEFAULT dark:bg-accent-light ml-0.5 animate-pulse align-middle" />
            )}

            {/* Metadata row */}
            {!message.isStreaming && (
              <div className="flex items-center gap-3 mt-2.5 flex-wrap">
                {/* Timestamp */}
                <span className="text-[10px] text-ink-faint dark:text-zinc-600">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>

                {/* Method badge */}
                {message.method && message.method !== 'generative' && (
                  <span className={clsx(
                    'text-[10px] font-mono px-1.5 py-0.5 rounded-full',
                    message.cached
                      ? 'bg-blue-50 text-blue-500 dark:bg-blue-900/20 dark:text-blue-400'
                      : 'bg-stone-100 text-ink-faint dark:bg-zinc-800 dark:text-zinc-500'
                  )}>
                    {message.cached ? '⚡ cached' : message.method}
                  </span>
                )}

                {/* Source count chip — mobile only */}
                {message.sources.length > 0 && (
                  <span className="lg:hidden text-[10px] text-accent-DEFAULT dark:text-accent-light font-medium">
                    {message.sources.length} source{message.sources.length !== 1 ? 's' : ''}
                  </span>
                )}

                {/* Feedback — appears on hover desktop, always on mobile */}
                <div className="opacity-0 group-hover:opacity-100 transition-opacity lg:block">
                  <FeedbackButtons
                    messageId={message.id}
                    current={message.feedback}
                    onFeedback={onFeedback}
                  />
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

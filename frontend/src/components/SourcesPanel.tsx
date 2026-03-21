'use client'

import { useState } from 'react'
import { Source } from '@/types'
import clsx from 'clsx'

interface Props {
  sources:   Source[]
  isLoading: boolean
}

function SkeletonCard() {
  return (
    <div className="rounded-xl border border-border dark:border-border-dark p-3.5 space-y-2 overflow-hidden">
      <div className="h-3 w-3/4 rounded bg-gradient-to-r from-stone-200 via-stone-100 to-stone-200 dark:from-zinc-700 dark:via-zinc-600 dark:to-zinc-700 animate-shimmer bg-[length:200%_100%]" />
      <div className="h-2.5 w-1/2 rounded bg-gradient-to-r from-stone-200 via-stone-100 to-stone-200 dark:from-zinc-700 dark:via-zinc-600 dark:to-zinc-700 animate-shimmer bg-[length:200%_100%]" />
      <div className="h-2 w-full rounded bg-gradient-to-r from-stone-200 via-stone-100 to-stone-200 dark:from-zinc-700 dark:via-zinc-600 dark:to-zinc-700 animate-shimmer bg-[length:200%_100%]" />
    </div>
  )
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const [expanded, setExpanded] = useState(false)
  const pct = Math.round(source.similarity * 100)

  return (
    <div
      className="rounded-xl border border-border dark:border-border-dark bg-panel dark:bg-panel-dark p-3.5 space-y-2 hover:border-accent/40 dark:hover:border-accent-light/30 transition-colors animate-slide-up cursor-pointer"
      style={{ animationDelay: `${index * 60}ms` }}
      onClick={() => setExpanded(e => !e)}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-ink-DEFAULT dark:text-zinc-100 truncate leading-tight">
            {source.document_name}
          </p>
          {source.section_title && (
            <p className="text-[11px] text-ink-muted dark:text-zinc-400 truncate mt-0.5">
              {source.section_title}
            </p>
          )}
        </div>
        {/* Similarity badge */}
        <span className={clsx(
          'shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded-full',
          pct >= 85
            ? 'bg-accent-muted text-accent dark:bg-accent/20 dark:text-accent-light'
            : pct >= 70
              ? 'bg-amber-50 text-amber-600 dark:bg-amber-900/20 dark:text-amber-400'
              : 'bg-stone-100 text-ink-muted dark:bg-zinc-800 dark:text-zinc-400'
        )}>
          {pct}%
        </span>
      </div>

      {/* Page info */}
      {(source.page_start != null) && (
        <p className="text-[10px] text-ink-faint dark:text-zinc-500 font-mono">
          p. {source.page_start}{source.page_end && source.page_end !== source.page_start ? `–${source.page_end}` : ''}
        </p>
      )}

      {/* Snippet — expandable */}
      {source.snippet && (
        <div>
          <p className={clsx(
            'text-[11px] text-ink-muted dark:text-zinc-400 leading-relaxed',
            !expanded && 'line-clamp-2'
          )}>
            {source.snippet}
          </p>
          {source.snippet.length > 100 && (
            <button className="text-[10px] text-accent dark:text-accent-light mt-0.5 hover:underline">
              {expanded ? 'less' : 'more'}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export function SourcesPanel({ sources, isLoading }: Props) {
  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 pt-4 pb-3 border-b border-border dark:border-border-dark">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-ink-muted dark:text-zinc-400">
            Sources
          </h2>
          {sources.length > 0 && (
            <span className="text-[10px] font-mono text-ink-faint dark:text-zinc-600">
              {sources.length} found
            </span>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {isLoading ? (
          // Skeleton cards while loading
          <>
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </>
        ) : sources.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center pt-12">
            <div className="w-10 h-10 rounded-full bg-stone-100 dark:bg-zinc-800 flex items-center justify-center mb-3">
              <svg className="w-5 h-5 text-ink-faint dark:text-zinc-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
              </svg>
            </div>
            <p className="text-xs text-ink-faint dark:text-zinc-600">
              No sources available for this answer yet.
            </p>
          </div>
        ) : (
          sources.map((s, i) => (
            <SourceCard key={i} source={s} index={i} />
          ))
        )}
      </div>
    </div>
  )
}

'use client'

import { useStats } from '@/hooks/useStats'

export function SystemInfoCard() {
  const stats = useStats()

  return (
    <div className="bg-panel dark:bg-panel-dark border border-border dark:border-border-dark rounded-xl p-4 space-y-4">
      {/* Header */}
      <div>
        <p className="text-[11px] uppercase tracking-widest text-ink-faint dark:text-zinc-500 mb-2">
          IOTA KSA
        </p>
        <div className="flex items-center gap-2.5">
          {/* Logo */}
          <img src="/logo.jpg" alt="IOTA Logo" className="h-10 w-auto shrink-0" />
          <div>
            <h2 className="text-xl font-bold font-display text-ink-DEFAULT dark:text-zinc-100 leading-tight">
              Regulation AI
            </h2>
          </div>
        </div>
        <p className="text-[12px] text-ink-muted dark:text-zinc-400 mt-1.5 leading-snug">
          AI answers with citations from SAMA rulebooks and NCA controls.
        </p>
      </div>

      {/* Status pills */}
      <div className="flex flex-wrap gap-1.5">
        {/* API status */}
        <span className="flex items-center gap-1 text-[11px] px-2.5 py-1 rounded-full bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 font-medium">
          <span className="w-1.5 h-1.5 rounded-full bg-green-500 dark:bg-green-400" />
          {stats?.api_status === 'ok' ? 'Live · API healthy' : 'Connecting…'}
        </span>

        {/* Model */}
        <span className="text-[11px] px-2.5 py-1 rounded-full bg-stone-100 dark:bg-zinc-800 text-ink-muted dark:text-zinc-400 font-medium">
          {stats?.model
            ? stats.model.replace('gpt-4o-mini', 'GPT-4o-Mini').replace('gpt-4o', 'GPT-4o')
            : 'GPT-4o-Mini'}
        </span>

        {/* Stack pills */}
        {['PGVector', 'Hybrid RAG', 'Redis Cache'].map(label => (
          <span key={label} className="text-[11px] px-2.5 py-1 rounded-full bg-stone-100 dark:bg-zinc-800 text-ink-muted dark:text-zinc-400">
            {label}
          </span>
        ))}
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3 pt-1">
        <StatBox label="DOCS INGESTED"   value={stats?.docs_ingested}      />
        <StatBox label="CHUNKS"          value={stats?.total_chunks}        />
        <StatBox label="CACHED ANSWERS"  value={stats?.cached_answers}      />
        <StatBox
          label="CACHE HIT RATE"
          value={stats ? `${stats.cache_hit_rate_pct}%` : undefined}
        />
      </div>
    </div>
  )
}

function StatBox({ label, value }: { label: string; value?: number | string }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-widest text-ink-faint dark:text-zinc-600 mb-0.5">
        {label}
      </p>
      <p className="text-2xl font-bold text-ink-DEFAULT dark:text-zinc-100 font-display">
        {value === undefined || value === null ? (
          <span className="text-base text-ink-faint dark:text-zinc-600">—</span>
        ) : (
          typeof value === 'number' ? value.toLocaleString() : value
        )}
      </p>
    </div>
  )
}
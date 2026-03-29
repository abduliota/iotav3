'use client'

import { useDocuments } from '@/hooks/useDocuments'
import { Document } from '@/types'

export function DocumentLibrary() {
  const { documents, total, search, setSearch, loading } = useDocuments()

  return (
    <div className="bg-panel dark:bg-panel-dark border border-border dark:border-border-dark rounded-xl flex flex-col overflow-hidden" style={{ maxHeight: '420px' }}>
      {/* Header */}
      <div className="px-4 pt-4 pb-3 shrink-0">
        <p className="text-[11px] uppercase tracking-widest text-ink-faint dark:text-zinc-500 mb-3">
          INDEXED DOCUMENTS
        </p>

        {/* Search input */}
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-border dark:border-border-dark bg-surface dark:bg-surface-dark focus-within:border-accent/50 transition-colors">
          <svg className="w-3.5 h-3.5 text-ink-faint dark:text-zinc-600 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
          </svg>
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search documents…"
            className="flex-1 bg-transparent text-xs text-ink-DEFAULT dark:text-zinc-200 placeholder:text-ink-faint dark:placeholder:text-zinc-600 focus:outline-none"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="text-ink-faint dark:text-zinc-600 hover:text-ink-DEFAULT dark:hover:text-zinc-400"
            >
              <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-1">
        {loading ? (
          // Skeleton
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-12 rounded-lg bg-stone-100 dark:bg-zinc-800 animate-pulse" />
          ))
        ) : documents.length === 0 ? (
          <p className="text-xs text-ink-faint dark:text-zinc-600 text-center py-6">
            No documents found
          </p>
        ) : (
          documents.map((doc, i) => (
            <DocumentRow key={i} doc={doc} />
          ))
        )}
      </div>

      {/* Footer */}
      {!loading && (
        <div className="px-4 py-2 border-t border-border dark:border-border-dark shrink-0">
          <p className="text-[10px] text-ink-faint dark:text-zinc-600">
            Showing {documents.length} of {total} documents
          </p>
        </div>
      )}
    </div>
  )
}

function DocumentRow({ doc }: { doc: Document }) {
  const isSAMA = doc.source_type?.toUpperCase() === 'SAMA'
  const isNCA  = doc.source_type?.toUpperCase() === 'NCA'

  return (
    <div className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg hover:bg-stone-50 dark:hover:bg-zinc-800/60 transition-colors group">
      {/* Source badge */}
      <span className={`shrink-0 text-[9px] font-bold px-1.5 py-0.5 rounded font-mono ${
        isSAMA
          ? 'bg-accent-muted text-accent dark:bg-accent/20 dark:text-accent-light'
          : isNCA
            ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400'
            : 'bg-stone-100 text-ink-muted dark:bg-zinc-800 dark:text-zinc-400'
      }`}>
        {doc.source_type || 'DOC'}
      </span>

      {/* Name */}
      <p className="flex-1 min-w-0 text-[11px] text-ink-DEFAULT dark:text-zinc-200 truncate" title={doc.document_name}>
        {doc.document_name}
      </p>

      {/* Meta */}
      <div className="shrink-0 text-right">
        <p className="text-[9px] font-mono text-ink-faint dark:text-zinc-600 leading-tight">
          {doc.total_pages}p · {doc.chunk_count}c
        </p>
      </div>
    </div>
  )
}
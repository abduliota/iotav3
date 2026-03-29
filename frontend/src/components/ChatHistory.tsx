'use client'

import { Conversation } from '@/types'

interface Props {
  conversations:       Conversation[]
  activeSessionId:     string
  onSelect:            (sessionId: string) => void
  onNewChat:           () => void
}

function groupByDate(conversations: Conversation[]): Record<string, Conversation[]> {
  const now       = new Date()
  const today     = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const yesterday = new Date(today.getTime() - 86_400_000)
  const thisWeek  = new Date(today.getTime() - 6 * 86_400_000)

  const groups: Record<string, Conversation[]> = {
    'Today':     [],
    'Yesterday': [],
    'This Week': [],
    'Older':     [],
  }

  for (const c of conversations) {
    const d = new Date(c.last_message_at)
    const day = new Date(d.getFullYear(), d.getMonth(), d.getDate())
    if (day >= today)          groups['Today'].push(c)
    else if (day >= yesterday) groups['Yesterday'].push(c)
    else if (day >= thisWeek)  groups['This Week'].push(c)
    else                       groups['Older'].push(c)
  }

  return groups
}

export function ChatHistory({ conversations, activeSessionId, onSelect, onNewChat }: Props) {
  const groups = groupByDate(conversations)

  return (
    <div className="h-full flex flex-col border-r border-border dark:border-border-dark bg-panel dark:bg-panel-dark">
      {/* New chat button */}
      <div className="px-3 py-3 shrink-0">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg border border-border dark:border-border-dark text-xs text-ink-muted dark:text-zinc-400 hover:bg-stone-50 dark:hover:bg-zinc-800 hover:text-ink-DEFAULT dark:hover:text-zinc-200 transition-colors"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
          </svg>
          New chat
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto px-2 pb-3">
        {conversations.length === 0 ? (
          <p className="text-[11px] text-ink-faint dark:text-zinc-600 text-center pt-8 px-3">
            No previous conversations
          </p>
        ) : (
          Object.entries(groups).map(([label, items]) => {
            if (items.length === 0) return null
            return (
              <div key={label} className="mb-3">
                {/* Group label */}
                <p className="text-[10px] uppercase tracking-widest text-ink-faint dark:text-zinc-600 px-2 py-1.5">
                  {label}
                </p>
                {/* Items */}
                {items.map(c => (
                  <button
                    key={c.session_id}
                    onClick={() => onSelect(c.session_id)}
                    className={`w-full text-left px-2.5 py-2 rounded-lg text-[11px] leading-snug transition-colors mb-0.5 ${
                      c.session_id === activeSessionId
                        ? 'bg-accent-muted/50 dark:bg-accent/10 text-ink-DEFAULT dark:text-zinc-100 border-l-2 border-accent dark:border-accent-light pl-2'
                        : 'text-ink-muted dark:text-zinc-400 hover:bg-stone-50 dark:hover:bg-zinc-800 hover:text-ink-DEFAULT dark:hover:text-zinc-200'
                    }`}
                  >
                    <span className="block truncate">{c.title}</span>
                    <span className="text-[9px] text-ink-faint dark:text-zinc-600 mt-0.5 block">
                      {c.message_count} message{c.message_count !== 1 ? 's' : ''}
                    </span>
                  </button>
                ))}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
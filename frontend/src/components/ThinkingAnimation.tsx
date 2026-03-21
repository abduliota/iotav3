'use client'

export function ThinkingAnimation() {
  return (
    <div className="flex items-center gap-1.5 px-1 py-2">
      {[0, 1, 2].map(i => (
        <span
          key={i}
          className="w-2 h-2 rounded-full bg-accent-DEFAULT dark:bg-accent-light animate-pulse-dot"
          style={{ animationDelay: `${i * 0.16}s` }}
        />
      ))}
      <span className="ml-2 text-xs text-ink-faint dark:text-zinc-500 font-mono tracking-wide">
        searching regulations…
      </span>
    </div>
  )
}
